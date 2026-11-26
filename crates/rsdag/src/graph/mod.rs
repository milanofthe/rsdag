use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet};

use crate::field::{Field, F64};

use crate::constants::POWF_AS_POWI_MAX;
use crate::extern_fn::ExternBundle;
use crate::func::{FuncId, Function, Output, OutputId};
use crate::node::{
    ArgList, BinOp, CmpOp, ConstId, Elementary, ExprId, Node, Operands, ReduceOp, SymbolId, UnaryOp,
};
use crate::role::{OutputRole, ParamRole};
use crate::semantics::{binary_f64, unary_f64};

/// Owns the hash-consed symbolic DAG and the symbol table.
///
/// All expressions are built through this context. Smart constructors fold
/// constants and apply a handful of identities so trivially-equal expressions
/// collapse to the same [`ExprId`]. Heavier canonicalisation (factoring,
/// term collection) is left to a later rewrite layer.
///
/// Storage is three dense arenas plus their dedup indices: the 16-byte
/// [`Node`]s, the exact constants they reference by [`ConstId`], and one shared
/// operand pool that every variadic node windows into by [`ArgList`]. A node
/// is interned by hashing 16 bytes; a constant is hashed once when it is first
/// seen; an operand list is interned by content so equal lists share one
/// window (which is what makes `Reduce`/`Dot`/`Solve`/`Call` hash-cons structurally).
pub struct Graph<K: Field = F64> {
    nodes: Vec<Node>,
    shape: Vec<u64>,
    dedup: HashMap<Node, ExprId>,
    consts: Vec<K>,
    const_dedup: HashMap<K, ConstId>,
    const_by_f64_bits: HashMap<u64, ExprId>,
    arg_pool: Vec<ExprId>,
    arg_dedup: HashMap<Box<[ExprId]>, ArgList>,
    list_shape: HashMap<(u32, ArgList), u64>,
    zero: ExprId,
    one: ExprId,
    symbol_names: Vec<String>,
    symbol_ids: HashMap<String, SymbolId>,
    funcs: Vec<Function>,
    outputs: Vec<(FuncId, u32)>,
    output_ctx: Vec<u32>,
    output_dedup: HashMap<(FuncId, u32, u32), OutputId>,
    contexts: Vec<Context>,
    context_dedup: HashMap<(FuncId, Box<[u32]>, ArgList), u32>,
    rebound: HashMap<(FuncId, Vec<(SymbolId, ExprId)>), FuncId>,
    vjps: HashMap<(FuncId, Box<[u32]>, Box<[u32]>), FuncId>,
    defined: HashMap<FuncKey, FuncId>,
    pub(crate) templates: std::sync::Mutex<TemplateCache>,
}

pub(crate) type TemplateCache =
    HashMap<(u32, Vec<u32>, Option<Vec<bool>>), Arc<crate::tape::compile::Program>>;

#[derive(PartialEq, Eq, Hash)]
pub(crate) struct FuncKey {
    pub(crate) params: Box<[SymbolId]>,
    pub(crate) roles: Box<[ParamRole]>,
    pub(crate) outputs_at_definition: Box<[(Output, OutputRole)]>,
    pub(crate) body_address: Option<usize>,
}

pub(crate) const NO_CONTEXT: u32 = u32::MAX;

pub(crate) struct Context {
    pub(crate) f: FuncId,
    pub(crate) at: Box<[u32]>,
    pub(crate) exprs: ArgList,
    pub(crate) slot: Box<[u32]>,
}

pub(crate) const BOUND: u32 = 1 << 31;

impl Context {
    #[inline]
    pub(crate) fn pick<T: Copy>(&self, p: usize, args: &[T], bound: &[T]) -> T {
        match self.slot[p] {
            s if s & BOUND == 0 => args[s as usize],
            s => bound[(s & !BOUND) as usize],
        }
    }
}

/// What [`Graph::partial`] does with a parameter: bind it to an
/// expression in the function, or pass one per call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    Bind(ExprId),
    Pass(ExprId),
}

/// A function with parameters bound (see [`Graph::bind`]): what a call
/// through it runs, its arguments the parameters left.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Bound {
    pub func: FuncId,
    pub(crate) ctx: u32,
}

pub(crate) struct NodeTable<T: Slot> {
    stamp: Vec<u32>,
    val: Vec<u32>,
    cur: u32,
    kind: std::marker::PhantomData<T>,
}

pub(crate) trait Slot: Copy {
    fn pack(self) -> u32;
    fn unpack(v: u32) -> Self;
}

impl Slot for ExprId {
    fn pack(self) -> u32 {
        self.0
    }
    fn unpack(v: u32) -> Self {
        ExprId(v)
    }
}

impl Slot for u32 {
    fn pack(self) -> u32 {
        self
    }
    fn unpack(v: u32) -> Self {
        v
    }
}

type TableStore = (Vec<u32>, Vec<u32>, u32);

impl<T: Slot> NodeTable<T> {
    pub(crate) fn new(n: usize) -> Self {
        let (stamp, val, cur) = Pooled::<TableStore>::take_uncleared().into_inner();
        let mut t = NodeTable {
            stamp,
            val,
            cur,
            kind: std::marker::PhantomData,
        };
        t.begin(n);
        t
    }

    pub(crate) fn begin(&mut self, n: usize) {
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            self.stamp.iter_mut().for_each(|e| *e = 0);
            self.cur = 1;
        }
        if self.stamp.len() < n {
            self.stamp.resize(n, 0);
            self.val.resize(n, 0);
        }
    }

    #[inline]
    pub(crate) fn get(&self, e: ExprId) -> Option<T> {
        let k = e.0 as usize;
        (k < self.stamp.len() && self.stamp[k] == self.cur).then(|| T::unpack(self.val[k]))
    }

    #[inline]
    pub(crate) fn contains(&self, e: ExprId) -> bool {
        let k = e.0 as usize;
        k < self.stamp.len() && self.stamp[k] == self.cur
    }

    #[inline]
    pub(crate) fn set(&mut self, e: ExprId, v: T) {
        let k = e.0 as usize;
        if k < self.stamp.len() {
            self.stamp[k] = self.cur;
            self.val[k] = v.pack();
        }
    }
}

impl<T: Slot> Drop for NodeTable<T> {
    fn drop(&mut self) {
        let store = (
            std::mem::take(&mut self.stamp),
            std::mem::take(&mut self.val),
            self.cur,
        );
        Pooled::give(store);
    }
}

thread_local! {
    static POOL: std::cell::RefCell<Vec<Box<dyn std::any::Any>>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) struct Pooled<T: Default + 'static>(std::mem::ManuallyDrop<T>);

impl<T: Default + 'static> Pooled<T> {
    pub(crate) fn take_uncleared() -> Self {
        let found = POOL.with(|p| {
            let mut p = p.borrow_mut();
            let at = p.iter().rposition(|b| b.is::<T>())?;
            p.swap_remove(at).downcast::<T>().ok()
        });
        Pooled(std::mem::ManuallyDrop::new(
            found.map_or_else(T::default, |b| *b),
        ))
    }

    pub(crate) fn into_inner(self) -> T {
        let mut held = std::mem::ManuallyDrop::new(self);
        // SAFETY: `held` is never dropped, so its value is taken once.
        unsafe { std::mem::ManuallyDrop::take(&mut held.0) }
    }

    pub(crate) fn give(t: T) {
        drop(Pooled(std::mem::ManuallyDrop::new(t)));
    }
}

impl<T: Default + 'static> std::ops::Deref for Pooled<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Default + 'static> std::ops::DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Default + 'static> Drop for Pooled<T> {
    fn drop(&mut self) {
        // SAFETY: `drop` runs once and `self.0` is not used after it.
        let t = unsafe { std::mem::ManuallyDrop::take(&mut self.0) };
        POOL.with(|p| p.borrow_mut().push(Box::new(t)));
    }
}

mod calls;
mod compose;
mod flow;

pub(crate) use calls::Deps;
pub(crate) use flow::{non_diff_prefix, Join, Set, Through};

impl<K: Field> Default for Graph<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Field> Graph<K> {
    pub fn new() -> Self {
        let mut ctx = Graph {
            nodes: Vec::new(),
            shape: Vec::new(),
            dedup: HashMap::default(),
            consts: Vec::new(),
            const_dedup: HashMap::default(),
            const_by_f64_bits: HashMap::default(),
            arg_pool: Vec::new(),
            arg_dedup: HashMap::default(),
            list_shape: HashMap::default(),
            zero: ExprId(0),
            one: ExprId(0),
            symbol_names: Vec::new(),
            symbol_ids: HashMap::default(),
            funcs: Vec::new(),
            outputs: Vec::new(),
            output_ctx: Vec::new(),
            output_dedup: HashMap::default(),
            contexts: Vec::new(),
            context_dedup: HashMap::default(),
            rebound: HashMap::default(),
            vjps: HashMap::default(),
            defined: HashMap::default(),
            templates: Default::default(),
        };
        ctx.zero = ctx.konst(K::zero());
        ctx.one = ctx.konst(K::one());
        ctx
    }

    /// Number of distinct nodes currently interned (useful for sharing checks).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Borrow the node behind an id.
    #[inline]
    pub fn node(&self, id: ExprId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Name of a symbol.
    pub fn symbol_name(&self, s: SymbolId) -> &str {
        &self.symbol_names[s.0 as usize]
    }

    #[inline]
    pub(crate) fn const_val(&self, c: ConstId) -> &K {
        &self.consts[c.0 as usize]
    }

    /// The operand slice of an interned argument list.
    #[inline]
    pub fn args(&self, l: ArgList) -> &[ExprId] {
        &self.arg_pool[l.start as usize..(l.start + l.len) as usize]
    }

    #[inline]
    pub(crate) fn dot_args(&self, l: ArgList) -> (&[ExprId], &[ExprId]) {
        debug_assert_eq!(l.len() % 2, 0, "a dot list holds two equal halves");
        self.args(l).split_at(l.len() / 2)
    }

    /// The operands a node reads (leaves have none); of a call, in its
    /// function's parameter order: an argument, or where the call's
    /// context binds the parameter, the bound expression.
    #[inline]
    pub fn operands(&self, id: ExprId) -> Operands<'_> {
        match *self.node(id) {
            Node::Call(o, l) if self.output_ctx[o.0 as usize] != NO_CONTEXT => {
                Operands::Owned(self.full_args(o, l).into_owned())
            }
            ref node => node.operands(&self.arg_pool),
        }
    }

    #[inline]
    fn intern(&mut self, node: Node) -> ExprId {
        if let Some(&id) = self.dedup.get(&node) {
            return id;
        }
        let id = ExprId(self.nodes.len() as u32);
        let shape = self.shape_of(&node);
        self.nodes.push(node);
        self.shape.push(shape);
        self.dedup.insert(node, id);
        id
    }

    fn shape_of(&self, node: &Node) -> u64 {
        use crate::node::shape::{mix, of_hash, Tag};
        let f = |e: ExprId| self.shape[e.0 as usize];
        let list = |t: Tag, extra: u64, l: ArgList| {
            self.args(l)
                .iter()
                .fold(mix(t as u64, extra), |h, &a| mix(h, f(a)))
        };
        match *node {
            Node::Const(c) => mix(Tag::Const as u64, self.consts[c.0 as usize].stable_hash()),
            Node::Symbol(s) => mix(
                Tag::Symbol as u64,
                of_hash(self.symbol_names[s.0 as usize].as_str()),
            ),
            Node::Add(a, b) | Node::Mul(a, b) => {
                let (x, y) = (f(a).min(f(b)), f(a).max(f(b)));
                let t = if matches!(node, Node::Add(..)) {
                    Tag::Add
                } else {
                    Tag::Mul
                };
                mix(mix(t as u64, x), y)
            }
            Node::Neg(a) => mix(Tag::Neg as u64, f(a)),
            Node::Pow(a, n) => mix(mix(Tag::Pow as u64, n as u64), f(a)),
            Node::Unary(op, a) => mix(mix(Tag::Unary as u64, op.code() as u64), f(a)),
            Node::Binary(op, a, b) => {
                mix(mix(mix(Tag::Binary as u64, op.code() as u64), f(a)), f(b))
            }
            Node::Cmp(op, a, b) => mix(mix(mix(Tag::Cmp as u64, op as u64), f(a)), f(b)),
            Node::Select(c, t, e) => mix(mix(mix(Tag::Select as u64, f(c)), f(t)), f(e)),
            Node::Reduce(op, l) => list(Tag::Reduce, op as u64, l),
            Node::Dot(l) => list(Tag::Dot, 0, l),
            Node::Solve(l, k) => list(Tag::Solve, k as u64, l),
            Node::Call(o, l) => {
                let (func, k) = self.output(o);
                let callee = of_hash(self.func(func).name());
                let operands = self.list_shape[&(self.output_ctx[o.0 as usize], l)];
                mix(mix(Tag::Call as u64, mix(callee, k as u64)), operands)
            }
        }
    }

    pub(crate) fn intern_args(&mut self, args: &[ExprId]) -> ArgList {
        if let Some(&l) = self.arg_dedup.get(args) {
            return l;
        }
        let l = ArgList {
            start: self.arg_pool.len() as u32,
            len: args.len() as u32,
        };
        self.arg_pool.extend_from_slice(args);
        self.arg_dedup.insert(args.into(), l);
        let shape = args.iter().fold(0, |h, &a| {
            crate::node::shape::mix(h, self.shape[a.0 as usize])
        });
        self.list_shape.insert((NO_CONTEXT, l), shape);
        l
    }

    #[inline]
    pub(crate) fn const_of(&self, id: ExprId) -> Option<&K> {
        match *self.node(id) {
            Node::Const(c) => Some(self.const_val(c)),
            _ => None,
        }
    }

    /// Value of `id` as an `f64` when it is a constant node, else `None`.
    pub fn const_f64(&self, id: ExprId) -> Option<f64> {
        self.const_of(id).map(|r| r.to_f64())
    }

    /// True if `id` is the constant zero.
    #[inline]
    pub fn is_zero(&self, id: ExprId) -> bool {
        id == self.zero
    }

    #[inline]
    pub(crate) fn is_one(&self, id: ExprId) -> bool {
        id == self.one
    }

    pub(crate) fn konst(&mut self, r: K) -> ExprId {
        let c = match self.const_dedup.get(&r) {
            Some(&c) => c,
            None => {
                let c = ConstId(self.consts.len() as u32);
                self.consts.push(r.clone());
                self.const_dedup.insert(r, c);
                c
            }
        };
        self.intern(Node::Const(c))
    }

    pub fn konst_int(&mut self, n: i64) -> ExprId {
        match n {
            0 => self.zero,
            1 => self.one,
            _ => self.konst(K::from_i64(n)),
        }
    }

    pub(crate) fn ratio(&mut self, num: i64, den: i64) -> ExprId {
        self.konst(K::from_ratio(num, den))
    }

    /// Exact rational constant from an `f64` (e.g. a model parameter threshold).
    /// Non-finite values fall back to zero.
    pub fn konst_f64(&mut self, x: f64) -> ExprId {
        if let Some(&id) = self.const_by_f64_bits.get(&x.to_bits()) {
            return id;
        }
        let id = match K::from_f64(x) {
            Some(r) => self.konst(r),
            None => self.zero,
        };
        self.const_by_f64_bits.insert(x.to_bits(), id);
        id
    }

    #[inline]
    pub fn zero(&mut self) -> ExprId {
        self.zero
    }

    #[inline]
    pub fn one(&mut self) -> ExprId {
        self.one
    }

    /// The expression node for an existing symbol id.
    pub fn symbol_expr(&mut self, s: SymbolId) -> ExprId {
        self.intern(Node::Symbol(s))
    }

    /// The symbol named `name`, made if new.
    pub fn symbol_id(&mut self, name: &str) -> SymbolId {
        let e = self.sym(name);
        match *self.node(e) {
            Node::Symbol(s) => s,
            _ => unreachable!("sym yields a symbol"),
        }
    }

    pub(crate) fn fresh_symbol(&mut self, prefix: &str) -> SymbolId {
        let mut k = self.symbol_names.len();
        loop {
            let name = format!("{prefix}{k}");
            if !self.symbol_ids.contains_key(name.as_str()) {
                return self.symbol_id(&name);
            }
            k += 1;
        }
    }

    /// The symbol named `name`, if the graph has one.
    pub fn find_symbol(&self, name: &str) -> Option<SymbolId> {
        self.symbol_ids.get(name).copied()
    }

    /// The symbol `e` is, if it is one.
    pub fn as_symbol(&self, e: ExprId) -> Option<SymbolId> {
        match *self.node(e) {
            Node::Symbol(s) => Some(s),
            _ => None,
        }
    }

    /// The terms `e` sums: the operands of a sum, none for zero, else `e`
    /// itself.
    pub fn summands(&self, e: ExprId) -> Vec<ExprId> {
        match *self.node(e) {
            Node::Reduce(ReduceOp::Sum, l) => self.args(l).to_vec(),
            _ if self.is_zero(e) => Vec::new(),
            _ => vec![e],
        }
    }

    /// Look up or create a free symbol by name.
    pub fn sym(&mut self, name: &str) -> ExprId {
        let sid = if let Some(&sid) = self.symbol_ids.get(name) {
            sid
        } else {
            let sid = SymbolId(self.symbol_names.len() as u32);
            self.symbol_names.push(name.to_string());
            self.symbol_ids.insert(name.to_string(), sid);
            sid
        };
        self.intern(Node::Symbol(sid))
    }

    pub fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if a == self.zero {
            return b;
        }
        if b == self.zero {
            return a;
        }
        if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
            let v = x.add(y);
            return self.konst(v);
        }
        let (a, b) = commutative_order(a, b);
        self.intern(Node::Add(a, b))
    }

    pub fn sub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let nb = self.neg(b);
        self.add(a, nb)
    }

    pub fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        if a == self.zero || b == self.zero {
            return self.zero;
        }
        if a == self.one {
            return b;
        }
        if b == self.one {
            return a;
        }
        if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
            let v = x.mul(y);
            return self.konst(v);
        }
        let (a, b) = commutative_order(a, b);
        self.intern(Node::Mul(a, b))
    }

    pub fn neg(&mut self, a: ExprId) -> ExprId {
        if a == self.zero {
            return a;
        }
        if let Some(x) = self.const_of(a) {
            let v = x.neg();
            return self.konst(v);
        }
        if let Node::Neg(inner) = *self.node(a) {
            return inner;
        }
        self.intern(Node::Neg(a))
    }

    /// Integer power. Folds constants and collapses nested powers. A
    /// `Pow` node's exponent fits `i32` (what every backend evaluates);
    /// a wider one becomes a real power, `Powf` of the exponent's value.
    pub fn pow_i(&mut self, a: ExprId, n: i64) -> ExprId {
        if n == 0 {
            return self.one;
        }
        if n == 1 {
            return a;
        }
        if i32::try_from(n).is_err() {
            let e = self.konst_f64(n as f64);
            return self.binary(BinOp::Powf, a, e);
        }
        if let Some(x) = self.const_of(a) {
            if let Some(v) = x.powi(n) {
                return self.konst(v);
            }
        }
        if let Node::Pow(base, m) = *self.node(a) {
            if let Some(k) = m.checked_mul(n).filter(|&k| i32::try_from(k).is_ok()) {
                return self.pow_i(base, k);
            }
        }
        self.intern(Node::Pow(a, n))
    }

    pub fn recip(&mut self, a: ExprId) -> ExprId {
        self.pow_i(a, -1)
    }

    pub fn div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let rb = self.recip(b);
        self.mul(a, rb)
    }

    /// Python's `a // b`.
    pub fn floordiv(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let q = self.div(a, b);
        self.floor(q)
    }

    /// Python's floored modulo: `a - b * floor(a / b)`.
    pub fn floormod(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let f = self.floordiv(a, b);
        let bf = self.mul(b, f);
        self.sub(a, bf)
    }

    /// Apply a unary function, with a few exact-at-special-points identities.
    pub fn unary(&mut self, op: UnaryOp, a: ExprId) -> ExprId {
        if !K::is_exact() {
            if let Some(x) = self.const_of(a) {
                let y = unary_f64(op, x.to_f64());
                let in_domain = !y.is_nan();
                if in_domain {
                    if let Some(v) = K::from_f64(y) {
                        return self.konst(v);
                    }
                }
            }
        }
        if op == UnaryOp::Abs {
            match *self.node(a) {
                Node::Unary(UnaryOp::Abs | UnaryOp::Sqrt, _) => return a,
                Node::Neg(inner) => return self.unary(UnaryOp::Abs, inner),
                _ => {}
            }
        }
        match op {
            UnaryOp::Exp if self.is_zero(a) => self.one,
            UnaryOp::Ln if self.is_one(a) => self.zero,
            UnaryOp::Sin if self.is_zero(a) => self.zero,
            UnaryOp::Cos if self.is_zero(a) => self.one,
            UnaryOp::Sinh if self.is_zero(a) => self.zero,
            UnaryOp::Cosh if self.is_zero(a) => self.one,
            UnaryOp::Tanh if self.is_zero(a) => self.zero,
            UnaryOp::Sqrt if self.is_zero(a) => self.zero,
            UnaryOp::Sqrt if self.is_one(a) => self.one,
            _ => self.intern(Node::Unary(op, a)),
        }
    }

    /// The elementary function `f` of `args` (as many as its arity).
    pub fn elementary(&mut self, f: Elementary, args: &[ExprId]) -> ExprId {
        debug_assert_eq!(
            args.len(),
            f.arity(),
            "{} takes {} arguments",
            f.name(),
            f.arity()
        );
        match f {
            Elementary::Unary(op) => self.unary(op, args[0]),
            Elementary::Binary(op) => self.binary(op, args[0], args[1]),
            Elementary::Min => self.reduce(ReduceOp::Min, args[..2].to_vec()),
            Elementary::Max => self.reduce(ReduceOp::Max, args[..2].to_vec()),
        }
    }

    /// Binary function node (`Powf`, `Mod`, `Atan2`, `Hypot`): the powers of
    /// zero and one resolve; a floating field folds constants through the
    /// reference math.
    pub fn binary(&mut self, op: BinOp, a: ExprId, b: ExprId) -> ExprId {
        if op == BinOp::Powf {
            if self.is_zero(b) {
                return self.one;
            }
            if self.is_one(b) {
                return a;
            }
            if let Some(n) = self.const_of(b).and_then(small_integer) {
                return self.pow_i(a, n);
            }
        }
        if !K::is_exact() {
            if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
                let z = binary_f64(op, x.to_f64(), y.to_f64());
                let in_domain = !z.is_nan();
                if in_domain {
                    if let Some(v) = K::from_f64(z) {
                        return self.konst(v);
                    }
                }
            }
        }
        self.intern(Node::Binary(op, a, b))
    }

    pub fn exp(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Exp, a)
    }
    pub fn ln(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Ln, a)
    }
    pub fn sqrt(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Sqrt, a)
    }
    pub fn sin(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Sin, a)
    }
    pub fn cos(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Cos, a)
    }
    pub fn floor(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Floor, a)
    }
    pub(crate) fn sinh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Sinh, a)
    }
    pub(crate) fn cosh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Cosh, a)
    }
    pub fn tanh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Tanh, a)
    }

    /// Comparison node (`1.0`/`0.0`); folds when both operands are constant.
    pub fn cmp(&mut self, op: CmpOp, a: ExprId, b: ExprId) -> ExprId {
        if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
            return if cmp_field(op, x, y) {
                self.one
            } else {
                self.zero
            };
        }
        self.intern(Node::Cmp(op, a, b))
    }

    /// `cond != 0 ? then : else_`. Folds a constant condition and collapses
    /// equal branches.
    pub fn select(&mut self, cond: ExprId, then: ExprId, else_: ExprId) -> ExprId {
        if let Some(c) = self.const_of(cond) {
            return if c.is_zero() { else_ } else { then };
        }
        if then == else_ {
            return then;
        }
        self.intern(Node::Select(cond, then, else_))
    }

    /// Associative reduction over `args` (one fused node instead of an
    /// Add/Mul-tree). Commutative operands are sorted to maximise sharing; an
    /// empty/singleton list collapses. Constants are kept (exact) and combined
    /// at evaluation time.
    pub fn reduce(&mut self, op: ReduceOp, args: Vec<ExprId>) -> ExprId {
        let rest = match op {
            ReduceOp::Sum => {
                let mut acc: Option<K> = None;
                let mut rest = Vec::with_capacity(args.len());
                for a in args {
                    if a == self.zero {
                        continue;
                    }
                    match self.const_of(a) {
                        Some(r) => {
                            acc = Some(match acc {
                                Some(s) => s.add(r),
                                None => r.clone(),
                            })
                        }
                        None => rest.push(a),
                    }
                }
                if let Some(acc) = acc {
                    if !acc.is_zero() {
                        let k = self.konst(acc);
                        rest.push(k);
                    }
                }
                rest
            }
            ReduceOp::Product => {
                let mut acc: Option<K> = None;
                let mut rest = Vec::with_capacity(args.len());
                for a in args {
                    if a == self.zero {
                        return self.zero;
                    }
                    if a == self.one {
                        continue;
                    }
                    match self.const_of(a) {
                        Some(r) => {
                            acc = Some(match acc {
                                Some(s) => s.mul(r),
                                None => r.clone(),
                            })
                        }
                        None => rest.push(a),
                    }
                }
                if let Some(acc) = acc {
                    if acc.is_zero() {
                        return self.zero;
                    }
                    if !acc.is_one() {
                        let k = self.konst(acc);
                        rest.push(k);
                    }
                }
                rest
            }
            ReduceOp::Min | ReduceOp::Max => args,
        };
        self.finish_reduce(op, rest)
    }

    fn finish_reduce(&mut self, op: ReduceOp, mut args: Vec<ExprId>) -> ExprId {
        match args.len() {
            0 => match op {
                ReduceOp::Sum => self.zero,
                ReduceOp::Product => self.one,
                ReduceOp::Min | ReduceOp::Max => self.zero,
            },
            1 => args[0],
            _ => {
                let fingerprint_then_id = |a: &ExprId| (self.shape[a.0 as usize], a.0);
                args.sort_unstable_by_key(fingerprint_then_id);
                let l = self.intern_args(&args);
                self.intern(Node::Reduce(op, l))
            }
        }
    }

    /// Inner product `sum_i a[i]*b[i]`. The two lists must be equal length; an
    /// empty product is zero and a single pair is a plain multiply.
    pub fn dot(&mut self, a: Vec<ExprId>, b: Vec<ExprId>) -> ExprId {
        assert_eq!(a.len(), b.len(), "dot: operand lists differ in length");
        match a.len() {
            0 => self.zero,
            1 => self.mul(a[0], b[0]),
            _ => {
                let mut all = a;
                all.extend(b);
                let l = self.intern_args(&all);
                self.intern(Node::Dot(l))
            }
        }
    }

    /// The solution `x` of the dense system `A x = b`, one expression per
    /// component: `a` is `n * n` row-major, `b` of `n`. One unknown folds
    /// to a division; otherwise the components are one kernel (see
    /// [`Node::Solve`]).
    pub fn solve_dense(&mut self, a: Vec<ExprId>, b: Vec<ExprId>) -> Vec<ExprId> {
        let n = b.len();
        assert_eq!(
            a.len(),
            n * n,
            "solve: a square matrix over the right-hand side"
        );
        match n {
            0 => Vec::new(),
            1 => vec![self.div(b[0], a[0])],
            _ => {
                let mut all = a;
                all.extend(b);
                let l = self.intern_args(&all);
                (0..n as u32)
                    .map(|i| self.intern(Node::Solve(l, i)))
                    .collect()
            }
        }
    }

    pub(crate) fn solve_args(&self, l: ArgList) -> (usize, &[ExprId], &[ExprId]) {
        let n = Self::solve_n(l.len());
        let all = self.args(l);
        (n, &all[..n * n], &all[n * n..])
    }

    pub(crate) fn solve_n(len: usize) -> usize {
        let n = ((len as f64).sqrt()) as usize;
        debug_assert_eq!(n * n + n, len, "a solve list is n*n + n long");
        n
    }

    fn solve_component(&mut self, all: &[ExprId], i: u32) -> ExprId {
        let n = Self::solve_n(all.len());
        let (a, b) = all.split_at(n * n);
        self.solve_dense(a.to_vec(), b.to_vec())[i as usize]
    }

    pub(crate) fn nodes_slice(&self) -> &[Node] {
        &self.nodes
    }
    pub(crate) fn consts_slice(&self) -> &[K] {
        &self.consts
    }
    pub(crate) fn arg_pool_slice(&self) -> &[ExprId] {
        &self.arg_pool
    }
    pub(crate) fn funcs_slice(&self) -> &[Function] {
        &self.funcs
    }
    pub(crate) fn call_outputs_slice(&self) -> &[(FuncId, u32)] {
        &self.outputs
    }

    pub(crate) fn build(&mut self, node: Node, ops: &[ExprId]) -> ExprId {
        match node {
            Node::Const(c) => {
                debug_assert!(
                    (c.0 as usize) < self.consts.len(),
                    "a constant leaf names a constant of this graph"
                );
                self.intern(node)
            }
            Node::Symbol(s) => {
                debug_assert!(
                    (s.0 as usize) < self.symbol_names.len(),
                    "a symbol leaf names a symbol of this graph"
                );
                self.intern(node)
            }
            Node::Add(..) => self.add(ops[0], ops[1]),
            Node::Mul(..) => self.mul(ops[0], ops[1]),
            Node::Neg(_) => self.neg(ops[0]),
            Node::Pow(_, n) => self.pow_i(ops[0], n),
            Node::Unary(op, _) => self.unary(op, ops[0]),
            Node::Binary(op, ..) => self.binary(op, ops[0], ops[1]),
            Node::Cmp(op, ..) => self.cmp(op, ops[0], ops[1]),
            Node::Select(..) => self.select(ops[0], ops[1], ops[2]),
            Node::Reduce(op, _) => self.reduce(op, ops.to_vec()),
            Node::Dot(_) => {
                let (a, b) = ops.split_at(ops.len() / 2);
                self.dot(a.to_vec(), b.to_vec())
            }
            Node::Solve(_, i) => self.solve_component(ops, i),
            Node::Call(..) => unreachable!("a call is rebuilt over its instance"),
        }
    }

    pub(crate) fn intern_unfolded(&mut self, node: Node, ops: &[ExprId]) -> ExprId {
        let list = if node.is_variadic() {
            self.intern_args(ops)
        } else {
            ArgList { start: 0, len: 0 }
        };
        self.intern(node.with_operands(ops, list))
    }
}

fn commutative_order(a: ExprId, b: ExprId) -> (ExprId, ExprId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn cmp_field<K: Field>(op: CmpOp, x: &K, y: &K) -> bool {
    use std::cmp::Ordering::*;
    match (op, x.partial_cmp(y)) {
        (_, None) => op == CmpOp::Ne,
        (CmpOp::Gt, Some(o)) => o == Greater,
        (CmpOp::Ge, Some(o)) => o != Less,
        (CmpOp::Lt, Some(o)) => o == Less,
        (CmpOp::Le, Some(o)) => o != Greater,
        (CmpOp::Eq, Some(o)) => o == Equal,
        (CmpOp::Ne, Some(o)) => o != Equal,
    }
}

fn small_integer<K: Field>(k: &K) -> Option<i64> {
    let x = k.to_f64();
    if x.fract() == 0.0 && x.abs() <= POWF_AS_POWI_MAX && K::from_i64(x as i64) == *k {
        Some(x as i64)
    } else {
        None
    }
}
