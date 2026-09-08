use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet};

use num_rational::BigRational;

use crate::field::Field;

use crate::extern_fn::ExternBundle;
use crate::func::{CompiledBody, FuncId, Function, FunctionBody, Output, OutputId};
use crate::node::{
    binary_f64, unary_f64, ArgList, BinOp, CmpOp, ConstId, ExprId, Node, Operands, ReduceOp,
    SymbolId, UnaryOp,
};
use crate::role::{OutputRole, ParamRole};

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
/// window (which is what makes `Reduce`/`Dot`/`Opaque` hash-cons structurally).
pub struct Graph<K: Field = BigRational> {
    nodes: Vec<Node>,
    dedup: HashMap<Node, ExprId>,
    consts: Vec<K>,
    const_dedup: HashMap<K, ConstId>,
    /// `f64` bit pattern -> constant node, so a numeric literal that recurs
    /// (model thresholds, `EXP_LIMIT`, ...) skips the rational conversion.
    f64_cache: HashMap<u64, ExprId>,
    arg_pool: Vec<ExprId>,
    arg_dedup: HashMap<Box<[ExprId]>, ArgList>,
    /// The interned constants `0` and `1` (created in `new`), so identity
    /// folding is an id compare and `zero()`/`one()` never hash.
    zero: ExprId,
    one: ExprId,
    symbol_names: Vec<String>,
    symbol_ids: HashMap<String, SymbolId>,
    /// Functions (see [`crate::func`]) and the interned `(function, output)`
    /// pairs the `Call` nodes name.
    funcs: Vec<Function>,
    outputs: Vec<(FuncId, u32)>,
    output_dedup: HashMap<(FuncId, u32), OutputId>,
    /// Reusable per-node memo for the graph traversals (differentiation,
    /// substitution); see [`Memo`].
    memo: Option<Box<Memo>>,
}

/// A per-node memo table over the arena, cleared in O(1) by bumping an epoch:
/// the traversal primitives (`differentiate`, `substitute*`) key their memo by
/// `ExprId`, and on a hash-consed graph a build pass runs thousands of them
/// (one per stamp and port, one per instance root), each visiting a few
/// thousand nodes -- a hash map per call measured as the dominant cost. This
/// is two dense arrays and an epoch instead: a lookup is an index compare.
/// Keys are always nodes of the graph being traversed (ids below the arena
/// length at `begin`), so nodes created during the traversal never alias a
/// key.
#[derive(Default)]
pub struct Memo {
    epoch: Vec<u32>,
    val: Vec<ExprId>,
    cur: u32,
}

impl Memo {
    /// Start a fresh traversal over an arena of `n` nodes.
    pub fn begin(&mut self, n: usize) {
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            // epoch wrapped: invalidate everything explicitly
            self.epoch.iter_mut().for_each(|e| *e = 0);
            self.cur = 1;
        }
        if self.epoch.len() < n {
            self.epoch.resize(n, 0);
            self.val.resize(n, ExprId(0));
        }
    }
    #[inline]
    pub fn get(&self, e: ExprId) -> Option<ExprId> {
        let k = e.0 as usize;
        if k < self.epoch.len() && self.epoch[k] == self.cur {
            Some(self.val[k])
        } else {
            None
        }
    }
    #[inline]
    pub fn set(&mut self, e: ExprId, v: ExprId) {
        let k = e.0 as usize;
        if k < self.epoch.len() {
            self.epoch[k] = self.cur;
            self.val[k] = v;
        }
    }
}

impl<K: Field> Default for Graph<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Field> Graph<K> {
    pub fn new() -> Self {
        let mut ctx = Graph {
            nodes: Vec::new(),
            dedup: HashMap::default(),
            consts: Vec::new(),
            const_dedup: HashMap::default(),
            f64_cache: HashMap::default(),
            arg_pool: Vec::new(),
            arg_dedup: HashMap::default(),
            zero: ExprId(0),
            one: ExprId(0),
            symbol_names: Vec::new(),
            symbol_ids: HashMap::default(),
            funcs: Vec::new(),
            outputs: Vec::new(),
            output_dedup: HashMap::default(),
            memo: None,
        };
        ctx.zero = ctx.konst(K::zero());
        ctx.one = ctx.konst(K::one());
        ctx
    }

    /// Take the reusable traversal memo (a fresh one if it is in use by an
    /// enclosing traversal), already begun over the current arena.
    pub fn take_memo(&mut self) -> Box<Memo> {
        let mut m = self.memo.take().unwrap_or_default();
        m.begin(self.nodes.len());
        m
    }

    /// Return a memo taken with [`take_memo`](Self::take_memo).
    pub fn put_memo(&mut self, m: Box<Memo>) {
        self.memo = Some(m);
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

    /// The exact rational behind a constant id.
    #[inline]
    pub fn const_val(&self, c: ConstId) -> &K {
        &self.consts[c.0 as usize]
    }

    /// The operand slice of an interned argument list.
    #[inline]
    pub fn args(&self, l: ArgList) -> &[ExprId] {
        &self.arg_pool[l.start as usize..(l.start + l.len) as usize]
    }

    /// The two equal-length halves `(a, b)` of a `Dot` node's operand list.
    #[inline]
    pub fn dot_args(&self, l: ArgList) -> (&[ExprId], &[ExprId]) {
        self.args(l).split_at(l.len() / 2)
    }

    /// The operands a node reads, without allocation (leaves have none).
    #[inline]
    pub fn operands(&self, id: ExprId) -> Operands<'_> {
        let z = ExprId(0);
        match *self.node(id) {
            Node::Const(_) | Node::Symbol(_) => Operands::Inline { buf: [z; 3], n: 0 },
            Node::Add(a, b) | Node::Mul(a, b) | Node::Cmp(_, a, b) | Node::Binary(_, a, b) => {
                Operands::Inline {
                    buf: [a, b, z],
                    n: 2,
                }
            }
            Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => Operands::Inline {
                buf: [a, z, z],
                n: 1,
            },
            Node::Select(c, t, e) => Operands::Inline {
                buf: [c, t, e],
                n: 3,
            },
            Node::Reduce(_, l) | Node::Dot(l) | Node::Call(_, l) => Operands::Slice(self.args(l)),
        }
    }

    /// Intern a node, reusing an existing id if structurally identical.
    #[inline]
    fn intern(&mut self, node: Node) -> ExprId {
        if let Some(&id) = self.dedup.get(&node) {
            return id;
        }
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.dedup.insert(node, id);
        id
    }

    /// Intern an operand list by content.
    fn intern_args(&mut self, args: &[ExprId]) -> ArgList {
        if let Some(&l) = self.arg_dedup.get(args) {
            return l;
        }
        let l = ArgList {
            start: self.arg_pool.len() as u32,
            len: args.len() as u32,
        };
        self.arg_pool.extend_from_slice(args);
        self.arg_dedup.insert(args.into(), l);
        l
    }

    /// Borrow the rational value if `id` is a constant.
    #[inline]
    pub fn const_of(&self, id: ExprId) -> Option<&K> {
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

    /// True if `id` is the constant one.
    #[inline]
    pub fn is_one(&self, id: ExprId) -> bool {
        id == self.one
    }

    // --- leaf constructors -------------------------------------------------

    pub fn konst(&mut self, r: K) -> ExprId {
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

    pub fn ratio(&mut self, num: i64, den: i64) -> ExprId {
        self.konst(K::from_ratio(num, den))
    }

    /// Exact rational constant from an `f64` (e.g. a model parameter threshold).
    /// Non-finite values fall back to zero.
    pub fn konst_f64(&mut self, x: f64) -> ExprId {
        // Keyed on the bit pattern, so `0.0` and `-0.0` are distinct keys but
        // both intern to the rational zero (as `from_float` yields for either).
        if let Some(&id) = self.f64_cache.get(&x.to_bits()) {
            return id;
        }
        let id = match K::from_f64(x) {
            Some(r) => self.konst(r),
            None => self.zero,
        };
        self.f64_cache.insert(x.to_bits(), id);
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

    // --- algebraic constructors -------------------------------------------

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
        let (a, b) = order(a, b);
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
        let (a, b) = order(a, b);
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

    /// Integer power. Folds constants and collapses nested powers.
    pub fn pow_i(&mut self, a: ExprId, n: i64) -> ExprId {
        if n == 0 {
            return self.one;
        }
        if n == 1 {
            return a;
        }
        if let Some(x) = self.const_of(a) {
            // `0^(negative)` has no exact rational value (it is `inf` numerically).
            // Don't fold it -- keep the `Pow` node, so the tape evaluates it as
            // `0.powi(n)` = `inf` (matching the numeric path) instead of panicking
            // on a division by zero. This makes the constructor total, which the
            // parameter-fold transform relies on (folding a zero-valued parameter
            // that sits in a denominator must not crash).
            if let Some(v) = x.powi(n) {
                return self.konst(v);
            }
        }
        if let Node::Pow(base, m) = *self.node(a) {
            return self.pow_i(base, m * n);
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

    // --- elementary functions ---------------------------------------------

    /// Apply a unary function, with a few exact-at-special-points identities.
    pub fn unary(&mut self, op: UnaryOp, a: ExprId) -> ExprId {
        // A floating field folds through the reference math; an exact field
        // keeps transcendental constants symbolic.
        if !K::is_exact() {
            if let Some(x) = self.const_of(a) {
                let y = unary_f64(op, x.to_f64());
                // Outside the domain the op stays (a NaN payload is the
                // backend's business, not a constant's).
                if !y.is_nan() {
                    if let Some(v) = K::from_f64(y) {
                        return self.konst(v);
                    }
                }
            }
        }
        // `abs` of a magnitude or an `abs` is itself; of a negation, of the
        // operand (exact in IEEE: abs only clears the sign bit).
        if op == UnaryOp::Abs {
            match *self.node(a) {
                Node::Unary(UnaryOp::Abs | UnaryOp::Sqrt, _) => return a,
                Node::Neg(inner) => return self.unary(UnaryOp::Abs, inner),
                _ => {}
            }
        }
        match op {
            UnaryOp::Exp if self.is_zero(a) => self.one, // exp(0) = 1
            UnaryOp::Ln if self.is_one(a) => self.zero,  // ln(1) = 0
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
                if !z.is_nan() {
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
    pub fn sinh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Sinh, a)
    }
    pub fn cosh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Cosh, a)
    }
    pub fn tanh(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Tanh, a)
    }
    pub fn atan(&mut self, a: ExprId) -> ExprId {
        self.unary(UnaryOp::Atan, a)
    }

    // --- conditions and selection ----------------------------------------

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

    // --- fused / variadic operators --------------------------------------

    /// Associative reduction over `args` (one fused node instead of an
    /// Add/Mul-tree). Commutative operands are sorted to maximise sharing; an
    /// empty/singleton list collapses. Constants are kept (exact) and combined
    /// at evaluation time.
    pub fn reduce(&mut self, op: ReduceOp, args: Vec<ExprId>) -> ExprId {
        // Combine the constant operands exactly (keeping sparsity exact and the
        // tape short); a zero factor zeroes a product, matching `mul`.
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
                // No finite rational identity; empty min/max is not used.
                ReduceOp::Min | ReduceOp::Max => self.zero,
            },
            1 => args[0],
            _ => {
                args.sort_unstable();
                let l = self.intern_args(&args);
                self.intern(Node::Reduce(op, l))
            }
        }
    }

    /// Inner product `Σ_i a[i]*b[i]`. The two lists must be equal length; an
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

    // --- module interchange (see `crate::module`) ------------------------

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

    /// Re-intern one node of a module in this graph, mapping its operand,
    /// constant, symbol and call ids through `map` and the module's tables.
    ///
    /// The raw interner, not the smart constructors: a module's nodes are
    /// already in the folded, canonical form the constructors produce, so
    /// re-running them would be work without an effect, and going through
    /// the interner reproduces the module exactly.
    pub(crate) fn rebuild_node(
        &mut self,
        node: &Node,
        map: &crate::module::IdMap,
        arg_pool: &[ExprId],
        consts: &[K],
        call_outputs: &rustc_hash::FxHashMap<OutputId, (FuncId, u32)>,
    ) -> ExprId {
        let e = |m: &crate::module::IdMap, x: ExprId| m.exprs[x.0 as usize];
        let list = |g: &mut Self, m: &crate::module::IdMap, l: ArgList| -> ArgList {
            let args: Vec<ExprId> = arg_pool[l.start as usize..(l.start + l.len) as usize]
                .iter()
                .map(|&x| e(m, x))
                .collect();
            g.intern_args(&args)
        };
        let n = match *node {
            Node::Const(c) => return self.konst(consts[c.0 as usize].clone()),
            Node::Symbol(s) => Node::Symbol(map.symbols[s.0 as usize]),
            Node::Add(a, b) => Node::Add(e(map, a), e(map, b)),
            Node::Mul(a, b) => Node::Mul(e(map, a), e(map, b)),
            Node::Neg(a) => Node::Neg(e(map, a)),
            Node::Pow(a, k) => Node::Pow(e(map, a), k),
            Node::Unary(op, a) => Node::Unary(op, e(map, a)),
            Node::Binary(op, a, b) => Node::Binary(op, e(map, a), e(map, b)),
            Node::Cmp(op, a, b) => Node::Cmp(op, e(map, a), e(map, b)),
            Node::Select(c, t, f) => Node::Select(e(map, c), e(map, t), e(map, f)),
            Node::Reduce(op, l) => Node::Reduce(op, list(self, map, l)),
            Node::Dot(l) => Node::Dot(list(self, map, l)),
            Node::Call(out, l) => {
                let (f, k) = call_outputs[&out];
                let f = map.funcs[f.0 as usize];
                let args: Vec<ExprId> = arg_pool[l.start as usize..(l.start + l.len) as usize]
                    .iter()
                    .map(|&x| e(map, x))
                    .collect();
                return self.call(f, k, &args);
            }
        };
        self.intern(n)
    }

    // --- functions and calls ---------------------------------------------

    /// Define a symbolic function: `outputs` are expressions over the formal
    /// `params` (free symbols of the outputs not listed in `params` are shared
    /// globals every call sees unchanged).
    pub fn define_func(
        &mut self,
        name: &str,
        params: Vec<SymbolId>,
        outputs: Vec<ExprId>,
    ) -> FuncId {
        let id = FuncId(self.funcs.len() as u32);
        let n_par = params.len();
        let n_out = outputs.len();
        self.funcs.push(Function {
            name: name.to_string(),
            params,
            param_roles: vec![ParamRole::Free; n_par],
            outputs: outputs.into_iter().map(Output::Expr).collect(),
            output_roles: vec![OutputRole::Plain; n_out],
            body: FunctionBody::Symbolic,
            deriv_index: HashMap::default(),
            compiled: None,
        });
        id
    }

    /// Close an open graph over `outputs` into a function: every free symbol
    /// the outputs depend on becomes a parameter, in symbol order. The
    /// `Scope` idiom: build with named symbols, then close.
    pub fn close(&mut self, name: &str, outputs: Vec<ExprId>) -> FuncId {
        let params: Vec<SymbolId> = self.free_symbols_in(&outputs).into_iter().collect();
        self.define_func(name, params, outputs)
    }

    /// Set the role of parameter `param` of function `f`.
    pub fn set_param_role(&mut self, f: FuncId, param: u32, role: ParamRole) {
        self.funcs[f.0 as usize].param_roles[param as usize] = role;
    }

    /// Set the role of output `out` of function `f`.
    pub fn set_output_role(&mut self, f: FuncId, out: u32, role: OutputRole) {
        self.funcs[f.0 as usize].output_roles[out as usize] = role;
    }

    /// Jacobian of the outputs of `f` with a role against its parameters with
    /// a role: `(output index, param index, derivative output index)` for
    /// every structurally nonzero pair, differentiating on demand.
    pub fn jacobian_by_role(
        &mut self,
        f: FuncId,
        out_role: impl Fn(&OutputRole) -> bool,
        param_role: impl Fn(&ParamRole) -> bool,
    ) -> Vec<(u32, u32, u32)> {
        let outs = self.funcs[f.0 as usize].outputs_with_role(out_role);
        let pars = self.funcs[f.0 as usize].params_with_role(param_role);
        let mut entries = Vec::new();
        for &o in &outs {
            for &p in &pars {
                let k = self.derivative_output(f, o, p);
                if !matches!(self.funcs[f.0 as usize].outputs[k as usize], Output::Zero) {
                    entries.push((o, p, k));
                }
            }
        }
        entries
    }

    /// Define an extern function over `arity` arguments whose outputs are the
    /// given slots of `body` (or [`Output::Zero`]). Derivative outputs the body
    /// carries are declared with [`declare_derivative`](Self::declare_derivative).
    pub fn define_extern_func(
        &mut self,
        name: &str,
        arity: usize,
        body: Arc<dyn ExternBundle>,
        outputs: Vec<Output>,
    ) -> FuncId {
        let params: Vec<SymbolId> = (0..arity)
            .map(|i| {
                let e = self.sym(&format!("{name}.${i}"));
                match *self.node(e) {
                    Node::Symbol(s) => s,
                    _ => unreachable!("sym yields a symbol"),
                }
            })
            .collect();
        let id = FuncId(self.funcs.len() as u32);
        let n_out = outputs.len();
        self.funcs.push(Function {
            name: name.to_string(),
            param_roles: vec![ParamRole::Free; params.len()],
            params,
            outputs,
            output_roles: vec![OutputRole::Plain; n_out],
            body: FunctionBody::Extern(body),
            deriv_index: HashMap::default(),
            compiled: None,
        });
        id
    }

    /// Declare `d outputs[out] / d params[param]` of an extern function as
    /// `deriv` (a slot of its body, or zero). Undeclared derivatives are zero.
    pub fn declare_derivative(&mut self, f: FuncId, out: u32, param: u32, deriv: Output) -> u32 {
        let func = &mut self.funcs[f.0 as usize];
        let k = func.outputs.len() as u32;
        func.outputs.push(deriv);
        func.output_roles.push(OutputRole::Derivative {
            of: out,
            wrt: param,
        });
        func.deriv_index.insert((out, param), k);
        k
    }

    pub fn func(&self, f: FuncId) -> &Function {
        &self.funcs[f.0 as usize]
    }

    /// Number of symbols.
    pub fn n_symbols(&self) -> usize {
        self.symbol_names.len()
    }

    /// Mutable access to a function (roles, memoised outputs).
    pub fn func_mut(&mut self, f: FuncId) -> &mut Function {
        &mut self.funcs[f.0 as usize]
    }

    /// Define an extern function over the given formal parameters (the
    /// symbols already exist), see [`define_extern_func`](Self::define_extern_func).
    pub fn define_extern_func_with_params(
        &mut self,
        name: &str,
        params: Vec<SymbolId>,
        body: Arc<dyn ExternBundle>,
        outputs: Vec<Output>,
    ) -> FuncId {
        let id = FuncId(self.funcs.len() as u32);
        let n_out = outputs.len();
        self.funcs.push(Function {
            name: name.to_string(),
            param_roles: vec![ParamRole::Free; params.len()],
            params,
            outputs,
            output_roles: vec![OutputRole::Plain; n_out],
            body: FunctionBody::Extern(body),
            deriv_index: HashMap::default(),
            compiled: None,
        });
        id
    }

    pub fn n_funcs(&self) -> usize {
        self.funcs.len()
    }

    /// Register the compiled body of a symbolic function (a solver's tape over
    /// the function's parameters), replacing any earlier one.
    pub fn set_func_body(&mut self, f: FuncId, body: CompiledBody) {
        self.funcs[f.0 as usize].compiled = Some(body);
    }

    /// The `(function, output index)` an output id names.
    #[inline]
    pub fn output(&self, o: OutputId) -> (FuncId, u32) {
        self.outputs[o.0 as usize]
    }

    /// The interned id of output `out` of `f`.
    pub fn output_id(&mut self, f: FuncId, out: u32) -> OutputId {
        if let Some(&o) = self.output_dedup.get(&(f, out)) {
            return o;
        }
        let o = OutputId(self.outputs.len() as u32);
        self.outputs.push((f, out));
        self.output_dedup.insert((f, out), o);
        o
    }

    /// The expression of a symbolic output, `None` for a slot or zero output.
    pub fn output_expr(&self, f: FuncId, out: u32) -> Option<ExprId> {
        match self.funcs[f.0 as usize].outputs[out as usize] {
            Output::Expr(e) => Some(e),
            _ => None,
        }
    }

    /// Output `out` of `f` applied to `args` (one argument per parameter). A
    /// zero output folds to the constant zero.
    pub fn call(&mut self, f: FuncId, out: u32, args: &[ExprId]) -> ExprId {
        debug_assert_eq!(
            args.len(),
            self.funcs[f.0 as usize].params.len(),
            "call arity"
        );
        if matches!(self.funcs[f.0 as usize].outputs[out as usize], Output::Zero) {
            return self.zero;
        }
        let o = self.output_id(f, out);
        let l = self.intern_args(args);
        self.intern(Node::Call(o, l))
    }

    /// [`call`](Self::call) by output id.
    pub fn call_output(&mut self, o: OutputId, args: &[ExprId]) -> ExprId {
        let (f, out) = self.output(o);
        self.call(f, out, args)
    }

    /// The index of the derivative output `d outputs[out] / d params[param]`,
    /// differentiating the body on first demand (symbolic functions) or
    /// looking up the declared slot (extern functions; zero if undeclared).
    pub fn derivative_output(&mut self, f: FuncId, out: u32, param: u32) -> u32 {
        if let Some(&k) = self.funcs[f.0 as usize].deriv_index.get(&(out, param)) {
            return k;
        }
        let d = match self.funcs[f.0 as usize].outputs[out as usize] {
            Output::Expr(e) => {
                let wrt = self.funcs[f.0 as usize].params[param as usize];
                let de = crate::autodiff::differentiate(self, e, wrt);
                if self.is_zero(de) {
                    Output::Zero
                } else {
                    Output::Expr(de)
                }
            }
            Output::Slot(_) | Output::Zero => Output::Zero,
        };
        let func = &mut self.funcs[f.0 as usize];
        let k = func.outputs.len() as u32;
        func.outputs.push(d);
        func.output_roles.push(OutputRole::Derivative {
            of: out,
            wrt: param,
        });
        func.deriv_index.insert((out, param), k);
        k
    }

    /// Inline a call: the output expression with the parameters replaced by
    /// `args`. `None` for an extern (slot) output, which has no body to inline.
    pub fn inline_call(&mut self, f: FuncId, out: u32, args: &[ExprId]) -> Option<ExprId> {
        match self.funcs[f.0 as usize].outputs[out as usize] {
            Output::Expr(e) => {
                let params = self.funcs[f.0 as usize].params.clone();
                let map: HashMap<SymbolId, ExprId> =
                    params.iter().copied().zip(args.iter().copied()).collect();
                Some(crate::transform::substitute_many(self, e, &map))
            }
            Output::Zero => Some(self.zero),
            Output::Slot(_) => None,
        }
    }

    /// Inline several outputs of a symbolic function at once, with one shared
    /// substitution pass (the outputs of a device template share its core, so
    /// per-output substitution would rebuild that core per output).
    pub fn inline_outputs(&mut self, f: FuncId, outs: &[u32], args: &[ExprId]) -> Vec<ExprId> {
        let params = self.funcs[f.0 as usize].params.clone();
        let map: HashMap<SymbolId, ExprId> =
            params.iter().copied().zip(args.iter().copied()).collect();
        let exprs: Vec<ExprId> = outs
            .iter()
            .map(|&o| match self.funcs[f.0 as usize].outputs[o as usize] {
                Output::Expr(e) => e,
                Output::Zero => self.zero,
                Output::Slot(_) => panic!("cannot inline an extern function output"),
            })
            .collect();
        crate::transform::substitute_many_all(self, &exprs, &map)
    }

    /// Every `(function, output)` called anywhere in `exprs` (one pass over
    /// the forest). A solver uses it to compile exactly the outputs a set of
    /// roots reads.
    pub fn free_calls_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<OutputId> {
        let mut set = std::collections::BTreeSet::new();
        let mut visited = FxHashSet::default();
        let mut stack: Vec<ExprId> = exprs.to_vec();
        while let Some(e) = stack.pop() {
            if !visited.insert(e) {
                continue;
            }
            if let Node::Call(o, _) = *self.node(e) {
                set.insert(o);
            }
            stack.extend_from_slice(&self.operands(e));
        }
        set
    }

    /// The set of free symbols reachable from `expr`.
    ///
    /// Memoised over shared subexpressions (a `visited` set): in a hash-consed
    /// DAG a node may be reachable by exponentially many paths, so without this
    /// the traversal is super-linear in the node count. With it, each node is
    /// visited once -- O(nodes reachable from `expr`).
    pub fn free_symbols(&self, expr: ExprId) -> std::collections::BTreeSet<SymbolId> {
        self.free_symbols_in(&[expr])
    }

    /// Union of the free symbols across many expressions, sharing one `visited`
    /// set so a subexpression hash-consed into several of them is traversed once
    /// (a single pass over the forest, not one per expression).
    pub fn free_symbols_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        let mut set = std::collections::BTreeSet::new();
        let mut visited = FxHashSet::default();
        let mut stack: Vec<ExprId> = exprs.to_vec();
        while let Some(e) = stack.pop() {
            if !visited.insert(e) {
                continue;
            }
            match *self.node(e) {
                Node::Const(_) => {}
                Node::Symbol(s) => {
                    set.insert(s);
                }
                _ => stack.extend_from_slice(&self.operands(e)),
            }
        }
        set
    }
}

/// Canonical operand order for commutative ops, to maximise hash-consing.
fn order(a: ExprId, b: ExprId) -> (ExprId, ExprId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// A comparison of two constants of the field; an unordered pair (NaN in a
/// floating field) compares false except for `Ne`.
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

/// A constant that is a small integer (the exponent of a `Powf` that is
/// really an integer power).
fn small_integer<K: Field>(k: &K) -> Option<i64> {
    let x = k.to_f64();
    if x.fract() == 0.0 && x.abs() <= 64.0 && K::from_i64(x as i64) == *k {
        Some(x as i64)
    } else {
        None
    }
}
