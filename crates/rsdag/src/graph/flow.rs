//! The one structural analysis: a value per node of a cone, a leaf's from
//! the caller, every other node's the join of its operands', through a call
//! the join of the arguments its output reads. Support, dependence,
//! feedthrough and the sparsity of a Jacobian are this with a different
//! leaf and a different way through calls.
//!
//! What an output of a function reads is found once per function, per way
//! through and per set of its parameters that carry a value (the moving
//! ones): a call that passes a value in a few arguments of hundreds asks
//! only about those few.

use std::rc::Rc;

use super::*;

/// The operands a value passes through, and how a call passes it on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Through {
    /// Every operand; of a call, the arguments its output reads.
    Reads,
    /// The operands a derivative reads (not a comparison's, not a
    /// selector's condition); of a call, the arguments its output's
    /// derivative carries.
    Carries,
}

/// A join-semilattice value of an analysis.
pub(crate) trait Join: Clone {
    fn bottom() -> Self;
    fn is_bottom(&self) -> bool;
    fn join(&mut self, other: &Self);
    /// The join of `parts`, at once (a set merges them in one pass, not
    /// one operand after the other).
    fn join_all<'a>(parts: impl Iterator<Item = &'a Self>) -> Self
    where
        Self: 'a,
    {
        let mut v = Self::bottom();
        for p in parts {
            v.join(p);
        }
        v
    }
}

impl Join for bool {
    fn bottom() -> bool {
        false
    }
    fn is_bottom(&self) -> bool {
        !*self
    }
    fn join(&mut self, other: &bool) {
        *self |= *other;
    }
}

/// A set of small integers: indices of a universe of at most
/// [`BITS_MAX`] as a bitset (a union an `or` over a few words), of a wider
/// one sorted. A node with one contributing operand shares that operand's.
#[derive(Clone, Default, PartialEq, Debug)]
pub(crate) struct Set(Option<Ids>);

#[derive(Clone, PartialEq, Debug)]
enum Ids {
    Bits(Rc<[u64]>),
    Sorted(Rc<[u32]>),
}

use crate::constants::BITS_MAX;

/// Members a word of a bitset holds.
const WORD: usize = u64::BITS as usize;

impl Set {
    /// `{k}`, of a universe of `n`.
    pub(crate) fn one(k: u32, n: usize) -> Set {
        if n <= BITS_MAX {
            let mut w = vec![0u64; k as usize / WORD + 1];
            w[k as usize / WORD] = 1 << (k % u64::BITS);
            Set(Some(Ids::Bits(w.into())))
        } else {
            Set(Some(Ids::Sorted(Rc::from([k]))))
        }
    }

    /// Whether `k` is a member.
    pub(crate) fn contains(&self, k: u32) -> bool {
        match &self.0 {
            None => false,
            Some(Ids::Bits(w)) => w
                .get(k as usize / WORD)
                .is_some_and(|&b| b >> (k % u64::BITS) & 1 == 1),
            Some(Ids::Sorted(v)) => v.binary_search(&k).is_ok(),
        }
    }

    /// The members, ascending.
    pub(crate) fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        let (bits, sorted): (&[u64], &[u32]) = match &self.0 {
            None => (&[], &[]),
            Some(Ids::Bits(w)) => (w, &[]),
            Some(Ids::Sorted(v)) => (&[], v),
        };
        let from_bits = bits.iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                (w != 0).then(|| {
                    let b = w.trailing_zeros();
                    w &= w - 1;
                    i as u32 * u64::BITS + b
                })
            })
        });
        from_bits.chain(sorted.iter().copied())
    }
}

impl Join for Set {
    fn join_all<'a>(parts: impl Iterator<Item = &'a Set>) -> Set {
        let mut first: Option<&Set> = None;
        let (mut bits, mut sorted): (Vec<u64>, Vec<u32>) = (Vec::new(), Vec::new());
        let add = |p: &Set, bits: &mut Vec<u64>, sorted: &mut Vec<u32>| match &p.0 {
            Some(Ids::Bits(w)) => {
                if bits.len() < w.len() {
                    bits.resize(w.len(), 0);
                }
                bits.iter_mut().zip(w.iter()).for_each(|(a, &b)| *a |= b);
            }
            Some(Ids::Sorted(v)) => sorted.extend_from_slice(v),
            None => {}
        };
        let mut many = false;
        for p in parts.filter(|p| !p.is_bottom()) {
            match first {
                None => first = Some(p),
                Some(f) => {
                    if !many {
                        add(f, &mut bits, &mut sorted);
                        many = true;
                    }
                    add(p, &mut bits, &mut sorted);
                }
            }
        }
        if !many {
            // none, or one set: shared as it is
            return first.cloned().unwrap_or_default();
        }
        if sorted.is_empty() {
            return Set(Some(Ids::Bits(bits.into())));
        }
        // one flow takes one kind; mixed, the bits join the list
        sorted.extend(Set(Some(Ids::Bits(bits.into()))).iter());
        sorted.sort_unstable();
        sorted.dedup();
        Set(Some(Ids::Sorted(sorted.into())))
    }
    fn bottom() -> Set {
        Set(None)
    }
    fn is_bottom(&self) -> bool {
        self.0.is_none()
    }
    fn join(&mut self, other: &Set) {
        if other.is_bottom() {
            return;
        }
        *self = Set::join_all([&*self, other].into_iter());
    }
}

/// Of a comparison its operands, of a selector its condition, carry no
/// derivative: the leading operands [`Through::Carries`] skips.
pub(crate) fn inert(node: &Node) -> usize {
    match node {
        Node::Cmp(..) => 2,
        Node::Select(..) => 1,
        _ => 0,
    }
}

/// The nodes under some roots in a topological order, each after its
/// operands, and each node's place in it (see [`Graph::cone`]).
pub(crate) struct Cone {
    pub(crate) nodes: Vec<ExprId>,
    at: NodeTable<u32>,
}

/// A node met but not finished in [`Graph::cone`]'s walk.
const OPEN: u32 = u32::MAX;

impl Cone {
    /// The place of a node of the cone.
    #[inline]
    pub(crate) fn pos(&self, e: ExprId) -> usize {
        self.at.get(e).expect("a node of the cone") as usize
    }
}

/// Values over a cone (see [`Graph::flow`]).
pub(crate) struct Flow<V> {
    cone: Cone,
    vals: Vec<V>,
}

impl<V> Flow<V> {
    /// The value of a node of the cone.
    pub(crate) fn get(&self, e: ExprId) -> &V {
        &self.vals[self.cone.pos(e)]
    }

    /// Every node of the cone with its value.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (ExprId, &V)> {
        self.cone.nodes.iter().copied().zip(&self.vals)
    }
}

impl<K: Field> Graph<K> {
    /// The cone of `roots`: the nodes under them, each once, in the order a
    /// recursive walk finishes them (operands in order, every argument list
    /// and every context once), a topological order. With `globals`, a
    /// call's globals as its operands too (what it reads, not only what it
    /// mentions).
    pub(crate) fn cone(&self, roots: &[ExprId], globals: bool) -> Cone {
        let mut at = NodeTable::new(self.len());
        let mut lists: FxHashSet<ArgList> = FxHashSet::default();
        let mut funcs: FxHashSet<FuncId> = FxHashSet::default();
        let mut contexts: FxHashSet<u32> = FxHashSet::default();
        let mut stack: Vec<(ExprId, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();
        let mut nodes = Vec::new();
        let mut ops: Vec<ExprId> = Vec::new();
        while let Some((e, done)) = stack.pop() {
            if done {
                at.set(e, nodes.len() as u32);
                nodes.push(e);
                continue;
            }
            if at.contains(e) {
                continue;
            }
            at.set(e, OPEN);
            stack.push((e, true));
            ops.clear();
            match *self.node(e) {
                Node::Call(o, l) => {
                    if lists.insert(l) {
                        ops.extend_from_slice(self.args(l));
                    }
                    let c = self.output_ctx[o.0 as usize];
                    if c != NO_CONTEXT && contexts.insert(c) {
                        ops.extend_from_slice(self.args(self.contexts[c as usize].exprs));
                    }
                    let f = self.output(o).0;
                    if globals && funcs.insert(f) {
                        ops.extend_from_slice(self.globals_of(f));
                    }
                }
                ref node => ops.extend_from_slice(&node.operands(&self.arg_pool)),
            }
            stack.extend(
                (ops.iter().rev())
                    .filter(|&&a| !at.contains(a))
                    .map(|&a| (a, false)),
            );
        }
        Cone { nodes, at }
    }

    /// The symbol nodes `f`'s body reads that are not its parameters, its
    /// calls' included, ascending: the globals every call of `f` reads
    /// besides its arguments. A call's operand `p` is its argument `p`, and
    /// past the arguments global `p - arity` (see
    /// [`operand_symbol`](Self::operand_symbol)).
    pub fn globals(&self, f: FuncId) -> Arc<[ExprId]> {
        self.funcs[f.0 as usize].globals_in(self)
    }

    /// The symbol a call of `f` binds by its operand `p`: parameter `p`, or
    /// past the parameters a global (see [`globals`](Self::globals)).
    pub fn operand_symbol(&self, f: FuncId, p: u32) -> SymbolId {
        let params = self.funcs[f.0 as usize].params();
        match params.get(p as usize) {
            Some(&s) => s,
            None => (self.as_symbol(self.globals(f)[p as usize - params.len()]))
                .expect("a global is a symbol"),
        }
    }

    /// Operand `p` of a call of output `o` over the list `l`: in its
    /// function's parameter order its argument or, where its context binds
    /// the parameter, the bound expression; past the parameters a global.
    pub fn call_operand(&self, o: OutputId, l: ArgList, p: u32) -> ExprId {
        let f = self.output(o).0;
        let n = self.funcs[f.0 as usize].params().len();
        if p as usize >= n {
            return self.globals_of(f)[p as usize - n];
        }
        let args = self.args(l);
        match self.output_ctx[o.0 as usize] {
            NO_CONTEXT => args[p as usize],
            c => {
                let c = &self.contexts[c as usize];
                c.pick(p as usize, args, self.args(c.exprs))
            }
        }
    }

    /// [`globals`](Self::globals), borrowed.
    pub(crate) fn globals_of(&self, f: FuncId) -> &[ExprId] {
        let func = &self.funcs[f.0 as usize];
        if func.globals.get().is_none() {
            func.globals_in(self);
        }
        func.globals.get().expect("just found")
    }

    /// Values over the cone of `roots`: a constant's and a symbol's from
    /// `leaf`, every other node's the join of its operands' as `through`
    /// passes them, a call's the join of the arguments its output reads
    /// (see [`reads`](Self::reads)).
    pub(crate) fn flow<V: Join>(
        &self,
        roots: &[ExprId],
        through: Through,
        leaf: impl Fn(&Node) -> V,
    ) -> Flow<V> {
        let cone = self.cone(roots, true);
        let mut vals: Vec<V> = Vec::with_capacity(cone.nodes.len());
        // per instance, what each output reads among its moving arguments
        let mut sites: HashMap<(FuncId, u32, ArgList), Arc<[Arc<[u32]>]>> = HashMap::default();
        for &e in &cone.nodes {
            let val = |c: &ExprId| &vals[cone.pos(*c)];
            let node = *self.node(e);
            let v = match node {
                Node::Const(_) | Node::Symbol(_) => leaf(&node),
                Node::Call(o, l) => {
                    let (f, k) = self.output(o);
                    let (args, globals) = (self.full_args(o, l), self.globals(f));
                    let operand = |p: u32| match args.get(p as usize) {
                        Some(a) => a,
                        None => &globals[p as usize - args.len()],
                    };
                    let site = (f, self.output_ctx[o.0 as usize], l);
                    let reads = match sites.get(&site) {
                        Some(r) => r.clone(),
                        None => {
                            let moving: Vec<u32> = (0..(args.len() + globals.len()) as u32)
                                .filter(|&p| !val(operand(p)).is_bottom())
                                .collect();
                            let r = self.reads(f, through, &moving);
                            sites.insert(site, r.clone());
                            r
                        }
                    };
                    let read = reads.get(k as usize).map_or(&[][..], |r| &r[..]);
                    V::join_all(read.iter().map(|&p| val(operand(p))))
                }
                _ => {
                    let ops = self.operands(e);
                    let skip = if through == Through::Carries {
                        inert(&node)
                    } else {
                        0
                    };
                    V::join_all(ops[skip..].iter().map(val))
                }
            };
            vals.push(v);
        }
        Flow { cone, vals }
    }

    /// Per output of `f`, the parameters among `moving` (indices, ascending)
    /// it reads through `through`, ascending. An extern output reads every
    /// one, a zero output none. Kept per function, way through and moving
    /// set; over all parameters with [`Through::Carries`] it is
    /// [`output_support`](Self::output_support).
    pub(crate) fn reads(&self, f: FuncId, through: Through, moving: &[u32]) -> Arc<[Arc<[u32]>]> {
        let func = &self.funcs[f.0 as usize];
        let outputs = func.outputs();
        let key = (through, Box::<[u32]>::from(moving));
        // outputs are only ever appended (derivatives): the known ones stay
        let known = func.cached_reads(&key);
        let done = known.as_ref().map_or(0, |r| r.len());
        if done == outputs.len() {
            return known.expect("all outputs known");
        }
        let index: HashMap<SymbolId, u32> = moving
            .iter()
            .map(|&p| (self.operand_symbol(f, p), p))
            .collect();
        let universe = moving.iter().max().map_or(0, |&p| p as usize + 1);
        let exprs: Vec<ExprId> = outputs[done..]
            .iter()
            .filter_map(|o| match *o {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let found = (!moving.is_empty() && !exprs.is_empty()).then(|| {
            self.flow(&exprs, through, |n| match *n {
                Node::Symbol(s) => index
                    .get(&s)
                    .map_or(Set::bottom(), |&p| Set::one(p, universe)),
                _ => Set::bottom(),
            })
        });
        let fresh = outputs[done..].iter().map(|o| match *o {
            Output::Zero => Arc::from([]),
            Output::Slot(_) => Arc::from(moving),
            Output::Expr(e) => found
                .as_ref()
                .map_or(Arc::from([]), |fl| fl.get(e).iter().collect()),
        });
        let r: Arc<[Arc<[u32]>]> = known
            .iter()
            .flat_map(|k| k.iter().cloned())
            .chain(fresh)
            .collect();
        func.cache_reads(key, r.clone());
        r
    }
}
