//! The one structural analysis: a value per node of a cone, a leaf's from
//! the caller, every other node's the join of its operands', through a call
//! the join of the arguments its output reads. Support, dependence and the
//! sparsity of a Jacobian are this with a different leaf and a different way
//! through calls.
//!
//! What an output of a function reads is found once per function, per way
//! through and per set of its parameters that carry a value (the moving
//! ones): a call that passes a value in a few arguments of hundreds asks
//! only about those few.

use std::rc::Rc;

use super::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Through {
    Reads,
    Carries,
}

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

#[derive(Clone, Default, PartialEq, Debug)]
pub(crate) struct Set(Option<Ids>);

#[derive(Clone, PartialEq, Debug)]
enum Ids {
    Bits(Rc<[u64]>),
    Sorted(Rc<[u32]>),
}

use crate::constants::BITS_MAX;

const BITS_PER_WORD: usize = u64::BITS as usize;

impl Set {
    pub(crate) fn one(k: u32, universe: usize) -> Set {
        if universe <= BITS_MAX {
            let mut w = vec![0u64; k as usize / BITS_PER_WORD + 1];
            w[k as usize / BITS_PER_WORD] = 1 << (k % u64::BITS);
            Set(Some(Ids::Bits(w.into())))
        } else {
            Set(Some(Ids::Sorted(Rc::from([k]))))
        }
    }

    pub(crate) fn contains(&self, k: u32) -> bool {
        match &self.0 {
            None => false,
            Some(Ids::Bits(w)) => w
                .get(k as usize / BITS_PER_WORD)
                .is_some_and(|&b| b >> (k % u64::BITS) & 1 == 1),
            Some(Ids::Sorted(v)) => v.binary_search(&k).is_ok(),
        }
    }

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
            return first.cloned().unwrap_or_default();
        }
        if sorted.is_empty() {
            return Set(Some(Ids::Bits(bits.into())));
        }
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

pub(crate) fn leading_operands_without_derivative(node: &Node) -> usize {
    match node {
        Node::Cmp(..) => 2,
        Node::Select(..) => 1,
        _ => 0,
    }
}

pub(crate) struct Cone {
    pub(crate) nodes: Pooled<Vec<ExprId>>,
    at: NodeTable<u32>,
}

#[derive(Default)]
struct Walk {
    lists: FxHashSet<ArgList>,
    funcs: FxHashSet<FuncId>,
    contexts: FxHashSet<u32>,
    stack: Vec<(ExprId, bool)>,
    ops: Vec<ExprId>,
}

const VISITED_NOT_FINISHED: u32 = u32::MAX;

impl Cone {
    #[inline]
    pub(crate) fn pos(&self, e: ExprId) -> usize {
        self.at.get(e).expect("a node of the cone") as usize
    }
}

pub(crate) struct Flow<V> {
    cone: Cone,
    vals: Vec<V>,
}

impl<V> Flow<V> {
    pub(crate) fn get(&self, e: ExprId) -> &V {
        &self.vals[self.cone.pos(e)]
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (ExprId, &V)> {
        self.cone.nodes.iter().copied().zip(&self.vals)
    }
}

impl<K: Field> Graph<K> {
    pub(crate) fn cone(&self, roots: &[ExprId], call_globals_as_operands: bool) -> Cone {
        let mut at = NodeTable::new(self.len());
        let mut walk: Pooled<Walk> = Pooled::take_uncleared();
        let Walk {
            lists,
            funcs,
            contexts,
            stack,
            ops,
        } = &mut *walk;
        lists.clear();
        funcs.clear();
        contexts.clear();
        stack.clear();
        stack.extend(roots.iter().rev().map(|&r| (r, false)));
        let mut nodes: Pooled<Vec<ExprId>> = Pooled::take_uncleared();
        nodes.clear();
        while let Some((e, done)) = stack.pop() {
            if done {
                at.set(e, nodes.len() as u32);
                nodes.push(e);
                continue;
            }
            if at.contains(e) {
                continue;
            }
            at.set(e, VISITED_NOT_FINISHED);
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
                    if call_globals_as_operands && funcs.insert(f) {
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
    /// past the arguments global `p - arity`.
    pub fn globals(&self, f: FuncId) -> Arc<[ExprId]> {
        self.funcs[f.0 as usize].globals_in(self).exprs.clone()
    }

    pub(crate) fn operand_symbol(&self, f: FuncId, p: u32) -> SymbolId {
        let params = self.funcs[f.0 as usize].params();
        match params.get(p as usize) {
            Some(&s) => s,
            None => self.funcs[f.0 as usize].globals_in(self).symbols[p as usize - params.len()],
        }
    }

    pub(crate) fn call_operand(&self, o: OutputId, l: ArgList, p: u32) -> ExprId {
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

    pub(crate) fn globals_of(&self, f: FuncId) -> &[ExprId] {
        &self.funcs[f.0 as usize].globals_in(self).exprs
    }

    pub(crate) fn flow<V: Join>(
        &self,
        roots: &[ExprId],
        through: Through,
        leaf: impl Fn(&Node) -> V,
    ) -> Flow<V> {
        let cone = self.cone(roots, true);
        let mut vals: Vec<V> = Vec::with_capacity(cone.nodes.len());
        let mut reads_per_instance: HashMap<(FuncId, u32, ArgList), Arc<[Arc<[u32]>]>> =
            HashMap::default();
        for &e in cone.nodes.iter() {
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
                    let reads = match reads_per_instance.get(&site) {
                        Some(r) => r.clone(),
                        None => {
                            let moving: Vec<u32> = (0..(args.len() + globals.len()) as u32)
                                .filter(|&p| !val(operand(p)).is_bottom())
                                .collect();
                            let r = self.reads(f, through, &moving);
                            reads_per_instance.insert(site, r.clone());
                            r
                        }
                    };
                    let read = reads.get(k as usize).map_or(&[][..], |r| &r[..]);
                    V::join_all(read.iter().map(|&p| val(operand(p))))
                }
                _ => {
                    let ops = self.operands(e);
                    let skip = if through == Through::Carries {
                        leading_operands_without_derivative(&node)
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

    pub(crate) fn reads(&self, f: FuncId, through: Through, moving: &[u32]) -> Arc<[Arc<[u32]>]> {
        let func = &self.funcs[f.0 as usize];
        let outputs = func.outputs();
        let key = (through, Box::<[u32]>::from(moving));
        let known = func.cached_reads(&key);
        let done = known.as_ref().map_or(0, |r| r.len());
        debug_assert!(
            done <= outputs.len(),
            "outputs are only appended, so the cached reads cover a prefix of them"
        );
        if let Some(all) = known.as_ref().filter(|r| r.len() == outputs.len()) {
            return all.clone();
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
