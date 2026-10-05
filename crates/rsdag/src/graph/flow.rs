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

use std::cell::RefCell;
use std::rc::Rc;

use super::*;

thread_local! {
    /// Position tables for the cones of this thread's analyses, reused.
    static MEMOS: RefCell<Vec<Memo>> = const { RefCell::new(Vec::new()) };
}

/// The operands a value passes through, and how a call passes it on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Through {
    /// Every operand; a call's arguments as written, its body unseen.
    Syntax,
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

/// A sorted set of small integers; a node with one contributing operand
/// shares that operand's.
#[derive(Clone, Default, PartialEq, Debug)]
pub(crate) struct Set(Option<Rc<[u32]>>);

impl Set {
    pub(crate) fn one(k: u32) -> Set {
        Set(Some(Rc::from([k])))
    }
    pub(crate) fn as_slice(&self) -> &[u32] {
        self.0.as_deref().unwrap_or(&[])
    }
}

impl Join for Set {
    fn join_all<'a>(parts: impl Iterator<Item = &'a Set>) -> Set {
        let mut first: Option<&Set> = None;
        let mut all: Vec<u32> = Vec::new();
        for p in parts.filter(|p| !p.is_bottom()) {
            match first {
                None => first = Some(p),
                Some(f) => {
                    if all.is_empty() {
                        all.extend_from_slice(f.as_slice());
                    }
                    all.extend_from_slice(p.as_slice());
                }
            }
        }
        if all.is_empty() {
            // none, or one set: shared as it is
            return first.cloned().unwrap_or_default();
        }
        all.sort_unstable();
        all.dedup();
        Set(Some(Rc::from(all)))
    }
    fn bottom() -> Set {
        Set(None)
    }
    fn is_bottom(&self) -> bool {
        self.as_slice().is_empty()
    }
    fn join(&mut self, other: &Set) {
        if other.is_bottom() {
            return;
        }
        if self.is_bottom() {
            *self = other.clone();
            return;
        }
        let (a, b) = (self.as_slice(), other.as_slice());
        let mut out = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    out.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    out.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&a[i..]);
        out.extend_from_slice(&b[j..]);
        if out.len() != a.len() {
            *self = Set(Some(Rc::from(out)));
        }
    }
}

/// Of a comparison its operands, of a selector its condition, carry no
/// derivative: the leading operands [`Through::Carries`] skips.
fn inert(node: &Node) -> usize {
    match node {
        Node::Cmp(..) => 2,
        Node::Select(..) => 1,
        _ => 0,
    }
}

/// Values over a cone (see [`Graph::flow`]).
pub(crate) struct Flow<V> {
    at: Memo,
    cone: Vec<ExprId>,
    vals: Vec<V>,
}

impl<V> Drop for Flow<V> {
    fn drop(&mut self) {
        let at = std::mem::take(&mut self.at);
        MEMOS.with(|m| m.borrow_mut().push(at));
    }
}

impl<V> Flow<V> {
    /// The value of a node of the cone.
    pub(crate) fn get(&self, e: ExprId) -> &V {
        &self.vals[self.at.get(e).expect("a node of the cone").0 as usize]
    }
    /// The cone, ascending (topological).
    pub(crate) fn cone(&self) -> &[ExprId] {
        &self.cone
    }
    /// A node's place in the cone, if it is in it.
    pub(crate) fn position(&self, e: ExprId) -> Option<usize> {
        self.at.get(e).map(|p| p.0 as usize)
    }
    /// The values, in the cone's order.
    pub(crate) fn values(&self) -> &[V] {
        &self.vals
    }
}

impl<K: Field> Graph<K> {
    /// The nodes under `roots`, every operand and every argument list once,
    /// in ascending id order (a topological one: a hash-consed node is
    /// newer than its operands), with each node's position in `at`.
    pub(crate) fn cone(&self, roots: &[ExprId], at: &mut Memo) -> Vec<ExprId> {
        let mut out = self.reach(roots, at);
        out.sort_unstable();
        for (k, &e) in out.iter().enumerate() {
            at.set(e, ExprId(k as u32));
        }
        out
    }

    /// The nodes under `roots`, every operand and every argument list once,
    /// in the order the walk meets them, marked in `at`.
    fn reach(&self, roots: &[ExprId], at: &mut Memo) -> Vec<ExprId> {
        at.begin(self.len());
        let mut lists: FxHashSet<ArgList> = FxHashSet::default();
        let mut stack: Vec<ExprId> = roots.to_vec();
        let mut out = Vec::new();
        while let Some(e) = stack.pop() {
            if at.get(e).is_some() {
                continue;
            }
            at.set(e, e);
            out.push(e);
            match *self.node(e) {
                Node::Call(_, l) => {
                    if lists.insert(l) {
                        stack.extend_from_slice(self.args(l));
                    }
                }
                _ => stack.extend_from_slice(&self.operands(e)),
            }
        }
        out
    }

    /// The nodes under `roots`, unordered (see [`cone`](Self::cone)).
    pub(crate) fn cone_nodes(&self, roots: &[ExprId]) -> Vec<ExprId> {
        let mut at = MEMOS.with(|m| m.borrow_mut().pop()).unwrap_or_default();
        let cone = self.reach(roots, &mut at);
        MEMOS.with(|m| m.borrow_mut().push(at));
        cone
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
        let mut at = MEMOS.with(|m| m.borrow_mut().pop()).unwrap_or_default();
        let cone = self.cone(roots, &mut at);
        let mut vals: Vec<V> = Vec::with_capacity(cone.len());
        // per instance, what each output reads among its moving arguments
        let mut sites: HashMap<(FuncId, ArgList), Arc<[Arc<[u32]>]>> = HashMap::default();
        for &e in &cone {
            let val = |c: &ExprId| &vals[at.get(*c).expect("in the cone").0 as usize];
            let node = *self.node(e);
            let v = match node {
                Node::Const(_) | Node::Symbol(_) => leaf(&node),
                Node::Call(o, l) if through != Through::Syntax => {
                    let (f, k) = self.output(o);
                    let args = self.args(l);
                    let reads = match sites.get(&(f, l)) {
                        Some(r) => r.clone(),
                        None => {
                            let moving: Vec<u32> = (0..args.len() as u32)
                                .filter(|&p| !val(&args[p as usize]).is_bottom())
                                .collect();
                            let r = self.reads(f, through, &moving);
                            sites.insert((f, l), r.clone());
                            r
                        }
                    };
                    let read = reads.get(k as usize).map_or(&[][..], |r| &r[..]);
                    V::join_all(read.iter().map(|&p| val(&args[p as usize])))
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
        Flow { at, cone, vals }
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
            .map(|&p| (func.params()[p as usize], p))
            .collect();
        let exprs: Vec<ExprId> = outputs[done..]
            .iter()
            .filter_map(|o| match *o {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let found = (!moving.is_empty() && !exprs.is_empty()).then(|| {
            self.flow(&exprs, through, |n| match *n {
                Node::Symbol(s) => index.get(&s).map_or(Set::bottom(), |&p| Set::one(p)),
                _ => Set::bottom(),
            })
        });
        let fresh = outputs[done..].iter().map(|o| match *o {
            Output::Zero => Arc::from([]),
            Output::Slot(_) => Arc::from(moving),
            Output::Expr(e) => found
                .as_ref()
                .map_or(Arc::from([]), |fl| Arc::from(fl.get(e).as_slice())),
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
