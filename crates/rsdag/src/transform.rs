//! Structural rewrites of the DAG: the bottom-up rewrite every transform is
//! written in, and substituting expressions for symbols.

use std::borrow::Cow;

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::func::FuncId;
use crate::graph::{Graph, Pooled, NO_CONTEXT};
use crate::node::{ArgList, ExprId, Node, SymbolId};

/// A bottom-up rewrite (see [`rewrite`]): the graph it reads and what it
/// makes of each node.
pub(crate) trait Rule<K: Field> {
    /// What a node becomes: a node of a graph, a drawn node.
    type V: Copy + 'static;
    /// The graph rewritten.
    fn source(&self) -> &Graph<K>;
    /// Node `e`, no call, over its operands rewritten, in
    /// [`Graph::operands`] order.
    fn node(&mut self, e: ExprId, node: Node, ops: &[Self::V]) -> Self::V;
    /// The calls `outs` (node, output index) of one instance, in cone
    /// order, over the instance's operands rewritten: one value per call
    /// pushed to `vals`.
    fn calls(&mut self, call: &Call<Self::V>, outs: &[(ExprId, u32)], vals: &mut Vec<Self::V>);
}

/// An instance a rewrite meets: function `f` called in context `ctx` of
/// the source over one argument list, the list rewritten, and the context's
/// bound expressions, rewritten once however many instances share them.
pub(crate) struct Call<'a, V> {
    pub f: FuncId,
    pub ctx: u32,
    pub args: &'a [V],
    pub bound: &'a [V],
}

impl<V: Copy> Call<'_, V> {
    /// The operands rewritten, in the function's parameter order (see
    /// [`Graph::full_args`]).
    pub(crate) fn params<K: Field>(&self, g: &Graph<K>) -> Cow<'_, [V]> {
        g.in_param_order(self.ctx, self.args, self.bound)
    }
}

/// Rewrite the DAG under `roots` bottom-up, every node once, in the order
/// of [`Graph::cone`] (the order a recursive walk finishes the nodes): a
/// call's instance at its first call, with every call of the instance under
/// `roots`, so an instance of a wide body costs its width once and its
/// outputs share what the rule makes of it.
pub(crate) fn rewrite<K: Field, R: Rule<K>>(r: &mut R, roots: &[ExprId]) -> Vec<R::V> {
    let cone = r.source().cone(roots, false);
    let mut scratch: Pooled<Scratch<R::V>> = Pooled::take();
    let Scratch {
        index,
        place,
        start,
        found,
        outs,
        next,
        at_done,
        done,
        vals,
        contexts,
        ops,
        args,
    } = &mut *scratch;
    index.clear();
    contexts.clear();
    place.clear();
    start.clear();
    found.clear();
    outs.clear();
    done.clear();
    vals.clear();
    // Per call its instance and its place among the instance's calls, and
    // the calls laid out by instance: `outs[start[s]..start[s + 1]]`.
    let g = r.source();
    for &e in cone.nodes.iter() {
        place.push(match *g.node(e) {
            Node::Call(o, l) => {
                let (f, k) = g.output(o);
                let fresh = index.len() as u32;
                let s = *index.entry((f, g.context_of(o), l)).or_insert(fresh);
                if s == fresh {
                    start.push(0);
                }
                start[s as usize] += 1;
                found.push((s, (e, k)));
                (s, start[s as usize] - 1)
            }
            _ => (u32::MAX, 0),
        });
    }
    let mut sum = 0;
    for n in start.iter_mut() {
        (*n, sum) = (sum, sum + *n);
    }
    start.push(sum);
    outs.resize(sum as usize, (ExprId(0), 0));
    next.clone_from(start);
    for &(s, call) in found.iter() {
        outs[next[s as usize] as usize] = call;
        next[s as usize] += 1;
    }
    // per instance, where its calls' values start in `done`
    at_done.clear();
    at_done.resize(index.len(), u32::MAX);
    for (&e, &(s, k)) in cone.nodes.iter().zip(place.iter()) {
        let g = r.source();
        let node = *g.node(e);
        let v = match node {
            Node::Call(o, l) => {
                let s = s as usize;
                if at_done[s] == u32::MAX {
                    let (f, c) = (g.output(o).0, g.context_of(o));
                    let at = |a: &ExprId| vals[cone.pos(*a)];
                    args.clear();
                    args.extend(g.args(l).iter().map(at));
                    let bound = match c {
                        NO_CONTEXT => &[][..],
                        c => contexts
                            .entry(c)
                            .or_insert_with(|| g.args(g.context_list(c)).iter().map(at).collect()),
                    };
                    let call = Call {
                        f,
                        ctx: c,
                        args,
                        bound,
                    };
                    at_done[s] = done.len() as u32;
                    let calls = &outs[start[s] as usize..start[s + 1] as usize];
                    r.calls(&call, calls, done);
                    debug_assert_eq!(done.len() as u32 - at_done[s], calls.len() as u32);
                }
                done[(at_done[s] + k) as usize]
            }
            _ => {
                ops.clear();
                ops.extend(g.operands(e).iter().map(|&a| vals[cone.pos(a)]));
                r.node(e, node, ops)
            }
        };
        vals.push(v);
    }
    roots.iter().map(|&e| vals[cone.pos(e)]).collect()
}

/// The buffers of [`rewrite`], kept in the per-thread pool.
struct Scratch<V> {
    index: HashMap<(FuncId, u32, ArgList), u32>,
    place: Vec<(u32, u32)>,
    start: Vec<u32>,
    found: Vec<(u32, (ExprId, u32))>,
    outs: Vec<(ExprId, u32)>,
    next: Vec<u32>,
    at_done: Vec<u32>,
    done: Vec<V>,
    vals: Vec<V>,
    contexts: HashMap<u32, Vec<V>>,
    ops: Vec<V>,
    args: Vec<V>,
}

impl<V> Default for Scratch<V> {
    fn default() -> Self {
        Scratch {
            index: HashMap::default(),
            place: Vec::new(),
            start: Vec::new(),
            found: Vec::new(),
            outs: Vec::new(),
            next: Vec::new(),
            at_done: Vec::new(),
            done: Vec::new(),
            vals: Vec::new(),
            contexts: HashMap::default(),
            ops: Vec::new(),
            args: Vec::new(),
        }
    }
}

/// [`substitute`]: a symbol `map` binds its expression, a call whose body
/// reads one as a global the copy bound so (see [`Graph::rebound`]).
struct Substitute<'a, K: Field> {
    g: &'a mut Graph<K>,
    map: &'a HashMap<SymbolId, ExprId>,
    /// per function, the copy its calls run
    copies: HashMap<FuncId, Option<FuncId>>,
}

impl<K: Field> Rule<K> for Substitute<'_, K> {
    type V = ExprId;
    fn source(&self) -> &Graph<K> {
        self.g
    }
    fn node(&mut self, e: ExprId, node: Node, ops: &[ExprId]) -> ExprId {
        match node {
            Node::Symbol(s) => self.map.get(&s).copied().unwrap_or(e),
            _ => self.g.build(node, ops),
        }
    }
    fn calls(&mut self, call: &Call<ExprId>, outs: &[(ExprId, u32)], vals: &mut Vec<ExprId>) {
        let (g, map) = (&mut *self.g, self.map);
        let f = *self
            .copies
            .entry(call.f)
            .or_insert_with(|| g.rebound(call.f, map));
        g.calls_over(f.unwrap_or(call.f), call, outs, vals)
    }
}

/// [`substitute`] of every free symbol of `roots` by what `f` maps it to,
/// `None` keeping it: a renaming into another frame, say, where the map is
/// a rule rather than a table.
pub fn substitute_with<K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    mut f: impl FnMut(&mut Graph<K>, SymbolId) -> Option<ExprId>,
) -> Vec<ExprId> {
    let free: Vec<SymbolId> = ctx.free_symbols_in(roots).into_iter().collect();
    let map: HashMap<SymbolId, ExprId> = (free.into_iter())
        .filter_map(|s| f(ctx, s).map(|e| (s, e)))
        .collect();
    substitute(ctx, roots, &map)
}

/// [`substitute`] where `map` pins symbols to what every evaluation reads
/// them at (a constant, mostly), the calls that pass one specialized to it
/// (see [`Graph::specialize_calls`]): what remains of `roots` once the
/// symbols are no inputs.
pub fn pin<K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    map: &HashMap<SymbolId, ExprId>,
) -> Vec<ExprId> {
    let roots = substitute(ctx, roots, map);
    ctx.specialize_calls(&roots)
}

/// Replace every symbol in `map` with its expression, in every root, in
/// one pass: a subexpression reachable from several roots is rebuilt once.
/// Rebuilt through the smart constructors, so the result is re-folded and
/// hash-consed. The substitution is simultaneous, so a target expression
/// that itself mentions a mapped symbol is not re-substituted. This is the
/// primitive behind merging two nodes (one voltage symbol for another),
/// eliminating one (its solved expression for its symbol) and instantiating
/// a template (its ports for the instance's).
pub fn substitute<K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    map: &HashMap<SymbolId, ExprId>,
) -> Vec<ExprId> {
    let mut rule = Substitute {
        g: ctx,
        map,
        copies: HashMap::default(),
    };
    rewrite(&mut rule, roots)
}
