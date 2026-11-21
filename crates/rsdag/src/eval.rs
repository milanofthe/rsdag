//! Numeric evaluation over the arena.
//!
//! One forward sweep over the nodes reachable from the roots (ascending
//! `ExprId` is topological, so every shared subexpression is computed once),
//! with the same per-node semantics the tape uses: arithmetic and fold
//! orders from [`crate::semantics`], so a value cannot depend on which
//! evaluator computed it. An unbound symbol is `NaN`, as a missing input is
//! for a tape. Calls evaluate their function once per distinct argument
//! list through the function's body, a bundle.

use std::collections::HashMap;

use crate::field::Field;
use crate::func::{Body, FuncId, Output, OutputId};
use crate::graph::Graph;
use crate::node::ArgList;
use crate::node::{ExprId, Node, SymbolId};
use crate::semantics::{binary_f64, cmp_bool, dot_slice, powi_f64, reduce_slice, unary_f64};

/// The value of one node from its operands.
fn node_value<K: Field>(
    ctx: &Graph<K>,
    node: &Node,
    mut get: impl FnMut(ExprId) -> f64,
    sym: &mut impl FnMut(SymbolId) -> f64,
    call: &mut impl FnMut(OutputId, ArgList, &[f64]) -> f64,
    solve: &mut impl FnMut(ArgList, &[f64], u32) -> f64,
) -> f64 {
    match *node {
        Node::Const(c) => ctx.const_val(c).to_f64(),
        Node::Symbol(s) => sym(s),
        Node::Add(a, b) => get(a) + get(b),
        Node::Mul(a, b) => get(a) * get(b),
        Node::Neg(a) => -get(a),
        Node::Pow(a, k) => powi_f64(get(a), k as i32),
        Node::Unary(op, a) => unary_f64(op, get(a)),
        Node::Binary(op, a, b) => binary_f64(op, get(a), get(b)),
        Node::Cmp(op, a, b) => match cmp_bool(op, get(a), get(b)) {
            true => 1.0,
            false => 0.0,
        },
        Node::Select(c, t, e) => {
            if get(c) != 0.0 {
                get(t)
            } else {
                get(e)
            }
        }
        Node::Reduce(op, l) => {
            let vals: Vec<f64> = ctx.args(l).iter().map(|&a| get(a)).collect();
            reduce_slice(op, &vals)
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            let va: Vec<f64> = a.iter().map(|&x| get(x)).collect();
            let vb: Vec<f64> = b.iter().map(|&y| get(y)).collect();
            dot_slice(&va, &vb)
        }
        Node::Call(o, l) => {
            // the operands in parameter order, then the globals the body reads
            let globals = ctx.globals(ctx.output(o).0);
            let full = ctx.full_args(o, l);
            let vals: Vec<f64> = full.iter().chain(&globals[..]).map(|&a| get(a)).collect();
            call(o, l, &vals)
        }
        Node::Solve(l, i) => {
            let vals: Vec<f64> = ctx.args(l).iter().map(|&a| get(a)).collect();
            solve(l, &vals, i)
        }
    }
}

/// The values of `roots` under the symbol bindings `env`.
///
/// Every node reachable from a root is evaluated once, in arena order; a
/// symbol absent from `env` is `NaN`. To see every node's value (a
/// diagnostic locating the first non-finite node, say), pass every id as a
/// root: the lowest-index non-finite value is the origin, since a node's
/// operands have smaller ids.
pub fn eval<K: Field>(ctx: &Graph<K>, roots: &[ExprId], env: &HashMap<SymbolId, f64>) -> Vec<f64> {
    // the nodes the roots reach, each after its operands
    let cone = ctx.cone(roots, true);
    let at = |e: ExprId| cone.pos(e);
    let mut fe = FuncEval::new();
    for &e in cone.nodes.iter() {
        if let Node::Call(o, _) = *ctx.node(e) {
            let (f, out) = ctx.output(o);
            fe.need(f, out);
        }
    }
    let mut w: Vec<f64> = Vec::with_capacity(cone.nodes.len());
    // A dense system is solved once for all its components.
    let mut solved: HashMap<ArgList, Vec<f64>> = HashMap::new();
    for &e in cone.nodes.iter() {
        let node = *ctx.node(e);
        let mut sym = |s: SymbolId| env.get(&s).copied().unwrap_or(f64::NAN);
        let mut call = |o: OutputId, l: ArgList, args: &[f64]| fe.output(ctx, o, l, args);
        let mut solve = |l: ArgList, vals: &[f64], i: u32| -> f64 {
            solved.entry(l).or_insert_with(|| {
                let n = Graph::<K>::solve_n(vals.len());
                let mut out = vec![0.0; n];
                crate::semantics::solve(&vals[..n * n], &vals[n * n..], n, &mut out);
                out
            })[i as usize]
        };
        let v = node_value(ctx, &node, |e| w[at(e)], &mut sym, &mut call, &mut solve);
        w.push(v);
    }
    roots.iter().map(|&r| w[at(r)]).collect()
}

/// Per-sweep function evaluation: a body per function and the outputs of
/// every distinct `(function, argument list)` already evaluated.
struct FuncEval {
    bodies: rustc_hash::FxHashMap<FuncId, Body>,
    /// Per instance (function, context, argument list), its outputs.
    vals: rustc_hash::FxHashMap<(FuncId, u32, ArgList), Vec<f64>>,
    /// The outputs the sweep calls per function, declared up front so one
    /// body serves the whole sweep (see [`Function::body_for`]).
    needed: rustc_hash::FxHashMap<FuncId, Vec<u32>>,
}

impl FuncEval {
    fn new() -> Self {
        Self {
            bodies: Default::default(),
            vals: Default::default(),
            needed: Default::default(),
        }
    }

    /// Declare that the sweep calls output `out` of `f`, before the first
    /// call; a body is chosen for the declared set.
    fn need(&mut self, f: FuncId, out: u32) {
        let v = self.needed.entry(f).or_default();
        if !v.contains(&out) {
            v.push(out);
        }
    }

    /// The value of a call of output `o` over the list `l` at the values
    /// `args` of its operands (in its function's parameter order, then its
    /// globals); the instance keys the memo.
    fn output<K: Field>(&mut self, ctx: &Graph<K>, o: OutputId, l: ArgList, args: &[f64]) -> f64 {
        let (f, out) = ctx.output(o);
        let site = (f, ctx.context_of(o), l);
        let func = ctx.func(f);
        if matches!(func.outputs()[out as usize], Output::Zero) {
            return 0.0;
        }
        let needed = &self.needed;
        let body = self
            .bodies
            .entry(f)
            .or_insert_with(|| match needed.get(&f) {
                Some(needed) => func.body_for(ctx, needed),
                None => func.body_for(ctx, &[out]),
            });
        let Some(slot) = body.slot_of.get(out as usize).copied().flatten() else {
            // The cached body predates this output (a derivative demanded
            // later): rebuild it over every output.
            let fresh = func.body(ctx);
            self.vals.retain(|k, _| k.0 != f);
            *body = fresh;
            let slot = body.slot_of[out as usize].expect("a body carries every output");
            return self.eval_slot(site, args, slot);
        };
        self.eval_slot(site, args, slot)
    }

    fn eval_slot(&mut self, site: (FuncId, u32, ArgList), args: &[f64], slot: u32) -> f64 {
        let body = &self.bodies[&site.0];
        let vals = self.vals.entry(site).or_insert_with(|| {
            let mut out = vec![0.0; body.bundle.n_outputs()];
            body.bundle.call(args, &mut out);
            out
        });
        vals[slot as usize]
    }
}
