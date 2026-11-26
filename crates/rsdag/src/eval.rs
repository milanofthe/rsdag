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
            let globals = ctx.globals(ctx.output(o).0);
            let params_in_order = ctx.full_args(o, l);
            let vals: Vec<f64> = (params_in_order.iter().chain(&globals[..]))
                .map(|&a| get(a))
                .collect();
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

struct FuncEval {
    bodies: rustc_hash::FxHashMap<FuncId, Body>,
    vals: rustc_hash::FxHashMap<(FuncId, u32, ArgList), Vec<f64>>,
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

    fn need(&mut self, f: FuncId, out: u32) {
        let v = self.needed.entry(f).or_default();
        if !v.contains(&out) {
            v.push(out);
        }
    }

    fn output<K: Field>(&mut self, ctx: &Graph<K>, o: OutputId, l: ArgList, args: &[f64]) -> f64 {
        let (f, out) = ctx.output(o);
        let site = (f, ctx.context_of(o), l);
        let func = ctx.func(f);
        let output = func.outputs()[out as usize];
        if matches!(output, Output::Zero) {
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
        let slot = match (body.slot_of.get(out as usize).copied().flatten(), output) {
            (Some(slot), _) | (None, Output::Slot(slot)) => slot,
            (None, Output::Expr(e)) => {
                let (fresh, slot) = func.body_leading(ctx, out, e);
                self.vals.retain(|k, _| k.0 != f);
                *body = fresh;
                slot
            }
            (None, Output::Zero) => return 0.0,
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
