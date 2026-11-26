//! Reverse mode differentiates a `Select` as forward mode does, by its taken
//! branch alone: a value or partial of the untaken branch that is not finite
//! at the point (a division by a variable that is zero there) does not reach
//! the gradient, also where the branches share a variable or a node, under
//! nested selects, through `min`, and in the rows of a sparse Jacobian above
//! and below `REVERSE_MIN_TOUCHED`, behind every rule of the reverse sweep.
//! The bound is the forward result's.

mod common;
use common::sym;

use rsdag::autodiff::REVERSE_MIN_TOUCHED;
use rsdag::eval::eval;
use rsdag::{
    differentiate, gradient, sparse_jacobian, BinOp, CmpOp, ExprId, Graph, OutputRole, ParamRole,
    ReduceOp, Scope, SymbolId, UnaryOp, F64,
};
use std::collections::HashMap;

fn reverse_matches_forward_and_is_finite(
    g: &mut Graph<F64>,
    f: ExprId,
    wrt: &[SymbolId],
    env: &HashMap<SymbolId, f64>,
) {
    let rev = gradient(g, f, wrt);
    let fwd: Vec<ExprId> = wrt.iter().map(|&s| differentiate(g, f, s)).collect();
    let (r, d) = (eval(g, &rev, env), eval(g, &fwd, env));
    for (k, (r, d)) in r.iter().zip(&d).enumerate() {
        assert!(d.is_finite(), "forward {k}: {d}");
        assert!(
            (r - d).abs() <= 4.0 * f64::EPSILON * d.abs(),
            "by {k}: reverse {r}, forward {d}"
        );
    }
}

#[test]
fn an_untaken_branch_with_an_infinite_partial_does_not_reach_the_gradient() {
    let mut g: Graph<F64> = Graph::new();
    let (x, xs) = sym(&mut g, "x");
    let (y, ys) = sym(&mut g, "y");
    let zero = g.zero();
    let pos = g.cmp(CmpOp::Gt, x, zero);
    let env: HashMap<SymbolId, f64> = [(xs, 0.0), (ys, 2.0)].into_iter().collect();
    let y_over_x_with_infinite_partial_at_0 = g.div(y, x);
    let yy = g.mul(y, y);
    let f = g.select(pos, y_over_x_with_infinite_partial_at_0, yy);
    reverse_matches_forward_and_is_finite(&mut g, f, &[xs, ys], &env);
    let u_shared_by_both_branches = g.mul(x, y);
    let ux = g.div(u_shared_by_both_branches, x);
    let uy = g.add(u_shared_by_both_branches, y);
    let f = g.select(pos, ux, uy);
    reverse_matches_forward_and_is_finite(&mut g, f, &[xs, ys], &env);
    let ypos = g.cmp(CmpOp::Gt, y, zero);
    let one = g.one();
    let ix = g.div(one, x);
    let iy = g.div(one, y);
    let inner_select_in_the_untaken_branch = g.select(ypos, ix, iy);
    let xy = g.mul(x, y);
    let f = g.select(pos, inner_select_in_the_untaken_branch, xy);
    reverse_matches_forward_and_is_finite(&mut g, f, &[xs, ys], &env);
    let used_again_past_the_select = g.mul(f, y);
    reverse_matches_forward_and_is_finite(&mut g, used_again_past_the_select, &[xs, ys], &env);
    let min_taking_y_over_an_infinite_1_over_x = g.reduce(ReduceOp::Min, vec![y, ix]);
    reverse_matches_forward_and_is_finite(
        &mut g,
        min_taking_y_over_an_infinite_1_over_x,
        &[xs, ys],
        &env,
    );
}

#[test]
fn sparse_jacobian_rows_agree_with_forward_mode_on_either_side_of_the_reverse_threshold() {
    for n in [REVERSE_MIN_TOUCHED - 1, REVERSE_MIN_TOUCHED + 4] {
        let mut g: Graph<F64> = Graph::new();
        let (x, xs) = sym(&mut g, "x");
        let zero = g.zero();
        let pos = g.cmp(CmpOp::Gt, x, zero);
        let mut wrt = vec![xs];
        let mut env: HashMap<SymbolId, f64> = [(xs, 0.0)].into_iter().collect();
        let mut terms = Vec::new();
        for k in 0..n {
            let (p, ps) = sym(&mut g, &format!("p{k}"));
            wrt.push(ps);
            env.insert(ps, 1.0 + k as f64);
            let q = g.div(p, x);
            let pp = g.mul(p, p);
            terms.push(g.select(pos, q, pp));
        }
        let row = g.reduce(ReduceOp::Sum, terms);
        let rows = sparse_jacobian(&mut g, &[row], &wrt);
        let fwd: Vec<ExprId> = wrt.iter().map(|&s| differentiate(&mut g, row, s)).collect();
        let d = eval(&g, &fwd, &env);
        let (cols, vals): (Vec<usize>, Vec<ExprId>) = rows[0].iter().copied().unzip();
        let r = eval(&g, &vals, &env);
        for (&j, r) in cols.iter().zip(&r) {
            assert!(d[j].is_finite(), "{n} columns, forward {j}: {}", d[j]);
            assert!(
                (r - d[j]).abs() <= 4.0 * f64::EPSILON * d[j].abs(),
                "{n} columns, by {j}: {r} vs {}",
                d[j]
            );
        }
    }
}

#[test]
fn every_sweep_rule_keeps_the_untaken_branch_out() {
    let mut g: Graph<F64> = Graph::new();
    let div = {
        let mut s = Scope::new(&mut g, "div");
        let (a, b) = (s.param("a", ParamRole::Free), s.param("b", ParamRole::Free));
        let q = s.div(a, b);
        s.close([(q, OutputRole::Plain)])
    };
    let (x, xs) = sym(&mut g, "x");
    let (y, ys) = sym(&mut g, "y");
    let env: HashMap<SymbolId, f64> = [(xs, 0.0), (ys, 2.0)].into_iter().collect();
    let zero = g.zero();
    let one = g.one();
    let pos = g.cmp(CmpOp::Gt, x, zero);
    let q = g.div(y, x);
    let ix = g.div(one, x);
    let yy = g.mul(y, y);
    let mut cases: Vec<(&str, ExprId)> = Vec::new();
    cases.push(("add", g.add(q, y)));
    let nq = g.neg(q);
    cases.push(("neg", nq));
    cases.push(("mul", g.mul(ix, y)));
    cases.push(("pow", g.pow_i(q, 2)));
    cases.push(("unary", g.unary(UnaryOp::Tanh, q)));
    cases.push(("binary", g.binary(BinOp::Powf, y, ix)));
    cases.push(("sum", g.reduce(ReduceOp::Sum, vec![q, y])));
    cases.push(("product", g.reduce(ReduceOp::Product, vec![q, y])));
    cases.push(("dot", g.dot(vec![ix, y], vec![y, y])));
    cases.push(("solve", g.solve_dense(vec![x], vec![y])[0]));
    cases.push(("call", g.call(div, 0, &[y, x])));
    cases.push(("neg-unary-div", g.unary(UnaryOp::Tanh, nq)));
    for (name, r) in cases {
        let into_y_squared = g.select(pos, r, yy);
        let into_zero = g.select(pos, r, zero);
        for f in [into_y_squared, into_zero] {
            let rev = gradient(&mut g, f, &[xs, ys]);
            let fwd: Vec<ExprId> = [xs, ys]
                .iter()
                .map(|&s| differentiate(&mut g, f, s))
                .collect();
            let (r, d) = (eval(&g, &rev, &env), eval(&g, &fwd, &env));
            for k in 0..2 {
                assert!(d[k].is_finite(), "{name}: forward {k}: {}", d[k]);
                assert!(
                    (r[k] - d[k]).abs() <= 4.0 * f64::EPSILON * d[k].abs(),
                    "{name} by {k}: reverse {}, forward {}",
                    r[k],
                    d[k]
                );
            }
        }
    }
}
