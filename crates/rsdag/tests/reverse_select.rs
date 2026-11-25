//! Reverse mode differentiates a `Select` as forward mode does, by its taken
//! branch alone: a value or partial of the untaken branch that is not finite
//! at the point (a division by a variable that is zero there) does not reach
//! the gradient, also where the branches share a variable or a node, under
//! nested selects, through `min`, and in the rows of a sparse Jacobian above
//! and below `REVERSE_MIN_TOUCHED`. The bound is the forward result's.

use rsdag::autodiff::REVERSE_MIN_TOUCHED;
use rsdag::eval::eval;
use rsdag::{
    differentiate, gradient, sparse_jacobian, CmpOp, ExprId, Graph, Node, ReduceOp, SymbolId, F64,
};
use std::collections::HashMap;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// Reverse against forward for `f` by `wrt` at `env`: equal within the
/// rounding of the forward value, and finite.
fn agree(g: &mut Graph<F64>, f: ExprId, wrt: &[SymbolId], env: &HashMap<SymbolId, f64>) {
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
    // select(x > 0, y / x, y^2) at x = 0: d(y/x)/dy = 1/x is infinite
    let q = g.div(y, x);
    let yy = g.mul(y, y);
    let f = g.select(pos, q, yy);
    agree(&mut g, f, &[xs, ys], &env);
    // the branches share a node: u = x y in both
    let u = g.mul(x, y);
    let ux = g.div(u, x);
    let uy = g.add(u, y);
    let f = g.select(pos, ux, uy);
    agree(&mut g, f, &[xs, ys], &env);
    // nested: the inner select sits in the untaken branch
    let ypos = g.cmp(CmpOp::Gt, y, zero);
    let one = g.one();
    let ix = g.div(one, x);
    let iy = g.div(one, y);
    let inner = g.select(ypos, ix, iy);
    let xy = g.mul(x, y);
    let f = g.select(pos, inner, xy);
    agree(&mut g, f, &[xs, ys], &env);
    // the result used again past the select
    let f2 = g.mul(f, y);
    agree(&mut g, f2, &[xs, ys], &env);
    // min(y, 1/x) at x = 0: y taken, 1/x infinite
    let m = g.reduce(ReduceOp::Min, vec![y, ix]);
    agree(&mut g, m, &[xs, ys], &env);
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
