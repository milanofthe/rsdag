use num_complex::Complex64;
use rsdag::eval::eval;
use rsdag::*;
use std::collections::HashMap;

fn sid<K: Field>(ctx: &mut Graph<K>, name: &str) -> SymbolId {
    let id = ctx.sym(name);
    match ctx.node(id) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

#[test]
fn derivative_matches_finite_difference() {
    let mut ctx: Graph = Graph::new();
    // f = Is*(exp(v/Vt) - 1) + g*v^2  (a diode current plus a quadratic term)
    let v = ctx.sym("v");
    let vt = ctx.sym("Vt");
    let is = ctx.sym("Is");
    let g = ctx.sym("g");
    let arg = ctx.div(v, vt);
    let e = ctx.exp(arg);
    let one = ctx.one();
    let em1 = ctx.sub(e, one);
    let diode = ctx.mul(is, em1);
    let v2 = ctx.pow_i(v, 2);
    let gv2 = ctx.mul(g, v2);
    let f = ctx.add(diode, gv2);

    let v_id = sid(&mut ctx, "v");
    let df = differentiate(&mut ctx, f, v_id);

    // Derivative wrt an absent symbol is structurally zero.
    let w_id = sid(&mut ctx, "w");
    let dfw = differentiate(&mut ctx, f, w_id);
    assert!(ctx.is_zero(dfw));

    let vt_id = sid(&mut ctx, "Vt");
    let is_id = sid(&mut ctx, "Is");
    let g_id = sid(&mut ctx, "g");

    let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
    env.insert(vt_id, Complex64::new(0.05, 0.0));
    env.insert(is_id, Complex64::new(1e-12, 0.0));
    env.insert(g_id, Complex64::new(0.3, 0.0));

    let h = 1e-6;
    for &x0 in &[-0.1, 0.0, 0.1, 0.3] {
        env.insert(v_id, Complex64::new(x0, 0.0));
        let analytic = eval(&ctx, &[df], &env)[0].re;
        env.insert(v_id, Complex64::new(x0 + h, 0.0));
        let fp = eval(&ctx, &[f], &env)[0].re;
        env.insert(v_id, Complex64::new(x0 - h, 0.0));
        let fm = eval(&ctx, &[f], &env)[0].re;
        let numeric = (fp - fm) / (2.0 * h);
        assert!(
            (analytic - numeric).abs() <= 1e-4 * (1.0 + analytic.abs()),
            "x0={x0}: analytic={analytic} numeric={numeric}"
        );
    }
}

#[test]
fn exp_derivative_shares_primal_and_matches_limexp_tail() {
    use rsdag::node::EXP_LIMIT;
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let e = ctx.exp(x);
    let xid = sid(&mut ctx, "x");
    let de = differentiate(&mut ctx, e, xid);
    // The derivative reuses the primal exp node (no second exp in the DAG).
    let n_exp = (0..ctx.len())
        .filter(|&i| matches!(ctx.node(ExprId(i as u32)), Node::Unary(UnaryOp::Exp, _)))
        .count();
    assert_eq!(n_exp, 1);
    let mut env: HashMap<SymbolId, f64> = HashMap::new();
    for &xv in &[-3.0, 0.0, 10.0, EXP_LIMIT, EXP_LIMIT + 5.0, 500.0] {
        env.insert(xid, xv);
        let v = rsdag::eval(&ctx, &[de], &env)[0];
        let want = if xv <= EXP_LIMIT {
            xv.exp()
        } else {
            EXP_LIMIT.exp()
        };
        assert_eq!(v.to_bits(), want.to_bits(), "x={xv}");
    }
}

#[test]
fn select_and_opaque_autodiff() {
    use rsdag::node::CmpOp;
    let mut ctx: Graph = Graph::new();
    // f = select(x > 0, x*x, -x);  df/dx = select(x>0, 2x, -1)
    let x = ctx.sym("x");
    let zero = ctx.zero();
    let cond = ctx.cmp(CmpOp::Gt, x, zero);
    let x2 = ctx.mul(x, x);
    let negx = ctx.neg(x);
    let f = ctx.select(cond, x2, negx);
    let xid = sid(&mut ctx, "x");
    let df = differentiate(&mut ctx, f, xid);

    let eval_at = |ctx: &Graph, e: ExprId, xv: f64| {
        let mut env = HashMap::new();
        env.insert(xid, Complex64::new(xv, 0.0));
        eval(ctx, &[e], &env)[0].re
    };
    assert!((eval_at(&ctx, df, 3.0) - 6.0).abs() < 1e-9); // 2*3
    assert!((eval_at(&ctx, df, -2.0) + 1.0).abs() < 1e-9); // -1

    // Chain rule through a call: g = f(2*x) with f(p) = p^3; dg/dx must
    // reference the derivative output of f (3p^2, a call) times 2:
    // at x = 1, 3*4*2 = 24.
    let two = ctx.konst_int(2);
    let two_x = ctx.mul(two, x);
    let p = ctx.sym("p");
    let ps = sid(&mut ctx, "p");
    let p3 = ctx.pow_i(p, 3);
    let f = ctx.define_func("cube", vec![ps], vec![p3]);
    let g = ctx.call(f, 0, &[two_x]);
    let dg = differentiate(&mut ctx, g, xid);
    assert!(!ctx.is_zero(dg));
    assert!(rsdag::to_string(&ctx, dg).contains("cube#1"));
    assert!((eval_at(&ctx, dg, 1.0) - 24.0).abs() < 1e-9);
}

#[test]
fn gradient_matches_forward_mode() {
    // A device-like expression exercising every node kind the reverse sweep
    // handles: exp/ln guards, select subgradients, powers, reduce, dot.
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let y = ctx.sym("y");
    let z = ctx.sym("z");
    let arg = ctx.div(x, y);
    let e = ctx.exp(arg);
    let x2 = ctx.pow_i(x, 3);
    let ln = ctx.ln(y);
    let zero = ctx.zero();
    let cond = ctx.cmp(CmpOp::Gt, z, zero);
    let t = ctx.mul(x2, ln);
    let nx = ctx.neg(x);
    let sel = ctx.select(cond, t, nx);
    let red = ctx.reduce(ReduceOp::Product, vec![x, y, z]);
    let mn = ctx.reduce(ReduceOp::Min, vec![x, y, z]);
    let dot = ctx.dot(vec![x, sel], vec![y, z]);
    let s1 = ctx.add(e, red);
    let s2 = ctx.add(mn, dot);
    let f = ctx.add(s1, s2);

    let (xs, ys, zs) = (sid(&mut ctx, "x"), sid(&mut ctx, "y"), sid(&mut ctx, "z"));
    let grad = gradient(&mut ctx, f, &[xs, ys, zs]);
    let fwd: Vec<ExprId> = [xs, ys, zs]
        .iter()
        .map(|&s| differentiate(&mut ctx, f, s))
        .collect();

    let pts = [
        (0.7, 1.3, 0.4),
        (-0.5, 2.0, -1.1),
        (1.9, 0.3, 2.5),
        (-2.0, -0.7, 0.9),
    ];
    for &(xv, yv, zv) in &pts {
        let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
        env.insert(xs, Complex64::new(xv, 0.0));
        env.insert(ys, Complex64::new(yv, 0.0));
        env.insert(zs, Complex64::new(zv, 0.0));
        for (g, d) in grad.iter().zip(&fwd) {
            let gv = eval(&ctx, &[*g], &env)[0].re;
            let dv = eval(&ctx, &[*d], &env)[0].re;
            assert!(
                (gv - dv).abs() <= 1e-12 * (1.0 + dv.abs()),
                "at ({xv},{yv},{zv}): reverse={gv} forward={dv}"
            );
        }
    }
}

#[test]
fn gradient_handles_calls_and_absent_symbols() {
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let two = ctx.konst_int(2);
    let two_x = ctx.mul(two, x);
    let p = ctx.sym("p");
    let ps = sid(&mut ctx, "p");
    let p3 = ctx.pow_i(p, 3);
    let f = ctx.define_func("cube", vec![ps], vec![p3]);
    let g = ctx.call(f, 0, &[two_x]);
    let xs = sid(&mut ctx, "x");
    let ws = sid(&mut ctx, "w");
    let grad = gradient(&mut ctx, g, &[xs, ws]);
    assert!(!ctx.is_zero(grad[0]));
    assert!(rsdag::to_string(&ctx, grad[0]).contains("cube#1"));
    // Absent symbol: structurally zero.
    assert!(ctx.is_zero(grad[1]));
}

#[test]
fn hessian_is_symmetric_and_correct() {
    // f = exp(x*y) + x^3*y  ->  d2f/dxdy = exp(xy)*(1 + xy) + 3x^2 (both orders).
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let y = ctx.sym("y");
    let xy = ctx.mul(x, y);
    let e = ctx.exp(xy);
    let x3 = ctx.pow_i(x, 3);
    let x3y = ctx.mul(x3, y);
    let f = ctx.add(e, x3y);
    let (xs, ys) = (sid(&mut ctx, "x"), sid(&mut ctx, "y"));
    let h = hessian(&mut ctx, f, &[xs, ys]);

    let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
    env.insert(xs, Complex64::new(0.6, 0.0));
    env.insert(ys, Complex64::new(-0.8, 0.0));
    let (xv, yv) = (0.6_f64, -0.8_f64);
    let want_xy = (xv * yv).exp() * (1.0 + xv * yv) + 3.0 * xv * xv;
    let h01 = eval(&ctx, &[h[0][1]], &env)[0].re;
    let h10 = eval(&ctx, &[h[1][0]], &env)[0].re;
    assert!((h01 - want_xy).abs() <= 1e-12 * (1.0 + want_xy.abs()));
    assert!((h10 - want_xy).abs() <= 1e-12 * (1.0 + want_xy.abs()));
    // Third order by repeated application: d3f/dx3 = 6y + y^3*exp(xy).
    let gx = gradient(&mut ctx, f, &[xs])[0];
    let gxx = differentiate(&mut ctx, gx, xs);
    let gxxx = differentiate(&mut ctx, gxx, xs);
    let want3 = 6.0 * yv + yv.powi(3) * (xv * yv).exp();
    let got3 = eval(&ctx, &[gxxx], &env)[0].re;
    assert!((got3 - want3).abs() <= 1e-12 * (1.0 + want3.abs()));
}

#[test]
fn the_jacobian_is_sparse() {
    let mut ctx: Graph = Graph::new();
    // r0 = a*x + b*y ; r1 = x  (so dr1/dy = 0)
    let x = ctx.sym("x");
    let y = ctx.sym("y");
    let a = ctx.sym("a");
    let b = ctx.sym("b");
    let ax = ctx.mul(a, x);
    let by = ctx.mul(b, y);
    let r0 = ctx.add(ax, by);
    let r1 = x;

    let x_id = sid(&mut ctx, "x");
    let y_id = sid(&mut ctx, "y");
    let jac = sparse_jacobian(&mut ctx, &[r0, r1], &[x_id, y_id]);
    let pattern: Vec<Vec<usize>> = jac
        .iter()
        .map(|row| row.iter().map(|&(j, _)| j).collect())
        .collect();
    assert_eq!(pattern, vec![vec![0, 1], vec![0]]);
    assert_eq!(jac[0][0].1, a);
    assert_eq!(jac[1][0].1, ctx.one());
}
