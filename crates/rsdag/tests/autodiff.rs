use num_complex::Complex64;
use rsdag::eval::eval;
use rsdag::BigRational;
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
    let mut ctx: Graph<BigRational> = Graph::new();
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
    use rsdag::semantics::EXP_LIMIT;
    let mut ctx: Graph<BigRational> = Graph::new();
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
    let mut ctx: Graph<BigRational> = Graph::new();
    // f = select(x > 0, x*x, -x);  df/dx = select(x>0, 2x, -1)
    let x = ctx.sym("x");
    let zero = ctx.zero();
    let cond = ctx.cmp(CmpOp::Gt, x, zero);
    let x2 = ctx.mul(x, x);
    let negx = ctx.neg(x);
    let f = ctx.select(cond, x2, negx);
    let xid = sid(&mut ctx, "x");
    let df = differentiate(&mut ctx, f, xid);

    let eval_at = |ctx: &Graph<BigRational>, e: ExprId, xv: f64| {
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
    let mut ctx: Graph<BigRational> = Graph::new();
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
    let mut ctx: Graph<BigRational> = Graph::new();
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
    let mut ctx: Graph<BigRational> = Graph::new();
    let x = ctx.sym("x");
    let y = ctx.sym("y");
    let xy = ctx.mul(x, y);
    let e = ctx.exp(xy);
    let x3 = ctx.pow_i(x, 3);
    let x3y = ctx.mul(x3, y);
    let f = ctx.add(e, x3y);
    let (xs, ys) = (sid(&mut ctx, "x"), sid(&mut ctx, "y"));
    let h = hessian(&mut ctx, f, &[xs, ys]);
    assert_eq!(h[0][1], h[1][0], "one expression for both orders");

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
    let mut ctx: Graph<BigRational> = Graph::new();
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

/// Rows touching many unknowns take a reverse sweep, the rest forward
/// sweeps shared across rows; either way every entry is the forward
/// derivative's value, and the pattern is the same.
#[test]
fn sparse_jacobian_modes_agree_with_forward_derivatives() {
    use rsdag::synth::{build, inputs, Spec, Vocabulary};
    use rsdag::{autodiff::REVERSE_MIN_TOUCHED, differentiate, sparse_jacobian, Tape, F64};
    for (seed, params) in [(1u64, 4usize), (2, 40), (3, 64)] {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(seed)
            .steps(1500)
            .params(params)
            .outputs(6)
            .vocab(Vocabulary::Elementary)
            .width(params.max(8))
            .smooth();
        let (roots, syms) = build(&mut g, &mut spec);
        let jac = sparse_jacobian(&mut g, &roots, &syms);
        let mut fast = Vec::new();
        let mut slow = Vec::new();
        let mut reverse_rows = 0;
        for (i, row) in jac.iter().enumerate() {
            if g.free_symbols(roots[i]).len() >= REVERSE_MIN_TOUCHED {
                reverse_rows += 1;
            }
            for &(j, e) in row {
                fast.push(e);
                slow.push(differentiate(&mut g, roots[i], syms[j]));
            }
        }
        if params >= REVERSE_MIN_TOUCHED {
            assert!(
                reverse_rows > 0,
                "seed {seed}: no row took the reverse sweep"
            );
        }
        let ins = inputs(&mut spec.rng(), syms.len());
        let (mut w, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
        Tape::compile(&g, &fast, &syms).eval(&ins, &mut w, &mut a);
        Tape::compile(&g, &slow, &syms).eval(&ins, &mut w, &mut b);
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            let tol = 1e-12 * (1.0 + x.abs().max(y.abs()));
            let same = (x.is_nan() && y.is_nan()) || x == y || (x - y).abs() <= tol;
            assert!(same, "seed {seed} entry {k}: {x} vs {y}");
        }
        // No structural nonzero is lost: a forward entry that is not zero
        // symbolically is in the sparse pattern.
        for (i, row) in jac.iter().enumerate() {
            for (j, &s) in syms.iter().enumerate() {
                let d = differentiate(&mut g, roots[i], s);
                if !g.is_zero(d) {
                    assert!(
                        row.iter().any(|&(c, _)| c == j),
                        "seed {seed}: ({i}, {j}) missing"
                    );
                }
            }
        }
    }
}

/// Both modes walk the graph with an explicit stack: a chain far deeper
/// than a small thread stack could recurse through is fine.
#[test]
fn deep_chains_differentiate_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let mut g: Graph = Graph::new();
            let x = g.sym("x");
            let xs = sid(&mut g, "x");
            let (mut sines, mut sum) = (x, x);
            for _ in 0..100_000 {
                sines = g.sin(sines);
                sum = g.add(sum, x);
            }
            // d/dx of x + x + ... + x folds to the count.
            let n = g.konst_int(100_001);
            assert_eq!(differentiate(&mut g, sum, xs), n);
            assert_eq!(gradient(&mut g, sum, &[xs])[0], n);
            let d = differentiate(&mut g, sines, xs);
            assert!(!g.is_zero(d));
            let r = gradient(&mut g, sines, &[xs])[0];
            assert!(!g.is_zero(r));
            let jac = sparse_jacobian(&mut g, &[sum, sines], &[xs]);
            assert_eq!(jac[0], vec![(0, n)]);
            assert_eq!(jac[1], vec![(0, d)]);
        })
        .unwrap()
        .join()
        .unwrap();
}

/// A subgraph that does not depend on the symbols asked for builds
/// nothing in either mode, and a call's constant arguments get no
/// derivative output.
#[test]
fn derivatives_skip_what_does_not_move() {
    let mut g: Graph = Graph::new();
    let x = g.sym("x");
    let z = g.sym("z");
    let (xs, zs) = (sid(&mut g, "x"), sid(&mut g, "z"));
    let mut deep = z;
    for _ in 0..1000 {
        deep = g.tanh(deep);
    }
    let f = g.mul(x, deep);
    let before = g.len();
    assert_eq!(differentiate(&mut g, f, xs), deep);
    assert_eq!(gradient(&mut g, f, &[xs])[0], deep);
    assert_eq!(g.len(), before, "nothing built below the constant factor");

    let p = g.sym("p");
    let q = g.sym("q");
    let (ps, qs) = (sid(&mut g, "p"), sid(&mut g, "q"));
    let pq = g.mul(p, q);
    let body = g.sin(pq);
    let h = g.define_func("h", vec![ps, qs], vec![body]);
    let three = g.konst_int(3);
    let call = g.call(h, 0, &[x, three]);
    let rev = gradient(&mut g, call, &[xs, zs]);
    assert_eq!(differentiate(&mut g, call, xs), rev[0]);
    assert!(g.is_zero(rev[1]));
    assert_eq!(g.func(h).outputs().len(), 2, "d/dp only");
}

/// The sparse Jacobian of a ring of nonlinear cells (each residual touching
/// its two neighbours) holds exactly the nonzero derivatives.
#[test]
fn the_sparse_jacobian_holds_every_nonzero_derivative() {
    let mut g: Graph<F64> = Graph::new();
    let n = 40;
    let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<SymbolId> = (0..n).map(|i| sid(&mut g, &format!("x{i}"))).collect();
    let f: Vec<ExprId> = (0..n)
        .map(|i| {
            let (l, c, r) = (xs[(i + n - 1) % n], xs[i], xs[(i + 1) % n]);
            let d1 = g.sub(c, l);
            let d2 = g.sub(r, c);
            let e1 = g.exp(d1);
            let e2 = g.exp(d2);
            let s = g.sub(e1, e2);
            let two = g.konst_f64(2.0);
            let t = g.mul(two, c);
            g.add(s, t)
        })
        .collect();
    let sparse = sparse_jacobian(&mut g, &f, &syms);
    for (i, row) in sparse.iter().enumerate() {
        assert_eq!(row.len(), 3, "row {i} touches three unknowns");
        for (j, &s) in syms.iter().enumerate() {
            let d = differentiate(&mut g, f[i], s);
            match row.iter().find(|&&(c, _)| c == j) {
                Some(&(_, e)) => assert_eq!(e, d, "entry ({i}, {j})"),
                None => assert!(g.is_zero(d), "entry ({i}, {j}) is missing"),
            }
        }
    }
}

#[test]
fn many_moving_call_arguments_derive_the_body_in_one_reverse_sweep() {
    // A body over 20 parameters, called with 20 symbols: the gradient derives
    // the body's derivative outputs in one reverse sweep; the values match one
    // forward sweep per argument on a twin graph.
    fn build(g: &mut Graph) -> (ExprId, Vec<SymbolId>, FuncId) {
        let n = 20;
        let ps: Vec<ExprId> = (0..n).map(|k| g.sym(&format!("p{k}"))).collect();
        let pids: Vec<SymbolId> = (0..n).map(|k| sid(g, &format!("p{k}"))).collect();
        let mut acc = g.zero();
        for k in 0..n {
            let e = g.exp(ps[(k + 1) % n]);
            let t = g.mul(ps[k], e);
            acc = g.add(acc, t);
        }
        let body = g.sin(acc);
        let f = g.define_func("dev", pids, vec![body]);
        let args: Vec<ExprId> = (0..n).map(|k| g.sym(&format!("a{k}"))).collect();
        let call = g.call(f, 0, &args);
        let aids = (0..n).map(|k| sid(g, &format!("a{k}"))).collect();
        (call, aids, f)
    }
    let (mut rg, mut fg): (Graph, Graph) = (Graph::new(), Graph::new());
    let (rc, rs, rf) = build(&mut rg);
    let (fc, fs, ff) = build(&mut fg);
    let rev = gradient(&mut rg, rc, &rs);
    let fwd: Vec<ExprId> = fs.iter().map(|&s| differentiate(&mut fg, fc, s)).collect();
    assert_eq!(rg.func(rf).outputs().len(), 21);
    assert_eq!(fg.func(ff).outputs().len(), 21);
    let env = |ids: &[SymbolId]| -> HashMap<SymbolId, f64> {
        ids.iter()
            .enumerate()
            .map(|(k, &s)| (s, 0.05 * (k as f64 + 1.0)))
            .collect()
    };
    let vr: Vec<f64> = eval(&rg, &rev, &env(&rs));
    let vf: Vec<f64> = eval(&fg, &fwd, &env(&fs));
    for (a, b) in vr.iter().zip(&vf) {
        assert!((a - b).abs() <= 1e-13 * b.abs().max(1.0), "{a} vs {b}");
    }
}

/// A call reads only what its output's body reads: through a nested call
/// the arguments the inner output's support names, and not a comparison's
/// operands (they carry no derivative).
#[test]
fn the_support_reads_through_nested_calls() {
    let mut g: Graph<F64> = Graph::new();
    // f(a, b, c) = [a b, a > 0 ? sin(c) : c]
    let mut s = Scope::new(&mut g, "f");
    let (a, b, c) = (s.param("a"), s.param("b"), s.param("c"));
    let ab = s.mul(a, b);
    let zero = s.zero();
    let pos = s.cmp(CmpOp::Gt, a, zero);
    let sc = s.sin(c);
    let pick = s.select(pos, sc, c);
    let f = s.close(vec![ab, pick]);
    // g(x, y, z, w) = [f0(x, y, z) + w, f1(x, y, z w)]
    let mut s = Scope::new(&mut g, "g");
    let (x, y, z, w) = (s.param("x"), s.param("y"), s.param("z"), s.param("w"));
    let f0 = s.call(f, 0, &[x, y, z]);
    let g0 = s.add(f0, w);
    let zw = s.mul(z, w);
    let g1 = s.call(f, 1, &[x, y, zw]);
    let gf = s.close(vec![g0, g1]);
    assert_eq!(&*g.output_support(f, 0), &[0, 1]);
    assert_eq!(&*g.output_support(f, 1), &[2]);
    assert_eq!(&*g.output_support(gf, 0), &[0, 1, 3]);
    assert_eq!(&*g.output_support(gf, 1), &[2, 3]);
    let args: Vec<ExprId> = ["p", "q", "r", "t"].iter().map(|n| g.sym(n)).collect();
    let call = g.call(gf, 1, &args);
    let names: Vec<String> = g
        .support_in(&[call])
        .into_iter()
        .map(|s| g.symbol_name(s).to_string())
        .collect();
    assert_eq!(names, ["r", "t"]);
    // the value reads the selector's condition too
    assert_eq!(&*g.output_reads(f, 1), &[0, 2]);
    assert_eq!(&*g.output_reads(gf, 1), &[0, 2, 3]);
    let (p, q) = (sid(&mut g, "p"), sid(&mut g, "q"));
    let g0 = g.call(gf, 0, &args);
    assert_eq!(g.depends_on(&[call, g0], &[p]), [true, true]);
    assert_eq!(g.depends_on(&[call, g0], &[q]), [false, true]);
}

/// A row that is a call of a wide function touches only the columns its
/// output reads, two levels of calls deep: the Jacobian makes one derivative
/// output per structural nonzero, and its entries and a gradient through the
/// calls are the inlined graph's.
#[test]
fn a_jacobian_through_wide_calls_makes_only_its_entries() {
    const N: usize = 48;
    let mut g: Graph<F64> = Graph::new();
    // f(x)_k = x_k exp(x_{k+1}), cyclic: each output reads two of N parameters
    let mut s = Scope::new(&mut g, "f");
    let xs: Vec<ExprId> = (0..N).map(|k| s.param(&format!("x{k}"))).collect();
    let outs: Vec<ExprId> = (0..N)
        .map(|k| {
            let e = s.exp(xs[(k + 1) % N]);
            s.mul(xs[k], e)
        })
        .collect();
    let f = s.close(outs);
    // h(y)_k = f(y)_k + y_k: a body that is all calls into f
    let mut s = Scope::new(&mut g, "h");
    let ys: Vec<ExprId> = (0..N).map(|k| s.param(&format!("y{k}"))).collect();
    let outs: Vec<ExprId> = (0..N)
        .map(|k| {
            let c = s.call(f, k as u32, &ys);
            s.add(c, ys[k])
        })
        .collect();
    let h = s.close(outs);
    let us: Vec<ExprId> = (0..N).map(|k| g.sym(&format!("u{k}"))).collect();
    let wrt: Vec<SymbolId> = (0..N).map(|k| sid(&mut g, &format!("u{k}"))).collect();
    let rows: Vec<ExprId> = (0..N).map(|k| g.call(h, k as u32, &us)).collect();

    let jac = sparse_jacobian(&mut g, &rows, &wrt);
    for (k, row) in jac.iter().enumerate() {
        let cols: Vec<usize> = row.iter().map(|&(j, _)| j).collect();
        let mut want = vec![k, (k + 1) % N];
        want.sort_unstable();
        assert_eq!(cols, want, "row {k}");
    }
    // N outputs and 2 N derivative outputs each, not N^2
    assert_eq!(g.func(f).outputs().len(), 3 * N);
    assert_eq!(g.func(h).outputs().len(), 3 * N);

    let flat = g.inline_all(&rows);
    let flat_jac = sparse_jacobian(&mut g, &flat, &wrt);
    let sum = g.reduce(ReduceOp::Sum, rows.clone());
    let flat_sum = g.reduce(ReduceOp::Sum, flat.clone());
    let grad = gradient(&mut g, sum, &wrt);
    let flat_grad = gradient(&mut g, flat_sum, &wrt);
    let mut a: Vec<ExprId> = jac.iter().flatten().map(|&(_, e)| e).collect();
    let mut b: Vec<ExprId> = flat_jac.iter().flatten().map(|&(_, e)| e).collect();
    a.extend(grad);
    b.extend(flat_grad);
    assert_eq!(a.len(), b.len());
    let ins: Vec<f64> = (0..N)
        .map(|k| 0.1 + 0.37 * ((k * 7) % 11) as f64 / 11.0)
        .collect();
    let (mut work, mut va, mut vb) = (Vec::new(), Vec::new(), Vec::new());
    Tape::compile(&g, &a, &wrt).eval(&ins, &mut work, &mut va);
    Tape::compile(&g, &b, &wrt).eval(&ins, &mut work, &mut vb);
    for (k, (p, q)) in va.iter().zip(&vb).enumerate() {
        assert!(
            (p - q).abs() <= 1e-12 * (1.0 + q.abs()),
            "entry {k}: {p} vs {q}"
        );
    }
}
