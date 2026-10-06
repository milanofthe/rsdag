use rsdag::*;

#[test]
fn charge_form_rows_select_by_role() {
    // One row in charge form: i(x) = x/r, q(x) = c x (an RC node), so
    // dF/dx' = dq/dx is the derivative of the charge output.
    let mut g: Graph<F64> = Graph::new();
    let (x, r, c) = (g.sym("x"), g.sym("r"), g.sym("c"));
    let i = g.div(x, r);
    let q = g.mul(c, x);
    let f = g.close("rc", vec![i, q]);
    for (k, s) in g.func(f).params().to_vec().into_iter().enumerate() {
        let role = match g.symbol_name(s) {
            "x" => ParamRole::State { id: 0 },
            _ => ParamRole::Param,
        };
        g.set_param_role(f, k as u32, role);
    }
    g.set_output_role(f, 0, OutputRole::Residual { id: 0 });
    g.set_output_role(f, 1, OutputRole::Charge { id: 0 });
    assert_eq!(
        g.func(f)
            .outputs_with_role(|o| matches!(o, OutputRole::Charge { .. })),
        vec![1]
    );
    let x = match g.node(x) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    assert_eq!(differentiate(&mut g, q, x), c);
}

#[test]
fn f64_field_builds_folds_and_evaluates() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let half = g.konst_f64(0.5);
    let quarter = g.konst_f64(0.25);
    let c = g.mul(half, quarter); // folds in f64
    assert_eq!(g.const_f64(c), Some(0.125));
    let e = g.add(x, c);
    let tape = Tape::compile(&g, &[e], &[SymbolId(0)]);
    let (mut work, mut out) = (Vec::new(), Vec::new());
    tape.eval(&[2.0f64], &mut work, &mut out);
    assert_eq!(out[0], 2.125);
    assert_eq!(to_string(&g, e), "(x + 0.125)");
}

use rsdag::BigRational;

#[test]
fn hash_consing_shares_identical_subexpressions() {
    let mut ctx: Graph<BigRational> = Graph::new();
    let a = ctx.sym("a");
    let b = ctx.sym("b");
    let s1 = ctx.add(a, b);
    let s2 = ctx.add(b, a); // commutative -> same node after ordering
    assert_eq!(s1, s2);

    // (a+b)*(a+b): the inner sum is interned exactly once.
    let before = ctx.len();
    let _prod = ctx.mul(s1, s2);
    // Only the product node is new.
    assert_eq!(ctx.len(), before + 1);
}

#[test]
fn folds_constants_and_identities() {
    let mut ctx: Graph<BigRational> = Graph::new();
    let a = ctx.sym("a");
    let zero = ctx.zero();
    let one = ctx.one();

    assert_eq!(ctx.add(a, zero), a);
    assert_eq!(ctx.mul(a, one), a);
    let az = ctx.mul(a, zero);
    assert!(ctx.is_zero(az));

    let two = ctx.konst_int(2);
    let three = ctx.konst_int(3);
    let six = ctx.mul(two, three);
    let q = ctx.div(six, six);
    assert!(ctx.is_one(q));
}

#[test]
fn unary_folding_and_eval() {
    let mut ctx: Graph<BigRational> = Graph::new();
    // exp(0) = 1, ln(1) = 0 fold structurally.
    let z = ctx.zero();
    let o = ctx.one();
    let e0 = ctx.exp(z);
    let l1 = ctx.ln(o);
    assert!(ctx.is_one(e0));
    assert!(ctx.is_zero(l1));

    // exp(ln(x)) evaluates to x; diode-like exp(v/Vt) evaluates correctly.
    let v = ctx.sym("v");
    let vt = ctx.sym("Vt");
    let arg = ctx.div(v, vt);
    let e = ctx.exp(arg);
    let sym = |e| match ctx.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let env: std::collections::HashMap<SymbolId, f64> =
        [(sym(v), 0.5), (sym(vt), 0.025)].into_iter().collect();
    let got = eval(&ctx, &[e], &env)[0];
    let want = (0.5_f64 / 0.025).exp();
    assert!((got - want).abs() <= want * 1e-12);
}

/// Printing is linear in the graph: a small expression is infix, a large
/// or deeply shared one a listing of its nodes, however deep.
#[test]
fn printing_a_shared_or_deep_dag_stays_linear() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let mut e = x;
    for _ in 0..40 {
        e = g.mul(e, e);
    }
    let text = rsdag::to_string(&g, e);
    assert!(text.lines().count() == 41, "{text}");
    let y = g.sym("y");
    let small = g.add(x, y);
    assert_eq!(rsdag::to_string(&g, small), "(x + y)");
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            let mut acc = x;
            for _ in 0..100_000 {
                acc = g.sin(acc);
            }
            assert!(rsdag::to_string(&g, acc).lines().count() > 100_000);
        })
        .unwrap()
        .join()
        .unwrap();
}
