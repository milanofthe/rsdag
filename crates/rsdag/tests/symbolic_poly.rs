use rsdag::display::to_string;
use rsdag::symbolic::{prune_poly, rational_form};
use rsdag::BigRational;
use rsdag::*;
use std::collections::HashMap;

fn sym_id<K: Field>(g: &Graph<K>, e: ExprId) -> SymbolId {
    match *g.node(e) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    }
}

#[test]
fn collects_a_transfer_function() {
    // H = 1 / (1 + s R C): numerator [1], denominator [1, R C]
    let mut g: Graph<BigRational> = Graph::new();
    let (s, r, c) = (g.sym("s"), g.sym("R"), g.sym("C"));
    let rc = g.mul(r, c);
    let src = g.mul(s, rc);
    let one = g.one();
    let den = g.add(one, src);
    let h = g.recip(den);
    let sid = sym_id(&g, s);
    let (n, d) = rational_form(&mut g, h, sid).unwrap();
    assert_eq!(n.len(), 1);
    assert!(g.is_one(n[0]));
    assert_eq!(d.len(), 2);
    assert!(g.is_one(d[0]));
    assert_eq!(to_string(&g, d[1]), "R*C");
    assert!(collect(&mut g, h, sid).is_none());
    let p = collect(&mut g, den, sid).unwrap();
    assert_eq!(p.len(), 2);
}

#[test]
fn prunes_small_terms() {
    let mut g: Graph<BigRational> = Graph::new();
    let (a, b) = (g.sym("a"), g.sym("b"));
    let sum = g.add(a, b);
    let poly = vec![sum];
    let mut env = HashMap::new();
    env.insert(sym_id(&g, a), 1.0);
    env.insert(sym_id(&g, b), 1e-9);
    let (pruned, total, kept) = prune_poly(&mut g, &poly, &env, 1.0, 1e-6);
    assert_eq!((total, kept), (2, 1));
    assert_eq!(pruned[0], a);
}

/// Over small random ring programs (a large one expands to polynomials of
/// high degree, which cancel in floating point): where a rational form
/// exists, `N(s) / D(s)`
/// is the expression, and where a polynomial exists, its coefficients are.
#[test]
fn rational_forms_evaluate_like_the_expression() {
    use rsdag::symbolic::poly::{collect, poly_to_expr, rational_form};
    use rsdag::synth::{build, inputs, Spec, Vocabulary};
    use rsdag::{eval, Graph, F64};
    let (mut forms, mut polys) = (0, 0);
    for seed in 0..60u64 {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(seed)
            .steps(15)
            .params(3)
            .vocab(Vocabulary::Ring)
            .smooth();
        let (roots, syms) = build(&mut g, &mut spec);
        let (e, s) = (roots[0], syms[0]);
        let s_e = g.symbol_expr(s);
        let row = inputs(&mut spec.rng(), syms.len());
        let env: std::collections::HashMap<_, _> = syms.iter().copied().zip(row).collect();
        let close = |x: f64, y: f64| {
            (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-8 * (1.0 + x.abs().max(y.abs()))
        };
        let want = eval(&g, &[e], &env)[0];
        if let Some((num, den)) = rational_form(&mut g, e, s) {
            let (n, d) = (
                poly_to_expr(&mut g, &num, s_e),
                poly_to_expr(&mut g, &den, s_e),
            );
            let got = eval(&g, &[n, d], &env);
            if got[1].abs() > 1e-9 && want.is_finite() {
                assert!(
                    close(got[0] / got[1], want),
                    "seed {seed}: {} vs {want}",
                    got[0] / got[1]
                );
                forms += 1;
            }
        }
        if let Some(coeffs) = collect(&mut g, e, s) {
            let p = poly_to_expr(&mut g, &coeffs, s_e);
            let got = eval(&g, &[p], &env)[0];
            if want.is_finite() {
                assert!(close(got, want), "seed {seed}: {got} vs {want}");
                polys += 1;
            }
        }
    }
    assert!(
        forms > 30 && polys > 5,
        "{forms} forms, {polys} polynomials checked"
    );
}
