//! The polynomial degree of an expression in a set of unknowns is exact, and
//! everything that breaks polynomial structure is named.

use std::collections::BTreeSet;

use rsdag::{nonlinearity, nonlinearity_of, CmpOp, Degree, ExprId, Graph, SymbolId, UnaryOp};

fn vars(g: &Graph, of: &[ExprId]) -> BTreeSet<SymbolId> {
    g.free_symbols_in(of)
}

#[test]
fn polynomial_degree_is_exact() {
    let mut g: Graph = Graph::new();
    let (x, a, b, c) = (g.sym("x"), g.sym("a"), g.sym("b"), g.sym("c"));
    let x2 = g.pow_i(x, 2);
    let ax2 = g.mul(a, x2);
    let bx = g.mul(b, x);
    let s = g.add(ax2, bx);
    let expr = g.add(s, c);
    let nl = nonlinearity(&g, expr, &vars(&g, &[x]));
    assert_eq!(nl.degree, Degree::Finite(2));
    assert!(nl.is_polynomial());
    assert!(nl.transcendental.is_empty());
    assert_eq!(nl.alias_free_samples(3), Some(2 * 2 * 3 + 1));
}

#[test]
fn parameters_do_not_raise_the_degree() {
    let mut g: Graph = Graph::new();
    let (x, a) = (g.sym("x"), g.sym("a"));
    let ea = g.exp(a);
    let expr = g.mul(ea, x); // exp(a) * x is linear in x
    let nl = nonlinearity(&g, expr, &vars(&g, &[x]));
    assert_eq!(nl.degree, Degree::Finite(1));
    assert!(nl.transcendental.is_empty());
}

#[test]
fn a_diode_is_transcendental() {
    let mut g: Graph = Graph::new();
    let (x, vt) = (g.sym("x"), g.sym("vt"));
    let q = g.div(x, vt);
    let expr = g.exp(q);
    let nl = nonlinearity(&g, expr, &vars(&g, &[x]));
    assert_eq!(nl.degree, Degree::Unbounded);
    assert!(!nl.is_polynomial());
    assert!(nl.transcendental.contains(&UnaryOp::Exp));
    assert_eq!(nl.alias_free_samples(5), None);
}

#[test]
fn a_reciprocal_of_an_unknown_is_rational() {
    let mut g: Graph = Graph::new();
    let x = g.sym("x");
    let expr = g.pow_i(x, -1);
    let nl = nonlinearity(&g, expr, &vars(&g, &[x]));
    assert!(nl.rational);
    assert_eq!(nl.degree, Degree::Unbounded);
}

#[test]
fn a_variable_branch_is_piecewise_and_a_fixed_one_is_not() {
    let mut g: Graph = Graph::new();
    let (x, p) = (g.sym("x"), g.sym("p"));
    let zero = g.zero();
    let on_x = g.cmp(CmpOp::Gt, x, zero);
    let nx = g.neg(x);
    let abs_like = g.select(on_x, x, nx);
    let nl = nonlinearity(&g, abs_like, &vars(&g, &[x]));
    assert!(nl.piecewise);
    assert_eq!(nl.degree, Degree::Unbounded);

    // The same select on a parameter is a fixed choice between two
    // polynomials: degree is the larger arm's.
    let on_p = g.cmp(CmpOp::Gt, p, zero);
    let x2 = g.pow_i(x, 2);
    let fixed = g.select(on_p, x2, x);
    let nl = nonlinearity(&g, fixed, &vars(&g, &[x]));
    assert!(!nl.piecewise);
    assert_eq!(nl.degree, Degree::Finite(2));
}

#[test]
fn reductions_and_dots_follow_sum_and_product_rules() {
    let mut g: Graph = Graph::new();
    let (x, y, a) = (g.sym("x"), g.sym("y"), g.sym("a"));
    let v = vars(&g, &[x, y]);
    let prod = g.reduce(rsdag::ReduceOp::Product, vec![x, y, a]);
    assert_eq!(nonlinearity(&g, prod, &v).degree, Degree::Finite(2));
    let sum = g.reduce(rsdag::ReduceOp::Sum, vec![x, prod]);
    assert_eq!(nonlinearity(&g, sum, &v).degree, Degree::Finite(2));
    // x*x + y*a: degree 2 from the first term.
    let dot = g.dot(vec![x, y], vec![x, a]);
    assert_eq!(nonlinearity(&g, dot, &v).degree, Degree::Finite(2));
    let m = g.reduce(rsdag::ReduceOp::Max, vec![x, a]);
    let nl = nonlinearity(&g, m, &v);
    assert!(nl.piecewise && nl.degree == Degree::Unbounded);
}

#[test]
fn a_call_on_an_unknown_is_opaque() {
    let mut g: Graph = Graph::new();
    let p = g.sym("p");
    let ps = match g.node(p) {
        rsdag::Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let body = g.mul(p, p);
    let f = g.define_func("sq", vec![ps], vec![body]);
    let x = g.sym("x");
    let call = g.call(f, 0, &[x]);
    let nl = nonlinearity(&g, call, &vars(&g, &[x]));
    assert!(nl.opaque);
    assert_eq!(nl.degree, Degree::Unbounded);
}

#[test]
fn a_system_takes_the_worst_case() {
    let mut g: Graph = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    let v = vars(&g, &[x, y]);
    let lin = g.add(x, y);
    let cubic = g.pow_i(y, 3);
    let nl = nonlinearity_of(&g, &[lin, cubic], &v);
    assert_eq!(nl.degree, Degree::Finite(3));
    let e = g.exp(x);
    let nl = nonlinearity_of(&g, &[lin, cubic, e], &v);
    assert_eq!(nl.degree, Degree::Unbounded);
    assert!(nl.transcendental.contains(&UnaryOp::Exp));
}
