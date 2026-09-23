//! The domain guards clamp wild iterates, they do not invent numbers: a
//! NaN argument stays NaN through `exp`, `ln` and `sqrt` and through their
//! derivatives, while the clamped values themselves are unchanged.

use rsdag::node::UnaryOp;
use rsdag::semantics::{unary_f64, LN_FLOOR};
use rsdag::BigRational;
use rsdag::{differentiate, ExprId, Graph, Node, SymbolId, Tape, F64};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

#[test]
fn nan_passes_every_guard() {
    for op in [UnaryOp::Exp, UnaryOp::Ln, UnaryOp::Sqrt] {
        assert!(unary_f64(op, f64::NAN).is_nan(), "{op:?}");
    }
    assert_eq!(unary_f64(UnaryOp::Ln, 0.0), LN_FLOOR.ln());
    assert_eq!(unary_f64(UnaryOp::Ln, -3.0), LN_FLOOR.ln());
    assert_eq!(unary_f64(UnaryOp::Sqrt, -1.0), 0.0);
    assert_eq!(unary_f64(UnaryOp::Sqrt, -0.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(unary_f64(UnaryOp::Sqrt, 4.0), 2.0);
}

#[test]
fn nan_passes_the_derivatives_and_a_missing_input_shows() {
    let mut g: Graph<F64> = Graph::new();
    let (x, xs) = sym(&mut g, "x");
    let (_, ys) = sym(&mut g, "y");
    let mut roots = Vec::new();
    for op in [UnaryOp::Exp, UnaryOp::Ln, UnaryOp::Sqrt] {
        let v = g.unary(op, x);
        let d = differentiate(&mut g, v, xs);
        roots.push(v);
        roots.push(d);
    }
    let tape = Tape::compile(&g, &roots, &[xs]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[f64::NAN], &mut w, &mut o);
    assert!(o.iter().all(|v| v.is_nan()), "{o:?}");
    // In range and clamped, derivatives as before.
    tape.eval(&[-2.0], &mut w, &mut o);
    assert_eq!(o[3], 0.0, "ln' below the floor");
    assert_eq!(o[5], 0.0, "sqrt' below zero");
    // An input the caller did not pass is NaN, and ln and sqrt no longer
    // turn it into a finite value.
    let tape = Tape::compile(&g, &roots, &[ys, xs]);
    tape.eval(&[1.0], &mut w, &mut o);
    assert!(o.iter().all(|v| v.is_nan()), "{o:?}");
}

#[test]
fn integer_powers_stay_in_range() {
    let mut g: Graph<F64> = Graph::new();
    let (x, xs) = sym(&mut g, "x");
    // An exponent past i32 is a real power, not a truncated integer one.
    let big = g.pow_i(x, 1 << 40);
    assert!(matches!(g.node(big), Node::Binary(rsdag::BinOp::Powf, ..)));
    // A nested power whose product overflows stays nested.
    let p = g.pow_i(x, 1 << 20);
    let pp = g.pow_i(p, 1 << 20);
    assert!(matches!(*g.node(pp), Node::Pow(b, n) if b == p && n == 1 << 20));
    let tape = Tape::compile(&g, &[big, pp], &[xs]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[1.0], &mut w, &mut o);
    assert_eq!(o, vec![1.0, 1.0]);
    tape.eval(&[0.5], &mut w, &mut o);
    assert_eq!(o, vec![0.0, 0.0]);

    // An exact constant to a huge power stays a node instead of growing
    // millions of digits.
    let mut q: Graph<BigRational> = Graph::new();
    let three = q.konst_int(3);
    let e = q.pow_i(three, 1_000_000);
    assert!(matches!(q.node(e), Node::Pow(..)));
    let one = q.konst_int(-1);
    let e = q.pow_i(one, 1_000_001);
    assert!(q.const_of(e).is_some());
}
