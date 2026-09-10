//! Named calls from a frontend: every name a Verilog-A or netlist lowering
//! uses reaches a node, and the ones with a native op use it.

use rsdag::mathfn::{is_known, lower_call};
use rsdag::{ExprId, Graph, Node, SymbolId, Tape, UnaryOp, F64};

fn eval1(g: &Graph<F64>, e: ExprId, s: SymbolId, x: f64) -> f64 {
    let tape = Tape::compile(g, &[e], &[s]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[x], &mut w, &mut o);
    o[0]
}

/// The set SANE's `mathfn` covers, which is what the Verilog-A lowering and
/// the netlist behavioural source call.
const VERILOG_A_UNARY: [&str; 20] = [
    "exp", "limexp", "ln", "log", "log10", "log2", "sqrt", "abs", "sin", "cos", "tan", "asin",
    "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
];
const VERILOG_A_BINARY: [&str; 5] = ["pow", "atan2", "hypot", "min", "max"];

#[test]
fn every_name_a_frontend_uses_lowers() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let y = g.sym("y");
    for name in VERILOG_A_UNARY {
        assert!(is_known(name, 1), "{name} is not known at arity 1");
        assert!(
            lower_call(&mut g, name, &[x]).is_some(),
            "{name} did not lower"
        );
    }
    for name in VERILOG_A_BINARY {
        assert!(is_known(name, 2), "{name} is not known at arity 2");
        assert!(
            lower_call(&mut g, name, &[x, y]).is_some(),
            "{name} did not lower"
        );
    }
    // `floor` and `ceil` come with the vocabulary rather than with a rule.
    for name in ["floor", "ceil", "trunc", "round", "erf", "lgamma"] {
        assert!(lower_call(&mut g, name, &[x]).is_some(), "{name}");
    }
    assert!(lower_call(&mut g, "no_such_function", &[x]).is_none());
    assert!(lower_call(&mut g, "exp", &[x, y]).is_none(), "wrong arity");
}

#[test]
fn a_name_with_a_native_op_uses_it() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    // `log10`, not `ln(x) / ln(10)`: one node, and the platform's own value.
    let e = lower_call(&mut g, "log", &[x]).unwrap();
    assert!(matches!(g.node(e), Node::Unary(UnaryOp::Log10, _)));
    assert_eq!(e, lower_call(&mut g, "log10", &[x]).unwrap());
    // `limexp` is the guarded exponential the backends already share.
    assert_eq!(
        lower_call(&mut g, "limexp", &[x]).unwrap(),
        lower_call(&mut g, "exp", &[x]).unwrap()
    );
}

#[test]
fn the_lowered_values_are_the_expected_ones() {
    let mut g: Graph<F64> = Graph::new();
    let xe = g.sym("x");
    let s = match g.node(xe) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let cases: [(&str, f64, f64); 6] = [
        ("log", 100.0, 2.0),
        ("log2", 8.0, 3.0),
        ("sqrt", 9.0, 3.0),
        ("abs", -2.5, 2.5),
        ("floor", 2.7, 2.0),
        ("tanh", 0.0, 0.0),
    ];
    for (name, arg, want) in cases {
        let e = lower_call(&mut g, name, &[xe]).unwrap();
        let got = eval1(&g, e, s, arg);
        assert!(
            (got - want).abs() < 1e-12,
            "{name}({arg}) = {got}, want {want}"
        );
    }
    // min / max are reductions, and they pick the right operand.
    let ye = g.sym("y");
    let sy = match g.node(ye) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    for (name, want) in [("min", 1.0), ("max", 4.0)] {
        let e = lower_call(&mut g, name, &[xe, ye]).unwrap();
        let tape = Tape::compile(&g, &[e], &[s, sy]);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&[4.0f64, 1.0], &mut w, &mut o);
        assert_eq!(o[0], want, "{name}");
    }
}

#[test]
fn a_lowered_call_differentiates() {
    let mut g: Graph<F64> = Graph::new();
    let xe = g.sym("x");
    let s = match g.node(xe) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    // d/dx log10(x) = 1 / (x ln 10)
    let e = lower_call(&mut g, "log", &[xe]).unwrap();
    let d = rsdag::differentiate(&mut g, e, s);
    let got = eval1(&g, d, s, 4.0);
    let want = 1.0 / (4.0 * std::f64::consts::LN_10);
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    // min carries a subgradient: below the other operand the derivative is 1.
    let ye = g.sym("y");
    let m = lower_call(&mut g, "min", &[xe, ye]).unwrap();
    let dm = rsdag::differentiate(&mut g, m, s);
    let sy = match g.node(ye) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let tape = Tape::compile(&g, &[dm], &[s, sy]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[1.0f64, 5.0], &mut w, &mut o);
    assert_eq!(o[0], 1.0, "d min(x, y)/dx with x below y");
}
