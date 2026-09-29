//! Integer powers take rsdag's reference (`semantics::powi_f64`) in the
//! interpreter and in native code, bit for bit: `x^-1` is `1 / x` and `x^2`
//! is `x * x` exactly, on every target (a platform `powi` with a runtime
//! exponent is not; the value below is one it rounds differently).

use rsdag::{Graph, Node, Tape, F64};
use rsdag_jit::NativeTape;

#[test]
fn integer_powers_agree_with_the_reference_natively() {
    let xs = [
        f64::from_bits(0x48e7e93a8799be39),
        0.1,
        -3.7,
        1.0 + f64::EPSILON,
        7.25e-200,
        0.0,
        -0.0,
    ];
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let s = match *g.node(x) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    };
    let ns: Vec<i64> = (-4..=5).collect();
    let roots: Vec<_> = ns.iter().map(|&n| g.pow_i(x, n)).collect();
    let tape = Tape::compile(&g, &roots, &[s]);
    let native = NativeTape::compile(&tape).expect("native code");
    for &v in &xs {
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&[v], &mut w, &mut o);
        let (mut wn, mut on) = (Vec::new(), Vec::new());
        native.eval(&[v], &mut wn, &mut on);
        for (k, &n) in ns.iter().enumerate() {
            let r = rsdag::semantics::powi_f64(v, n as i32);
            assert_eq!(o[k].to_bits(), r.to_bits(), "interpreter {v:e}^{n}");
            assert_eq!(on[k].to_bits(), r.to_bits(), "native {v:e}^{n}");
        }
        assert_eq!(
            rsdag::semantics::powi_f64(v, -1).to_bits(),
            (1.0 / v).to_bits()
        );
        assert_eq!(
            rsdag::semantics::powi_f64(v, 2).to_bits(),
            (v * v).to_bits()
        );
    }
}
