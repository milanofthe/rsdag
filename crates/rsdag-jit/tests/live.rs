//! A specialized tape's prolog guards read from the work array after the
//! prolog pass: the native backend keeps the guard slots written when asked
//! to, and the check then agrees with the interpreter's.

use rsdag::{CmpOp, Graph, Node, SymbolId, Tape, F64};
use rsdag_jit::{NativeTape, Options};

#[test]
fn prolog_guards_are_checked_against_the_native_prolog() {
    let mut g: Graph<F64> = Graph::new();
    let (p, x) = (g.sym("p"), g.sym("x"));
    let syms: Vec<SymbolId> = [p, x]
        .iter()
        .map(|&e| match g.node(e) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        })
        .collect();
    // The choice depends on the parameter alone: its guard lives in the
    // prolog.
    let zero = g.zero();
    let c = g.cmp(CmpOp::Gt, p, zero);
    let two = g.konst_f64(2.0);
    let three = g.konst_f64(3.0);
    let a = g.mul(two, x);
    let b = g.mul(three, x);
    let root = g.select(c, a, b);
    let tape = Tape::compile_split(&g, &[root], &syms, &[true, false]);
    let (mut w, mut o, mut choices) = (Vec::new(), Vec::new(), Vec::new());
    tape.eval_with(&[1.0, 0.5], &mut w, &mut o, &mut choices);
    let spec = tape.specialize(&choices, &vec![true; tape.n_selects()]);
    let guards: Vec<u32> = spec.prolog_guards().iter().map(|&(s, _)| s).collect();
    assert!(!guards.is_empty(), "{}", spec.tape().dump());
    let native =
        NativeTape::compile_opts(spec.tape(), &Options::default(), &guards).expect("compile");
    for (inputs, holds) in [
        ([1.0, 0.5], true),
        ([-1.0, 0.5], false),
        ([2.0, -4.0], true),
    ] {
        let mut wn = Vec::new();
        native.eval_prolog(&inputs, &mut wn);
        assert_eq!(spec.check_prolog_guards(&inputs, &wn), holds, "{inputs:?}");
        let mut wi = Vec::new();
        assert_eq!(
            spec.eval_prolog_checked(&inputs, &mut wi),
            holds,
            "{inputs:?}"
        );
        if holds {
            let (mut on, mut oi) = (Vec::new(), Vec::new());
            native.eval_main(&inputs, &mut wn, &mut on);
            assert!(spec.eval_main_checked(&inputs, &mut wi, &mut oi));
            assert_eq!(on[0].to_bits(), oi[0].to_bits());
        }
    }
}
