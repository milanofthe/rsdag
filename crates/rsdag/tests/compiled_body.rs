//! A consumer may register a body it compiled itself for a symbolic
//! function: tapes and sweeps call that body while it covers every
//! expression output, and fall back to the interpreted body once a
//! derivative output it lacks exists; the symbolic outputs stay throughout.

use std::collections::HashMap;
use std::sync::Arc;

use rsdag::func::Body;
use rsdag::{ExprId, ExternBundle, Graph, Node, SymbolId, Tape, F64};

/// `10 x`, where the symbolic body says `2 x`.
struct TenTimes;

impl ExternBundle for TenTimes {
    fn n_outputs(&self) -> usize {
        1
    }
    fn call(&self, args: &[f64], out: &mut [f64]) {
        out[0] = 10.0 * args[0];
    }
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        for g in 0..n_groups {
            out[g] = 10.0 * args[g * n_args];
        }
    }
}

#[test]
fn a_registered_body_serves_the_tape_and_the_symbolic_outputs_stay() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let sx = match g.node(x) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let two = g.konst_f64(2.0);
    let twice = g.mul(two, x);
    let f = g.define_func("twice", vec![sx], vec![twice]);
    let y = g.sym("y");
    let sy = match g.node(y) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let call = g.call(f, 0, &[y]);
    let before = Tape::compile(&g, &[call], &[sy]);
    g.set_func_body(
        f,
        Body {
            bundle: Arc::new(TenTimes),
            slot_of: vec![Some(0)],
        },
    );
    let after = Tape::compile(&g, &[call], &[sy]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    before.eval(&[1.5], &mut w, &mut o);
    assert_eq!(o, vec![3.0], "a tape compiled before keeps its body");
    after.eval(&[1.5], &mut w, &mut o);
    assert_eq!(
        o,
        vec![15.0],
        "a tape compiled after calls the registered body"
    );
    // The sweep honours the registered body too.
    let env: HashMap<SymbolId, f64> = [(sy, 1.5)].into_iter().collect();
    assert_eq!(rsdag::eval(&g, &[call], &env), vec![15.0]);
    // A derivative demands an output the registered body does not carry:
    // calls revert to the interpreted body, the derivative is symbolic.
    let d: ExprId = rsdag::differentiate(&mut g, call, sy);
    assert_eq!(rsdag::eval(&g, &[d], &env), vec![2.0]);
    assert_eq!(rsdag::eval(&g, &[call], &env), vec![3.0]);
    let later = Tape::compile(&g, &[call, d], &[sy]);
    later.eval(&[1.5], &mut w, &mut o);
    assert_eq!(o, vec![3.0, 2.0]);
    assert!(
        g.func(f).compiled.is_some(),
        "the registration stays for the consumer to refresh"
    );
}
