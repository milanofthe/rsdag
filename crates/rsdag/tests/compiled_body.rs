//! A consumer may register a body it compiled itself for a symbolic
//! function: a program whose calls the body covers uses it, a program that
//! calls an expression output it lacks (a derivative demanded later) takes
//! the interpreted body; the symbolic outputs stay throughout.

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
    fn call_into(&self, args: &[f64], _work: &mut [f64], out: &mut [f64]) {
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
    assert_eq!(rsdag::eval(&g, &[call], &env), vec![15.0]);
    // One sweep over both: one body for the whole sweep, the interpreted one.
    assert_eq!(rsdag::eval(&g, &[call, d], &env), vec![3.0, 2.0]);
    let later = Tape::compile(&g, &[call, d], &[sy]);
    later.eval(&[1.5], &mut w, &mut o);
    assert_eq!(o, vec![3.0, 2.0]);
    // A program that calls only what the registered body carries keeps it.
    let only = Tape::compile(&g, &[call], &[sy]);
    only.eval(&[1.5], &mut w, &mut o);
    assert_eq!(o, vec![15.0]);
    assert!(
        !g.func(f).compiled.is_empty(),
        "the registration stays for the consumer to refresh"
    );
}

/// `100 x` for output 0 only.
struct Hundred;

impl ExternBundle for Hundred {
    fn n_outputs(&self) -> usize {
        1
    }
    fn call_into(&self, args: &[f64], _work: &mut [f64], out: &mut [f64]) {
        out[0] = 100.0 * args[0];
    }
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        for g in 0..n_groups {
            out[g] = 100.0 * args[g * n_args];
        }
    }
}

/// `1000 x, 2000 x` for both outputs.
struct Thousands;

impl ExternBundle for Thousands {
    fn n_outputs(&self) -> usize {
        2
    }
    fn call_into(&self, args: &[f64], _work: &mut [f64], out: &mut [f64]) {
        out[0] = 1000.0 * args[0];
        out[1] = 2000.0 * args[0];
    }
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        for g in 0..n_groups {
            out[2 * g] = 1000.0 * args[g * n_args];
            out[2 * g + 1] = 2000.0 * args[g * n_args];
        }
    }
}

/// Two registered bodies: a call site that needs one output takes the
/// body computing the fewest outputs that covers it, a site that needs
/// both takes the one carrying both.
#[test]
fn each_call_site_takes_the_smallest_body_covering_its_outputs() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let sx = match g.node(x) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let (two, three) = (g.konst_f64(2.0), g.konst_f64(3.0));
    let twice = g.mul(two, x);
    let thrice = g.mul(three, x);
    let f = g.define_func("both", vec![sx], vec![twice, thrice]);
    g.set_func_body(
        f,
        Body {
            bundle: Arc::new(Thousands),
            slot_of: vec![Some(0), Some(1)],
        },
    );
    g.set_func_body(
        f,
        Body {
            bundle: Arc::new(Hundred),
            slot_of: vec![Some(0), None],
        },
    );
    assert_eq!(g.func(f).compiled.len(), 2);
    let (y, z) = (g.sym("y"), g.sym("z"));
    let syms: Vec<SymbolId> = [y, z]
        .iter()
        .map(|&e| match g.node(e) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        })
        .collect();
    let only0 = g.call(f, 0, &[y]);
    let both0 = g.call(f, 0, &[z]);
    let both1 = g.call(f, 1, &[z]);
    let tape = Tape::compile(&g, &[only0, both0, both1], &syms);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[1.0, 2.0], &mut w, &mut o);
    assert_eq!(o, vec![100.0, 2000.0, 4000.0]);
}
