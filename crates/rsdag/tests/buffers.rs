//! A caller's work buffer is reused across tapes: a buffer left larger by a
//! bigger tape serves a smaller one's prolog and main passes unchanged, and
//! nothing an evaluation reads depends on what the buffer held before.

use rsdag::{ExprId, Graph, Node, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// `sum_k sin(p * k) * x^k` over `n` terms: `p` parameter-pure, `x` not.
fn chain(g: &mut Graph<F64>, n: usize) -> (ExprId, Vec<SymbolId>) {
    let (p, x) = (g.sym("p"), g.sym("x"));
    let mut acc = g.zero();
    let mut xp = g.one();
    for k in 0..n {
        let kk = g.konst_f64(k as f64);
        let pk = g.mul(p, kk);
        let s = g.sin(pk);
        let t = g.mul(s, xp);
        acc = g.add(acc, t);
        xp = g.mul(xp, x);
    }
    (acc, vec![sym(g, p), sym(g, x)])
}

#[test]
fn a_larger_buffer_from_another_tape_serves_prolog_and_main() {
    let mut g: Graph<F64> = Graph::new();
    let (big, syms) = chain(&mut g, 200);
    let (small, _) = chain(&mut g, 5);
    let big_tape = Tape::compile_split(&g, &[big], &syms, &[true, false]);
    let small_tape = Tape::compile_split(&g, &[small], &syms, &[true, false]);
    let inputs: [f64; 2] = [0.3, 0.7];
    let (mut work, mut out) = (Vec::new(), Vec::new());
    big_tape.eval(&inputs, &mut work, &mut out);
    let stale = work.len();
    small_tape.eval_prolog(&inputs, &mut work);
    assert_eq!(work.len(), stale, "the buffer keeps its larger size");
    small_tape.eval_main(&inputs, &mut work, &mut out);
    let (mut fresh_w, mut fresh_o) = (Vec::new(), Vec::new());
    small_tape.eval(&inputs, &mut fresh_w, &mut fresh_o);
    assert_eq!(out[0].to_bits(), fresh_o[0].to_bits());
    // And a buffer full of NaN changes nothing either.
    let mut poisoned = vec![f64::NAN; stale];
    small_tape.eval_prolog(&inputs, &mut poisoned);
    small_tape.eval_main(&inputs, &mut poisoned, &mut out);
    assert_eq!(out[0].to_bits(), fresh_o[0].to_bits());
}
