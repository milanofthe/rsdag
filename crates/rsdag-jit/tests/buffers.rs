//! The native backend over a reused, larger or poisoned work buffer: the
//! same result as over a fresh one.

use rsdag::{ExprId, Graph, Node, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

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
fn a_reused_buffer_gives_the_fresh_result() {
    let mut g: Graph<F64> = Graph::new();
    let (big, syms) = chain(&mut g, 300);
    let (small, _) = chain(&mut g, 7);
    let big_tape = Tape::compile_split(&g, &[big], &syms, &[true, false]);
    let small_tape = Tape::compile_split(&g, &[small], &syms, &[true, false]);
    let big_native = NativeTape::compile(&big_tape).expect("compile");
    let small_native = NativeTape::compile(&small_tape).expect("compile");
    let inputs: [f64; 2] = [0.3, 0.7];
    let (mut work, mut out) = (Vec::new(), Vec::new());
    big_native.eval(&inputs, &mut work, &mut out);
    let stale = work.len();
    small_native.eval_prolog(&inputs, &mut work);
    assert_eq!(work.len(), stale);
    small_native.eval_main(&inputs, &mut work, &mut out);
    let (mut fw, mut fo) = (Vec::new(), Vec::new());
    small_native.eval(&inputs, &mut fw, &mut fo);
    assert_eq!(out[0].to_bits(), fo[0].to_bits());
    let mut poisoned = vec![f64::NAN; stale];
    small_native.eval_prolog(&inputs, &mut poisoned);
    small_native.eval_main(&inputs, &mut poisoned, &mut out);
    assert_eq!(out[0].to_bits(), fo[0].to_bits());
}
