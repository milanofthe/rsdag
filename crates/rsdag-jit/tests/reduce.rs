//! Min and max reductions natively: the reference's left fold with its
//! NaN rule (a NaN term yields the other) and its signed-zero outcome,
//! bit-identical to the interpreter on short lists (inline instructions)
//! and long ones (the host routine).

use rsdag::{ExprId, Graph, ReduceOp, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

#[test]
fn min_and_max_match_the_interpreter_with_nans_and_signed_zeros() {
    let values: Vec<Vec<f64>> = vec![
        vec![1.0],
        vec![2.0, -3.0],
        vec![f64::NAN, 1.0],
        vec![1.0, f64::NAN],
        vec![f64::NAN, f64::NAN, 2.0],
        vec![0.0, -0.0],
        vec![-0.0, 0.0],
        vec![3.0, 1.0, 2.0, -7.5, 4.0, 4.0, 0.5],
        (0..20).map(|k| ((k * 7919) % 23) as f64 - 11.0).collect(),
        (0..37)
            .map(|k| {
                if k % 9 == 4 {
                    f64::NAN
                } else {
                    (k as f64).sin()
                }
            })
            .collect(),
    ];
    for op in [ReduceOp::Min, ReduceOp::Max] {
        for vals in &values {
            let n = vals.len();
            let mut g: Graph<F64> = Graph::new();
            let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
            let syms: Vec<SymbolId> = (0..n as u32).map(SymbolId).collect();
            // Through a sine so the terms are slots, and once as inputs.
            let sines: Vec<ExprId> = xs.iter().map(|&x| g.sin(x)).collect();
            let r1 = g.reduce(op, sines);
            let r2 = g.reduce(op, xs);
            let tape = Tape::compile(&g, &[r1, r2], &syms);
            let native = NativeTape::compile(&tape).expect("compile");
            let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            tape.eval(vals, &mut w1, &mut o1);
            native.eval(vals, &mut w2, &mut o2);
            for k in 0..2 {
                assert_eq!(
                    o1[k].to_bits(),
                    o2[k].to_bits(),
                    "{op:?} over {vals:?}, root {k}: {} vs {}",
                    o1[k],
                    o2[k]
                );
            }
        }
    }
}
