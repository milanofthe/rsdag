//! The native backend's matrix-vector kernel: inputs read in place and
//! gathered slots, bit-identical to the interpreter.

use rsdag::{ExprId, Graph, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

#[test]
fn the_kernel_matches_the_interpreter() {
    for (n, m, computed) in [(8usize, 2usize, false), (40, 3, false), (12, 1, true)] {
        let mut g: Graph<F64> = Graph::new();
        let a: Vec<ExprId> = (0..n * n).map(|k| g.sym(&format!("a{k}"))).collect();
        let x: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
        let b: Vec<ExprId> = (0..n * m).map(|k| g.sym(&format!("b{k}"))).collect();
        let u: Vec<ExprId> = (0..m).map(|i| g.sym(&format!("u{i}"))).collect();
        let syms: Vec<SymbolId> = (0..(n * n + n + n * m + m) as u32).map(SymbolId).collect();
        let (a, x) = if computed {
            // Entries computed from the inputs, so the kernel gathers.
            let a2 = a.iter().map(|&e| g.exp(e)).collect();
            let x2 = x.iter().map(|&e| g.sin(e)).collect();
            (a2, x2)
        } else {
            (a, x)
        };
        let roots: Vec<ExprId> = (0..n)
            .map(|i| {
                let ax = g.dot(a[i * n..(i + 1) * n].to_vec(), x.clone());
                let bu = g.dot(b[i * m..(i + 1) * m].to_vec(), u.clone());
                g.add(ax, bu)
            })
            .collect();
        let tape = Tape::compile(&g, &roots, &syms);
        assert!(tape.dump().contains("Gemv"));
        let native = NativeTape::compile(&tape).expect("compile");
        let inputs: Vec<f64> = (0..syms.len())
            .map(|k| 0.03 * (k % 17) as f64 - 0.2)
            .collect();
        let (mut w1, mut o1, mut w2, mut o2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        tape.eval(&inputs, &mut w1, &mut o1);
        native.eval(&inputs, &mut w2, &mut o2);
        for (k, (p, q)) in o1.iter().zip(&o2).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "n {n} computed {computed}: output {k}"
            );
        }
    }
}
