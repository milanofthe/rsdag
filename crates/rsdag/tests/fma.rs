//! Contraction: `a*b + c` as one fused multiply-add when a compile asks for
//! it. Off, the tape is the bit-exact reference; on, every backend fuses the
//! same way, so they still agree with each other, and the whole program
//! stays within a rounding of the uncontracted one.

use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::{CompileOptions, Graph, Tape, F64};

fn corpus(
    seed: u64,
) -> (
    Graph<F64>,
    Vec<rsdag::ExprId>,
    Vec<rsdag::SymbolId>,
    Vec<f64>,
) {
    let mut g: Graph<F64> = Graph::new();
    // Ring only, so the difference between the two programs is the
    // contraction and nothing else.
    let mut spec = Spec::new(seed)
        .steps(80 + (seed as usize % 40))
        .params(6)
        .outputs(3)
        .vocab(Vocabulary::Ring)
        .smooth();
    let (roots, syms) = build(&mut g, &mut spec);
    let row = inputs(&mut spec.rng(), syms.len());
    (g, roots, syms, row)
}

#[test]
fn contraction_is_off_by_default() {
    let (g, roots, syms, _) = corpus(1);
    let plain = Tape::compile(&g, &roots, &syms);
    assert!(
        !plain.dump().contains("Fma"),
        "the default tape must not contract"
    );
    let fused = Tape::compile_with(&g, &roots, &syms, None, CompileOptions { contract: true });
    assert!(fused.dump().contains("Fma"), "the contracted tape fuses");
    assert!(
        !fused.dump().contains("MulAdd"),
        "and leaves no unfused pair behind"
    );
}

#[test]
fn a_contracted_program_stays_within_a_rounding_of_the_reference() {
    for seed in 0..40u64 {
        let (g, roots, syms, row) = corpus(seed);
        let plain = Tape::compile(&g, &roots, &syms);
        let fused = Tape::compile_with(&g, &roots, &syms, None, CompileOptions { contract: true });
        let (mut w, mut o) = (Vec::new(), Vec::new());
        plain.eval(&row, &mut w, &mut o);
        let (mut fw, mut fo) = (Vec::new(), Vec::new());
        fused.eval(&row, &mut fw, &mut fo);
        for (k, (a, b)) in o.iter().zip(&fo).enumerate() {
            if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
                continue;
            }
            let tol = 1e-12 * (1.0 + a.abs());
            assert!(
                (a - b).abs() <= tol,
                "seed {seed} output {k}: plain {a:?} vs contracted {b:?}"
            );
        }
    }
}

/// The interpreter computes the contracted program with `f64::mul_add`,
/// which is a correctly rounded fused multiply-add. A hand-written fold with
/// the same instruction must match it to the bit.
#[test]
fn the_interpreter_fuses_with_the_hardware_instruction() {
    let mut g: Graph<F64> = Graph::new();
    let (x, y, z) = (g.sym("x"), g.sym("y"), g.sym("z"));
    let m = g.mul(x, y);
    let e = g.add(m, z);
    let syms: Vec<_> = [x, y, z]
        .iter()
        .map(|&s| match g.node(s) {
            rsdag::Node::Symbol(id) => *id,
            _ => unreachable!(),
        })
        .collect();
    let fused = Tape::compile_with(&g, &[e], &syms, None, CompileOptions { contract: true });
    let (mut w, mut o) = (Vec::new(), Vec::new());
    for &(a, b, c) in &[(1.1, 3.7, -4.07), (1e16, 1e-16, -1.0), (0.1, 0.2, 0.3)] {
        fused.eval(&[a, b, c], &mut w, &mut o);
        assert_eq!(o[0].to_bits(), a.mul_add(b, c).to_bits(), "{a} * {b} + {c}");
    }
}
