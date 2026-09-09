//! The direct emitter against the arena sweep, bit for bit, over the
//! synthetic corpus. Programs it refuses (ordered reductions, bundles) are
//! counted, not hidden.
#![cfg(target_arch = "aarch64")]

use rsdag::synth::{cases, Spec, Vocabulary};
use rsdag_jit::{EmittedTape, CHUNK_OPS};

#[test]
fn the_emitter_matches_the_arena_at_every_chunk_size() {
    let mut lowered = 0;
    let mut refused = 0;
    for (i, case) in cases(0..300, |seed| {
        let spec = Spec::new(seed)
            .steps(6 + (seed as usize % 40))
            .params(1 + seed as usize % 4)
            .outputs(1 + seed as usize % 3)
            // Ordered reductions are what the emitter refuses; the smooth
            // draw keeps most programs inside its vocabulary.
            .smooth();
        match seed % 3 {
            0 => spec.vocab(Vocabulary::Ring).max_list(20),
            1 => spec.vocab(Vocabulary::Elementary),
            _ => spec.vocab(Vocabulary::Full),
        }
    })
    .enumerate()
    {
        let chunk = [3, 7, CHUNK_OPS][i % 3];
        let emitted = match EmittedTape::compile_with(&case.tape, chunk) {
            Ok(e) => e,
            Err(_) => {
                refused += 1;
                continue;
            }
        };
        lowered += 1;
        let (mut w, mut o) = (Vec::new(), Vec::new());
        case.expect_bits(&format!("emitter (chunk {chunk})"), |row| {
            emitted.eval(row, &mut w, &mut o);
            o.clone()
        });
    }
    assert!(lowered > 250, "lowered {lowered}, refused {refused}");
}

#[test]
fn the_emitter_pads_short_inputs_with_nan() {
    use rsdag::{Graph, Tape, F64};
    let mut g: Graph<F64> = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    let e = g.add(x, y);
    let syms: Vec<_> = [x, y]
        .iter()
        .map(|&s| match g.node(s) {
            rsdag::Node::Symbol(id) => *id,
            _ => unreachable!(),
        })
        .collect();
    let tape = Tape::compile(&g, &[e], &syms);
    let em = EmittedTape::compile(&tape).unwrap();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    em.eval(&[1.0], &mut w, &mut o);
    assert!(o[0].is_nan());
    em.eval(&[1.0, 2.0], &mut w, &mut o);
    assert_eq!(o[0], 3.0);
}
