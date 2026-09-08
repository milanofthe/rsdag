//! Three-leg parity for the chunked JIT over synthetic programs.
//!
//! Arena sweep, tape interpreter and native code must agree on every program
//! the generator draws. Chunk sizes down to 3 ops force values across chunk
//! boundaries constantly, so the work-array store-through contract is
//! exercised hard rather than incidentally.

use rsgb::synth::{cases, Spec, Vocabulary};
use rsgb::{Graph, Tape};
use rsgb_jit::ChunkedTape;

/// The corpus both fuzz tests draw from: sizes and vocabularies varying with
/// the seed, three symbols' worth of inputs.
fn corpus(seeds: std::ops::Range<u64>) -> impl Iterator<Item = rsgb::synth::Case> {
    cases(seeds, |seed| {
        let spec = Spec::new(seed)
            .steps(6 + (seed as usize % 30))
            .params(3)
            .outputs(1 + seed as usize % 3);
        match seed % 3 {
            0 => spec.vocab(Vocabulary::Ring).max_list(20),
            1 => spec.vocab(Vocabulary::Elementary),
            _ => spec.vocab(Vocabulary::Full),
        }
    })
}

#[test]
fn chunked_jit_matches_the_arena_at_every_chunk_size() {
    for (i, case) in corpus(0..250).enumerate() {
        // The tape leg first, then native code at an adversarial chunk size.
        let (mut w, mut o) = (Vec::new(), Vec::new());
        case.expect_bits("tape", |row| {
            case.tape.eval(row, &mut w, &mut o);
            o.clone()
        });
        let chunk_ops = [3, 7, rsgb_jit::CHUNK_OPS][i % 3];
        let jit = ChunkedTape::compile_with(&case.tape, chunk_ops).expect("compile chunks");
        let (mut jw, mut jo) = (Vec::new(), Vec::new());
        case.expect_bits(&format!("chunked jit (chunk {chunk_ops})"), |row| {
            jit.eval(row, &mut jw, &mut jo);
            jo.clone()
        });
    }
}

/// Specialize-then-compile: a choice-specialized (shortened) tape compiled
/// with the chunked backend must reproduce the interpreted specialization,
/// real outputs and guard outputs alike.
#[test]
fn compiled_specialized_tape_matches_the_interpreter() {
    for (i, case) in corpus(1000..1150).enumerate() {
        let row = &case.rows[0];
        let (mut w, mut o, mut choices) = (Vec::new(), Vec::new(), Vec::new());
        case.tape.eval_traced(row, &mut w, &mut o, &mut choices);
        let spec = case.tape.specialize(&choices);
        let jit = ChunkedTape::compile_with(spec.tape(), [3, rsgb_jit::CHUNK_OPS][i % 2])
            .expect("compile specialized tape");

        // Every row, not just the one the choices were traced on: outside
        // the region the two paths must still agree with each other.
        let (mut wi, mut oi) = (Vec::new(), Vec::new());
        let (mut wn, mut on) = (Vec::new(), Vec::new());
        for probe in &case.rows {
            spec.tape().eval(probe, &mut wi, &mut oi);
            jit.eval(probe, &mut wn, &mut on);
            assert_eq!(oi.len(), on.len());
            for (k, (a, b)) in oi.iter().zip(&on).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                    "seed {}: specialized output {k}, interpreter {a:?} vs native {b:?}",
                    case.seed
                );
            }
        }
    }
}

/// The symbol behind a symbol node, for the hand-written case below.
fn symbol_of(g: &Graph, e: rsgb::ExprId) -> rsgb::SymbolId {
    match *g.node(e) {
        rsgb::Node::Symbol(s) => s,
        _ => unreachable!(),
    }
}

/// A call into a symbolic function evaluates through its interpreted body on
/// both backends (bit-identical), and a short input array pads to NaN instead
/// of reading out of bounds.
#[test]
fn function_call_and_short_input_parity() {
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let y = ctx.sym("y");
    // f(p) = p*p + 1, applied to x.
    let p = ctx.sym("p");
    let ps = symbol_of(&ctx, p);
    let pp = ctx.mul(p, p);
    let one = ctx.one();
    let body = ctx.add(pp, one);
    let f = ctx.define_func("sq1", vec![ps], vec![body]);
    let o = ctx.call(f, 0, &[x]);
    let s = ctx.add(o, y);
    let e = ctx.exp(s);

    let (xi, yi) = (symbol_of(&ctx, x), symbol_of(&ctx, y));
    let tape = Tape::compile(&ctx, &[e, s], &[xi, yi]);
    let jit = ChunkedTape::compile_with(&tape, 2).expect("compile");

    let (mut w1, mut o1) = (Vec::new(), Vec::new());
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    tape.eval(&[0.5, 2.0], &mut w1, &mut o1);
    jit.eval(&[0.5, 2.0], &mut w2, &mut o2);
    assert_eq!(o1[1], 0.25 + 1.0 + 2.0);
    assert_eq!(o1[0].to_bits(), o2[0].to_bits());
    assert_eq!(o1[1].to_bits(), o2[1].to_bits());

    // Short input array: interpreter yields NaN for the missing input; the
    // JIT's padding must reproduce that instead of reading out of bounds.
    tape.eval(&[0.5], &mut w1, &mut o1);
    jit.eval(&[0.5], &mut w2, &mut o2);
    assert!(o1[0].is_nan() && o2[0].is_nan());
    assert!(o1[1].is_nan() && o2[1].is_nan());
}
