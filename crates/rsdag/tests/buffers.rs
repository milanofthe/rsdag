//! A caller's work buffer is reused across tapes: a buffer left larger by a
//! bigger tape serves a smaller one's prolog and main passes unchanged, and
//! nothing an evaluation reads depends on what the buffer held before.

mod common;
use common::sine_series;

use rsdag::Program;
use rsdag::{Graph, Tape, F64};

#[test]
fn a_larger_buffer_from_another_tape_serves_prolog_and_main() {
    let mut g: Graph<F64> = Graph::new();
    let (big, syms) = sine_series(&mut g, 200);
    let (small, _) = sine_series(&mut g, 5);
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
    let mut poisoned = vec![f64::NAN; stale];
    small_tape.eval_prolog(&inputs, &mut poisoned);
    small_tape.eval_main(&inputs, &mut poisoned, &mut out);
    assert_eq!(out[0].to_bits(), fresh_o[0].to_bits());
}
