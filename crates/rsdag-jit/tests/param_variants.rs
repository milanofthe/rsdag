//! Per-binding variants natively: the variants compiled like the full body,
//! bit for bit the interpreter through bindings that flip the branches,
//! built in the background or by the prolog that needs them, and a state
//! one backend's prolog left read alike by the other's main phase.

#[path = "../../rsdag/tests/param_variants.rs"]
mod variants;

use rsdag::Tape;
use rsdag_jit::{NativeTape, Options};
use variants::{circuit, inputs, same, variant_of};

#[test]
fn variants_run_natively() {
    for n in [1usize, 12] {
        let c = circuit(n);
        let tape = Tape::compile_split(&c.g, &c.roots, &c.syms, &c.pure);
        let native = NativeTape::compile(&tape).expect("native");
        let (mut w, mut want) = (Vec::new(), Vec::new());
        let (mut nw, mut got) = (Vec::new(), Vec::new());
        for binding in [0u32, 0b1010_0101, 0xff, 0] {
            // The first prolog runs the full body and finds the selects in
            // the background, the next one builds the pattern's variant
            // there, the one after has it compiled; an instance runs on the
            // interpreted variant meanwhile. Each is checked.
            native.eval_prolog(&inputs(n, 0.0, binding), &mut nw);
            for t in [0.0, 0.5] {
                let ins = inputs(n, t, binding);
                tape.eval(&ins, &mut w, &mut want);
                native.eval_main(&ins, &mut nw, &mut got);
                assert!(same(&got, &want), "{n} at {binding:#x}, t = {t}, full body");
            }
            for _ in 0..3 {
                rsdag_jit::background::drain();
                native.eval_prolog(&inputs(n, 0.0, binding), &mut nw);
            }
            for t in [0.0, 0.5, -1.25] {
                let ins = inputs(n, t, binding);
                tape.eval(&ins, &mut w, &mut want);
                native.eval_main(&ins, &mut nw, &mut got);
                assert!(same(&got, &want), "{n} at {binding:#x}, t = {t}");
            }
        }
    }
}

/// Without the background, the native prolog builds and compiles its
/// instance's variant itself: the first one runs on it.
#[test]
fn a_native_prolog_builds_its_variant_without_the_background() {
    let n = 12;
    let c = circuit(n);
    let tape = Tape::compile_split(&c.g, &c.roots, &c.syms, &c.pure);
    let opts = Options {
        background: false,
        ..Options::default()
    };
    let native = NativeTape::compile_opts(&tape, &opts, &[]).expect("native");
    let body = native.bundles()[0].clone();
    assert_ne!(variant_of(body.as_ref(), &[0.4, 2.0]), 0.0);
    let (mut w, mut want) = (Vec::new(), Vec::new());
    let (mut nw, mut got) = (Vec::new(), Vec::new());
    for binding in [0u32, 0b0110_1001] {
        native.eval_prolog(&inputs(n, 0.0, binding), &mut nw);
        let ins = inputs(n, 0.5, binding);
        tape.eval(&ins, &mut w, &mut want);
        native.eval_main(&ins, &mut nw, &mut got);
        assert!(same(&got, &want), "at {binding:#x}");
    }
}

#[test]
fn a_prolog_of_one_backend_serves_the_main_phase_of_the_other() {
    let n = 12;
    let c = circuit(n);
    let tape = Tape::compile_split(&c.g, &c.roots, &c.syms, &c.pure);
    let native = NativeTape::compile(&tape).expect("native");
    let s = tape.state_len();
    let (mut w, mut want) = (Vec::new(), Vec::new());
    let mut got = Vec::new();
    for binding in [0x3cu32, 0xc3] {
        let ins = inputs(n, 0.75, binding);
        tape.eval(&ins, &mut w, &mut want);
        // The interpreter's prolog under the native main phase.
        let mut iw = vec![0.0; tape.work_len()];
        tape.eval_prolog_into(&inputs(n, 0.0, binding), &mut iw);
        let mut nw = Vec::new();
        native.eval_prolog(&inputs(n, 0.0, binding ^ 0xff), &mut nw);
        nw[..s].copy_from_slice(&iw[..s]);
        native.eval_main(&ins, &mut nw, &mut got);
        assert!(same(&got, &want), "interpreter prolog, native main");
        // The native prolog under the interpreter's main phase.
        native.eval_prolog(&inputs(n, 0.0, binding), &mut nw);
        let mut iw2 = vec![f64::NAN; tape.work_len()];
        iw2[..s].copy_from_slice(&nw[..s]);
        let mut o = vec![0.0; tape.out_len()];
        tape.eval_main_into(&ins, &mut iw2, &mut o);
        assert!(same(&o, &want), "native prolog, interpreter main");
    }
}
