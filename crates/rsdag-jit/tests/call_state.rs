//! Stateful calls natively: the caller's prolog runs the body's prolog per
//! instance, the main phase reads the state, bit for bit the interpreter;
//! and since both backends lay out the state the same way, a prolog run by
//! one serves the main phase of the other.

#[path = "../../rsdag/tests/common/mod.rs"]
mod common;
use common::{circuit, exp_ln_device, identical, Circuit};

use rsdag::Program;
use rsdag::Tape;
use rsdag_jit::NativeTape;

#[test]
fn stateful_calls_run_natively_and_share_the_state_layout() {
    for n in [1usize, 12] {
        let Circuit {
            g,
            roots,
            syms,
            pure,
        } = circuit(n, |g| exp_ln_device(g, 30), &["q"]);
        let tape = Tape::compile_split(&g, &roots, &syms, &pure);
        let native = NativeTape::compile(&tape).expect("native");
        let ins = |t: f64| -> Vec<f64> {
            (0..syms.len())
                .map(|k| {
                    if k % 2 == 1 {
                        0.2 + 0.01 * k as f64
                    } else {
                        t + 0.1 * k as f64
                    }
                })
                .collect()
        };
        let (mut w, mut want) = (Vec::new(), Vec::new());
        let (mut nw, mut got) = (Vec::new(), Vec::new());
        native.eval_prolog(&ins(0.0), &mut nw);
        for t in [0.0, 0.5, -1.25] {
            tape.eval(&ins(t), &mut w, &mut want);
            native.eval_main(&ins(t), &mut nw, &mut got);
            assert!(identical(&got, &want), "{n}: native at t = {t}");
        }
        let s = tape.state_len();
        let mut iw = vec![0.0; tape.work_len()];
        tape.eval_prolog_into(&ins(0.0), &mut iw);
        let mut nw2 = vec![f64::NAN; nw.len()];
        nw2[..s].copy_from_slice(&iw[..s]);
        native.eval_main(&ins(0.75), &mut nw2, &mut got);
        tape.eval(&ins(0.75), &mut w, &mut want);
        assert!(
            identical(&got, &want),
            "{n}: interpreter prolog, native main"
        );
        let mut iw2 = vec![f64::NAN; tape.work_len()];
        iw2[..s].copy_from_slice(&nw[..s]);
        let mut o = vec![0.0; tape.out_len()];
        tape.eval_main_into(&ins(0.75), &mut iw2, &mut o);
        assert!(identical(&o, &want), "{n}: native prolog, interpreter main");
    }
}
