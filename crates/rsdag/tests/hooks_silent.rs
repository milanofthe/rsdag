//! With nobody listening for debug reports there is nothing to report, so a
//! compile never reads the clock. That is what keeps rsdag running on a target
//! without one (`wasm32-unknown-unknown`, where `std::time::Instant::now()`
//! panics): the clock is installed by hosts that want timings, not required of
//! the rest. Two ways of not listening: no sink at all, and a sink whose
//! logger is off.

use rsdag::hooks::{self, Clock, Level, Log};
use rsdag::{Graph, Node, Tape, F64};

struct Trap;
impl Clock for Trap {
    fn now_ns(&self) -> u64 {
        panic!("time not implemented on this platform");
    }
}
static CLOCK: Trap = Trap;

struct Off;
impl Log for Off {
    fn log(&self, _: Level, msg: &str) {
        panic!("reported to a sink that is off: {msg}");
    }
    fn enabled(&self, _: Level) -> bool {
        false
    }
}
static OFF: Off = Off;

#[test]
fn a_compile_nobody_listens_to_never_reads_the_clock() {
    hooks::set_clock(&CLOCK);

    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let e = g.mul(x, x);
    let Node::Symbol(s) = *g.node(x) else {
        unreachable!()
    };
    let tape = Tape::compile(&g, &[e], &[s]);

    let (mut work, mut out) = (Vec::new(), Vec::new());
    tape.eval(&[3.0], &mut work, &mut out);
    assert_eq!(out, vec![9.0]);

    // Same with a sink installed whose logger is off.
    hooks::set_log(&OFF);
    let tape = Tape::compile(&g, &[e], &[s]);
    tape.eval(&[3.0], &mut work, &mut out);
    assert_eq!(out, vec![9.0]);
}
