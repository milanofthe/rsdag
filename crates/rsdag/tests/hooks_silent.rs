//! Without a log sink there is nothing to report, so a compile never reads the
//! clock. That is what keeps rsdag running on a target without one
//! (`wasm32-unknown-unknown`, where `std::time::Instant::now()` panics): the
//! clock is installed by hosts that want timings, not required of the rest.

use rsdag::hooks::{self, Clock};
use rsdag::{Graph, Node, Tape, F64};

struct Trap;
impl Clock for Trap {
    fn now_ns(&self) -> u64 {
        panic!("time not implemented on this platform");
    }
}
static CLOCK: Trap = Trap;

#[test]
fn a_compile_without_a_sink_never_reads_the_clock() {
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
}
