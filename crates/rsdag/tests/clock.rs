//! A clock is read where an rsprobe context records, or where a policy
//! times its work (`rspool::Instant`), and nowhere else: a compile and an
//! evaluation with nothing recording read none. Two ways of nothing
//! recording: a context current on no thread of the work, and one whose
//! threshold leaves rsdag's sites inactive.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use rsdag::{Graph, Node, Program, Tape, F64};
use rsprobe::{Context, Level};

static READS: AtomicU64 = AtomicU64::new(0);

fn clock_counting_its_reads() -> u64 {
    READS.fetch_add(1, Relaxed);
    0
}

fn compile_and_eval() {
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

#[test]
fn a_compile_nothing_records_reads_no_clock() {
    let idle = Context::builder()
        .threshold(Level::TRACE)
        .clock(clock_counting_its_reads)
        .build();
    let before = READS.load(Relaxed);
    compile_and_eval();
    assert_eq!(READS.load(Relaxed), before);
    drop(idle);
    let quiet = Context::builder()
        .threshold(Level::ERROR)
        .clock(clock_counting_its_reads)
        .build();
    let _on = quiet.enter();
    let reads_after_the_entry = READS.load(Relaxed);
    compile_and_eval();
    assert_eq!(READS.load(Relaxed), reads_after_the_entry);
}
