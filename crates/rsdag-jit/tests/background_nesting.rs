//! Background compile jobs run one after the other on their threads, never
//! one inside another: a compile that waits for its chunks on the compile
//! pool does not start the next queued job on its own stack, so the depth
//! of a thread's stack does not grow with the number of queued jobs.
//! The jobs compile one small tape cut into many chunks, the shape that
//! keeps the pool's workers waiting in their joins.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rsdag::synth::{cases, Spec, Vocabulary};
use rsdag_jit::{NativeTape, Options};

const QUEUED_JOBS: usize = 8000;
const TAPE_STEPS: usize = 400;
const CHUNK_OPS: usize = 16;

thread_local! {
    static JOBS_ON_THIS_STACK: Cell<usize> = const { Cell::new(0) };
}

#[test]
fn queued_compile_jobs_do_not_nest() {
    let case = cases(0..1, |seed| {
        Spec::new(seed)
            .steps(TAPE_STEPS)
            .params(3)
            .outputs(2)
            .vocab(Vocabulary::Full)
    })
    .next()
    .expect("a case");
    let tape = Arc::new(case.tape);
    let opts = Options {
        chunk_ops: CHUNK_OPS,
        ..Options::default()
    };
    let deepest = Arc::new(AtomicUsize::new(0));
    let compiled = Arc::new(AtomicUsize::new(0));
    for _ in 0..QUEUED_JOBS {
        let (tape, deepest, compiled) = (tape.clone(), deepest.clone(), compiled.clone());
        rsdag_jit::background::submit(move || {
            let depth = JOBS_ON_THIS_STACK.with(|d| {
                d.set(d.get() + 1);
                d.get()
            });
            deepest.fetch_max(depth, Ordering::Relaxed);
            if NativeTape::compile_opts(&tape, &opts, &[]).is_ok() {
                compiled.fetch_add(1, Ordering::Relaxed);
            }
            JOBS_ON_THIS_STACK.with(|d| d.set(d.get() - 1));
        });
    }
    rsdag_jit::background::drain();
    assert_eq!(compiled.load(Ordering::Relaxed), QUEUED_JOBS);
    assert_eq!(
        deepest.load(Ordering::Relaxed),
        1,
        "compile jobs nested on one thread's stack"
    );
}
