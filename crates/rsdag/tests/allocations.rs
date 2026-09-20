//! A solver drives the evaluation paths from its inner loop, so they must not
//! allocate once the caller's buffers exist. The counting allocator below is
//! process-wide, which is why this file holds exactly one test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::Complex64;
use rsdag::{Graph, Node, Scalar, Tape, F64};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Allocations of `runs` runs of `f`, after one warm-up run.
fn allocs(runs: usize, mut f: impl FnMut()) -> usize {
    f();
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..runs {
        f();
    }
    ALLOCS.load(Ordering::Relaxed) - before
}

#[test]
fn the_evaluation_paths_do_not_allocate() {
    let mut g: Graph<F64> = Graph::new();
    let xs: Vec<_> = (0..4).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<_> = xs
        .iter()
        .map(|&x| match g.node(x) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        })
        .collect();
    let roots: Vec<_> = (0..4)
        .map(|i| {
            let a = g.mul(xs[i], xs[(i + 1) % 4]);
            let b = g.exp(xs[(i + 2) % 4]);
            g.add(a, b)
        })
        .collect();
    let tape = Tape::compile(&g, &roots, &syms);
    let inputs = [0.3, 0.7, 1.1, 1.9];

    let mut work = vec![0.0; tape.work_len()];
    let mut out = vec![0.0; tape.out_len()];
    assert_eq!(
        allocs(50, || tape.eval_into(&inputs, &mut work, &mut out)),
        0,
        "eval_into allocated"
    );

    let mut run = tape.runner::<f64>();
    assert_eq!(
        allocs(50, || {
            let _ = run.eval(&inputs);
        }),
        0,
        "runner allocated"
    );

    let f = g.define_func("body", syms, roots);
    let body = g.func(f).body(&g);
    let bundle = &*body.bundle;
    let mut bwork = vec![0.0; bundle.work_len()];
    let mut bout = vec![0.0; bundle.n_outputs()];
    assert_eq!(
        allocs(50, || bundle.call_into(&inputs, &mut bwork, &mut bout)),
        0,
        "call_into allocated"
    );
    assert_eq!(
        allocs(50, || bundle.call(&inputs, &mut bout)),
        0,
        "call allocated beyond its thread-local buffer"
    );

    // The generic scalars convert through f64 and must borrow their buffers.
    let f32_in: Vec<f32> = inputs.iter().map(|&v| v as f32).collect();
    let mut f32_out = vec![0.0f32; bundle.n_outputs()];
    assert_eq!(
        allocs(50, || f32::call_bundle(bundle, &f32_in, &mut f32_out)),
        0,
        "call_bundle in f32 allocated"
    );
    let cx_in: Vec<Complex64> = inputs.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut cx_out = vec![Complex64::new(0.0, 0.0); bundle.n_outputs()];
    assert_eq!(
        allocs(50, || Complex64::call_bundle(bundle, &cx_in, &mut cx_out)),
        0,
        "call_bundle in Complex64 allocated"
    );
}
