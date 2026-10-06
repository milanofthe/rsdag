//! A solver drives the evaluation paths from its inner loop, so they must not
//! allocate once the caller's buffers exist. The count is the calling
//! thread's: the idle workers of a thread pool tidy their queues on their own
//! threads now and then, which a process-wide count would pick up.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use rsdag::{Graph, Node, Tape, F64};

thread_local!(static ALLOCS: Cell<usize> = const { Cell::new(0) });

fn count() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

fn counted() -> usize {
    ALLOCS.with(Cell::get)
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Allocations of `runs` runs of `f`, after one warm-up run.
fn allocs(runs: usize, mut f: impl FnMut()) -> usize {
    f();
    let before = counted();
    for _ in 0..runs {
        f();
    }
    counted() - before
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

    // A dense solve works in the scratch the work buffer lends, on both
    // sides of the blocked elimination.
    for n in [3usize, 80] {
        let mut g: Graph<F64> = Graph::new();
        let xs: Vec<_> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
        let syms: Vec<_> = xs
            .iter()
            .map(|&x| match g.node(x) {
                Node::Symbol(s) => *s,
                _ => unreachable!(),
            })
            .collect();
        let one = g.one();
        let a: Vec<_> = (0..n * n)
            .map(|k| {
                if k / n == k % n {
                    g.add(xs[k % n], one)
                } else {
                    xs[(k + 1) % n]
                }
            })
            .collect();
        let x = g.solve_dense(a, xs.clone());
        let tape = Tape::compile(&g, &x, &syms);
        let vals: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.01).collect();
        let mut work = vec![0.0; tape.work_len()];
        let mut out = vec![0.0; tape.out_len()];
        assert_eq!(
            allocs(5, || tape.eval_into(&vals, &mut work, &mut out)),
            0,
            "solve of {n} allocated"
        );
    }
}
