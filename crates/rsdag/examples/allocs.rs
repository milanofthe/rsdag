//! Allocation audit of the evaluation path: how many times a caller that
//! already owns its buffers still hits the allocator per evaluation.
//!
//! A solver drives these in its inner loop (residual and Jacobian per Newton
//! iteration, a factorization and two triangular solves per system), so every
//! allocation there is one the caller cannot avoid from outside.
//!
//!   cargo run -q --release --example allocs

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rsdag::{Graph, Node, Tape, F64};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Allocations and bytes of `f`, after `warm` warm-up runs.
fn audit(what: &str, warm: usize, runs: usize, mut f: impl FnMut()) {
    for _ in 0..warm {
        f();
    }
    let (a0, b0) = (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    );
    for _ in 0..runs {
        f();
    }
    let a = ALLOCS.load(Ordering::Relaxed) - a0;
    let b = BYTES.load(Ordering::Relaxed) - b0;
    println!(
        "{what:<34} {:6.1} allocs/run  {:8.1} bytes/run",
        a as f64 / runs as f64,
        b as f64 / runs as f64
    );
}

fn main() {
    // A small dense residual with a few transcendentals, the shape of a device
    // body, plus its Jacobian: what a Newton iteration evaluates.
    let mut g: Graph<F64> = Graph::new();
    let xs: Vec<_> = (0..8).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<_> = xs
        .iter()
        .map(|&x| match g.node(x) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        })
        .collect();
    let res: Vec<_> = (0..8)
        .map(|i| {
            let a = g.mul(xs[i], xs[(i + 1) % 8]);
            let b = g.exp(xs[(i + 3) % 8]);
            let c = g.add(a, b);
            g.sub(c, xs[i])
        })
        .collect();
    let jac: Vec<_> = res
        .iter()
        .flat_map(|&r| syms.iter().map(move |&s| (r, s)).collect::<Vec<_>>())
        .map(|(r, s)| rsdag::autodiff::differentiate(&mut g, r, s))
        .collect();

    let tape_res = Tape::compile(&g, &res, &syms);
    let mut roots = res.clone();
    roots.extend(jac.iter().copied());
    let tape_all = Tape::compile(&g, &roots, &syms);

    let inputs: Vec<f64> = (0..8).map(|i| 0.1 * (i as f64 + 1.0)).collect();
    let (mut work, mut out) = (Vec::new(), Vec::new());

    audit("Tape::eval residual", 4, 200, || {
        tape_all.eval(&inputs, &mut work, &mut out)
    });
    audit("Tape::eval residual + jacobian", 4, 200, || {
        tape_res.eval(&inputs, &mut work, &mut out)
    });

    // The solve program: a static LU of a fixed pattern, factored and
    // substituted as one tape over the entry values and the right-hand side.
    let n = 8;
    let mut rows: Vec<Vec<(usize, rsdag::ExprId)>> = Vec::new();
    let mut val_syms = Vec::new();
    for r in 0..n {
        let mut row = Vec::new();
        for c in [(r + n - 1) % n, r, (r + 1) % n] {
            let e = g.sym(&format!("a{r}_{c}"));
            val_syms.push(e);
            row.push((c, e));
        }
        row.sort_by_key(|&(c, _)| c);
        rows.push(row);
    }
    let rhs: Vec<_> = (0..n).map(|i| g.sym(&format!("b{i}"))).collect();
    let pattern: Vec<Vec<usize>> = rows
        .iter()
        .map(|r| r.iter().map(|&(c, _)| c).collect())
        .collect();
    let plan = rsdag::symbolic::solve::plan(&pattern).expect("plan");
    let solved = rsdag::symbolic::solve::solve_planned(&mut g, &rows, &plan, &rhs);
    let solve_syms: Vec<_> = val_syms
        .iter()
        .chain(rhs.iter())
        .map(|&e| match g.node(e) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        })
        .collect();
    let mut solve_roots = solved.x.clone();
    solve_roots.push(solved.pivots_ok);
    let tape_solve = Tape::compile(&g, &solve_roots, &solve_syms);

    let sv: Vec<f64> = (0..val_syms.len() + n)
        .map(|k| if k % 3 == 1 { 4.0 } else { 0.5 })
        .collect();
    audit("solve program, 8x8 tridiagonal", 4, 200, || {
        tape_solve.eval(&sv, &mut work, &mut out)
    });

    println!(
        "\ntotal since start: {} allocs, {} bytes",
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed)
    );
}
