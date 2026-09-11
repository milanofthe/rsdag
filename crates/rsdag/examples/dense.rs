//! The dense kernels' throughput on one core, as CSV on stdout
//! (`docs/bench/data/dense.csv` for the README figures).
//!
//!     cargo run --release -p rsdag --example dense > docs/bench/data/dense.csv

use std::time::Instant;

use rsdag::semantics::{gemm, gemv, solve};

fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() {
    println!("kernel,n,gflops,ms");
    for &n in &[64usize, 128, 256, 512, 1024] {
        let a: Vec<f64> = (0..n * n).map(|i| 1.0 + (i % 7) as f64 * 1e-3).collect();
        let b: Vec<f64> = (0..n * n).map(|i| 1.0 - (i % 5) as f64 * 1e-3).collect();
        let x: Vec<f64> = (0..n).map(|i| 1.0 - (i % 5) as f64 * 1e-3).collect();
        let mut y = vec![0.0; n];
        let t = best(50, || gemv(&a, &x, n, n, &mut y));
        println!(
            "gemv,{n},{:.2},{:.4}",
            2.0 * (n * n) as f64 / t / 1e9,
            t * 1e3
        );
        let mut c = vec![0.0; n * n];
        let t = best(if n >= 1024 { 3 } else { 8 }, || {
            gemm(&a, &b, n, n, n, &mut c)
        });
        println!(
            "gemm,{n},{:.2},{:.4}",
            2.0 * (n * n * n) as f64 / t / 1e9,
            t * 1e3
        );
        let d: Vec<f64> = (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    4.0
                } else {
                    0.5 * ((i % 11) as f64 - 5.0) / 5.0
                }
            })
            .collect();
        let mut s = vec![0.0; n];
        let t = best(if n >= 1024 { 3 } else { 8 }, || solve(&d, &x, n, &mut s));
        println!(
            "solve,{n},{:.2},{:.4}",
            (2.0 / 3.0) * (n * n * n) as f64 / t / 1e9,
            t * 1e3
        );
    }
}
