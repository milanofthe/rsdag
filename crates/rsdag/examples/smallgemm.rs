//! Gemm and the multi-solve at the shapes a block elimination produces.
use rsdag::semantics::{gemm, solve_many};
use std::time::Instant;
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
    for &(m, k, n) in &[
        (26usize, 26usize, 26usize),
        (26, 52, 26),
        (24, 24, 24),
        (32, 32, 32),
        (26, 1300, 26),
    ] {
        let a: Vec<f64> = (0..m * k).map(|i| 1.0 + (i % 7) as f64 * 1e-3).collect();
        let b: Vec<f64> = (0..n * k).map(|i| 1.0 - (i % 5) as f64 * 1e-3).collect();
        let mut c = vec![0.0; m * n];
        let t = best(2000, || gemm(&a, &b, m, k, n, &mut c));
        println!(
            "gemm {m}x{k}x{n}: {:.2} GF/s ({:.2} us)",
            2.0 * (m * k * n) as f64 / t / 1e9,
            t * 1e6
        );
    }
    for &(n, k) in &[(26usize, 26usize), (26, 1), (13, 13), (52, 52)] {
        let a: Vec<f64> = (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    4.0 * n as f64
                } else {
                    0.5 * ((i % 11) as f64 - 5.0) / 5.0
                }
            })
            .collect();
        let b: Vec<f64> = (0..n * k).map(|i| 1.0 - (i % 5) as f64 * 1e-3).collect();
        let mut x = vec![0.0; n * k];
        let t = best(2000, || solve_many(&a, &b, n, k, &mut x));
        let flops = (2.0 / 3.0) * (n * n * n) as f64 + 2.0 * (n * n * k) as f64;
        println!(
            "solve_many n={n} k={k}: {:.2} GF/s ({:.2} us)",
            flops / t / 1e9,
            t * 1e6
        );
    }
}
