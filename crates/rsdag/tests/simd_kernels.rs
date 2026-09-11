//! The `f64` vector twins of the dot-fold kernels are bit-identical to the
//! generic reference on every shape: chunked, with tails, with remainder
//! rows and columns, on values whose rounding differs between folds.

use rsdag::semantics::{
    dot_slice, dot_slice_t, gemm, gemm_t, gemv, gemv_t, solve_many, solve_many_generic,
};
use rsdag::synth::Spec;

fn values(rng: &mut rsdag::synth::Rng, len: usize) -> Vec<f64> {
    (0..len)
        .map(|_| {
            let v = rng.val() - 0.5;
            // Spread magnitudes so that fold order matters.
            v * 10f64.powi((rng.val() * 8.0) as i32 - 4)
        })
        .collect()
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn dot_matches_the_reference_on_every_length() {
    let mut rng = Spec::new(3).rng();
    for n in 0..70 {
        let a = values(&mut rng, n);
        let b = values(&mut rng, n);
        assert_eq!(
            dot_slice(&a, &b).to_bits(),
            dot_slice_t(&a, &b).to_bits(),
            "n {n}"
        );
    }
}

#[test]
fn gemv_matches_the_reference_on_every_shape() {
    let mut rng = Spec::new(4).rng();
    for (m, n) in (0..11)
        .flat_map(|m| (0..13).map(move |n| (m, n)))
        .chain([(37, 129), (100, 100)])
    {
        let a = values(&mut rng, m * n);
        let x = values(&mut rng, n);
        let (mut y1, mut y2) = (vec![0.0; m], vec![0.0; m]);
        gemv(&a, &x, m, n, &mut y1);
        gemv_t(&a, &x, m, n, &mut y2);
        assert!(same(&y1, &y2), "m {m} n {n}");
    }
}

#[test]
fn gemm_matches_the_reference_on_every_shape() {
    let mut rng = Spec::new(5).rng();
    let shapes = (0..7)
        .flat_map(|m| (0..7).flat_map(move |k| (0..5).map(move |n| (m, k, n))))
        .chain([(9, 33, 7), (16, 64, 16), (33, 17, 6)]);
    for (m, k, n) in shapes {
        let a = values(&mut rng, m * k);
        let b = values(&mut rng, n * k);
        let (mut c1, mut c2) = (vec![0.0; m * n], vec![0.0; m * n]);
        gemm(&a, &b, m, k, n, &mut c1);
        gemm_t(&a, &b, m, k, n, &mut c2);
        assert!(same(&c1, &c2), "m {m} k {k} n {n}");
    }
}

#[test]
fn solve_many_matches_the_generic_reference_on_every_shape() {
    let mut rng = Spec::new(9).rng();
    let shapes = (1..30usize)
        .flat_map(|n| [1usize, 2, 5].into_iter().map(move |k| (n, k)))
        .chain([(37, 4), (100, 7), (520, 2)]);
    for (n, k) in shapes {
        let mut a = values(&mut rng, n * n);
        for i in 0..n {
            a[i * n + i] += 4.0 * n as f64;
        }
        let b = values(&mut rng, n * k);
        let (mut x1, mut x2) = (vec![0.0; n * k], vec![0.0; n * k]);
        solve_many(&a, &b, n, k, &mut x1);
        solve_many_generic(&a, &b, n, k, &mut x2);
        assert!(same(&x1, &x2), "n {n} k {k}");
    }
}
