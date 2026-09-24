//! The sparse solve of a Newton step, factor and substitute on fresh values,
//! as rsdag's `LuProgram` in native code (the default `Panels` choice)
//! against rslab's KLU (numeric refactor and solve), over pattern families
//! and sizes. CSV on stdout, for `docs/bench/data/solve.csv`:
//!
//!     cargo run --release --manifest-path bench/Cargo.toml
//!
//! `scripts/bench.sh` runs it with the other benchmarks and draws the plots.

use rsdag::symbolic::solve::{plan, LuProgram, Panels, Pattern};
use rsdag_jit::NativeTape;
use rslab::{GeneralCsc, KluSettings, KluSolver};
use std::time::Instant;

/// The entries of an `n` by `n` pattern of a family.
fn pattern(family: &str, n: usize) -> Vec<(usize, usize)> {
    let mut e = Vec::new();
    match family {
        // A chain of devices closed into a loop.
        "ring" => {
            for i in 0..n {
                let mut c = vec![(i + n - 1) % n, i, (i + 1) % n];
                c.sort_unstable();
                c.dedup();
                e.extend(c.into_iter().map(|j| (i, j)));
            }
        }
        // Three neighbours on each side.
        "band" => {
            for i in 0..n {
                e.extend((i.saturating_sub(3)..(i + 4).min(n)).map(|j| (i, j)));
            }
        }
        // The five-point stencil of a square mesh.
        "grid" => {
            let m = (n as f64).sqrt() as usize;
            for r in 0..m {
                for c in 0..m {
                    let i = r * m + c;
                    e.push((i, i));
                    if r > 0 {
                        e.push((i, i - m));
                    }
                    if r + 1 < m {
                        e.push((i, i + m));
                    }
                    if c > 0 {
                        e.push((i, i - 1));
                    }
                    if c + 1 < m {
                        e.push((i, i + 1));
                    }
                }
            }
        }
        // The ring with one long coupling per row: an expander, dense fill.
        "random" => {
            for i in 0..n {
                let mut c = vec![(i + n - 1) % n, i, (i + 1) % n, (i * 7 + 3) % n];
                c.sort_unstable();
                c.dedup();
                e.extend(c.into_iter().map(|j| (i, j)));
            }
        }
        _ => unreachable!("unknown family {family}"),
    }
    e
}

/// Diagonally dominant values, varied by `step`.
fn values(e: &[(usize, usize)], step: usize) -> Vec<f64> {
    e.iter()
        .enumerate()
        .map(|(k, &(i, j))| {
            let s = 1.0 + 0.01 * ((k + step) % 13) as f64;
            if i == j {
                8.0 * s
            } else {
                -s / (1.0 + ((i + j) % 5) as f64)
            }
        })
        .collect()
}

/// Microseconds per call, best of five runs of `reps`.
fn best(mut f: impl FnMut(), reps: usize) -> f64 {
    f();
    (0..5)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / reps as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    println!("family,n,program,ops_per_unknown,build_ms,rsdag_us,klu_build_ms,klu_us");
    let cases: &[(&str, &[usize])] = &[
        ("ring", &[1000, 10000, 100000]),
        ("band", &[1000, 10000, 100000]),
        ("grid", &[1024, 4096, 16384]),
        ("random", &[300, 1000]),
    ];
    for &(family, sizes) in cases {
        for &n in sizes {
            let e = pattern(family, n);
            let mut pat: Pattern = vec![Vec::new(); n];
            for &(i, j) in &e {
                pat[i].push(j);
            }
            let vals: Vec<Vec<f64>> = (0..13).map(|s| values(&e, s)).collect();
            let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();

            let t = Instant::now();
            let lu = LuProgram::build(n, e.clone(), plan(&pat).unwrap(), Some(Panels::default()));
            let native = NativeTape::compile(lu.tape()).unwrap();
            let build_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut inputs = vec![0.0; lu.input_len()];
            let (mut work, mut out) = (Vec::new(), Vec::new());
            let reps = (2_000_000 / lu.tape().n_ops().max(1)).clamp(3, 2000);
            let mut step = 0;
            let rsdag_us = best(
                || {
                    step += 1;
                    lu.write_values(&vals[step % 13], None, &mut inputs);
                    lu.write_rhs(&b, None, &mut inputs);
                    native.eval_prolog(&inputs, &mut work);
                    native.eval_main(&inputs, &mut work, &mut out);
                },
                reps,
            );
            assert!(lu.factored(&inputs, &work));
            let x = lu.solution(&out).to_vec();

            let rows: Vec<usize> = e.iter().map(|p| p.0).collect();
            let cols: Vec<usize> = e.iter().map(|p| p.1).collect();
            let mats: Vec<GeneralCsc<f64>> = vals
                .iter()
                .map(|v| GeneralCsc::from_triplets(n, &rows, &cols, v).unwrap())
                .collect();
            let t = Instant::now();
            let mut klu = KluSolver::factor(&mats[0], &KluSettings::default()).unwrap();
            let klu_build_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut k = 0;
            let klu_us = best(
                || {
                    k += 1;
                    klu.refactor(&mats[k % 13]).unwrap();
                    std::hint::black_box(klu.solve(&b).unwrap());
                },
                reps,
            );
            // Both solved the same system last.
            klu.refactor(&mats[step % 13]).unwrap();
            let y = klu.solve(&b).unwrap();
            let diff = x
                .iter()
                .zip(&y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max);
            assert!(diff < 1e-9, "{family} {n}: solutions differ by {diff}");

            let program = if lu.supernodal().is_some() {
                "supernodal"
            } else {
                "scalar"
            };
            println!(
                "{family},{n},{program},{:.1},{build_ms:.1},{rsdag_us:.2},{klu_build_ms:.2},{klu_us:.2}",
                lu.tape().n_ops() as f64 / n as f64
            );
        }
    }
}
