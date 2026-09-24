//! The LU as one program: a factorization is the prolog over the entry
//! values, a solve the main phase over a right-hand side, and the pivot
//! guard is read from the prolog's state.

use rsdag::symbolic::solve::{plan, supernodes, LuProgram, Panels, Pattern};
use rsdag::synth::Spec;
use rsdag::{Builder, Numeric};

/// The positions of a matrix's nonzeros, row by row, and their values.
fn sparse(num: &[Vec<f64>]) -> (Vec<(usize, usize)>, Vec<f64>) {
    let mut entries = Vec::new();
    let mut values = Vec::new();
    for (i, row) in num.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                entries.push((i, j));
                values.push(v);
            }
        }
    }
    (entries, values)
}

fn program(n: usize, entries: &[(usize, usize)], panels: Option<Panels>) -> LuProgram {
    let mut pattern: Pattern = vec![Vec::new(); n];
    for &(i, j) in entries {
        pattern[i].push(j);
    }
    LuProgram::build(n, entries.to_vec(), plan(&pattern).unwrap(), panels)
}

/// Factor `values`, then solve each right-hand side: whether the guard held,
/// and the solutions.
fn run(
    lu: &LuProgram,
    values: &[f64],
    scale: Option<&[f64]>,
    rhs: &[Vec<f64>],
) -> (bool, Vec<Vec<f64>>) {
    let mut inputs = vec![0.0; lu.input_len()];
    let (mut work, mut out) = (Vec::new(), Vec::new());
    lu.write_values(values, scale, &mut inputs);
    lu.tape().eval_prolog(&inputs, &mut work);
    let ok = lu.factored(&inputs, &work);
    let xs = rhs
        .iter()
        .map(|b| {
            lu.write_rhs(b, scale, &mut inputs);
            lu.tape().eval_main(&inputs, &mut work, &mut out);
            lu.solution(&out).to_vec()
        })
        .collect();
    (ok, xs)
}

fn close(want: &[f64], got: &[f64], what: &str) {
    let scale = want.iter().map(|v| v.abs()).fold(1.0f64, f64::max);
    for (k, (p, q)) in want.iter().zip(got).enumerate() {
        assert!((p - q).abs() <= 1e-10 * scale, "{what} x[{k}]: {p} vs {q}");
    }
}

/// A random sparse matrix with a dominant diagonal.
fn dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Spec::new(seed).rng();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        4.0 + rng.val()
                    } else if rng.val() < 0.25 {
                        rng.val() - 0.5
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn a_factorization_serves_several_right_hand_sides() {
    let n = 12;
    let num = dominant(n, 3);
    let (entries, values) = sparse(&num);
    let lu = program(n, &entries, None);
    assert!(lu.supernodal().is_none());
    let rhs: Vec<Vec<f64>> = (0..3)
        .map(|r| (0..n).map(|i| (i + r) as f64 - 5.0).collect())
        .collect();
    let (ok, xs) = run(&lu, &values, None, &rhs);
    assert!(ok, "a dominant diagonal holds the guard");
    for (b, x) in rhs.iter().zip(&xs) {
        close(&Numeric.solve(&num, b), x, "unscaled");
    }
    // Row scaling scales the values and the right-hand side alike.
    let scale: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect();
    let (ok, scaled) = run(&lu, &values, Some(&scale), &rhs);
    assert!(ok);
    for (b, x) in rhs.iter().zip(&scaled) {
        close(&Numeric.solve(&num, b), x, "scaled");
    }
}

#[test]
fn a_zero_pivot_fails_the_guard_until_repivoted() {
    let num = vec![
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let mut entries = sparse(&num).0;
    entries.push((0, 0));
    let values: Vec<f64> = entries.iter().map(|&(i, j)| num[i][j]).collect();
    let lu = program(3, &entries, None);
    let b = vec![1.0, 2.0, 3.0];
    let (ok, _) = run(&lu, &values, None, std::slice::from_ref(&b));
    assert!(!ok, "the structural diagonal has a zero pivot");
    let mags: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    let re = lu.repivot(&mags);
    let (ok, xs) = run(&re, &values, None, std::slice::from_ref(&b));
    assert!(ok, "the repivoted program holds on its values");
    close(&Numeric.solve(&num, &b), &xs[0], "repivoted");
}

#[test]
fn a_program_without_a_guard_is_factored() {
    // One candidate per column: the guard is the constant one.
    let n = 5;
    let num: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if j >= i { 1.0 + j as f64 } else { 0.0 })
                .collect()
        })
        .collect();
    let (entries, values) = sparse(&num);
    let lu = program(n, &entries, None);
    let b: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let (ok, xs) = run(&lu, &values, None, std::slice::from_ref(&b));
    assert!(ok);
    close(&Numeric.solve(&num, &b), &xs[0], "triangular");
}

#[test]
fn a_non_finite_value_fails_the_factorization() {
    let n = 6;
    let num = dominant(n, 5);
    let (entries, mut values) = sparse(&num);
    let lu = program(n, &entries, None);
    values[0] = f64::NAN;
    assert!(!run(&lu, &values, None, &[]).0);
    values[0] = f64::INFINITY;
    assert!(!run(&lu, &values, None, &[]).0);
}

#[test]
fn the_supernodal_program_solves_alike() {
    let n = 16;
    let num: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        8.0
                    } else {
                        1.0 / (1.0 + (i + 2 * j) as f64)
                    }
                })
                .collect()
        })
        .collect();
    let (entries, values) = sparse(&num);
    let panels = Panels {
        min_n: 0,
        min_width: 1,
        min_share: 0.0,
        ..Panels::default()
    };
    let lu = program(n, &entries, Some(panels));
    assert!(lu.supernodal().is_some());
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let (ok, xs) = run(&lu, &values, None, std::slice::from_ref(&b));
    assert!(ok);
    close(&Numeric.solve(&num, &b), &xs[0], "supernodal");
}

/// The five-point Laplacian pattern of a `side` by `side` grid.
fn grid(side: usize) -> Vec<(usize, usize)> {
    let mut entries = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let i = r * side + c;
            entries.push((i, i));
            if r > 0 {
                entries.push((i, i - side));
            }
            if r + 1 < side {
                entries.push((i, i + side));
            }
            if c > 0 {
                entries.push((i, i - 1));
            }
            if c + 1 < side {
                entries.push((i, i + 1));
            }
        }
    }
    entries
}

#[test]
fn a_mesh_goes_supernodal_on_its_flops_not_its_unknowns() {
    // On a grid the wide panels are the separators: a small share of the
    // unknowns, most of the flops.
    let side = 24;
    let n = side * side;
    let entries = grid(side);
    let mut pattern: Pattern = vec![Vec::new(); n];
    for &(i, j) in &entries {
        pattern[i].push(j);
    }
    let p = plan(&pattern).unwrap();
    let sn = supernodes(&pattern, &p);
    let unknowns: usize = sn.widths().iter().filter(|&&w| w >= 8).sum();
    assert!((unknowns as f64) < 0.5 * n as f64);
    let share = sn.flop_share(8);
    assert!(share > 0.5 && share <= 1.0, "{share}");
    assert!((sn.flop_share(1) - 1.0).abs() < 1e-12);
    assert!(sn.flop_share(16) <= share);
    // The default wants more flops than this system has.
    assert!(p.cost.flops < Panels::default().min_flops);
    assert!(program(n, &entries, Some(Panels::default()))
        .supernodal()
        .is_none());
    let at_scale = Panels {
        min_flops: 0,
        min_flop_share: share,
        ..Panels::default()
    };
    let lu = program(n, &entries, Some(at_scale));
    assert!(lu.supernodal().is_some());
    let values: Vec<f64> = entries
        .iter()
        .map(|&(i, j)| if i == j { 4.5 } else { -1.0 })
        .collect();
    let mut num = vec![vec![0.0; n]; n];
    for (&(i, j), &v) in entries.iter().zip(&values) {
        num[i][j] = v;
    }
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    let (ok, xs) = run(&lu, &values, None, std::slice::from_ref(&b));
    assert!(ok);
    close(&Numeric.solve(&num, &b), &xs[0], "mesh");
    let above = Panels {
        min_flops: 0,
        min_flop_share: share + 1e-9,
        ..Panels::default()
    };
    assert!(program(n, &entries, Some(above)).supernodal().is_none());
}
