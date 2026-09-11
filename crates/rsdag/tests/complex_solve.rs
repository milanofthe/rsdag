//! Complex systems as programs over pairs of real expressions: the sparse
//! solve and the block solve over `Cx` match a complex dense elimination,
//! with dense and with diagonal blocks.

use num_complex::Complex64;
use rsdag::symbolic::solve::{
    block_pattern, pattern_of, plan, solve_block_planned, solve_planned, Block, BlockRows, Cx,
};
use rsdag::synth::Spec;
use rsdag::{Builder, ExprId, Graph, Node, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// A complex symbol: two real inputs.
fn csym(g: &mut Graph<F64>, name: &str, syms: &mut Vec<SymbolId>) -> Cx {
    let re = g.sym(&format!("{name}r"));
    let im = g.sym(&format!("{name}i"));
    syms.push(sym(g, re));
    syms.push(sym(g, im));
    Cx::new(re, im)
}

/// Dense complex Gaussian elimination with partial pivoting.
fn dense_solve(mut a: Vec<Vec<Complex64>>, mut b: Vec<Complex64>) -> Vec<Complex64> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| a[i][k].norm().total_cmp(&a[j][k].norm()))
            .unwrap();
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            for j in k..n {
                let t = a[k][j];
                a[i][j] -= f * t;
            }
            let t = b[k];
            b[i] -= f * t;
        }
    }
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in i + 1..n {
            acc -= a[i][j] * x[j];
        }
        x[i] = acc / a[i][i];
    }
    x
}

fn close(want: &[Complex64], got: &[f64]) {
    let scale = want.iter().map(|v| v.norm()).fold(1.0f64, f64::max);
    for (k, w) in want.iter().enumerate() {
        let g = Complex64::new(got[2 * k], got[2 * k + 1]);
        assert!((w - g).norm() <= 1e-10 * scale, "x[{k}]: {w} vs {g}");
    }
}

#[test]
fn a_sparse_complex_system_matches_the_dense_elimination_and_passes_the_guard() {
    let n = 12;
    let mut rng = Spec::new(5).rng();
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    let mut vals = Vec::new();
    let mut dense = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut rows: Vec<Vec<(usize, Cx)>> = vec![Vec::new(); n];
    for i in 0..n {
        let mut cols = vec![i, (i + 1) % n, (i * 5 + 3) % n];
        cols.sort_unstable();
        cols.dedup();
        for j in cols {
            let e = csym(&mut g, &format!("a{i}_{j}"), &mut syms);
            // Imaginary-dominant diagonals: a real-part pivot rule would fail.
            let v = if i == j {
                Complex64::new(0.1 * (rng.val() - 0.5), 6.0 + rng.val())
            } else {
                Complex64::new(rng.val() - 0.5, rng.val() - 0.5)
            };
            vals.push(v.re);
            vals.push(v.im);
            dense[i][j] = v;
            rows[i].push((j, e));
        }
    }
    let mut rhs = Vec::new();
    let mut b = Vec::new();
    for k in 0..n {
        rhs.push(csym(&mut g, &format!("b{k}"), &mut syms));
        let v = Complex64::new(1.0 + k as f64 * 0.1, 0.5 - k as f64 * 0.05);
        vals.push(v.re);
        vals.push(v.im);
        b.push(v);
    }
    let plan = plan(&pattern_of(
        &rows
            .iter()
            .map(|r| r.iter().map(|&(j, e)| (j, e.re)).collect())
            .collect(),
    ))
    .expect("plan");
    let solved = solve_planned(&mut g, &rows, &plan, &rhs);
    let mut roots: Vec<ExprId> = solved.x.iter().flat_map(|c| [c.re, c.im]).collect();
    roots.push(solved.pivots_ok);
    let tape = Tape::compile(&g, &roots, &syms);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    assert_eq!(o[2 * n], 1.0, "the pivot guard, by modulus");
    close(&dense_solve(dense, b), &o[..2 * n]);
}

/// A block system: dense pivot blocks, diagonal off-diagonal blocks (the
/// shape of a harmonic-balance Jacobian with linear couplings).
fn block_system(
    g: &mut Graph<F64>,
    nb: usize,
    b: usize,
    diagonal_couplings: bool,
    syms: &mut Vec<SymbolId>,
    vals: &mut Vec<f64>,
) -> (BlockRows<Cx>, Vec<Cx>, Vec<Vec<Complex64>>, Vec<Complex64>) {
    let mut rng = Spec::new(9).rng();
    let n = nb * b;
    let mut dense = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut rows: BlockRows<Cx> = vec![Vec::new(); nb];
    let mut pattern = Vec::new();
    for i in 0..nb {
        pattern.push((i, i));
        pattern.push((i, (i + 1) % nb));
        pattern.push(((i + 1) % nb, i));
    }
    pattern.sort();
    pattern.dedup();
    for &(i, j) in &pattern {
        let mut val = |r: usize, c: usize, g: &mut Graph<F64>| -> Cx {
            let e = csym(g, &format!("a{i}_{j}_{r}_{c}"), syms);
            let v = Complex64::new(rng.val() - 0.5, rng.val() - 0.5)
                + if i == j && r == c {
                    Complex64::new(2.0 * b as f64, 3.0 * b as f64)
                } else {
                    Complex64::new(0.0, 0.0)
                };
            vals.push(v.re);
            vals.push(v.im);
            dense[i * b + r][j * b + c] = v;
            e
        };
        if i != j && diagonal_couplings {
            let d: Vec<Cx> = (0..b).map(|r| val(r, r, g)).collect();
            rows[i].push((j, Block::Diag(d)));
        } else {
            let mut block = Vec::with_capacity(b * b);
            for r in 0..b {
                for c in 0..b {
                    block.push(val(r, c, g));
                }
            }
            rows[i].push((j, Block::Dense(block)));
        }
    }
    let mut rhs = Vec::new();
    let mut bv = Vec::new();
    for k in 0..n {
        rhs.push(csym(g, &format!("b{k}"), syms));
        let v = Complex64::new(1.0 + k as f64 * 0.1, -0.2 * k as f64);
        vals.push(v.re);
        vals.push(v.im);
        bv.push(v);
    }
    (rows, rhs, dense, bv)
}

fn check_blocks(nb: usize, b: usize, diagonal_couplings: bool) {
    let mut g: Graph<F64> = Graph::new();
    let (mut syms, mut vals) = (Vec::new(), Vec::new());
    let (rows, rhs, dense, bv) =
        block_system(&mut g, nb, b, diagonal_couplings, &mut syms, &mut vals);
    let plan = plan(&block_pattern(&rows)).expect("plan");
    let solved = solve_block_planned(&mut g, &rows, b, &plan, &rhs);
    let roots: Vec<ExprId> = solved.x.iter().flat_map(|c| [c.re, c.im]).collect();
    let tape = Tape::compile(&g, &roots, &syms);
    let d = tape.dump();
    assert!(d.contains("SolveMany("), "{d}");
    assert!(d.contains("Gemm("), "{d}");
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    close(&dense_solve(dense, bv), &o);
}

#[test]
fn complex_block_systems_match_the_dense_elimination() {
    check_blocks(5, 6, false);
    check_blocks(5, 6, true);
}

#[test]
fn real_block_systems_with_diagonal_blocks_match_the_dense_solve() {
    let nb = 4;
    let b = 5;
    let mut rng = Spec::new(3).rng();
    let mut g: Graph<F64> = Graph::new();
    let (mut syms, mut vals) = (Vec::new(), Vec::new());
    let n = nb * b;
    let mut dense = vec![vec![0.0; n]; n];
    let mut rows: BlockRows = vec![Vec::new(); nb];
    let mut val = |i: usize, j: usize, r: usize, c: usize, g: &mut Graph<F64>| -> ExprId {
        let e = g.sym(&format!("a{i}_{j}_{r}_{c}"));
        syms.push(sym(g, e));
        let v = rng.val() - 0.5
            + if i == j && r == c {
                4.0 * b as f64
            } else {
                0.0
            };
        vals.push(v);
        dense[i * b + r][j * b + c] = v;
        e
    };
    for i in 0..nb {
        for j in 0..nb {
            if i == j {
                let mut block = Vec::new();
                for r in 0..b {
                    for c in 0..b {
                        block.push(val(i, j, r, c, &mut g));
                    }
                }
                rows[i].push((j, Block::Dense(block)));
            } else if (i + j) % 2 == 1 {
                rows[i].push((
                    j,
                    Block::Diag((0..b).map(|r| val(i, j, r, r, &mut g)).collect()),
                ));
            }
        }
    }
    let rhs: Vec<ExprId> = (0..n)
        .map(|k| {
            let e = g.sym(&format!("b{k}"));
            syms.push(sym(&g, e));
            vals.push(1.0 + k as f64 * 0.1);
            e
        })
        .collect();
    let plan = plan(&block_pattern(&rows)).expect("plan");
    let solved = solve_block_planned(&mut g, &rows, b, &plan, &rhs);
    let tape = Tape::compile(&g, &solved.x, &syms);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    let want = rsdag::Numeric.solve(&dense, &vals[vals.len() - n..]);
    let scale = want.iter().map(|v| v.abs()).fold(1.0f64, f64::max);
    for k in 0..n {
        assert!(
            (want[k] - o[k]).abs() <= 1e-10 * scale,
            "x[{k}]: {} vs {}",
            want[k],
            o[k]
        );
    }
}
