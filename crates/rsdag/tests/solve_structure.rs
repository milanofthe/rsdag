//! The structure of a solve, worked out before the arithmetic: the block
//! triangular form finds the blocks and the singular case, the ordering
//! keeps fill down without storing it, the predictor knows the fill before
//! the build, and the planned solve is the dense solve to rounding.

use rsdag::symbolic::solve::{amd::amd, btf::block_triangular, plan, predict::cost};
use rsdag::symbolic::solve::{lu_static, pattern_of, solve_planned, SparseRows};
use rsdag::synth::Spec;
use rsdag::{Builder, ExprId, Graph, Numeric, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        rsdag::Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

#[test]
fn block_triangular_form_finds_the_blocks_and_the_singular_case() {
    // Two coupled pairs and a singleton, coupled one way: three blocks.
    // Rows: 0:{0,1,4} 1:{0,1} 2:{2,3} 3:{2,3,0} 4:{4}
    let pattern = vec![
        vec![0, 1, 4],
        vec![0, 1],
        vec![2, 3],
        vec![0, 2, 3],
        vec![4],
    ];
    let btf = block_triangular(&pattern).expect("nonsingular");
    assert_eq!(btf.n_blocks(), 3);
    // Every entry lies in a diagonal block or above it.
    let mut col_pos = [0; 5];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        col_pos[j] = k;
    }
    let block_of = |k: usize| {
        (0..btf.n_blocks())
            .find(|&b| btf.block(b).contains(&k))
            .unwrap()
    };
    for (k, &i) in btf.row_perm.iter().enumerate() {
        for &j in &pattern[i] {
            assert!(
                block_of(col_pos[j]) >= block_of(k),
                "entry ({i}, {j}) below its block"
            );
        }
        assert!(pattern[i].contains(&btf.col_perm[k]), "zero-free diagonal");
    }
    // A column nothing can match: structurally singular.
    let singular = vec![vec![0], vec![0], vec![0, 1, 2]];
    assert!(block_triangular(&singular).is_none());
}

/// A 2D grid's symmetric adjacency, `side * side` vertices on a torus.
fn grid(side: usize) -> Vec<Vec<usize>> {
    let n = side * side;
    (0..n)
        .map(|i| {
            let (r, c) = (i / side, i % side);
            vec![
                ((r + side - 1) % side) * side + c,
                ((r + 1) % side) * side + c,
                r * side + (c + side - 1) % side,
                r * side + (c + 1) % side,
            ]
        })
        .collect()
}

#[test]
fn the_quotient_graph_ordering_is_a_permutation_that_reduces_fill() {
    let adj = grid(24);
    let order = amd(&adj);
    let mut seen = order.clone();
    seen.sort_unstable();
    assert_eq!(seen, (0..adj.len()).collect::<Vec<_>>());
    let pattern: Vec<Vec<usize>> = adj
        .iter()
        .enumerate()
        .map(|(i, nb)| {
            let mut row = nb.clone();
            row.push(i);
            row
        })
        .collect();
    let natural: Vec<usize> = (0..adj.len()).collect();
    let by_amd = cost(&pattern, &order);
    let by_natural = cost(&pattern, &natural);
    // Natural order on a 24-grid fills the band, about 24 per vertex; a
    // fill-reducing order stays well under half of that.
    assert!(
        by_amd.fill < by_natural.fill / 2,
        "amd {} natural {}",
        by_amd.fill,
        by_natural.fill
    );
}

#[test]
fn the_predicted_fill_is_the_factorization_fill() {
    for seed in 0..12u64 {
        let mut rng = Spec::new(seed).rng();
        let n = 12 + seed as usize * 3;
        // A random symmetric pattern with a full diagonal.
        let mut pattern: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        for i in 0..n {
            for j in 0..i {
                if rng.below(4) == 0 {
                    pattern[i].push(j);
                    pattern[j].push(i);
                }
            }
        }
        let adj: Vec<Vec<usize>> = pattern
            .iter()
            .enumerate()
            .map(|(i, r)| r.iter().copied().filter(|&j| j != i).collect())
            .collect();
        let order = amd(&adj);
        let predicted = cost(&pattern, &order);
        let mut g: Graph<F64> = Graph::new();
        let rows: SparseRows = pattern
            .iter()
            .enumerate()
            .map(|(i, r)| {
                r.iter()
                    .map(|&j| (j, g.sym(&format!("a{i}_{j}"))))
                    .collect()
            })
            .collect();
        let lu = lu_static(&mut g, &rows, &order);
        assert_eq!(lu.fill, predicted.fill, "seed {seed}");
    }
}

#[test]
fn the_planned_solve_is_the_dense_solve() {
    let mut rng = Spec::new(9).rng();
    for trial in 0..16 {
        // Block upper triangular by construction: blocks of 1 to 4, dense
        // within, random coupling above, diagonal dominance throughout.
        let mut sizes = Vec::new();
        let mut n = 0;
        while n < 6 + trial {
            let s = 1 + rng.below(4);
            sizes.push(s);
            n += s;
        }
        let mut dense = vec![vec![0.0; n]; n];
        let mut lo = 0;
        for &s in &sizes {
            for i in lo..lo + s {
                for j in lo..lo + s {
                    dense[i][j] = if i == j { 10.0 + rng.val() } else { rng.val() };
                }
                for j in lo + s..n {
                    if rng.below(3) == 0 {
                        dense[i][j] = rng.val();
                    }
                }
            }
            lo += s;
        }
        // Scramble rows and columns so the form has to be found.
        let mut rp: Vec<usize> = (0..n).collect();
        let mut cp: Vec<usize> = (0..n).collect();
        for k in (1..n).rev() {
            rp.swap(k, rng.below(k + 1));
            cp.swap(k, rng.below(k + 1));
        }
        let scrambled: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| dense[rp[i]][cp[j]]).collect())
            .collect();
        let rhs: Vec<f64> = (0..n).map(|_| rng.val()).collect();
        let want = Numeric.solve(&scrambled, &rhs);

        let mut g: Graph<F64> = Graph::new();
        let (mut vals, mut syms) = (Vec::new(), Vec::new());
        let a: Vec<Vec<ExprId>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if scrambled[i][j] != 0.0 {
                            let e = g.sym(&format!("a{i}_{j}"));
                            vals.push(scrambled[i][j]);
                            syms.push(sym(&g, e));
                            e
                        } else {
                            g.zero()
                        }
                    })
                    .collect()
            })
            .collect();
        let b: Vec<ExprId> = (0..n)
            .map(|i| {
                let e = g.sym(&format!("b{i}"));
                vals.push(rhs[i]);
                syms.push(sym(&g, e));
                e
            })
            .collect();
        let rows = rsdag::symbolic::solve::sparse_rows(&g, &a);
        let p = plan(&pattern_of(&rows)).expect("nonsingular");
        assert!(
            p.btf.n_blocks() >= sizes.len(),
            "trial {trial}: {} blocks of {}",
            p.btf.n_blocks(),
            sizes.len()
        );
        let x = solve_planned(&mut g, &rows, &p, &b).x;
        let tape = Tape::compile(&g, &x, &syms);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&vals, &mut w, &mut o);
        for (k, (p, q)) in want.iter().zip(&o).enumerate() {
            assert!(
                (p - q).abs() <= 1e-11 * (1.0 + p.abs()),
                "trial {trial} x[{k}]: {p} vs {q}"
            );
        }
    }
}

#[test]
fn a_newton_step_on_a_grid_converges() {
    let side = 20;
    let n = side * side;
    let adj = grid(side);
    let mut g: Graph<F64> = Graph::new();
    let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<SymbolId> = xs.iter().map(|&e| sym(&g, e)).collect();
    let f: Vec<ExprId> = (0..n)
        .map(|i| {
            let e = g.exp(xs[i]);
            let five = g.konst_f64(5.0);
            let mut acc = g.mul(five, xs[i]);
            acc = g.add(acc, e);
            for &j in &adj[i] {
                acc = g.sub(acc, xs[j]);
            }
            let one = g.one();
            g.sub(acc, one)
        })
        .collect();
    let step = rsdag::newton_step(&mut g, &f, &syms)
        .expect("nonsingular")
        .x;
    let step_tape = Tape::compile(&g, &step, &syms);
    let res_tape = Tape::compile(&g, &f, &syms);
    let mut x = vec![0.3; n];
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let norm = |v: &[f64]| v.iter().map(|t| t * t).sum::<f64>().sqrt();
    let mut last = f64::INFINITY;
    for _ in 0..8 {
        step_tape.eval(&x, &mut w, &mut o);
        x.copy_from_slice(&o);
        res_tape.eval(&x, &mut w, &mut o);
        last = norm(&o);
        if last < 1e-10 {
            break;
        }
    }
    assert!(last < 1e-9, "residual {last}");
}
