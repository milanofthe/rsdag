//! A block-sparse system solved as one program: the result is the dense
//! numeric solve's to rounding, the pivot blocks run through the dense
//! solve kernel (one factorization per block for its right-hand sides) and
//! the block updates through Gemm kernels.

use rsdag::symbolic::solve::{block_pattern, plan, solve_block_planned, BlockRows};
use rsdag::synth::Spec;
use rsdag::{Builder, ExprId, Graph, Node, Numeric, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// A block-sparse system over `pattern` (block positions) with `b` by `b`
/// blocks of symbols, diagonally dominant values, and its dense image.
fn system(
    g: &mut Graph<F64>,
    pattern: &[(usize, usize)],
    nb: usize,
    b: usize,
    seed: u64,
) -> (
    BlockRows,
    Vec<ExprId>,
    Vec<SymbolId>,
    Vec<f64>,
    Vec<Vec<f64>>,
) {
    let mut rng = Spec::new(seed).rng();
    let n = nb * b;
    let (mut syms, mut vals) = (Vec::new(), Vec::new());
    let mut dense = vec![vec![0.0; n]; n];
    let mut rows: BlockRows = vec![Vec::new(); nb];
    for &(i, j) in pattern {
        let mut block = Vec::with_capacity(b * b);
        for r in 0..b {
            for c in 0..b {
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
                block.push(e);
            }
        }
        rows[i].push((j, block));
    }
    let rhs: Vec<ExprId> = (0..n)
        .map(|k| {
            let e = g.sym(&format!("b{k}"));
            syms.push(sym(g, e));
            vals.push(1.0 + k as f64 * 0.1);
            e
        })
        .collect();
    (rows, rhs, syms, vals, dense)
}

fn check(pattern: &[(usize, usize)], nb: usize, b: usize, kernels: bool) {
    let mut g: Graph<F64> = Graph::new();
    let (rows, rhs, syms, vals, dense) = system(&mut g, pattern, nb, b, 11);
    let plan = plan(&block_pattern(&rows)).expect("plan");
    let solved = solve_block_planned(&mut g, &rows, b, &plan, &rhs);
    let tape = Tape::compile(&g, &solved.x, &syms);
    let d = tape.dump();
    if kernels {
        assert!(d.contains("SolveMany("), "{d}");
        assert!(d.contains("Gemm("), "{d}");
    }
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    let n = nb * b;
    let want = Numeric.solve(&dense, &vals[vals.len() - n..]);
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

#[test]
fn a_block_ring_matches_the_dense_solve_and_runs_on_kernels() {
    let nb = 5;
    let mut pattern = Vec::new();
    for i in 0..nb {
        pattern.push((i, i));
        pattern.push((i, (i + 1) % nb));
        pattern.push(((i + 1) % nb, i));
    }
    pattern.sort();
    pattern.dedup();
    check(&pattern, nb, 10, true);
}

#[test]
fn small_blocks_and_two_decoupled_components_match_the_dense_solve() {
    // Two components of the block pattern: BTF splits them.
    let pattern = vec![
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 2),
    ];
    check(&pattern, 4, 3, false);
    // A dense block pattern.
    let mut full = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            full.push((i, j));
        }
    }
    check(&full, 3, 4, false);
}
