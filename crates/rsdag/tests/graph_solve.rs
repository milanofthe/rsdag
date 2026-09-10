//! A linear solve as graph ops: correct against a pivoted dense solve, the
//! ordering keeps fill down, and a Newton step composed of it converges.

use rsdag::symbolic::solve::{amd::amd, lu_static, pattern_of, Pattern, SparseRows};
use rsdag::synth::Spec;
use rsdag::{newton_step, Builder, ExprId, Graph, Node, Numeric, Scope, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// The circuit-like residual: row i touches x_i, x_{i+1}, x_{i+7}, with an
/// exponential per row.
fn residual(g: &mut Graph<F64>, n: usize) -> (Vec<ExprId>, Vec<SymbolId>) {
    let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<SymbolId> = xs.iter().map(|&e| sym(g, e)).collect();
    let vt = g.konst_f64(0.25);
    let is = g.konst_f64(1e-3);
    let outs = (0..n)
        .map(|i| {
            let (a, b, c) = (xs[i], xs[(i + 1) % n], xs[(i + 7) % n]);
            let q = g.div(a, vt);
            let e = g.exp(q);
            let one = g.one();
            let m = g.sub(e, one);
            let diode = g.mul(is, m);
            let two = g.konst_f64(2.0);
            let t = g.mul(two, a);
            let s = g.add(t, b);
            let lin = g.sub(s, c);
            g.add(diode, lin)
        })
        .collect();
    (outs, syms)
}

fn fill_of(pattern: &Pattern, order: &[usize]) -> usize {
    // Count fill by running the structural elimination on symbols.
    let mut g: Graph<F64> = Graph::new();
    let m: SparseRows = pattern
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .map(|&j| (j, g.sym(&format!("a{i}_{j}"))))
                .collect()
        })
        .collect();
    lu_static(&mut g, &m, order).fill
}

#[test]
fn the_ordering_reduces_fill() {
    let mut g: Graph<F64> = Graph::new();
    let (f, syms) = residual(&mut g, 128);
    let jac = rsdag::sparse_jacobian(&mut g, &f, &syms);
    let pattern = pattern_of(&jac);
    let natural: Vec<usize> = (0..128).collect();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); 128];
    for (i, row) in pattern.iter().enumerate() {
        for &j in row {
            if i != j {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let md = amd(&adj);
    let (a, b) = (fill_of(&pattern, &natural), fill_of(&pattern, &md));
    assert!(b <= a, "minimum degree fill {b} against natural {a}");
    // Every index exactly once.
    let mut seen = md.clone();
    seen.sort_unstable();
    assert_eq!(seen, natural);
}

#[test]
fn the_static_solve_matches_a_pivoted_dense_solve() {
    let mut rng = Spec::new(5).rng();
    for trial in 0..20 {
        let n = 3 + trial % 9;
        let mut g: Graph<F64> = Graph::new();
        // A diagonally dominant random matrix over symbols, so static
        // pivoting is safe and the comparison is about the arithmetic.
        let a: Vec<Vec<ExprId>> = (0..n)
            .map(|i| (0..n).map(|j| g.sym(&format!("a{i}_{j}"))).collect())
            .collect();
        let b: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("b{i}"))).collect();
        let mut vals: Vec<f64> = Vec::new();
        let mut syms: Vec<SymbolId> = Vec::new();
        let mut dense = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let v = if i == j { 10.0 + rng.val() } else { rng.val() };
                dense[i][j] = v;
                vals.push(v);
                syms.push(sym(&g, a[i][j]));
            }
        }
        let rhs: Vec<f64> = (0..n).map(|_| rng.val()).collect();
        for (i, &v) in rhs.iter().enumerate() {
            vals.push(v);
            syms.push(sym(&g, b[i]));
        }
        let want = Numeric.solve(&dense, &rhs);

        let x = g.solve(&a, &b);
        let tape = Tape::compile(&g, &x, &syms);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&vals, &mut w, &mut o);
        for (k, (p, q)) in want.iter().zip(&o).enumerate() {
            assert!(
                (p - q).abs() <= 1e-12 * (1.0 + p.abs()),
                "trial {trial} x[{k}]: numeric {p} vs graph {q}"
            );
        }
    }
}

#[test]
fn a_newton_step_composed_of_graph_ops_converges() {
    let n = 64;
    let mut g: Graph<F64> = Graph::new();
    let (f, syms) = residual(&mut g, n);
    let (step, fill) = newton_step(&mut g, &f, &syms);
    assert!(
        fill < 2 * n * 8,
        "fill {fill} for a banded pattern of width 8"
    );
    let step_tape = Tape::compile(&g, &step, &syms);
    let res_tape = Tape::compile(&g, &f, &syms);

    let mut x: Vec<f64> = (0..n).map(|k| 0.1 + 0.3 * ((k % 5) as f64) / 5.0).collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let norm = |v: &[f64]| v.iter().map(|t| t * t).sum::<f64>().sqrt();
    res_tape.eval(&x, &mut w, &mut o);
    let start = norm(&o);
    let mut last = start;
    for it in 0..12 {
        step_tape.eval(&x, &mut w, &mut o);
        x.copy_from_slice(&o);
        res_tape.eval(&x, &mut w, &mut o);
        let now = norm(&o);
        assert!(now.is_finite(), "iteration {it} diverged");
        last = now;
        if now < 1e-12 {
            break;
        }
    }
    assert!(
        last < 1e-10 * start.max(1.0),
        "residual {last} after Newton from {start}"
    );
}

/// The solve is reachable from `Builder`: a model with an implicit update
/// computes it numerically and records it symbolically, and the two agree.
#[test]
fn a_block_with_an_implicit_update_records_and_computes_the_same() {
    fn implicit<B: Builder>(b: &mut B, u: B::N, k: B::N) -> Vec<B::N> {
        // (I + k*A) x = [u, 0] for a 2x2 coupling A = [[1, -1], [-1, 1]].
        let one = b.cst(1.0);
        let zero = b.cst(0.0);
        let d = b.add(one, k);
        let nk = b.neg(k);
        let a = vec![vec![d, nk], vec![nk, d]];
        b.solve(&a, &[u, zero])
    }
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "implicit");
    let (u, k) = (s.param("u"), s.param("k"));
    let out = implicit(&mut *s, u, k);
    let f = s.close(out.clone());
    let params = g.func(f).params.clone();
    let tape = Tape::compile(&g, &out, &params);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[3.0f64, 0.5], &mut w, &mut o);
    let want = implicit(&mut Numeric, 3.0, 0.5);
    for (p, q) in want.iter().zip(&o) {
        assert!((p - q).abs() <= 1e-14 * (1.0 + p.abs()), "{p} vs {q}");
    }
    // And it differentiates through the solve.
    let d = rsdag::differentiate(&mut g, out[0], params[1]);
    let dt = Tape::compile(&g, &[d], &params);
    dt.eval(&[3.0f64, 0.5], &mut w, &mut o);
    let h = 1e-6;
    let fd = (implicit(&mut Numeric, 3.0, 0.5 + h)[0] - implicit(&mut Numeric, 3.0, 0.5 - h)[0])
        / (2.0 * h);
    assert!(
        (o[0] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
        "d/dk: {} vs fd {fd}",
        o[0]
    );
}
