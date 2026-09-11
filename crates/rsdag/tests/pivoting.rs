//! The planned solve is a static elimination with a guard: each step whose
//! column has more than one structural candidate checks that its pivot row
//! still dominates the column by the tolerance. When the guard fails, the
//! plan is repivoted on the current values and the program rebuilt; the
//! result is then the pivoted numeric solve's.

use rsdag::symbolic::solve::{pattern_of, plan, solve_planned, sparse_rows};
use rsdag::synth::Spec;
use rsdag::{Builder, ExprId, Graph, Node, Numeric, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// A dense symbolic system, and a numeric instance of it.
fn dense(
    g: &mut Graph<F64>,
    n: usize,
    values: impl Fn(usize, usize) -> f64,
) -> (
    Vec<Vec<ExprId>>,
    Vec<ExprId>,
    Vec<SymbolId>,
    Vec<f64>,
    Vec<Vec<f64>>,
) {
    let (mut syms, mut vals) = (Vec::new(), Vec::new());
    let mut num = vec![vec![0.0; n]; n];
    let a: Vec<Vec<ExprId>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let e = g.sym(&format!("a{i}_{j}"));
                    syms.push(sym(g, e));
                    num[i][j] = values(i, j);
                    vals.push(num[i][j]);
                    e
                })
                .collect()
        })
        .collect();
    let b: Vec<ExprId> = (0..n)
        .map(|i| {
            let e = g.sym(&format!("b{i}"));
            syms.push(sym(g, e));
            vals.push(1.0 + i as f64);
            e
        })
        .collect();
    (a, b, syms, vals, num)
}

/// Evaluates the solve `x` and its guard on `vals`.
fn run(
    g: &Graph<F64>,
    x: &[ExprId],
    guard: ExprId,
    syms: &[SymbolId],
    vals: &[f64],
) -> (Vec<f64>, f64) {
    let mut roots = x.to_vec();
    roots.push(guard);
    let tape = Tape::compile(g, &roots, syms);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(vals, &mut w, &mut o);
    let guard = o.pop().unwrap();
    (o, guard)
}

#[test]
fn a_zero_diagonal_fails_the_guard_and_repivots() {
    // [[0, 1], [1, 1]] x = b: the transversal's diagonal is zero.
    let mut g: Graph<F64> = Graph::new();
    let (a, b, syms, vals, num) = dense(&mut g, 2, |i, j| if i == 0 && j == 0 { 0.0 } else { 1.0 });
    let rows = sparse_rows(&g, &a);
    let pattern = pattern_of(&rows);
    let p = plan(&pattern).unwrap();
    let s = solve_planned(&mut g, &rows, &p, &b);
    let (_, ok) = run(&g, &s.x, s.pivots_ok, &syms, &vals);
    assert_eq!(ok, 0.0, "the zero pivot fails the guard");

    let p2 = p.repivot(&pattern, |i, j| num[i][j]);
    let s2 = solve_planned(&mut g, &rows, &p2, &b);
    let (x, ok) = run(&g, &s2.x, s2.pivots_ok, &syms, &vals);
    assert_eq!(ok, 1.0, "the repivoted plan holds");
    let want = Numeric.solve(&num, &vals[4..]);
    for (k, (p, q)) in want.iter().zip(&x).enumerate() {
        assert!(
            (p - q).abs() <= 1e-12 * (1.0 + p.abs()),
            "x[{k}]: {p} vs {q}"
        );
    }
}

#[test]
fn random_systems_repivoted_match_the_pivoted_numeric_solve() {
    let mut rng = Spec::new(21).rng();
    for trial in 0..30 {
        let n = 2 + trial % 7;
        let mut g: Graph<F64> = Graph::new();
        let entries: Vec<f64> = (0..n * n).map(|_| rng.val() - 0.5).collect();
        let (a, b, syms, vals, num) = dense(&mut g, n, |i, j| entries[i * n + j]);
        let rows = sparse_rows(&g, &a);
        let pattern = pattern_of(&rows);
        let p = plan(&pattern).unwrap().repivot(&pattern, |i, j| num[i][j]);
        let s = solve_planned(&mut g, &rows, &p, &b);
        let (x, ok) = run(&g, &s.x, s.pivots_ok, &syms, &vals);
        assert_eq!(
            ok, 1.0,
            "trial {trial}: the guard holds on the values it was pivoted for"
        );
        let want = Numeric.solve(&num, &vals[n * n..]);
        let scale = want.iter().map(|v| v.abs()).fold(1.0f64, f64::max);
        for (k, (p, q)) in want.iter().zip(&x).enumerate() {
            assert!(
                (p - q).abs() <= 1e-9 * scale,
                "trial {trial} x[{k}]: {p} vs {q}"
            );
        }
    }
}

#[test]
fn a_single_candidate_step_has_no_guard() {
    // An upper-triangular pattern has one structural candidate per step:
    // no guard anywhere, the program is the static elimination's.
    let n = 6;
    let mut g: Graph<F64> = Graph::new();
    let (a, b, syms, _, _) = dense(&mut g, n, |_, _| 1.0);
    let upper: Vec<Vec<ExprId>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if j >= i { a[i][j] } else { g.zero() })
                .collect()
        })
        .collect();
    let rows = sparse_rows(&g, &upper);
    let p = plan(&pattern_of(&rows)).unwrap();
    let s = solve_planned(&mut g, &rows, &p, &b);
    assert_eq!(s.fill, 0);
    let one = g.one();
    assert_eq!(s.pivots_ok, one, "the guard is the constant one");
    let tape = Tape::compile(&g, &s.x, &syms);
    let d = tape.dump();
    assert!(!d.contains("Cmp") && !d.contains("Abs"), "{d}");
}

#[test]
fn a_dominant_diagonal_passes_the_guard_without_repivoting() {
    let n = 5;
    let mut g: Graph<F64> = Graph::new();
    let (a, b, syms, vals, num) = dense(&mut g, n, |i, j| if i == j { 10.0 } else { 1.0 });
    let rows = sparse_rows(&g, &a);
    let p = plan(&pattern_of(&rows)).unwrap();
    let s = solve_planned(&mut g, &rows, &p, &b);
    let (x, ok) = run(&g, &s.x, s.pivots_ok, &syms, &vals);
    assert_eq!(ok, 1.0);
    let want = Numeric.solve(&num, &vals[n * n..]);
    for (p, q) in want.iter().zip(&x) {
        assert!((p - q).abs() <= 1e-12 * (1.0 + p.abs()));
    }
}
