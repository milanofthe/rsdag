//! The supernodal program: the planned scalar elimination over panels,
//! matching the dense solve, with kernels where the pattern has panels,
//! and guarded across panels.

use num_complex::Complex64;
use rsdag::symbolic::solve::{
    pattern_of, plan, solve_planned, solve_supernodal_planned, supernodes, Cx, Plan,
};
use rsdag::synth::Spec;
use rsdag::{Builder, ExprId, Graph, Node, Numeric, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// A random circuit-like pattern: the diagonal plus `extra` entries per
/// row, values diagonally dominant.
fn random_system(
    n: usize,
    extra: usize,
    seed: u64,
) -> (
    Graph<F64>,
    Vec<Vec<(usize, ExprId)>>,
    Vec<ExprId>,
    Vec<SymbolId>,
    Vec<f64>,
    Vec<Vec<f64>>,
) {
    let mut rng = Spec::new(seed).rng();
    let mut g: Graph<F64> = Graph::new();
    let (mut syms, mut vals) = (Vec::new(), Vec::new());
    let mut dense = vec![vec![0.0; n]; n];
    let mut rows: Vec<Vec<(usize, ExprId)>> = vec![Vec::new(); n];
    for i in 0..n {
        let mut cols = vec![i];
        for _ in 0..extra {
            cols.push((rng.val() * n as f64) as usize % n);
        }
        cols.sort_unstable();
        cols.dedup();
        for j in cols {
            let e = g.sym(&format!("a{i}_{j}"));
            syms.push(sym(&g, e));
            let v = rng.val() - 0.5 + if i == j { 3.0 + extra as f64 } else { 0.0 };
            vals.push(v);
            dense[i][j] = v;
            rows[i].push((j, e));
        }
    }
    let rhs: Vec<ExprId> = (0..n)
        .map(|k| {
            let e = g.sym(&format!("b{k}"));
            syms.push(sym(&g, e));
            vals.push(1.0 + 0.1 * k as f64);
            e
        })
        .collect();
    (g, rows, rhs, syms, vals, dense)
}

#[test]
fn random_patterns_match_the_dense_solve_and_pass_the_guard() {
    for (n, extra, seed) in [(12, 2, 1), (40, 3, 2), (120, 4, 3), (60, 1, 4)] {
        let (mut g, rows, rhs, syms, vals, dense) = random_system(n, extra, seed);
        let plan = plan(&pattern_of(&rows)).expect("plan");
        let sn = supernodes(&pattern_of(&rows), &plan);
        assert_eq!(sn.widths().iter().sum::<usize>(), n);
        let solved = solve_supernodal_planned(&mut g, &rows, &plan, &sn, &rhs);
        let mut roots = solved.x.clone();
        roots.push(solved.pivots_ok);
        let tape = Tape::compile(&g, &roots, &syms);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&vals, &mut w, &mut o);
        assert_eq!(o[n], 1.0, "guard on a dominant diagonal (n={n})");
        let want = Numeric.solve(&dense, &vals[vals.len() - n..]);
        let scale = want.iter().map(|v| v.abs()).fold(1.0f64, f64::max);
        for k in 0..n {
            assert!(
                (want[k] - o[k]).abs() <= 1e-9 * scale,
                "n={n} x[{k}]: {} vs {}",
                want[k],
                o[k]
            );
        }
    }
}

/// A ring of `nb` unknown groups of `b` scalars: dense diagonal groups,
/// diagonal couplings (the shape of a harmonic-balance Jacobian).
fn hb_like(
    nb: usize,
    b: usize,
) -> (
    Graph<F64>,
    Vec<Vec<(usize, Cx)>>,
    Vec<Cx>,
    Vec<SymbolId>,
    Vec<f64>,
    Vec<Vec<Complex64>>,
) {
    let mut rng = Spec::new(7).rng();
    let mut g: Graph<F64> = Graph::new();
    let n = nb * b;
    let (mut syms, mut vals) = (Vec::new(), Vec::new());
    let mut dense = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut rows: Vec<Vec<(usize, Cx)>> = vec![Vec::new(); n];
    let mut entry = |g: &mut Graph<F64>, i: usize, j: usize, big: bool| {
        let re = g.sym(&format!("a{i}_{j}r"));
        let im = g.sym(&format!("a{i}_{j}i"));
        syms.push(sym(g, re));
        syms.push(sym(g, im));
        let v = Complex64::new(rng.val() - 0.5, rng.val() - 0.5)
            + if big {
                Complex64::new(3.0 * b as f64, 2.0 * b as f64)
            } else {
                Complex64::new(0.0, 0.0)
            };
        vals.push(v.re);
        vals.push(v.im);
        dense[i][j] = v;
        rows[i].push((j, Cx::new(re, im)));
    };
    for gi in 0..nb {
        for r in 0..b {
            for c in 0..b {
                entry(&mut g, gi * b + r, gi * b + c, r == c);
            }
            let gj = (gi + 1) % nb;
            entry(&mut g, gi * b + r, gj * b + r, false);
            entry(&mut g, gj * b + r, gi * b + r, false);
        }
    }
    let mut rhs = Vec::new();
    for k in 0..n {
        let re = g.sym(&format!("b{k}r"));
        let im = g.sym(&format!("b{k}i"));
        syms.push(sym(&g, re));
        syms.push(sym(&g, im));
        vals.push(1.0 + 0.1 * k as f64);
        vals.push(-0.3);
        rhs.push(Cx::new(re, im));
    }
    (g, rows, rhs, syms, vals, dense)
}

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

#[test]
fn a_harmonic_balance_shaped_complex_system_runs_on_panels_and_kernels() {
    let (nb, b) = (6, 5);
    let (mut g, rows, rhs, syms, vals, dense) = hb_like(nb, b);
    let real_pattern = pattern_of(
        &rows
            .iter()
            .map(|r| r.iter().map(|&(j, e)| (j, e.re)).collect())
            .collect(),
    );
    let plan = plan(&real_pattern).expect("plan");
    let sn = supernodes(&real_pattern, &plan);
    assert!(
        sn.max_width() >= b,
        "panels of the dense groups: {:?}",
        sn.widths()
    );
    let solved = solve_supernodal_planned(&mut g, &rows, &plan, &sn, &rhs);
    let mut roots: Vec<ExprId> = solved.x.iter().flat_map(|c| [c.re, c.im]).collect();
    roots.push(solved.pivots_ok);
    let tape = Tape::compile(&g, &roots, &syms);
    let d = tape.dump();
    assert!(d.contains("SolveMany("), "{d}");
    assert!(d.contains("Gemm("), "{d}");
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    let n = nb * b;
    assert_eq!(o[2 * n], 1.0, "guard");
    let want = dense_solve(
        dense,
        (0..n)
            .map(|k| Complex64::new(1.0 + 0.1 * k as f64, -0.3))
            .collect(),
    );
    let scale = want.iter().map(|v| v.norm()).fold(1.0f64, f64::max);
    for k in 0..n {
        let got = Complex64::new(o[2 * k], o[2 * k + 1]);
        assert!(
            (want[k] - got).norm() <= 1e-9 * scale,
            "x[{k}]: {} vs {got}",
            want[k]
        );
    }
}

#[test]
fn the_panel_guard_rejects_a_tiny_pivot_and_the_repivoted_plan_passes() {
    // A star: hub 0, leaves 1..3. The leaves are eliminated first, each a
    // panel of its own with the hub's row below it; leaf 1's tiny pivot
    // against the hub's entry fails the guard (a panel of several would
    // pivot inside the dense kernel and need none).
    let mut g: Graph<F64> = Graph::new();
    let names = [
        "a00", "a01", "a02", "a03", "a10", "a11", "a20", "a22", "a30", "a33", "b0", "b1", "b2",
        "b3",
    ];
    let es: Vec<ExprId> = names.iter().map(|n| g.sym(n)).collect();
    let syms: Vec<SymbolId> = es.iter().map(|&e| sym(&g, e)).collect();
    let rows = vec![
        vec![(0, es[0]), (1, es[1]), (2, es[2]), (3, es[3])],
        vec![(0, es[4]), (1, es[5])],
        vec![(0, es[6]), (2, es[7])],
        vec![(0, es[8]), (3, es[9])],
    ];
    let rhs = vec![es[10], es[11], es[12], es[13]];
    let vals = [
        4.0, 1.0, 1.0, 1.0, 1.0, 1e-9, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 3.0, 4.0,
    ];
    let dense: Vec<Vec<f64>> = vec![
        vec![4.0, 1.0, 1.0, 1.0],
        vec![1.0, 1e-9, 0.0, 0.0],
        vec![1.0, 0.0, 2.0, 0.0],
        vec![1.0, 0.0, 0.0, 3.0],
    ];
    let pattern = pattern_of(&rows);
    let plan: Plan = plan(&pattern).expect("plan");
    let run = |g: &mut Graph<F64>, plan: &Plan| -> (f64, Vec<f64>) {
        let sn = supernodes(&pattern, plan);
        let solved = solve_supernodal_planned(g, &rows, plan, &sn, &rhs);
        let mut roots = solved.x.clone();
        roots.push(solved.pivots_ok);
        let tape = Tape::compile(g, &roots, &syms);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&vals, &mut w, &mut o);
        (o[4], o[..4].to_vec())
    };
    let (guard, _) = run(&mut g, &plan);
    let scalar = solve_planned(&mut g, &rows, &plan, &rhs);
    let tape = Tape::compile(&g, &[scalar.pivots_ok], &syms);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    assert_eq!(guard, o[0], "the panel guard agrees with the scalar guard");
    assert_eq!(guard, 0.0);
    let re = plan.repivot(&pattern, |i, j| dense[i][j].abs());
    let (guard, x) = run(&mut g, &re);
    assert_eq!(guard, 1.0);
    let want = Numeric.solve(&dense, &[1.0, 2.0, 3.0, 4.0]);
    for k in 0..4 {
        assert!(
            (x[k] - want[k]).abs() < 1e-9 * want[k].abs().max(1.0),
            "{x:?} vs {want:?}"
        );
    }
}
