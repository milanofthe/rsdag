//! The sparse forms scale: a Jacobian is built from what each row touches,
//! the ordering and the factorization touch only nonzeros, and a Newton step
//! over a hundred thousand unknowns is a program of tens of ops per unknown
//! that converges.

use rsdag::symbolic::solve::ordering;
use rsdag::{newton_step, sparse_jacobian, ExprId, Graph, Node, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// A ring of nonlinear cells, each residual touching its two neighbours:
/// the pattern of a discretised line, tridiagonal plus the wrap.
fn ring(g: &mut Graph<F64>, n: usize) -> (Vec<ExprId>, Vec<SymbolId>) {
    let xs: Vec<ExprId> = (0..n).map(|i| g.sym(&format!("x{i}"))).collect();
    let syms: Vec<SymbolId> = xs.iter().map(|&e| sym(g, e)).collect();
    let f = (0..n)
        .map(|i| {
            let (l, c, r) = (xs[(i + n - 1) % n], xs[i], xs[(i + 1) % n]);
            let d1 = g.sub(c, l);
            let d2 = g.sub(r, c);
            let e1 = g.exp(d1);
            let e2 = g.exp(d2);
            let s = g.sub(e1, e2);
            let two = g.konst_f64(2.0);
            let t = g.mul(two, c);
            let one = g.one();
            let u = g.sub(t, one);
            g.add(s, u)
        })
        .collect();
    (f, syms)
}

#[test]
fn the_sparse_jacobian_is_the_dense_one() {
    let mut g: Graph<F64> = Graph::new();
    let (f, syms) = ring(&mut g, 40);
    let sparse = sparse_jacobian(&mut g, &f, &syms);
    let dense = rsdag::jacobian(&mut g, &f, &syms);
    for (i, row) in sparse.iter().enumerate() {
        assert_eq!(row.len(), 3, "row {i} touches three unknowns");
        for &(j, e) in row {
            assert_eq!(e, dense[i][j]);
        }
        for (j, &e) in dense[i].iter().enumerate() {
            let present = row.iter().any(|&(c, _)| c == j);
            assert_eq!(!g.is_zero(e), present, "entry ({i}, {j})");
        }
    }
}

#[test]
fn a_global_net_is_eliminated_last_without_fill() {
    // A star: every row touches itself and column 0, like a ground net.
    let n = 200;
    let pattern: Vec<Vec<usize>> = (0..n)
        .map(|i| if i == 0 { (0..n).collect() } else { vec![0, i] })
        .collect();
    let order = ordering(&pattern);
    // The hub's degree only falls as the leaves go; it is eliminated once
    // it ties the last leaf at degree one, so it is one of the last two.
    let at = order.iter().position(|&k| k == 0).unwrap();
    assert!(at >= n - 2, "the hub is eliminated at position {at} of {n}");
    let mut g: Graph<F64> = Graph::new();
    let rows: rsdag::SparseRows = pattern
        .iter()
        .enumerate()
        .map(|(i, r)| {
            r.iter()
                .map(|&j| (j, g.sym(&format!("a{i}_{j}"))))
                .collect()
        })
        .collect();
    let lu = rsdag::symbolic::solve::lu_static(&mut g, &rows, &order);
    assert_eq!(lu.fill, 0);
}

#[test]
fn a_newton_step_over_a_hundred_thousand_unknowns_builds_and_converges() {
    let n = 100_000;
    let mut g: Graph<F64> = Graph::new();
    let (f, syms) = ring(&mut g, n);
    let (step, fill) = newton_step(&mut g, &f, &syms);
    // Eliminating a degree-two vertex of a cycle creates one fill entry.
    assert!(fill <= 2 * n, "fill {fill} on a ring of {n}");
    let step_tape = Tape::compile(&g, &step, &syms);
    let res_tape = Tape::compile(&g, &f, &syms);
    assert!(
        step_tape.n_ops() < 80 * n,
        "{} ops for a Newton step over {n} unknowns",
        step_tape.n_ops()
    );
    let mut x: Vec<f64> = (0..n).map(|k| 0.4 + 0.2 * ((k % 7) as f64) / 7.0).collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let norm = |v: &[f64]| v.iter().map(|t| t * t).sum::<f64>().sqrt();
    res_tape.eval(&x, &mut w, &mut o);
    let start = norm(&o);
    let mut last = start;
    for _ in 0..10 {
        step_tape.eval(&x, &mut w, &mut o);
        x.copy_from_slice(&o);
        res_tape.eval(&x, &mut w, &mut o);
        last = norm(&o);
        assert!(last.is_finite());
        if last < 1e-10 {
            break;
        }
    }
    assert!(last < 1e-9 * start.max(1.0), "residual {last} from {start}");
}
