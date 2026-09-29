use rsdag::display::to_string;
use rsdag::symbolic::det::count_det_terms;
use rsdag::BigRational;
use rsdag::*;

#[test]
fn determinant_of_a_diagonal_and_a_2x2() {
    let mut g: Graph<BigRational> = Graph::new();
    let (a, b, c, d) = (g.sym("a"), g.sym("b"), g.sym("c"), g.sym("d"));
    let z = g.zero();
    let det = determinant(&mut g, &[vec![a, z, z], vec![z, b, z], vec![z, z, c]]);
    let mut env = std::collections::HashMap::new();
    for (i, v) in [2.0, 3.0, 5.0].iter().enumerate() {
        env.insert(rsdag::node::SymbolId(i as u32), *v);
    }
    assert_eq!(rsdag::eval(&g, &[det], &env)[0], 30.0);
    let det2 = determinant(&mut g, &[vec![a, b], vec![c, d]]);
    assert_eq!(to_string(&g, det2), "(a*d + -b*c)");
    assert_eq!(
        count_det_terms(&[vec![true, false], vec![true, true]], 100),
        1
    );
    assert_eq!(
        count_det_terms(&[vec![true, true], vec![true, true]], 100),
        2
    );
}

/// The expansion against a plain numeric one on dense matrices, the term
/// count against `n!`, and a size whose `n!` expansion would not finish.
#[test]
fn determinants_expand_each_minor_once() {
    use rsdag::symbolic::count_det_terms;
    use rsdag::{eval, Graph, SymbolId, F64};
    fn laplace(m: &[Vec<f64>]) -> f64 {
        if m.len() == 1 {
            return m[0][0];
        }
        (0..m.len())
            .map(|j| {
                let minor: Vec<Vec<f64>> = m[1..]
                    .iter()
                    .map(|r| {
                        r.iter()
                            .enumerate()
                            .filter(|&(c, _)| c != j)
                            .map(|(_, &v)| v)
                            .collect()
                    })
                    .collect();
                let s = if j % 2 == 0 { 1.0 } else { -1.0 };
                s * m[0][j] * laplace(&minor)
            })
            .sum()
    }
    for n in 1..=7usize {
        let mut g: Graph<F64> = Graph::new();
        let xs: Vec<_> = (0..n * n).map(|k| g.sym(&format!("a{k}"))).collect();
        let m: Vec<Vec<_>> = (0..n).map(|i| xs[i * n..(i + 1) * n].to_vec()).collect();
        let d = determinant(&mut g, &m);
        let vals: Vec<f64> = (0..n * n)
            .map(|k| ((k * 37 + 11) % 17) as f64 / 7.0 - 1.0)
            .collect();
        let env = (0..n * n).map(|k| (SymbolId(k as u32), vals[k])).collect();
        let want = laplace(
            &(0..n)
                .map(|i| vals[i * n..(i + 1) * n].to_vec())
                .collect::<Vec<_>>(),
        );
        let got = eval(&g, &[d], &env)[0];
        assert!(
            (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
            "n {n}: {got} vs {want}"
        );
        let fact: u64 = (1..=n as u64).product();
        assert_eq!(count_det_terms(&vec![vec![true; n]; n], u64::MAX), fact);
    }
    let n = 14;
    let mut g: Graph<F64> = Graph::new();
    let xs: Vec<_> = (0..n * n).map(|k| g.sym(&format!("a{k}"))).collect();
    let m: Vec<Vec<_>> = (0..n).map(|i| xs[i * n..(i + 1) * n].to_vec()).collect();
    let _ = determinant(&mut g, &m);
    assert_eq!(
        count_det_terms(&vec![vec![true; n]; n], 1 << 40),
        87_178_291_200
    );
}
