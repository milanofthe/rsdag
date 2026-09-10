use rsdag::display::to_string;
use rsdag::symbolic::det::count_det_terms;
use rsdag::*;

#[test]
fn determinant_of_a_diagonal_and_a_2x2() {
    let mut g: Graph = Graph::new();
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
