//! Symbolic determinants.

use crate::field::Field;
use crate::graph::Graph;
use crate::node::ExprId;

/// Determinant of a square matrix of expressions by Laplace expansion along
/// the first row, skipping structurally zero entries (the sparse case a
/// circuit matrix is). Exponential in general; use [`count_det_terms`] to
/// bound the term count first.
pub fn determinant<K: Field>(g: &mut Graph<K>, m: &[Vec<ExprId>]) -> ExprId {
    let n = m.len();
    match n {
        0 => g.one(),
        1 => m[0][0],
        2 => {
            let ad = g.mul(m[0][0], m[1][1]);
            let bc = g.mul(m[0][1], m[1][0]);
            g.sub(ad, bc)
        }
        _ => {
            let mut acc = g.zero();
            for j in 0..n {
                let entry = m[0][j];
                if g.is_zero(entry) {
                    continue;
                }
                let minor = minor_matrix(m, 0, j);
                let sub = determinant(g, &minor);
                let mut term = g.mul(entry, sub);
                if j % 2 == 1 {
                    term = g.neg(term);
                }
                acc = g.add(acc, term);
            }
            acc
        }
    }
}

/// Number of nonzero terms a determinant expansion of a matrix with the
/// given nonzero `pattern` produces, saturating at `cap` (matrices above
/// 100 rows report `cap`).
pub fn count_det_terms(pattern: &[Vec<bool>], cap: u64) -> u64 {
    let n = pattern.len();
    if n > 100 || cap == 0 {
        return cap;
    }
    let rows: Vec<u128> = pattern
        .iter()
        .map(|r| {
            let mut m = 0u128;
            for (j, &b) in r.iter().enumerate() {
                if b {
                    m |= 1u128 << j;
                }
            }
            m
        })
        .collect();
    fn rec(rows: &[u128], depth: usize, avail: u128, cap: u64) -> u64 {
        if depth == rows.len() {
            return 1;
        }
        let mut acc: u64 = 0;
        let mut bits = rows[depth] & avail;
        while bits != 0 {
            let j = bits.trailing_zeros();
            bits &= bits - 1;
            acc = acc.saturating_add(rec(
                rows,
                depth + 1,
                avail & !(1u128 << j),
                cap.saturating_sub(acc),
            ));
            if acc >= cap {
                return cap;
            }
        }
        acc
    }
    rec(&rows, 0, !0u128, cap).min(cap)
}

fn minor_matrix(m: &[Vec<ExprId>], row: usize, col: usize) -> Vec<Vec<ExprId>> {
    let n = m.len();
    let mut out = Vec::with_capacity(n - 1);
    for (i, r) in m.iter().enumerate() {
        if i == row {
            continue;
        }
        let mut new_row = Vec::with_capacity(n - 1);
        for (j, &v) in r.iter().enumerate() {
            if j == col {
                continue;
            }
            new_row.push(v);
        }
        out.push(new_row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::to_string;

    #[test]
    fn determinant_of_a_diagonal_and_a_2x2() {
        let mut g: Graph = Graph::new();
        let (a, b, c, d) = (g.sym("a"), g.sym("b"), g.sym("c"), g.sym("d"));
        let z = g.zero();
        let det = determinant(&mut g, &[vec![a, z, z], vec![z, b, z], vec![z, z, c]]);
        let mut env = std::collections::HashMap::new();
        for (i, v) in [2.0, 3.0, 5.0].iter().enumerate() {
            env.insert(crate::node::SymbolId(i as u32), *v);
        }
        assert_eq!(crate::eval::eval_real(&g, &env, &[det])[0], 30.0);
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
}
