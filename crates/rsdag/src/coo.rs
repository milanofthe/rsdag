//! Sparse matrices of expressions in coordinate form: what a sparse
//! Jacobian becomes once its rows are laid out as blocks of a larger matrix.

use crate::autodiff::{sparse_jacobian, SparseRows};
use crate::field::Field;
use crate::graph::Graph;
use crate::node::{ExprId, SymbolId};

/// A sparse matrix of expressions: entry `k` is `vals[k]` at row `rows[k]`
/// and column `cols[k]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coo {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<ExprId>,
}

impl Coo {
    /// The entries of `rows`, row by row, each row's in its order.
    pub fn from_rows(rows: SparseRows) -> Self {
        let mut out = Coo::default();
        for (i, row) in rows.into_iter().enumerate() {
            for (j, e) in row {
                out.push(i, j, e);
            }
        }
        out
    }

    /// The sparse Jacobian of `roots` by `wrt` (see [`sparse_jacobian`]).
    pub fn jacobian<K: Field>(ctx: &mut Graph<K>, roots: &[ExprId], wrt: &[SymbolId]) -> Self {
        Self::from_rows(sparse_jacobian(ctx, roots, wrt))
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.vals.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }

    /// Add `val` at `(row, col)`.
    pub fn push(&mut self, row: usize, col: usize, val: ExprId) {
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(val);
    }

    /// Add the entries of `other`, their rows and columns shifted by
    /// `(row, col)`: a block of a larger matrix.
    pub fn append(&mut self, other: &Coo, row: usize, col: usize) {
        self.rows.extend(other.rows.iter().map(|&r| r + row));
        self.cols.extend(other.cols.iter().map(|&c| c + col));
        self.vals.extend_from_slice(&other.vals);
    }

    /// The columns relabelled by `f`.
    pub fn map_cols(mut self, f: impl Fn(usize) -> usize) -> Self {
        for c in &mut self.cols {
            *c = f(*c);
        }
        self
    }

    /// The matrix cut into row blocks at the ascending row indices `at`:
    /// block `b` holds the rows from `at[b - 1]` (from `0` for the first) up
    /// to `at[b]` (to the end for the last), counted from its first. The
    /// entries keep their order.
    pub fn split_rows(self, at: &[usize]) -> Vec<Coo> {
        let mut blocks = vec![Coo::default(); at.len() + 1];
        for ((r, c), v) in self.rows.into_iter().zip(self.cols).zip(self.vals) {
            let b = at.partition_point(|&a| a <= r);
            let start = if b == 0 { 0 } else { at[b - 1] };
            blocks[b].push(r - start, c, v);
        }
        blocks
    }
}

/// The rows of the transpose of `rows`, a matrix of `n_cols` columns: row
/// `j` lists `(i, e)` for each entry `e` of `rows[i]` in column `j`, by `i`.
pub fn transpose(rows: &SparseRows, n_cols: usize) -> SparseRows {
    let mut out: SparseRows = vec![Vec::new(); n_cols];
    for (i, row) in rows.iter().enumerate() {
        for &(j, e) in row {
            out[j].push((i, e));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coo_blocks_and_transpose() {
        let mut g: Graph = Graph::new();
        let (x, y) = (g.sym("x"), g.sym("y"));
        let (sx, sy) = (g.as_symbol(x).unwrap(), g.as_symbol(y).unwrap());
        let xy = g.mul(x, y);
        let x2 = g.mul(x, x);
        // rows: x*y, y, x*x
        let coo = Coo::jacobian(&mut g, &[xy, y, x2], &[sx, sy]);
        assert_eq!(coo.rows, vec![0, 0, 1, 2]);
        assert_eq!(coo.cols, vec![0, 1, 1, 0]);
        let blocks = coo.clone().split_rows(&[1]);
        assert_eq!(blocks[0].rows, vec![0, 0]);
        assert_eq!(blocks[1].rows, vec![0, 1]);
        assert_eq!(blocks[1].cols, vec![1, 0]);
        let shifted = coo.map_cols(|c| c + 10);
        assert_eq!(shifted.cols, vec![10, 11, 11, 10]);
        let mut whole = Coo::default();
        whole.append(&blocks[1], 1, 0);
        assert_eq!(whole.rows, vec![1, 2]);
        let rows = sparse_jacobian(&mut g, &[xy, y, x2], &[sx, sy]);
        let t = transpose(&rows, 2);
        assert_eq!(t[0].iter().map(|&(i, _)| i).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(t[1].iter().map(|&(i, _)| i).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn symbols_and_summands() {
        let mut g: Graph = Graph::new();
        assert_eq!(g.find_symbol("a"), None);
        let a = g.symbol_id("a");
        assert_eq!(g.find_symbol("a"), Some(a));
        let (ea, b) = (g.symbol_expr(a), g.sym("b"));
        let s = g.reduce(crate::node::ReduceOp::Sum, vec![ea, b]);
        assert_eq!(g.summands(s).len(), 2);
        let zero = g.zero();
        assert!(g.summands(zero).is_empty());
        assert_eq!(g.summands(ea), vec![ea]);
        assert_eq!(g.as_symbol(s), None);
    }
}
