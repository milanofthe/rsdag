//! The solve of a block-sparse system as graph ops: a sparse pattern of
//! `b` by `b` blocks (a harmonic-balance Jacobian, a multi-port coupling),
//! eliminated block by block along the block pattern's [`Plan`]. A block
//! is dense or diagonal ([`Block`]); a dense pivot block is inverted
//! through the dense [`Graph::solve_dense`] kernel (its `b` right-hand
//! sides fuse into one factorization), a diagonal one by reciprocals, and
//! every block update is a set of dot products over whole block rows and
//! columns where both factors are dense, which the tape compiler fuses
//! into `Gemm` kernels, and a product per entry where one is diagonal.
//! The scalar is a [`Num`]: real, or complex as pairs of real expressions.
//!
//! The pivot blocks are the block pattern's transversal; the dense kernel
//! pivots within a block, so no guard is needed across blocks (a singular
//! diagonal block breaks the elimination as it would any block solver).

use super::num::Num;
use super::{Plan, Solved};
use crate::field::Field;
use crate::graph::Graph;
use crate::node::ExprId;
use rustc_hash::FxHashMap as HashMap;

/// One block of a block-sparse matrix: dense (`b * b` entries, row-major)
/// or diagonal (`b` entries).
#[derive(Clone, Debug)]
pub enum Block<N = ExprId> {
    Dense(Vec<N>),
    Diag(Vec<N>),
}

impl<N: Num> Block<N> {
    fn is_diag(&self) -> bool {
        matches!(self, Block::Diag(_))
    }

    /// Entry `(r, c)`, if structurally present.
    fn at(&self, r: usize, c: usize, b: usize) -> Option<N> {
        match self {
            Block::Dense(v) => Some(v[r * b + c]),
            Block::Diag(v) => (r == c).then(|| v[r]),
        }
    }

    /// The terms of entry `(r, c)` of `self * u`, appended to `ls, us`. A
    /// diagonal block against a dense one is taken dense (its zeros in the
    /// dot): the dots then fuse into one kernel, where the products of the
    /// diagonal alone would be `b * b` scalar ops, slower than the kernel.
    fn product_terms(
        &self,
        u: &Block<N>,
        r: usize,
        c: usize,
        b: usize,
        ls: &mut Vec<N>,
        us: &mut Vec<N>,
        zero: N,
    ) {
        match (self, u) {
            (Block::Dense(l), Block::Dense(u)) => {
                ls.extend_from_slice(&l[r * b..(r + 1) * b]);
                us.extend((0..b).map(|q| u[q * b + c]));
            }
            (Block::Dense(l), Block::Diag(u)) => {
                ls.extend_from_slice(&l[r * b..(r + 1) * b]);
                us.extend((0..b).map(|q| if q == c { u[c] } else { zero }));
            }
            (Block::Diag(l), Block::Dense(u)) => {
                ls.extend((0..b).map(|q| if q == r { l[r] } else { zero }));
                us.extend((0..b).map(|q| u[q * b + c]));
            }
            (Block::Diag(l), Block::Diag(u)) => {
                if r == c {
                    ls.push(l[r]);
                    us.push(u[r]);
                }
            }
        }
    }

    /// The terms of entry `r` of `self * y`, appended to `ls, ys`.
    fn apply_terms(&self, y: &[N], r: usize, b: usize, ls: &mut Vec<N>, ys: &mut Vec<N>) {
        match self {
            Block::Dense(l) => {
                ls.extend_from_slice(&l[r * b..(r + 1) * b]);
                ys.extend_from_slice(y);
            }
            Block::Diag(l) => {
                ls.push(l[r]);
                ys.push(y[r]);
            }
        }
    }

    /// `self * u` as a block: diagonal when both are.
    fn product<K: Field>(&self, g: &mut Graph<K>, u: &Block<N>, b: usize) -> Block<N> {
        if let (Block::Diag(l), Block::Diag(d)) = (self, u) {
            return Block::Diag((0..b).map(|r| N::mul(g, l[r], d[r])).collect());
        }
        let zero = N::zero(g);
        let mut out = Vec::with_capacity(b * b);
        for r in 0..b {
            for c in 0..b {
                let (mut ls, mut us) = (Vec::new(), Vec::new());
                self.product_terms(u, r, c, b, &mut ls, &mut us, zero);
                out.push(N::dot(g, ls, us));
            }
        }
        Block::Dense(out)
    }

    /// The inverse: reciprocals of a diagonal block, the dense kernel's
    /// solves against the unit vectors of a dense one.
    fn inverse<K: Field>(&self, g: &mut Graph<K>, b: usize) -> Block<N> {
        match self {
            Block::Diag(d) => Block::Diag(d.iter().map(|&v| N::recip(g, v)).collect()),
            Block::Dense(a) => {
                let cols = N::inverse_columns(g, a, b);
                let mut out = Vec::with_capacity(b * b);
                for r in 0..b {
                    for col in cols.iter().take(b) {
                        out.push(col[r]);
                    }
                }
                Block::Dense(out)
            }
        }
    }
}

/// A block-sparse matrix: one list per block row of `(block column, block)`.
pub type BlockRows<N = ExprId> = Vec<Vec<(usize, Block<N>)>>;

/// The block pattern of `m` (the input of [`super::plan`]).
pub fn block_pattern<N>(m: &BlockRows<N>) -> super::Pattern {
    m.iter()
        .map(|row| row.iter().map(|&(j, _)| j).collect())
        .collect()
}

/// `y - sum of the products in terms`, entry by entry over `b` rows.
fn subtract_terms<K: Field, N: Num>(
    g: &mut Graph<K>,
    y: &[N],
    b: usize,
    mut terms: impl FnMut(usize, &mut Vec<N>, &mut Vec<N>),
) -> Vec<N> {
    (0..b)
        .map(|r| {
            let (mut ls, mut us) = (Vec::new(), Vec::new());
            terms(r, &mut ls, &mut us);
            if ls.is_empty() {
                y[r]
            } else {
                let d = N::dot(g, ls, us);
                N::sub(g, y[r], d)
            }
        })
        .collect()
}

/// Solve `A x = rhs` for a block-sparse `A` of `b` by `b` blocks along
/// `plan` (planned over [`block_pattern`]); `rhs` has `b` entries per block
/// row, `x` comes back the same way. `fill` counts blocks.
pub fn solve_block_planned<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &BlockRows<N>,
    b: usize,
    plan: &Plan,
    rhs: &[N],
) -> Solved<N> {
    let nb = m.len();
    assert_eq!(
        rhs.len(),
        nb * b,
        "b entries of right-hand side per block row"
    );
    let btf = &plan.btf;
    let mut col_pos = vec![0usize; nb];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        col_pos[j] = k;
    }
    // Permuted block rows: (permuted block column, block).
    let rows: Vec<Vec<(usize, &Block<N>)>> = btf
        .row_perm
        .iter()
        .map(|&i| {
            let mut r: Vec<(usize, &Block<N>)> =
                m[i].iter().map(|(j, e)| (col_pos[*j], e)).collect();
            r.sort_by_key(|&(c, _)| c);
            r
        })
        .collect();
    let zero = N::zero(g);
    let mut x: Vec<Vec<N>> = vec![vec![zero; b]; nb]; // permuted block coordinates
    let mut fill = 0usize;
    for blk in (0..btf.n_blocks()).rev() {
        let range = btf.block(blk);
        let (lo, hi) = (range.start, range.end);
        let nl = hi - lo;
        let mut local: Vec<Vec<(usize, &Block<N>)>> = Vec::with_capacity(nl);
        let mut local_rhs: Vec<Vec<N>> = Vec::with_capacity(nl);
        for k in lo..hi {
            let mut row = Vec::new();
            let mut couplings: Vec<(&Block<N>, &Vec<N>)> = Vec::new();
            for &(c, e) in &rows[k] {
                if c >= hi {
                    couplings.push((e, &x[c]));
                } else if c >= lo {
                    row.push((c - lo, e));
                }
            }
            local.push(row);
            let i = btf.row_perm[k];
            let bk = &rhs[i * b..(i + 1) * b];
            // rhs_k = b_k - sum over solved blocks of A_kc x_c.
            let r = subtract_terms(g, bk, b, |r, ls, us| {
                for (e, xc) in &couplings {
                    e.apply_terms(xc, r, b, ls, us);
                }
            });
            local_rhs.push(r);
        }
        let (sol, f) = eliminate_blocks(g, &local, b, &plan.orders[blk], &local_rhs);
        fill += f;
        for (k, v) in (lo..hi).zip(sol) {
            x[k] = v;
        }
    }
    let mut out = vec![zero; nb * b];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        out[j * b..(j + 1) * b].copy_from_slice(&x[k]);
    }
    Solved {
        x: out,
        pivots_ok: g.one(),
        fill,
    }
}

/// One irreducible block of the block pattern: Crout elimination over
/// blocks in `order`, the right-hand side carried along.
fn eliminate_blocks<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &[Vec<(usize, &Block<N>)>],
    b: usize,
    order: &[usize],
    rhs: &[Vec<N>],
) -> (Vec<Vec<N>>, usize) {
    let n = m.len();
    assert_eq!(order.len(), n);
    let mut pos = vec![0usize; n];
    for (k, &j) in order.iter().enumerate() {
        pos[j] = k;
    }
    // Both rows and columns permuted by the order: the pivot blocks are the
    // diagonal of the permuted pattern (the transversal).
    let mut orig: HashMap<(usize, usize), Block<N>> = HashMap::default();
    let mut in_row: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_col: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut y: Vec<Vec<N>> = vec![Vec::new(); n];
    for (i, row) in m.iter().enumerate() {
        for &(j, e) in row {
            let (r, c) = (pos[i], pos[j]);
            if orig.insert((r, c), e.clone()).is_none() {
                in_row[r].push(c);
                in_col[c].push(r);
            }
        }
        y[pos[i]] = rhs[i].clone();
    }
    // Pending updates per block position: the (L, U) block pairs whose
    // product is subtracted, in elimination order; per right-hand side
    // row the (L, y) pairs.
    let mut pending: HashMap<(usize, usize), Vec<(Block<N>, Block<N>)>> = HashMap::default();
    let mut pending_y: Vec<Vec<(Block<N>, Vec<N>)>> = vec![Vec::new(); n];
    // The block at `at` after its pending updates: diagonal when the base
    // and every update are, dense otherwise.
    fn finalize<K: Field, N: Num>(
        g: &mut Graph<K>,
        orig: &mut HashMap<(usize, usize), Block<N>>,
        pending: &mut HashMap<(usize, usize), Vec<(Block<N>, Block<N>)>>,
        at: (usize, usize),
        b: usize,
    ) -> Block<N> {
        let base = orig.get(&at).cloned();
        let Some(updates) = pending.remove(&at) else {
            return base.expect("an occupied block position has entries");
        };
        let zero = N::zero(g);
        let diag = base.as_ref().is_none_or(|k| k.is_diag())
            && updates.iter().all(|(l, u)| l.is_diag() && u.is_diag());
        let out = if diag {
            Block::Diag(
                (0..b)
                    .map(|r| {
                        let (mut ls, mut us) = (Vec::new(), Vec::new());
                        for (l, u) in &updates {
                            l.product_terms(u, r, r, b, &mut ls, &mut us, zero);
                        }
                        let d = N::dot(g, ls, us);
                        match base.as_ref().and_then(|k| k.at(r, r, b)) {
                            Some(o) => N::sub(g, o, d),
                            None => N::neg(g, d),
                        }
                    })
                    .collect(),
            )
        } else {
            let mut out = Vec::with_capacity(b * b);
            for r in 0..b {
                for c in 0..b {
                    let (mut ls, mut us) = (Vec::new(), Vec::new());
                    for (l, u) in &updates {
                        l.product_terms(u, r, c, b, &mut ls, &mut us, zero);
                    }
                    let o = base.as_ref().and_then(|k| k.at(r, c, b));
                    out.push(match (o, ls.is_empty()) {
                        (Some(o), true) => o,
                        (None, true) => N::zero(g),
                        (Some(o), false) => {
                            let d = N::dot(g, ls, us);
                            N::sub(g, o, d)
                        }
                        (None, false) => {
                            let d = N::dot(g, ls, us);
                            N::neg(g, d)
                        }
                    });
                }
            }
            Block::Dense(out)
        };
        orig.insert(at, out.clone());
        out
    }
    let mut fill = 0usize;
    let mut upper: Vec<Vec<(usize, Block<N>)>> = vec![Vec::new(); n];
    let mut inv: Vec<Option<Block<N>>> = vec![None; n];
    for k in 0..n {
        assert!(
            orig.contains_key(&(k, k)) || pending.contains_key(&(k, k)),
            "a structurally nonzero pivot block at {k}"
        );
        let pivot = finalize(g, &mut orig, &mut pending, (k, k), b);
        let pinv = pivot.inverse(g, b);
        // The right-hand side row of this step, after its updates.
        let yk = {
            let updates = std::mem::take(&mut pending_y[k]);
            let base = std::mem::take(&mut y[k]);
            subtract_terms(g, &base, b, |r, ls, us| {
                for (l, v) in &updates {
                    l.apply_terms(v, r, b, ls, us);
                }
            })
        };
        let mut row_k: Vec<usize> = in_row[k].iter().copied().filter(|&c| c > k).collect();
        let mut col_k: Vec<usize> = in_col[k].iter().copied().filter(|&r| r > k).collect();
        row_k.sort_unstable();
        row_k.dedup();
        col_k.sort_unstable();
        col_k.dedup();
        let us: Vec<Block<N>> = row_k
            .iter()
            .map(|&j| finalize(g, &mut orig, &mut pending, (k, j), b))
            .collect();
        for (&j, u) in row_k.iter().zip(&us) {
            upper[k].push((j, u.clone()));
        }
        // L_ik = W A_kk^-1.
        let ls: Vec<Block<N>> = col_k
            .iter()
            .map(|&i| {
                let w = finalize(g, &mut orig, &mut pending, (i, k), b);
                w.product(g, &pinv, b)
            })
            .collect();
        for (&i, l) in col_k.iter().zip(&ls) {
            for (&j, u) in row_k.iter().zip(&us) {
                let entry = pending.entry((i, j)).or_default();
                if entry.is_empty() && !orig.contains_key(&(i, j)) {
                    fill += 1;
                    in_row[i].push(j);
                    in_col[j].push(i);
                }
                entry.push((l.clone(), u.clone()));
            }
            pending_y[i].push((l.clone(), yk.clone()));
        }
        y[k] = yk;
        inv[k] = Some(pinv);
    }
    // Back substitution over blocks: x_i = A_ii^-1 (y_i - sum U_ij x_j).
    let mut x: Vec<Vec<N>> = vec![Vec::new(); n];
    for i in (0..n).rev() {
        let r = subtract_terms(g, &y[i], b, |r, ls, us| {
            for (j, u) in &upper[i] {
                u.apply_terms(&x[*j], r, b, ls, us);
            }
        });
        let pinv = inv[i].as_ref().unwrap();
        x[i] = match pinv {
            Block::Diag(d) => (0..b).map(|q| N::mul(g, d[q], r[q])).collect(),
            Block::Dense(a) => (0..b)
                .map(|q| N::dot(g, a[q * b..(q + 1) * b].to_vec(), r.clone()))
                .collect(),
        };
    }
    let mut out = vec![Vec::new(); n];
    for (k, &j) in order.iter().enumerate() {
        out[j] = std::mem::take(&mut x[k]);
    }
    (out, fill)
}
