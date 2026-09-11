//! The solve of a block-sparse system as graph ops: a sparse pattern of
//! dense `b` by `b` blocks (a harmonic-balance Jacobian, a multi-port
//! coupling), eliminated block by block along the block pattern's
//! [`Plan`]. A pivot block is applied through the dense [`Graph::solve_dense`]
//! kernel (its `b` right-hand sides fuse into one factorization), and every
//! block update is a set of dot products over whole block rows and columns,
//! which the tape compiler fuses into `Gemm` kernels. The program is a few
//! kernel ops per block plus one subtraction per entry, and its flops run at
//! the kernels' rate.
//!
//! The pivot blocks are the block pattern's transversal; the dense kernel
//! pivots within a block, so no guard is needed across blocks (a singular
//! diagonal block breaks the elimination as it would any block solver).

use super::{Plan, Solved};
use crate::field::Field;
use crate::graph::Graph;
use crate::node::ExprId;
use rustc_hash::FxHashMap as HashMap;

/// A block-sparse matrix: one list per block row of `(block column,
/// entries)`, the entries a row-major `b * b` list of expressions.
pub type BlockRows = Vec<Vec<(usize, Vec<ExprId>)>>;

/// The block pattern of `m` (the input of [`super::plan`]).
pub fn block_pattern(m: &BlockRows) -> super::Pattern {
    m.iter()
        .map(|row| row.iter().map(|&(j, _)| j).collect())
        .collect()
}

/// Solve `A x = rhs` for a block-sparse `A` of `b` by `b` blocks along
/// `plan` (planned over [`block_pattern`]); `rhs` has `b` entries per block
/// row, `x` comes back the same way. `fill` counts blocks.
pub fn solve_block_planned<K: Field>(
    g: &mut Graph<K>,
    m: &BlockRows,
    b: usize,
    plan: &Plan,
    rhs: &[ExprId],
) -> Solved {
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
    // Permuted block rows: (permuted block column, entries).
    let rows: Vec<Vec<(usize, &[ExprId])>> = btf
        .row_perm
        .iter()
        .map(|&i| {
            let mut r: Vec<(usize, &[ExprId])> = m[i]
                .iter()
                .map(|(j, e)| (col_pos[*j], e.as_slice()))
                .collect();
            r.sort_by_key(|&(c, _)| c);
            r
        })
        .collect();
    let zero = g.zero();
    let mut x: Vec<Vec<ExprId>> = vec![vec![zero; b]; nb]; // permuted block coordinates
    let mut fill = 0usize;
    for blk in (0..btf.n_blocks()).rev() {
        let range = btf.block(blk);
        let (lo, hi) = (range.start, range.end);
        let nl = hi - lo;
        let mut local: Vec<Vec<(usize, &[ExprId])>> = Vec::with_capacity(nl);
        let mut local_rhs: Vec<Vec<ExprId>> = Vec::with_capacity(nl);
        for k in lo..hi {
            let mut row = Vec::new();
            let mut couplings: Vec<(&[ExprId], &Vec<ExprId>)> = Vec::new();
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
            // rhs_k = b_k - sum over solved blocks of A_kc x_c, one dot per
            // entry over the concatenated coupling rows and solutions.
            let r: Vec<ExprId> = (0..b)
                .map(|r| {
                    if couplings.is_empty() {
                        return bk[r];
                    }
                    let mut ls = Vec::with_capacity(couplings.len() * b);
                    let mut xs = Vec::with_capacity(couplings.len() * b);
                    for (e, xc) in &couplings {
                        ls.extend_from_slice(&e[r * b..(r + 1) * b]);
                        xs.extend_from_slice(xc);
                    }
                    let d = g.dot(ls, xs);
                    g.sub(bk[r], d)
                })
                .collect();
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
/// blocks in `order`, the right-hand side as one more block column.
fn eliminate_blocks<K: Field>(
    g: &mut Graph<K>,
    m: &[Vec<(usize, &[ExprId])>],
    b: usize,
    order: &[usize],
    rhs: &[Vec<ExprId>],
) -> (Vec<Vec<ExprId>>, usize) {
    let n = m.len();
    assert_eq!(order.len(), n);
    let mut pos = vec![0usize; n];
    for (k, &j) in order.iter().enumerate() {
        pos[j] = k;
    }
    // Both rows and columns permuted by the order: the pivot blocks are the
    // diagonal of the permuted pattern (the transversal).
    let mut orig: HashMap<(usize, usize), Vec<ExprId>> = HashMap::default();
    let mut in_row: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_col: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (i, row) in m.iter().enumerate() {
        for &(j, e) in row {
            let (r, c) = (pos[i], pos[j]);
            if orig.insert((r, c), e.to_vec()).is_none() {
                in_row[r].push(c);
                in_col[c].push(r);
            }
        }
        orig.insert((pos[i], n), rhs[i].clone());
        in_row[pos[i]].push(n);
    }
    // Pending updates per block position: the (L, U) block pairs whose
    // product is subtracted, in elimination order.
    let mut pending: HashMap<(usize, usize), Vec<(Vec<ExprId>, Vec<ExprId>)>> = HashMap::default();
    let width = |c: usize| if c == n { 1 } else { b };
    // The block at `at` after its pending updates: entry (r, c) minus the
    // dot of row r of the L blocks with column c of the U blocks.
    fn finalize<K: Field>(
        g: &mut Graph<K>,
        orig: &mut HashMap<(usize, usize), Vec<ExprId>>,
        pending: &mut HashMap<(usize, usize), Vec<(Vec<ExprId>, Vec<ExprId>)>>,
        at: (usize, usize),
        b: usize,
        w: usize,
    ) -> Vec<ExprId> {
        let base = orig.get(&at).cloned();
        let Some(updates) = pending.remove(&at) else {
            return base.expect("an occupied block position has entries");
        };
        let mut out = Vec::with_capacity(b * w);
        for r in 0..b {
            for c in 0..w {
                let mut ls = Vec::with_capacity(updates.len() * b);
                let mut us = Vec::with_capacity(updates.len() * b);
                for (l, u) in &updates {
                    ls.extend_from_slice(&l[r * b..(r + 1) * b]);
                    us.extend((0..b).map(|q| u[q * w + c]));
                }
                let d = g.dot(ls, us);
                out.push(match &base {
                    Some(o) => g.sub(o[r * w + c], d),
                    None => g.neg(d),
                });
            }
        }
        orig.insert(at, out.clone());
        out
    }
    let mut fill = 0usize;
    let mut upper: Vec<Vec<(usize, Vec<ExprId>)>> = vec![Vec::new(); n];
    let mut diag: Vec<Vec<ExprId>> = vec![Vec::new(); n];
    for k in 0..n {
        assert!(
            orig.contains_key(&(k, k)) || pending.contains_key(&(k, k)),
            "a structurally nonzero pivot block at {k}"
        );
        let pivot = finalize(g, &mut orig, &mut pending, (k, k), b, b);
        // The pivot block transposed, for `L = W A^-1` as `A^T L^T = W^T`.
        let pivot_t: Vec<ExprId> = (0..b * b).map(|q| pivot[(q % b) * b + q / b]).collect();
        diag[k] = pivot;
        let mut row_k: Vec<usize> = in_row[k].iter().copied().filter(|&c| c > k).collect();
        let mut col_k: Vec<usize> = in_col[k].iter().copied().filter(|&r| r > k).collect();
        row_k.sort_unstable();
        row_k.dedup();
        col_k.sort_unstable();
        col_k.dedup();
        let us: Vec<Vec<ExprId>> = row_k
            .iter()
            .map(|&j| finalize(g, &mut orig, &mut pending, (k, j), b, width(j)))
            .collect();
        for (&j, u) in row_k.iter().zip(&us) {
            if j < n {
                upper[k].push((j, u.clone()));
            }
        }
        let ls: Vec<Vec<ExprId>> = col_k
            .iter()
            .map(|&i| {
                let w = finalize(g, &mut orig, &mut pending, (i, k), b, b);
                // L_ik = W A_kk^-1: row r of L is the solve of A_kk^T against
                // row r of W; the b solves share the matrix and fuse.
                let mut l = Vec::with_capacity(b * b);
                for r in 0..b {
                    let row = w[r * b..(r + 1) * b].to_vec();
                    l.extend(g.solve_dense(pivot_t.clone(), row));
                }
                l
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
        }
    }
    // Back substitution over blocks: x_i = A_ii^-1 (y_i - sum U_ij x_j).
    let mut x: Vec<Vec<ExprId>> = vec![Vec::new(); n];
    for i in (0..n).rev() {
        let y = finalize(g, &mut orig, &mut pending, (i, n), b, 1);
        let r: Vec<ExprId> = (0..b)
            .map(|r| {
                if upper[i].is_empty() {
                    return y[r];
                }
                let mut us = Vec::with_capacity(upper[i].len() * b);
                let mut xs = Vec::with_capacity(upper[i].len() * b);
                for (j, u) in &upper[i] {
                    us.extend_from_slice(&u[r * b..(r + 1) * b]);
                    xs.extend_from_slice(&x[*j]);
                }
                let d = g.dot(us, xs);
                g.sub(y[r], d)
            })
            .collect();
        x[i] = g.solve_dense(diag[i].clone(), r);
    }
    let mut out = vec![Vec::new(); n];
    for (k, &j) in order.iter().enumerate() {
        out[j] = std::mem::take(&mut x[k]);
    }
    (out, fill)
}
