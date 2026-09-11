//! The solve of a block-sparse system as graph ops: a sparse pattern of
//! blocks (a harmonic-balance Jacobian, a multi-port coupling, the panels
//! of a supernodal factorization), eliminated block by block along the
//! block pattern's [`Plan`]. Block row `i` and block column `i` have
//! `sizes[i]` scalars; a block is dense or diagonal ([`Block`]). A dense
//! pivot block is inverted through the dense [`Graph::solve_dense`] kernel
//! (its right-hand sides fuse into one factorization), a diagonal one by
//! reciprocals, and every block update is a set of dot products over whole
//! block rows and columns, which the tape compiler fuses into `Gemm`
//! kernels. The scalar is a [`Num`]: real, or complex as pairs of real
//! expressions.
//!
//! The pivot blocks are the block pattern's transversal; the dense kernel
//! pivots within a block. Across blocks the pivot rows are static, and
//! guarded as in the scalar solve: a pivot block whose column has entries
//! below it checks that its own largest entry in that column is at least
//! [`super::PIVOT_TOLERANCE`] times the largest below ([`Solved::pivots_ok`]).

use super::num::Num;
use super::{Plan, Solved};
use crate::field::Field;
use crate::graph::Graph;
use crate::node::{CmpOp, ExprId, ReduceOp};
use rustc_hash::FxHashMap as HashMap;

/// One block of a block-sparse matrix: dense (`rows * cols` entries,
/// row-major) or diagonal (`rows` entries, a square block).
#[derive(Clone, Debug)]
pub enum Block<N = ExprId> {
    Dense(Vec<N>),
    Diag(Vec<N>),
}

/// A block-sparse matrix: one list per block row of `(block column, block)`.
pub type BlockRows<N = ExprId> = Vec<Vec<(usize, Block<N>)>>;

/// The block pattern of `m` (the input of [`super::plan`]).
pub fn block_pattern<N>(m: &BlockRows<N>) -> super::Pattern {
    m.iter()
        .map(|row| row.iter().map(|&(j, _)| j).collect())
        .collect()
}

/// A block with its shape.
#[derive(Clone, Debug)]
struct Blk<N> {
    r: usize,
    c: usize,
    d: Block<N>,
}

impl<N: Num> Blk<N> {
    fn from(block: &Block<N>, r: usize, c: usize) -> Self {
        match block {
            Block::Dense(v) => assert_eq!(v.len(), r * c, "a dense block of {r} by {c}"),
            Block::Diag(v) => {
                assert_eq!(r, c, "a diagonal block is square");
                assert_eq!(v.len(), r, "a diagonal block of {r}");
            }
        }
        Blk {
            r,
            c,
            d: block.clone(),
        }
    }

    fn is_diag(&self) -> bool {
        matches!(self.d, Block::Diag(_))
    }

    /// Entry `(r, c)`, if structurally present.
    fn at(&self, r: usize, c: usize) -> Option<N> {
        match &self.d {
            Block::Dense(v) => Some(v[r * self.c + c]),
            Block::Diag(v) => (r == c).then(|| v[r]),
        }
    }

    /// The terms of entry `(r, c)` of `self * u`, appended to `ls, us`. A
    /// diagonal block against a dense one is taken dense (its zeros in the
    /// dot): the dots then fuse into one kernel, where the products of the
    /// diagonal alone would be scalar ops, slower than the kernel.
    fn product_terms(
        &self,
        u: &Blk<N>,
        r: usize,
        c: usize,
        ls: &mut Vec<N>,
        us: &mut Vec<N>,
        zero: N,
    ) {
        let k = self.c;
        debug_assert_eq!(k, u.r, "conformable blocks");
        match (&self.d, &u.d) {
            (Block::Dense(l), Block::Dense(ud)) => {
                ls.extend_from_slice(&l[r * k..(r + 1) * k]);
                us.extend((0..k).map(|q| ud[q * u.c + c]));
            }
            (Block::Dense(l), Block::Diag(ud)) => {
                ls.extend_from_slice(&l[r * k..(r + 1) * k]);
                us.extend((0..k).map(|q| if q == c { ud[c] } else { zero }));
            }
            (Block::Diag(l), Block::Dense(ud)) => {
                ls.extend((0..k).map(|q| if q == r { l[r] } else { zero }));
                us.extend((0..k).map(|q| ud[q * u.c + c]));
            }
            (Block::Diag(l), Block::Diag(ud)) => {
                if r == c {
                    ls.push(l[r]);
                    us.push(ud[r]);
                }
            }
        }
    }

    /// The terms of entry `r` of `self * y`, appended to `ls, ys`.
    fn apply_terms(&self, y: &[N], r: usize, ls: &mut Vec<N>, ys: &mut Vec<N>) {
        debug_assert_eq!(y.len(), self.c);
        match &self.d {
            Block::Dense(l) => {
                ls.extend_from_slice(&l[r * self.c..(r + 1) * self.c]);
                ys.extend_from_slice(y);
            }
            Block::Diag(l) => {
                ls.push(l[r]);
                ys.push(y[r]);
            }
        }
    }

    /// `self * u` as a block: diagonal when both are.
    fn product<K: Field>(&self, g: &mut Graph<K>, u: &Blk<N>) -> Blk<N> {
        if let (Block::Diag(l), Block::Diag(d)) = (&self.d, &u.d) {
            let v = (0..self.r).map(|r| N::mul(g, l[r], d[r])).collect();
            return Blk {
                r: self.r,
                c: u.c,
                d: Block::Diag(v),
            };
        }
        let zero = N::zero(g);
        let mut out = Vec::with_capacity(self.r * u.c);
        for r in 0..self.r {
            for c in 0..u.c {
                let (mut ls, mut us) = (Vec::new(), Vec::new());
                self.product_terms(u, r, c, &mut ls, &mut us, zero);
                out.push(N::dot(g, ls, us));
            }
        }
        Blk {
            r: self.r,
            c: u.c,
            d: Block::Dense(out),
        }
    }

    /// The inverse of a square block: reciprocals of a diagonal one, the
    /// dense kernel's solves against the unit vectors of a dense one.
    fn inverse<K: Field>(&self, g: &mut Graph<K>) -> Blk<N> {
        let s = self.r;
        let d = match &self.d {
            Block::Diag(d) => Block::Diag(d.iter().map(|&v| N::recip(g, v)).collect()),
            Block::Dense(a) => {
                let cols = N::inverse_columns(g, a, s);
                let mut out = Vec::with_capacity(s * s);
                for r in 0..s {
                    for col in cols.iter().take(s) {
                        out.push(col[r]);
                    }
                }
                Block::Dense(out)
            }
        };
        Blk { r: s, c: s, d }
    }

    /// The largest size of column `c`'s entries.
    fn column_size<K: Field>(&self, g: &mut Graph<K>, c: usize, sizes: &mut Vec<ExprId>) {
        match &self.d {
            Block::Dense(v) => {
                for r in 0..self.r {
                    sizes.push(N::size(g, v[r * self.c + c]));
                }
            }
            Block::Diag(d) => sizes.push(N::size(g, d[c])),
        }
    }
}

/// `y - sum of the products in terms`, entry by entry.
fn subtract_terms<K: Field, N: Num>(
    g: &mut Graph<K>,
    y: &[N],
    mut terms: impl FnMut(usize, &mut Vec<N>, &mut Vec<N>),
) -> Vec<N> {
    (0..y.len())
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

fn max_of<K: Field>(g: &mut Graph<K>, mut v: Vec<ExprId>) -> ExprId {
    if v.len() == 1 {
        v.pop().unwrap()
    } else {
        g.reduce(ReduceOp::Max, v)
    }
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
    let sizes = vec![b; m.len()];
    solve_block_planned_sizes(g, m, &sizes, plan, rhs)
}

/// [`solve_block_planned`] with block row and column `i` of `sizes[i]`
/// scalars: block `(i, j)` is `sizes[i]` by `sizes[j]`, the right-hand
/// side and the solution are the blocks' entries back to back.
pub fn solve_block_planned_sizes<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &BlockRows<N>,
    sizes: &[usize],
    plan: &Plan,
    rhs: &[N],
) -> Solved<N> {
    let nb = m.len();
    assert_eq!(sizes.len(), nb, "a size per block row");
    let total: usize = sizes.iter().sum();
    assert_eq!(rhs.len(), total, "the right-hand side over every block row");
    let mut starts = Vec::with_capacity(nb + 1);
    let mut at = 0;
    for &s in sizes {
        starts.push(at);
        at += s;
    }
    starts.push(at);
    let btf = &plan.btf;
    let mut col_pos = vec![0usize; nb];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        col_pos[j] = k;
    }
    // Permuted block rows: (permuted block column, block with its shape).
    let rows: Vec<Vec<(usize, Blk<N>)>> = btf
        .row_perm
        .iter()
        .map(|&i| {
            let mut r: Vec<(usize, Blk<N>)> = m[i]
                .iter()
                .map(|(j, e)| (col_pos[*j], Blk::from(e, sizes[i], sizes[*j])))
                .collect();
            r.sort_by_key(|&(c, _)| c);
            r
        })
        .collect();
    let zero = N::zero(g);
    // Solutions per permuted block column.
    let mut x: Vec<Vec<N>> = (0..nb)
        .map(|k| vec![zero; sizes[btf.col_perm[k]]])
        .collect();
    let mut fill = 0usize;
    let mut guards: Vec<ExprId> = Vec::new();
    for blk in (0..btf.n_blocks()).rev() {
        let range = btf.block(blk);
        let (lo, hi) = (range.start, range.end);
        let nl = hi - lo;
        let mut local: Vec<Vec<(usize, &Blk<N>)>> = Vec::with_capacity(nl);
        let mut local_rhs: Vec<Vec<N>> = Vec::with_capacity(nl);
        let local_sizes: Vec<usize> = (lo..hi).map(|k| sizes[btf.row_perm[k]]).collect();
        for k in lo..hi {
            let mut row = Vec::new();
            let mut couplings: Vec<(&Blk<N>, &Vec<N>)> = Vec::new();
            for (c, e) in &rows[k] {
                if *c >= hi {
                    couplings.push((e, &x[*c]));
                } else if *c >= lo {
                    row.push((c - lo, e));
                }
            }
            local.push(row);
            let i = btf.row_perm[k];
            let bk = &rhs[starts[i]..starts[i + 1]];
            // rhs_k = b_k - sum over solved blocks of A_kc x_c.
            let r = subtract_terms(g, bk, |r, ls, us| {
                for (e, xc) in &couplings {
                    e.apply_terms(xc, r, ls, us);
                }
            });
            local_rhs.push(r);
        }
        let (sol, gs, f) = eliminate_blocks(g, &local, &local_sizes, &plan.orders[blk], &local_rhs);
        fill += f;
        guards.extend(gs);
        for (k, v) in (lo..hi).zip(sol) {
            x[k] = v;
        }
    }
    let mut out = vec![zero; total];
    for (k, &j) in btf.col_perm.iter().enumerate() {
        out[starts[j]..starts[j + 1]].copy_from_slice(&x[k]);
    }
    let pivots_ok = match guards.len() {
        0 => g.one(),
        1 => guards[0],
        _ => g.reduce(ReduceOp::Min, guards),
    };
    Solved {
        x: out,
        pivots_ok,
        fill,
    }
}

/// One irreducible block of the block pattern: Crout elimination over
/// blocks in `order`, the right-hand side carried along. Returns the
/// solution per block, the guards, and the fill in blocks.
fn eliminate_blocks<K: Field, N: Num>(
    g: &mut Graph<K>,
    m: &[Vec<(usize, &Blk<N>)>],
    sizes: &[usize],
    order: &[usize],
    rhs: &[Vec<N>],
) -> (Vec<Vec<N>>, Vec<ExprId>, usize) {
    let n = m.len();
    assert_eq!(order.len(), n);
    let mut pos = vec![0usize; n];
    for (k, &j) in order.iter().enumerate() {
        pos[j] = k;
    }
    // Both rows and columns permuted by the order: the pivot blocks are the
    // diagonal of the permuted pattern (the transversal). Sizes follow.
    let psize: Vec<usize> = (0..n).map(|k| sizes[order[k]]).collect();
    let mut orig: HashMap<(usize, usize), Blk<N>> = HashMap::default();
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
    let mut pending: HashMap<(usize, usize), Vec<(Blk<N>, Blk<N>)>> = HashMap::default();
    let mut pending_y: Vec<Vec<(Blk<N>, Vec<N>)>> = vec![Vec::new(); n];
    // The block at `at` after its pending updates: diagonal when the base
    // and every update are, dense otherwise.
    fn finalize<K: Field, N: Num>(
        g: &mut Graph<K>,
        orig: &mut HashMap<(usize, usize), Blk<N>>,
        pending: &mut HashMap<(usize, usize), Vec<(Blk<N>, Blk<N>)>>,
        at: (usize, usize),
        shape: (usize, usize),
    ) -> Blk<N> {
        let base = orig.get(&at).cloned();
        let Some(updates) = pending.remove(&at) else {
            return base.expect("an occupied block position has entries");
        };
        let (r, c) = shape;
        let zero = N::zero(g);
        let diag = r == c
            && base.as_ref().is_none_or(|k| k.is_diag())
            && updates.iter().all(|(l, u)| l.is_diag() && u.is_diag());
        let d = if diag {
            Block::Diag(
                (0..r)
                    .map(|q| {
                        let (mut ls, mut us) = (Vec::new(), Vec::new());
                        for (l, u) in &updates {
                            l.product_terms(u, q, q, &mut ls, &mut us, zero);
                        }
                        let d = N::dot(g, ls, us);
                        match base.as_ref().and_then(|k| k.at(q, q)) {
                            Some(o) => N::sub(g, o, d),
                            None => N::neg(g, d),
                        }
                    })
                    .collect(),
            )
        } else {
            let mut out = Vec::with_capacity(r * c);
            for rr in 0..r {
                for cc in 0..c {
                    let (mut ls, mut us) = (Vec::new(), Vec::new());
                    for (l, u) in &updates {
                        l.product_terms(u, rr, cc, &mut ls, &mut us, zero);
                    }
                    let o = base.as_ref().and_then(|k| k.at(rr, cc));
                    out.push(match (o, ls.is_empty()) {
                        (Some(o), true) => o,
                        (None, true) => zero,
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
        let out = Blk { r, c, d };
        orig.insert(at, out.clone());
        out
    }
    let mut fill = 0usize;
    let mut guards: Vec<ExprId> = Vec::new();
    let mut upper: Vec<Vec<(usize, Blk<N>)>> = vec![Vec::new(); n];
    let mut inv: Vec<Option<Blk<N>>> = vec![None; n];
    let tol = N::tolerance(g);
    for k in 0..n {
        assert!(
            orig.contains_key(&(k, k)) || pending.contains_key(&(k, k)),
            "a structurally nonzero pivot block at {k}"
        );
        let s = psize[k];
        let pivot = finalize(g, &mut orig, &mut pending, (k, k), (s, s));
        // The right-hand side row of this step, after its updates.
        let yk = {
            let updates = std::mem::take(&mut pending_y[k]);
            let base = std::mem::take(&mut y[k]);
            subtract_terms(g, &base, |r, ls, us| {
                for (l, v) in &updates {
                    l.apply_terms(v, r, ls, us);
                }
            })
        };
        let mut row_k: Vec<usize> = in_row[k].iter().copied().filter(|&c| c > k).collect();
        let mut col_k: Vec<usize> = in_col[k].iter().copied().filter(|&r| r > k).collect();
        row_k.sort_unstable();
        row_k.dedup();
        col_k.sort_unstable();
        col_k.dedup();
        let us: Vec<Blk<N>> = row_k
            .iter()
            .map(|&j| finalize(g, &mut orig, &mut pending, (k, j), (s, psize[j])))
            .collect();
        for (&j, u) in row_k.iter().zip(&us) {
            upper[k].push((j, u.clone()));
        }
        let ws: Vec<Blk<N>> = col_k
            .iter()
            .map(|&i| finalize(g, &mut orig, &mut pending, (i, k), (psize[i], s)))
            .collect();
        if !ws.is_empty() {
            // The guard: in every column of the panel, the pivot block's
            // largest entry against the largest below it.
            for c in 0..s {
                let mut ps = Vec::new();
                pivot.column_size(g, c, &mut ps);
                let mut below = Vec::new();
                for w in &ws {
                    w.column_size(g, c, &mut below);
                }
                let pm = max_of(g, ps);
                let bm = max_of(g, below);
                let bound = g.mul(tol, bm);
                guards.push(g.cmp(CmpOp::Ge, pm, bound));
            }
        }
        let pinv = pivot.inverse(g);
        // L_ik = W A_kk^-1.
        let ls: Vec<Blk<N>> = ws.iter().map(|w| w.product(g, &pinv)).collect();
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
        let r = subtract_terms(g, &y[i], |r, ls, us| {
            for (j, u) in &upper[i] {
                u.apply_terms(&x[*j], r, ls, us);
            }
        });
        let pinv = inv[i].as_ref().unwrap();
        let s = psize[i];
        x[i] = match &pinv.d {
            Block::Diag(d) => (0..s).map(|q| N::mul(g, d[q], r[q])).collect(),
            Block::Dense(a) => (0..s)
                .map(|q| N::dot(g, a[q * s..(q + 1) * s].to_vec(), r.clone()))
                .collect(),
        };
    }
    let mut out = vec![Vec::new(); n];
    for (k, &j) in order.iter().enumerate() {
        out[j] = std::mem::take(&mut x[k]);
    }
    (out, guards, fill)
}
