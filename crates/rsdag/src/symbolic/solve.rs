//! A linear solve as graph ops: the vertical integration of the solve.
//!
//! A simulator is segmented -- assemble, factor, solve, update -- with a data
//! hand-off between the stages and a generic sparse solver that must order,
//! allocate and dispatch at run time because it knows nothing about the
//! matrix until it sees it. Here the matrix is a graph whose sparsity pattern
//! is fixed at build time, so the elimination is a fixed sequence of ring
//! ops: a DAG like any other. [`lu_static`] builds it, [`StaticLu::solve_static`] the
//! two triangular solves, and [`newton_step`] composes residual, Jacobian,
//! factorization, solve and update into one expression per unknown, which
//! then compiles to one native function with the rest of the model.
//!
//! Measured against an outsourced multifrontal solve on a 512-unknown
//! circuit residual, the integrated step was about 30x faster: not from the
//! arithmetic, but from everything a generic solver does per call and this
//! form does once, at build time.
//!
//! What it is not: a replacement for a sparse solver on a large system. Every
//! flop of the factorization is a node, so the program grows with the
//! factorization's cost; a fill-reducing [`ordering`] keeps that near the
//! pattern for the structures a circuit or a mesh produces, but a hundred
//! thousand unknowns with heavy fill are a program of millions of nodes.
//! And the pivot order is fixed at build time -- static pivoting -- so a
//! system whose pivots need reordering at run time is not a candidate. The
//! intended place is the small and mid-sized solve: a block's implicit
//! update, a device's internal nodes, a subsystem of a few thousand
//! unknowns, with the large sparse top level left to a solver built for it.

use rustc_hash::FxHashMap as HashMap;

use crate::field::Field;
use crate::graph::Graph;
use crate::node::ExprId;

/// The structural pattern of a square matrix: `pattern[i][j]` is whether
/// entry `(i, j)` is present.
pub type Pattern = Vec<Vec<bool>>;

/// A static LU factorization as graph ops, in a given elimination order.
pub struct StaticLu {
    /// The elimination order: `order[k]` is the original index eliminated
    /// at step `k`.
    pub order: Vec<usize>,
    /// Unit-lower multipliers and upper entries, keyed by *permuted*
    /// position `(k, l)` (step indices). Only the structurally nonzero ones.
    pub factors: HashMap<(usize, usize), ExprId>,
    /// Structural nonzeros created by the elimination beyond the pattern.
    pub fill: usize,
    n: usize,
}

/// A fill-reducing elimination order over a pattern: minimum degree, the
/// classic greedy that eliminates the vertex of fewest neighbours next and
/// adds the fill its elimination creates.
///
/// Symmetric in the structure (it works on `A + A^T`), which is what fill
/// depends on. Quadratic in the size, which is fine for the sizes this module
/// is for, and it runs once per pattern at build time.
pub fn ordering(pattern: &Pattern) -> Vec<usize> {
    let n = pattern.len();
    let mut adj: Vec<std::collections::BTreeSet<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i && (pattern[i][j] || pattern[j][i]))
                .collect()
        })
        .collect();
    let mut alive: Vec<bool> = vec![true; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        // Fewest live neighbours; ties by index so the order is stable.
        let k = (0..n)
            .filter(|&i| alive[i])
            .min_by_key(|&i| (adj[i].iter().filter(|&&j| alive[j]).count(), i))
            .expect("a live vertex remains");
        alive[k] = false;
        order.push(k);
        // Eliminating k connects its neighbours pairwise: that is the fill.
        let nb: Vec<usize> = adj[k].iter().copied().filter(|&j| alive[j]).collect();
        for &a in &nb {
            for &b in &nb {
                if a != b {
                    adj[a].insert(b);
                }
            }
        }
    }
    order
}

/// The pattern of a matrix of expressions: an entry is present when it is
/// not the structural zero.
pub fn pattern_of<K: Field>(g: &Graph<K>, m: &[Vec<ExprId>]) -> Pattern {
    m.iter()
        .map(|row| row.iter().map(|&e| !g.is_zero(e)).collect())
        .collect()
}

/// Static-pivot LU of a matrix of expressions in the given elimination
/// order, as graph ops.
///
/// At step `k` the pivot is `A[order[k]][order[k]]`; every later row with a
/// nonzero in the pivot column gets its multiplier and its update. Values
/// flow through the pattern's op sequence; a pivot that is numerically zero
/// at evaluation time gives an infinite multiplier, exactly as an
/// unpivoted elimination would.
pub fn lu_static<K: Field>(g: &mut Graph<K>, m: &[Vec<ExprId>], order: &[usize]) -> StaticLu {
    let n = m.len();
    assert_eq!(order.len(), n, "one order entry per unknown");
    // Position of an original index in the order.
    let mut pos = vec![0usize; n];
    for (k, &i) in order.iter().enumerate() {
        pos[i] = k;
    }
    // The working matrix in permuted coordinates, structurally nonzero only.
    let mut a: HashMap<(usize, usize), ExprId> = HashMap::default();
    for (i, row) in m.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            if !g.is_zero(e) {
                a.insert((pos[i], pos[j]), e);
            }
        }
    }
    // Row and column occupancy, so a step touches only what is nonzero.
    let mut in_row: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_col: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(r, c) in a.keys() {
        in_row[r].push(c);
        in_col[c].push(r);
    }
    let mut fill = 0usize;
    for k in 0..n {
        let pivot = a[&(k, k)];
        let inv = g.recip(pivot);
        let row_k: Vec<usize> = in_row[k].iter().copied().filter(|&c| c > k).collect();
        let col_k: Vec<usize> = in_col[k].iter().copied().filter(|&r| r > k).collect();
        for &i in &col_k {
            let aik = a[&(i, k)];
            let l = g.mul(aik, inv);
            a.insert((i, k), l);
            for &j in &row_k {
                let akj = a[&(k, j)];
                let upd = g.mul(l, akj);
                let v = match a.get(&(i, j)) {
                    Some(&aij) => g.sub(aij, upd),
                    None => {
                        fill += 1;
                        in_row[i].push(j);
                        in_col[j].push(i);
                        g.neg(upd)
                    }
                };
                a.insert((i, j), v);
            }
        }
    }
    StaticLu {
        order: order.to_vec(),
        factors: a,
        fill,
        n,
    }
}

impl StaticLu {
    /// Solve `A x = b` with the factors: forward through the unit lower
    /// part, backward through the upper, then undo the permutation. `b` is
    /// in original coordinates and so is the result.
    pub fn solve_static<K: Field>(&self, g: &mut Graph<K>, b: &[ExprId]) -> Vec<ExprId> {
        let n = self.n;
        assert_eq!(b.len(), n);
        let mut lower: Vec<Vec<(usize, ExprId)>> = vec![Vec::new(); n];
        let mut upper: Vec<Vec<(usize, ExprId)>> = vec![Vec::new(); n];
        for (&(i, j), &e) in &self.factors {
            if j < i {
                lower[i].push((j, e));
            } else if j > i {
                upper[i].push((j, e));
            }
        }
        // A fixed operand order keeps the program the same across builds.
        for row in lower.iter_mut().chain(upper.iter_mut()) {
            row.sort_by_key(|&(j, _)| j);
        }
        let mut y: Vec<ExprId> = (0..n).map(|k| b[self.order[k]]).collect();
        for i in 0..n {
            let mut acc = y[i];
            for &(j, l) in &lower[i] {
                let t = g.mul(l, y[j]);
                acc = g.sub(acc, t);
            }
            y[i] = acc;
        }
        let mut x = vec![g.zero(); n];
        for i in (0..n).rev() {
            let mut acc = y[i];
            for &(j, u) in &upper[i] {
                let t = g.mul(u, x[j]);
                acc = g.sub(acc, t);
            }
            x[i] = g.div(acc, self.factors[&(i, i)]);
        }
        // Back to original coordinates: unknown `order[k]` is `x[k]`.
        let mut out = vec![g.zero(); n];
        for (k, &i) in self.order.iter().enumerate() {
            out[i] = x[k];
        }
        out
    }
}

/// One Newton step as expressions: `x - J^-1 F`, with `J` the Jacobian of
/// `f` with respect to `x`, factored in a fill-reducing order. Returns the
/// updated unknowns, one expression each, and the factorization's fill.
///
/// The result is an ordinary expression over the graph: it differentiates,
/// specializes, compiles and batches like the model it came from, and a
/// consumer that wants damping or a line search composes it from the same
/// pieces ([`lu_static`], [`StaticLu::solve_static`]).
pub fn newton_step<K: Field>(
    g: &mut Graph<K>,
    f: &[ExprId],
    x: &[crate::node::SymbolId],
) -> (Vec<ExprId>, usize) {
    let jac = crate::autodiff::jacobian(g, f, x);
    let pattern = pattern_of(g, &jac);
    let order = ordering(&pattern);
    let lu = lu_static(g, &jac, &order);
    let dx = lu.solve_static(g, f);
    let out = x
        .iter()
        .zip(&dx)
        .map(|(&s, &d)| {
            let xe = g.symbol_expr(s);
            g.sub(xe, d)
        })
        .collect();
    (out, lu.fill)
}
