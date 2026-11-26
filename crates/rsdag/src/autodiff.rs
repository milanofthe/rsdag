//! Symbolic differentiation over the DAG.
//!
//! Builds the derivative as new nodes in the same context, so common
//! subexpressions are shared (hash-consing) and the result can be exported or
//! differentiated again. Used to produce analytic Jacobians for DAE export.

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::func::{FuncId, OutputId};
use crate::graph::{leading_operands_without_derivative, Graph, Join, NodeTable, Set, Through};
use crate::node::{ArgList, BinOp, CmpOp, ExprId, Node, ReduceOp, SymbolId, UnaryOp};

/// Derivative of `expr` with respect to the symbol `wrt`.
pub fn differentiate<K: Field>(ctx: &mut Graph<K>, expr: ExprId, wrt: SymbolId) -> ExprId {
    let _s = rsprobe::scope!("differentiate");
    let mut memo = NodeTable::new(ctx.len());
    let mut calls = CallOperands::new(&|_| true);
    forward(ctx, &[expr], wrt, &mut memo, &|_| true, &mut calls)[0]
}

fn unary_factor<K: Field>(ctx: &mut Graph<K>, op: UnaryOp, a: ExprId) -> ExprId {
    match op {
        UnaryOp::Exp => {
            let limit = ctx.konst_f64(crate::semantics::EXP_LIMIT);
            let above_limit = ctx.cmp(CmpOp::Gt, a, limit);
            let primal_exp = ctx.exp(a);
            let tail_slope = ctx.konst_f64(crate::semantics::EXP_AT_LIMIT);
            ctx.select(above_limit, tail_slope, primal_exp)
        }
        UnaryOp::Ln => {
            let floor = ctx.konst_f64(crate::semantics::LN_FLOOR);
            let clamped_flat = ctx.cmp(CmpOp::Le, a, floor);
            let inv_a = ctx.recip(a);
            let zero = ctx.zero();
            ctx.select(clamped_flat, zero, inv_a)
        }
        UnaryOp::Sqrt => {
            let s = ctx.sqrt(a);
            let rs = ctx.recip(s);
            let half = ctx.ratio(1, 2);
            let d = ctx.mul(half, rs);
            let zero = ctx.zero();
            let clamped_to_zero = ctx.cmp(CmpOp::Le, a, zero);
            ctx.select(clamped_to_zero, zero, d)
        }
        UnaryOp::Sin => ctx.cos(a),
        UnaryOp::Cos => {
            let s = ctx.sin(a);
            ctx.neg(s)
        }
        UnaryOp::Floor => ctx.zero(),
        UnaryOp::Sinh => ctx.cosh(a),
        UnaryOp::Cosh => ctx.sinh(a),
        UnaryOp::Tanh => {
            let t = ctx.tanh(a);
            let t2 = ctx.mul(t, t);
            let one = ctx.one();
            ctx.sub(one, t2)
        }
        UnaryOp::Atan => {
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let denom = ctx.add(one, a2);
            ctx.recip(denom)
        }
        UnaryOp::Tan => {
            let t = ctx.unary(UnaryOp::Tan, a);
            let t2 = ctx.mul(t, t);
            let one = ctx.one();
            ctx.add(one, t2)
        }
        UnaryOp::Log10 => {
            let r = ctx.recip(a);
            let c = ctx.konst_f64(1.0 / std::f64::consts::LN_10);
            ctx.mul(c, r)
        }
        UnaryOp::Log2 => {
            let r = ctx.recip(a);
            let c = ctx.konst_f64(1.0 / std::f64::consts::LN_2);
            ctx.mul(c, r)
        }
        UnaryOp::Log1p => {
            let one = ctx.one();
            let d = ctx.add(one, a);
            ctx.recip(d)
        }
        UnaryOp::Expm1 => {
            let limit = ctx.konst_f64(crate::semantics::EXP_LIMIT);
            let above_limit = ctx.cmp(CmpOp::Gt, a, limit);
            let primal_expm1 = ctx.unary(UnaryOp::Expm1, a);
            let one = ctx.one();
            let unguarded_exp = ctx.add(primal_expm1, one);
            let guarded_exp = ctx.exp(a);
            ctx.select(above_limit, unguarded_exp, guarded_exp)
        }
        UnaryOp::Cbrt => {
            let c = ctx.unary(UnaryOp::Cbrt, a);
            let c2 = ctx.mul(c, c);
            let degree = ctx.konst_int(crate::constants::CBRT_DEGREE);
            let d = ctx.mul(degree, c2);
            ctx.recip(d)
        }
        UnaryOp::Abs => ctx.unary(UnaryOp::Sign, a),
        UnaryOp::Sign | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Trunc | UnaryOp::RandUniform => {
            ctx.zero()
        }
        UnaryOp::Asin | UnaryOp::Acos => {
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let d = ctx.sub(one, a2);
            let s = ctx.sqrt(d);
            let r = ctx.recip(s);
            if op == UnaryOp::Asin {
                r
            } else {
                ctx.neg(r)
            }
        }
        UnaryOp::Asinh => {
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let d = ctx.add(one, a2);
            let s = ctx.sqrt(d);
            ctx.recip(s)
        }
        UnaryOp::Acosh => {
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let d = ctx.sub(a2, one);
            let s = ctx.sqrt(d);
            ctx.recip(s)
        }
        UnaryOp::Atanh => {
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let d = ctx.sub(one, a2);
            ctx.recip(d)
        }
        UnaryOp::Erf | UnaryOp::Erfc => {
            let a2 = ctx.mul(a, a);
            let na2 = ctx.neg(a2);
            let e = ctx.exp(na2);
            let c = ctx.konst_f64(std::f64::consts::FRAC_2_SQRT_PI);
            let d = ctx.mul(c, e);
            if op == UnaryOp::Erf {
                d
            } else {
                ctx.neg(d)
            }
        }
        UnaryOp::Lgamma => ctx.unary(UnaryOp::Digamma, a),
        UnaryOp::Tgamma => {
            let g = ctx.unary(UnaryOp::Tgamma, a);
            let p = ctx.unary(UnaryOp::Digamma, a);
            ctx.mul(g, p)
        }
        UnaryOp::Digamma => ctx.unary(UnaryOp::Trigamma, a),
        UnaryOp::Trigamma => {
            panic!("rsdag: the derivative of trigamma (polygamma of order 2) is not available")
        }
    }
}

fn binary_partials<K: Field>(
    ctx: &mut Graph<K>,
    op: BinOp,
    a: ExprId,
    b: ExprId,
    want: [bool; 2],
) -> [ExprId; 2] {
    let zero = ctx.zero();
    match op {
        BinOp::Powf => {
            let da = if want[0] {
                let one = ctx.one();
                let bm1 = ctx.sub(b, one);
                let p = ctx.binary(BinOp::Powf, a, bm1);
                ctx.mul(b, p)
            } else {
                zero
            };
            let db = if want[1] {
                let ab = ctx.binary(BinOp::Powf, a, b);
                let ln = ctx.ln(a);
                ctx.mul(ab, ln)
            } else {
                zero
            };
            [da, db]
        }
        BinOp::Mod => {
            let db = if want[1] {
                let q = ctx.div(a, b);
                let t = ctx.unary(UnaryOp::Trunc, q);
                ctx.neg(t)
            } else {
                zero
            };
            [ctx.one(), db]
        }
        BinOp::Atan2 => {
            let a2 = ctx.mul(a, a);
            let b2 = ctx.mul(b, b);
            let s = ctx.add(a2, b2);
            let r = ctx.recip(s);
            let da = if want[0] { ctx.mul(b, r) } else { zero };
            let db = if want[1] {
                let ar = ctx.mul(a, r);
                ctx.neg(ar)
            } else {
                zero
            };
            [da, db]
        }
        BinOp::Hypot => {
            let h = ctx.binary(BinOp::Hypot, a, b);
            let r = ctx.recip(h);
            let da = if want[0] { ctx.mul(a, r) } else { zero };
            let db = if want[1] { ctx.mul(b, r) } else { zero };
            [da, db]
        }
        BinOp::RandStream => [zero, zero],
    }
}

fn cofactors<K: Field>(
    ctx: &mut Graph<K>,
    args: &[ExprId],
    want: impl Fn(usize) -> bool,
) -> Vec<Option<ExprId>> {
    let mut prefix = Vec::with_capacity(args.len());
    let mut acc = ctx.one();
    for &x in args {
        prefix.push(acc);
        acc = ctx.mul(acc, x);
    }
    let mut out = vec![None; args.len()];
    let mut suffix = ctx.one();
    for i in (0..args.len()).rev() {
        if want(i) {
            out[i] = Some(ctx.mul(prefix[i], suffix));
        }
        suffix = ctx.mul(suffix, args[i]);
    }
    out
}

pub(crate) struct CallOperands<'a> {
    live: &'a dyn Fn(ExprId) -> bool,
    live_by_call: HashMap<ExprId, Box<[(u32, ExprId)]>>,
}

impl<'a> CallOperands<'a> {
    pub(crate) fn new(live: &'a dyn Fn(ExprId) -> bool) -> Self {
        CallOperands {
            live,
            live_by_call: HashMap::default(),
        }
    }

    fn live_operands<K: Field>(
        &mut self,
        ctx: &Graph<K>,
        e: ExprId,
        o: OutputId,
        l: ArgList,
    ) -> &[(u32, ExprId)] {
        let live = self.live;
        self.live_by_call.entry(e).or_insert_with(|| {
            let (f, k) = ctx.output(o);
            (ctx.output_support(f, k).iter())
                .map(|&p| (p, ctx.call_operand(o, l, p)))
                .filter(|&(_, c)| live(c))
                .collect()
        })
    }
}

pub(crate) fn carrying<K: Field>(ctx: &Graph<K>, e: ExprId, out: &mut Vec<ExprId>) {
    out.clear();
    match *ctx.node(e) {
        Node::Call(o, l) => {
            let (f, k) = ctx.output(o);
            out.extend(
                ctx.output_support(f, k)
                    .iter()
                    .map(|&p| ctx.call_operand(o, l, p)),
            );
        }
        ref node => {
            out.extend_from_slice(&ctx.operands(e)[leading_operands_without_derivative(node)..])
        }
    }
}

fn topological_cone<K: Field>(
    ctx: &Graph<K>,
    root: ExprId,
    seen: &mut NodeTable<u32>,
) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    let mut ops = Vec::new();
    while let Some(e) = stack.pop() {
        if !seen.contains(e) {
            seen.set(e, 0);
            out.push(e);
            carrying(ctx, e, &mut ops);
            debug_assert!(ops.iter().all(|&c| c < e));
            stack.extend_from_slice(&ops);
        }
    }
    out.sort_unstable();
    out
}

pub(crate) fn forward<K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    wrt: SymbolId,
    memo: &mut NodeTable<ExprId>,
    moves: &dyn Fn(ExprId) -> bool,
    calls: &mut CallOperands,
) -> Vec<ExprId> {
    let zero = ctx.zero();
    let mut stack: Vec<(ExprId, bool)> =
        Vec::with_capacity(crate::constants::FORWARD_STACK_CAPACITY);
    stack.extend(roots.iter().rev().map(|&r| (r, false)));
    let mut ops = Vec::new();
    while let Some((e, expanded)) = stack.pop() {
        let differentiated_already = memo.get(e).is_some();
        if differentiated_already {
            continue;
        }
        if expanded {
            let d = tangent(ctx, e, wrt, memo, calls);
            memo.set(e, d);
        } else {
            stack.push((e, true));
            match *ctx.node(e) {
                Node::Call(o, l) => {
                    ops.clear();
                    ops.extend(calls.live_operands(ctx, e, o, l).iter().map(|&(_, c)| c));
                }
                _ => carrying(ctx, e, &mut ops),
            }
            for &c in ops.iter().rev() {
                if memo.get(c).is_none() {
                    if moves(c) {
                        stack.push((c, false));
                    } else {
                        memo.set(c, zero);
                    }
                }
            }
        }
    }
    roots
        .iter()
        .map(|&r| memo.get(r).expect("root differentiated"))
        .collect()
}

fn tangent<K: Field>(
    ctx: &mut Graph<K>,
    e: ExprId,
    wrt: SymbolId,
    memo: &mut NodeTable<ExprId>,
    calls: &mut CallOperands,
) -> ExprId {
    let zero = ctx.zero();
    let node = *ctx.node(e);
    if let Node::Symbol(s) = node {
        return if s == wrt { ctx.one() } else { zero };
    }
    let d = |c: ExprId| memo.get(c).expect("operand differentiated");
    let no_operand_moves = match node {
        Node::Call(o, l) => calls
            .live_operands(ctx, e, o, l)
            .iter()
            .all(|&(_, c)| d(c) == zero),
        _ => ctx.operands(e)[leading_operands_without_derivative(&node)..]
            .iter()
            .all(|&c| d(c) == zero),
    };
    if no_operand_moves {
        return zero;
    }
    match node {
        Node::Const(_) | Node::Symbol(_) | Node::Cmp(..) => unreachable!("no moving operand"),
        Node::Add(a, b) => ctx.add(d(a), d(b)),
        Node::Mul(a, b) => {
            let t1 = ctx.mul(d(a), b);
            let t2 = ctx.mul(a, d(b));
            ctx.add(t1, t2)
        }
        Node::Neg(a) => ctx.neg(d(a)),
        Node::Pow(a, n) => {
            let coeff = ctx.konst_int(n);
            let p = ctx.pow_i(a, n - 1);
            let cp = ctx.mul(coeff, p);
            ctx.mul(cp, d(a))
        }
        Node::Unary(op, a) => {
            let factor = unary_factor(ctx, op, a);
            ctx.mul(factor, d(a))
        }
        Node::Binary(op, a, b) => {
            let (da, db) = (d(a), d(b));
            let [pa, pb] = binary_partials(ctx, op, a, b, [da != zero, db != zero]);
            let t1 = ctx.mul(pa, da);
            let t2 = ctx.mul(pb, db);
            ctx.add(t1, t2)
        }
        Node::Select(c, t, f) => ctx.select(c, d(t), d(f)),
        Node::Reduce(op, l) => match op {
            ReduceOp::Sum => {
                let dargs: Vec<ExprId> = ctx.args(l).iter().map(|&a| d(a)).collect();
                ctx.reduce(ReduceOp::Sum, dargs)
            }
            ReduceOp::Product => {
                let args = ctx.args(l).to_vec();
                let others = cofactors(ctx, &args, |i| d(args[i]) != zero);
                let mut terms = Vec::with_capacity(args.len());
                for (&a, other) in args.iter().zip(others) {
                    if let Some(other) = other {
                        terms.push(ctx.mul(d(a), other));
                    }
                }
                ctx.reduce(ReduceOp::Sum, terms)
            }
            ReduceOp::Min | ReduceOp::Max => {
                let args = ctx.args(l).to_vec();
                let strictly_more_extreme = if op == ReduceOp::Max {
                    CmpOp::Gt
                } else {
                    CmpOp::Lt
                };
                let mut m = args[0];
                let mut dm = d(args[0]);
                for &a in &args[1..] {
                    let a_replaces_extremum = ctx.cmp(strictly_more_extreme, a, m);
                    dm = ctx.select(a_replaces_extremum, d(a), dm);
                    m = ctx.reduce(op, vec![m, a]);
                }
                dm
            }
        },
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let mut terms = Vec::with_capacity(a.len());
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let t1 = ctx.mul(d(ai), bi);
                let t2 = ctx.mul(ai, d(bi));
                terms.push(ctx.add(t1, t2));
            }
            ctx.reduce(ReduceOp::Sum, terms)
        }
        Node::Solve(l, i) => {
            let (n, a, b) = ctx.solve_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let x = ctx.solve_dense(a.clone(), b.clone());
            let mut rhs = Vec::with_capacity(n);
            for r in 0..n {
                let da: Vec<ExprId> = a[r * n..(r + 1) * n].iter().map(|&c| d(c)).collect();
                let dax = ctx.dot(da, x.clone());
                rhs.push(ctx.sub(d(b[r]), dax));
            }
            let dx = ctx.solve_dense(a, rhs);
            for (&xk, &dk) in x.iter().zip(&dx) {
                memo.set(xk, dk);
            }
            dx[i as usize]
        }
        Node::Call(o, l) => {
            let (f, out) = ctx.output(o);
            let moving: Vec<(u32, ExprId)> = (calls.live_operands(ctx, e, o, l).iter())
                .map(|&(i, c)| (i, d(c)))
                .filter(|&(_, da)| da != zero)
                .collect();
            let params: Vec<u32> = moving.iter().map(|&(i, _)| i).collect();
            let partial_outputs = ctx.derivative_outputs(f, out, &params);
            let mut acc = zero;
            for (&(_, dai), k) in moving.iter().zip(partial_outputs) {
                let partial = ctx.call_list_in(f, k, ctx.context_of(o), l);
                let term = ctx.mul(partial, dai);
                acc = ctx.add(acc, term);
            }
            acc
        }
    }
}

/// A sparse matrix of expressions: for each row, the `(column, entry)`
/// pairs that are not the structural zero, in ascending column order.
pub type SparseRows = Vec<Vec<(usize, ExprId)>>;

pub use crate::constants::REVERSE_MIN_TOUCHED;

/// Sparse Jacobian of `residuals` with respect to `wrt`: row `i` lists the
/// nonzero `(column, d residuals[i] / d wrt[column])`, by column.
///
/// Only the symbols a row reads through operands that carry a derivative
/// can have a nonzero one: through a call, the ones the called output reads
/// (its output support).
/// A row that touches few unknowns is differentiated forward, one sweep
/// per unknown, and a sweep is shared by every such row that touches that
/// unknown, so a subexpression common to several rows (a device current
/// in two node equations) is differentiated once per unknown, not once per
/// row. A row that touches [`REVERSE_MIN_TOUCHED`] unknowns or more (a
/// scalar output over a deep shared graph, a node many devices meet at)
/// takes one reverse sweep instead: its cost is the row's graph once, not
/// once per unknown.
pub fn sparse_jacobian<K: Field>(
    ctx: &mut Graph<K>,
    residuals: &[ExprId],
    wrt: &[SymbolId],
) -> SparseRows {
    let _s = rsprobe::scope!("sparse jacobian");
    let nodes = ctx.len();
    let col: rustc_hash::FxHashMap<SymbolId, u32> = wrt
        .iter()
        .enumerate()
        .map(|(j, &s)| (s, j as u32))
        .collect();
    let flow = ctx.flow(residuals, Through::Carries, |n| match *n {
        Node::Symbol(s) => col
            .get(&s)
            .map_or(Set::bottom(), |&j| Set::one(j, wrt.len())),
        _ => Set::bottom(),
    });
    let touched: Vec<Vec<(usize, SymbolId)>> = residuals
        .iter()
        .map(|&r| {
            flow.get(r)
                .iter()
                .map(|j| (j as usize, wrt[j as usize]))
                .collect()
        })
        .collect();
    let mut rows: SparseRows = vec![Vec::new(); residuals.len()];
    let mut forward_rows_by_col: Vec<Vec<usize>> = vec![Vec::new(); wrt.len()];
    for (i, t) in touched.iter().enumerate() {
        if t.len() >= REVERSE_MIN_TOUCHED {
            let syms: Vec<SymbolId> = t.iter().map(|&(_, s)| s).collect();
            let g = gradient(ctx, residuals[i], &syms);
            rows[i] = t.iter().map(|&(j, _)| j).zip(g).collect();
        } else {
            for &(j, _) in t {
                forward_rows_by_col[j].push(i);
            }
        }
    }
    {
        let _s = rsprobe::scope!("forward");
        let mut memo = NodeTable::new(ctx.len());
        let moved_by_some_column = |c: ExprId| !flow.get(c).is_bottom();
        let mut calls = CallOperands::new(&moved_by_some_column);
        for (j, members) in forward_rows_by_col.iter().enumerate() {
            if members.is_empty() {
                continue;
            }
            rsprobe::add!("sweeps", 1);
            memo.begin(ctx.len());
            let roots: Vec<ExprId> = members.iter().map(|&i| residuals[i]).collect();
            let support_holds_column = |c: ExprId| flow.get(c).contains(j as u32);
            let col = forward(
                ctx,
                &roots,
                wrt[j],
                &mut memo,
                &support_holds_column,
                &mut calls,
            );
            for (&i, d) in members.iter().zip(col) {
                rows[i].push((j, d));
            }
        }
    }
    for row in rows.iter_mut() {
        row.retain(|&(_, e)| !ctx.is_zero(e));
    }
    rsprobe::add!("entries", rows.iter().map(Vec::len).sum::<usize>());
    rsprobe::add!("nodes", ctx.len() - nodes);
    rows
}

/// `roots` as affine functions of `syms`: per root its value at `syms` all
/// zero and its coefficients, `(column, d root / d syms[column])` by
/// column, so `root = rest + sum coef * sym`. `Err(i)` where root `i` is not
/// affine in them (a coefficient reads one of `syms`).
pub fn affine_in<K: Field>(
    ctx: &mut Graph<K>,
    roots: &[ExprId],
    syms: &[SymbolId],
) -> Result<(Vec<ExprId>, SparseRows), usize> {
    let coefs = sparse_jacobian(ctx, roots, syms);
    let all: Vec<ExprId> = coefs.iter().flatten().map(|&(_, c)| c).collect();
    let mut nonaffine = ctx.depends_on(&all, syms).into_iter();
    for (i, row) in coefs.iter().enumerate() {
        if nonaffine.by_ref().take(row.len()).any(|d| d) {
            return Err(i);
        }
    }
    let zero = ctx.zero();
    let at_zero: HashMap<SymbolId, ExprId> = syms.iter().map(|&s| (s, zero)).collect();
    Ok((crate::transform::substitute(ctx, roots, &at_zero), coefs))
}

/// The vector-Jacobian product of `outputs` by `wrt`, over weights of its
/// own: the gradient of `sum_k w_k outputs[k]` by `wrt` (one reverse sweep),
/// and the weights `w_k`, symbols of their own (a name no symbol of the
/// graph has) an evaluation binds. Every entry is linear in the weights.
pub fn vjp<K: Field>(
    ctx: &mut Graph<K>,
    outputs: &[ExprId],
    wrt: &[SymbolId],
) -> (Vec<SymbolId>, Vec<ExprId>) {
    let mut weights = Vec::with_capacity(outputs.len());
    let mut terms = Vec::with_capacity(outputs.len());
    for &e in outputs {
        let w = ctx.fresh_symbol("vjp$w");
        weights.push(w);
        let we = ctx.symbol_expr(w);
        terms.push(ctx.mul(we, e));
    }
    let f = ctx.reduce(ReduceOp::Sum, terms);
    (weights, gradient(ctx, f, wrt))
}

/// Reverse-mode symbolic gradient: `d(f)/d(wrt[j])` for every `j`, built in ONE
/// adjoint sweep over the reachable sub-DAG instead of one forward sweep per
/// symbol -- the right shape for a scalar objective over many leaves (a
/// parameter gradient). The local rules (domain-guard mirroring, `Select` /
/// `Min` / `Max` subgradients) are shared with [`differentiate`], so both modes
/// return the same values everywhere; only the graph shape of the result
/// differs. The result is an ordinary expression in the same context, so it can
/// be differentiated again.
pub fn gradient<K: Field>(ctx: &mut Graph<K>, f: ExprId, wrt: &[SymbolId]) -> Vec<ExprId> {
    let _s = rsprobe::scope!("gradient");
    let mut pos = NodeTable::new(ctx.len());
    let nodes = topological_cone(ctx, f, &mut pos);
    for (k, &e) in nodes.iter().enumerate() {
        pos.set(e, k as u32);
    }
    let at = |e: ExprId| pos.get(e).expect("in the cone") as usize;
    let wanted: rustc_hash::FxHashSet<SymbolId> = wrt.iter().copied().collect();
    let mut active = vec![false; nodes.len()];
    let mut ops = Vec::new();
    for (k, &e) in nodes.iter().enumerate() {
        active[k] = match *ctx.node(e) {
            Node::Symbol(s) => wanted.contains(&s),
            _ => {
                carrying(ctx, e, &mut ops);
                ops.iter().any(|&c| active[at(c)])
            }
        };
    }
    let act = |e: ExprId| active[at(e)];

    let mut adj = Adjoints::new(nodes.len());
    if act(f) {
        let one = ctx.one();
        adj.push(at(f), one, false);
    }
    let zero = ctx.zero();
    let mut sym_adj: HashMap<SymbolId, ExprId> = HashMap::default();
    let mut carry_at_first_node: HashMap<usize, Vec<CallSite>> = HashMap::default();
    {
        let mut first: HashMap<CallSite, usize> = HashMap::default();
        for (k, &e) in nodes.iter().enumerate() {
            if let Node::Call(o, l) = *ctx.node(e) {
                first.entry(CallSite::of(ctx, o, l)).or_insert(k);
            }
        }
        for (site, k) in first {
            carry_at_first_node.entry(k).or_default().push(site);
        }
    }
    let mut pending: HashMap<CallSite, Vec<CarriedAdjoint>> = HashMap::default();
    for k in (0..nodes.len()).rev() {
        'node: {
            if adj.terms[k].is_empty() {
                break 'node;
            }
            let e = nodes[k];
            let terms = std::mem::take(&mut adj.terms[k]);
            let gated = adj.gated[k];
            let a_bar = ctx.reduce(ReduceOp::Sum, terms);
            if a_bar == zero {
                break 'node;
            }
            let sweep = Sweep {
                act: &act,
                at: &at,
                pos: &pos,
            };
            let a = (a_bar, gated);
            sweep_node(ctx, e, a, &sweep, &mut adj, &mut sym_adj, &mut pending);
        }
        for site in carry_at_first_node.remove(&k).unwrap_or_default() {
            if let Some(calls) = pending.remove(&site) {
                carry(ctx, site, calls, &act, &mut adj, &at);
            }
        }
    }
    wrt.iter()
        .map(|s| sym_adj.get(s).copied().unwrap_or(zero))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CallSite {
    func: FuncId,
    context: u32,
    args: ArgList,
}

impl CallSite {
    fn of<K: Field>(ctx: &Graph<K>, o: OutputId, args: ArgList) -> Self {
        CallSite {
            func: ctx.output(o).0,
            context: ctx.context_of(o),
            args,
        }
    }
}

fn carry<K: Field>(
    ctx: &mut Graph<K>,
    site: CallSite,
    mut calls: Vec<CarriedAdjoint>,
    act: &dyn Fn(ExprId) -> bool,
    adj: &mut Adjoints,
    at: &dyn Fn(ExprId) -> usize,
) {
    let CallSite {
        func,
        context,
        args: l,
    } = site;
    let o0 = calls[0].output;
    let mut moving: Vec<u32> = Vec::new();
    for c in &calls {
        for &i in ctx.output_support(func, c.out).iter() {
            if act(ctx.call_operand(c.output, l, i)) {
                moving.push(i);
            }
        }
    }
    moving.sort_unstable();
    moving.dedup();
    let symbolic = (calls.iter()).all(|c| {
        !matches!(
            ctx.func(func).outputs()[c.out as usize],
            crate::func::Output::Slot(_)
        )
    });
    let any_adjoint_gated = calls.iter().any(|c| c.gated);
    if symbolic && !any_adjoint_gated && moving.len() >= REVERSE_MIN_TOUCHED {
        calls.sort_unstable_by_key(|c| c.out);
        let outs: Vec<u32> = calls.iter().map(|c| c.out).collect();
        let g = ctx.vjp_function(func, &outs, &moving);
        let mut args = ctx.full_args(o0, l).into_owned();
        args.extend(calls.iter().map(|c| c.adjoint));
        let partials = ctx.calls(g, &(0..moving.len() as u32).collect::<Vec<_>>(), &args);
        for (&i, d) in moving.iter().zip(partials) {
            let arg = ctx.call_operand(o0, l, i);
            adj.push(at(arg), d, false);
        }
        return;
    }
    for CarriedAdjoint {
        out,
        adjoint: a_bar,
        output: o,
        gated,
    } in calls
    {
        let moving: Vec<(u32, ExprId)> = ctx
            .output_support(func, out)
            .iter()
            .map(|&i| (i, ctx.call_operand(o, l, i)))
            .filter(|&(_, a)| act(a))
            .collect();
        let params: Vec<u32> = moving.iter().map(|&(i, _)| i).collect();
        let partial_outputs = ctx.derivative_outputs(func, out, &params);
        for (&(_, arg), k) in moving.iter().zip(partial_outputs) {
            let partial = ctx.call_list_in(func, k, context, l);
            let t = scaled(ctx, (a_bar, gated), partial);
            adj.push(at(arg), t, gated);
        }
    }
}

struct Adjoints {
    terms: Vec<Vec<ExprId>>,
    gated: Vec<bool>,
}

impl Adjoints {
    fn new(n: usize) -> Self {
        Adjoints {
            terms: vec![Vec::new(); n],
            gated: vec![false; n],
        }
    }

    fn push(&mut self, k: usize, term: ExprId, gated: bool) {
        self.terms[k].push(term);
        self.gated[k] |= gated;
    }
}

struct CarriedAdjoint {
    out: u32,
    adjoint: ExprId,
    output: OutputId,
    gated: bool,
}

fn scaled<K: Field>(ctx: &mut Graph<K>, (a_bar, gated): (ExprId, bool), p: ExprId) -> ExprId {
    let ap = ctx.mul(a_bar, p);
    if !gated {
        return ap;
    }
    let zero = ctx.zero();
    let branch_not_taken = ctx.cmp(CmpOp::Eq, a_bar, zero);
    ctx.select(branch_not_taken, zero, ap)
}

struct Sweep<'s> {
    act: &'s dyn Fn(ExprId) -> bool,
    at: &'s dyn Fn(ExprId) -> usize,
    pos: &'s NodeTable<u32>,
}

fn sweep_node<K: Field>(
    ctx: &mut Graph<K>,
    e: ExprId,
    a: (ExprId, bool),
    sweep: &Sweep<'_>,
    adj: &mut Adjoints,
    sym_adj: &mut HashMap<SymbolId, ExprId>,
    pending: &mut HashMap<CallSite, Vec<CarriedAdjoint>>,
) {
    let (a_bar, gated) = a;
    let (act, at, pos) = (sweep.act, sweep.at, sweep.pos);
    let zero = ctx.zero();
    let push_gated_as_node = |adj: &mut Adjoints, child: ExprId, term: ExprId| {
        adj.push(at(child), term, gated);
    };
    {
        match *ctx.node(e) {
            Node::Const(_) | Node::Cmp(..) => {}
            Node::Symbol(s) => {
                sym_adj.insert(s, a_bar);
            }
            Node::Add(x, y) => {
                for c in [x, y] {
                    if act(c) {
                        push_gated_as_node(adj, c, a_bar);
                    }
                }
            }
            Node::Mul(x, y) => {
                if act(x) {
                    let t = scaled(ctx, a, y);
                    push_gated_as_node(adj, x, t);
                }
                if act(y) {
                    let t = scaled(ctx, a, x);
                    push_gated_as_node(adj, y, t);
                }
            }
            Node::Neg(x) => {
                let t = ctx.neg(a_bar);
                push_gated_as_node(adj, x, t);
            }
            Node::Pow(x, n) => {
                let coeff = ctx.konst_int(n);
                let p = ctx.pow_i(x, n - 1);
                let cp = ctx.mul(coeff, p);
                let t = scaled(ctx, a, cp);
                push_gated_as_node(adj, x, t);
            }
            Node::Unary(op, x) => {
                let factor = unary_factor(ctx, op, x);
                let t = scaled(ctx, a, factor);
                push_gated_as_node(adj, x, t);
            }
            Node::Binary(op, x, y) => {
                let p = binary_partials(ctx, op, x, y, [act(x), act(y)]);
                for (c, p) in [(x, p[0]), (y, p[1])] {
                    if act(c) {
                        let t = scaled(ctx, a, p);
                        push_gated_as_node(adj, c, t);
                    }
                }
            }
            Node::Select(c, t, e2) => {
                if act(t) {
                    let tt = ctx.select(c, a_bar, zero);
                    adj.push(at(t), tt, true);
                }
                if act(e2) {
                    let te = ctx.select(c, zero, a_bar);
                    adj.push(at(e2), te, true);
                }
            }
            Node::Reduce(op, l) => match op {
                ReduceOp::Sum => {
                    for &x in ctx.args(l) {
                        if act(x) {
                            push_gated_as_node(adj, x, a_bar);
                        }
                    }
                }
                ReduceOp::Product => {
                    let args = ctx.args(l).to_vec();
                    let others = cofactors(ctx, &args, |i| act(args[i]));
                    for (&arg, other) in args.iter().zip(others) {
                        if let Some(other) = other {
                            let t = scaled(ctx, a, other);
                            push_gated_as_node(adj, arg, t);
                        }
                    }
                }
                ReduceOp::Min | ReduceOp::Max => {
                    let args = ctx.args(l).to_vec();
                    let strictly_more_extreme = if op == ReduceOp::Max {
                        CmpOp::Gt
                    } else {
                        CmpOp::Lt
                    };
                    let n = args.len();
                    let mut m = args[0];
                    let mut replaces_extremum = Vec::with_capacity(n.saturating_sub(1));
                    for &x in &args[1..] {
                        let c = ctx.cmp(strictly_more_extreme, x, m);
                        replaces_extremum.push(c);
                        m = ctx.reduce(op, vec![m, x]);
                    }
                    let one = ctx.one();
                    let mut none_later_replaces = one;
                    for i in (0..n).rev() {
                        if act(args[i]) {
                            let c_i = if i == 0 {
                                one
                            } else {
                                replaces_extremum[i - 1]
                            };
                            let is_extremum = ctx.mul(c_i, none_later_replaces);
                            let t = ctx.select(is_extremum, a_bar, zero);
                            adj.push(at(args[i]), t, true);
                        }
                        if i > 0 {
                            let not_c = ctx.sub(one, replaces_extremum[i - 1]);
                            none_later_replaces = ctx.mul(none_later_replaces, not_c);
                        }
                    }
                }
            },
            Node::Dot(l) => {
                let (xs, ys) = ctx.dot_args(l);
                let (xs, ys) = (xs.to_vec(), ys.to_vec());
                for (&x, &y) in xs.iter().zip(ys.iter()) {
                    if act(x) {
                        let t = scaled(ctx, a, y);
                        push_gated_as_node(adj, x, t);
                    }
                    if act(y) {
                        let t = scaled(ctx, a, x);
                        push_gated_as_node(adj, y, t);
                    }
                }
            }
            Node::Solve(l, _) => {
                let (n, a, b) = ctx.solve_args(l);
                let (a, b) = (a.to_vec(), b.to_vec());
                let x = ctx.solve_dense(a.clone(), b.clone());
                debug_assert!(x.windows(2).all(|w| w[1].0 == w[0].0 + 1));
                let mut x_bar = vec![zero; n];
                let mut rhs_gated = gated;
                for (k, &xk) in x.iter().enumerate() {
                    x_bar[k] = if xk == e {
                        a_bar
                    } else {
                        match pos.get(xk) {
                            Some(p) => {
                                let t = std::mem::take(&mut adj.terms[p as usize]);
                                rhs_gated |= adj.gated[p as usize];
                                ctx.reduce(ReduceOp::Sum, t)
                            }
                            None => zero,
                        }
                    };
                }
                let a_transposed: Vec<ExprId> =
                    (0..n * n).map(|k| a[(k % n) * n + k / n]).collect();
                let mut b_bar = ctx.solve_dense(a_transposed, x_bar.clone());
                if rhs_gated {
                    let squares = (x_bar.iter()).map(|&v| ctx.mul(v, v)).collect();
                    let x_bar_norm2 = ctx.reduce(ReduceOp::Sum, squares);
                    let branch_not_taken = ctx.cmp(CmpOp::Eq, x_bar_norm2, zero);
                    for v in b_bar.iter_mut() {
                        *v = ctx.select(branch_not_taken, zero, *v);
                    }
                }
                for r in 0..n {
                    if act(b[r]) {
                        adj.push(at(b[r]), b_bar[r], rhs_gated);
                    }
                    for j in 0..n {
                        if act(a[r * n + j]) {
                            let t = scaled(ctx, (b_bar[r], rhs_gated), x[j]);
                            let nt = ctx.neg(t);
                            adj.push(at(a[r * n + j]), nt, rhs_gated);
                        }
                    }
                }
            }
            Node::Call(o, l) => {
                let out = ctx.output(o).1;
                pending
                    .entry(CallSite::of(ctx, o, l))
                    .or_default()
                    .push(CarriedAdjoint {
                        out,
                        adjoint: a_bar,
                        output: o,
                        gated,
                    });
            }
        }
    }
}
