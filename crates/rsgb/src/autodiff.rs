//! Symbolic differentiation over the DAG.
//!
//! Builds the derivative as new nodes in the same context, so common
//! subexpressions are shared (hash-consing) and the result can be exported or
//! differentiated again. Used to produce analytic Jacobians for DAE export.

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Context, Memo};
use crate::node::{CmpOp, ExprId, Node, ReduceOp, SymbolId, UnaryOp};

/// Derivative of `expr` with respect to the symbol `wrt`.
pub fn differentiate<K: Field>(ctx: &mut Context<K>, expr: ExprId, wrt: SymbolId) -> ExprId {
    let mut memo = ctx.take_memo();
    let d = diff(ctx, expr, wrt, &mut memo);
    ctx.put_memo(memo);
    d
}

/// Total time derivative `d/dt(e) = Σ_s (∂e/∂s) · deriv_of[s]`, summed over the
/// free symbols of `e` that have a known time-derivative entry in `deriv_of`
/// (mapping a state symbol to its derivative expression, e.g. `v{k}` -> `vdot{k}`).
/// The symbolic analogue of forming a capacitive current `dq/dt` from a charge
/// `q(v)`; used to lower Verilog-A `ddt(...)` in arbitrary residual rows.
pub fn time_derivative<K: Field>(
    ctx: &mut Context<K>,
    e: ExprId,
    deriv_of: &HashMap<SymbolId, ExprId>,
) -> ExprId {
    let syms: Vec<SymbolId> = ctx.free_symbols(e).into_iter().collect();
    let mut terms: Vec<ExprId> = Vec::new();
    for s in syms {
        if let Some(&sdot) = deriv_of.get(&s) {
            let de = differentiate(ctx, e, s);
            if !ctx.is_zero(de) {
                let term = ctx.mul(de, sdot);
                terms.push(term);
            }
        }
    }
    ctx.reduce(ReduceOp::Sum, terms)
}

/// Local derivative `d(op(a))/da` of a unary op, shared by the forward
/// ([`differentiate`]) and reverse ([`gradient`]) sweeps so both modes apply the
/// identical rule. The rules mirror the domain guards in
/// [`crate::node::unary_f64`] exactly, so the Jacobian stays finite wherever the
/// residual does (an out-of-range internal-node guess must not produce an
/// `inf`/`NaN` Jacobian entry that derails Newton).
fn unary_factor<K: Field>(ctx: &mut Context<K>, op: UnaryOp, a: ExprId) -> ExprId {
    match op {
        UnaryOp::Exp => {
            // d/dx limexp(a) = exp(a) below the threshold, the constant slope
            // exp(EXP_LIMIT) on the linear tail. Written as a `select` over the
            // PRIMAL `exp(a)` node (not `exp(min(a, EXP_LIMIT))`), so
            // hash-consing shares the one exp between residual and Jacobian:
            // one transcendental per junction per evaluation instead of two.
            // Values are identical: for a <= EXP_LIMIT `unary_f64` evaluates
            // the bare `a.exp()`.
            let hi = ctx.konst_f64(crate::node::EXP_LIMIT);
            let below = ctx.cmp(CmpOp::Le, a, hi);
            let ea = ctx.exp(a);
            let slope = ctx.konst_f64(crate::node::EXP_LIMIT.exp());
            ctx.select(below, ea, slope)
        }
        UnaryOp::Ln => {
            // 1/a above the floor, 0 below it (ln is clamped flat there).
            let lo = ctx.konst_f64(crate::node::LN_FLOOR);
            let above = ctx.cmp(CmpOp::Gt, a, lo);
            let inv_a = ctx.recip(a);
            let zero = ctx.zero();
            ctx.select(above, inv_a, zero)
        }
        UnaryOp::Sqrt => {
            // 1/(2*sqrt(a)) for a>0, else 0 (matches sqrt clamped to 0).
            let s = ctx.sqrt(a);
            let rs = ctx.recip(s);
            let half = ctx.ratio(1, 2);
            let d = ctx.mul(half, rs);
            let zero = ctx.zero();
            let pos = ctx.cmp(CmpOp::Gt, a, zero);
            ctx.select(pos, d, zero)
        }
        UnaryOp::Sin => ctx.cos(a),
        UnaryOp::Cos => {
            let s = ctx.sin(a);
            ctx.neg(s)
        }
        UnaryOp::Floor => ctx.zero(), // piecewise-constant: 0 a.e. (factor 0)
        UnaryOp::Sinh => ctx.cosh(a),
        UnaryOp::Cosh => ctx.sinh(a),
        UnaryOp::Tanh => {
            // 1 - tanh(a)^2
            let t = ctx.tanh(a);
            let t2 = ctx.mul(t, t);
            let one = ctx.one();
            ctx.sub(one, t2)
        }
        UnaryOp::Atan => {
            // 1 / (1 + a^2)
            let a2 = ctx.mul(a, a);
            let one = ctx.one();
            let denom = ctx.add(one, a2);
            ctx.recip(denom)
        }
    }
}

fn diff<K: Field>(ctx: &mut Context<K>, expr: ExprId, wrt: SymbolId, memo: &mut Memo) -> ExprId {
    if let Some(d) = memo.get(expr) {
        return d;
    }
    // Copy the node (16 bytes) so we can mutate the context while building the
    // derivative; variadic operand lists are copied out of the pool below.
    let node = *ctx.node(expr);
    let d = match node {
        Node::Const(_) => ctx.zero(),
        Node::Symbol(s) => {
            if s == wrt {
                ctx.one()
            } else {
                ctx.zero()
            }
        }
        Node::Add(a, b) => {
            let da = diff(ctx, a, wrt, memo);
            let db = diff(ctx, b, wrt, memo);
            ctx.add(da, db)
        }
        Node::Mul(a, b) => {
            // product rule: da*b + a*db
            let da = diff(ctx, a, wrt, memo);
            let db = diff(ctx, b, wrt, memo);
            let t1 = ctx.mul(da, b);
            let t2 = ctx.mul(a, db);
            ctx.add(t1, t2)
        }
        Node::Neg(a) => {
            let da = diff(ctx, a, wrt, memo);
            ctx.neg(da)
        }
        Node::Pow(a, n) => {
            // power rule (integer exponent): n * a^(n-1) * da
            let da = diff(ctx, a, wrt, memo);
            let coeff = ctx.konst_int(n);
            let p = ctx.pow_i(a, n - 1);
            let cp = ctx.mul(coeff, p);
            ctx.mul(cp, da)
        }
        Node::Unary(op, a) => {
            let da = diff(ctx, a, wrt, memo);
            let factor = unary_factor(ctx, op, a);
            ctx.mul(factor, da)
        }
        // Comparisons are piecewise-constant: derivative is zero a.e.
        Node::Cmp(..) => ctx.zero(),
        // Subgradient: differentiate through both branches, keep the condition.
        Node::Select(c, t, e) => {
            let dt = diff(ctx, t, wrt, memo);
            let de = diff(ctx, e, wrt, memo);
            ctx.select(c, dt, de)
        }
        Node::Reduce(op, l) => match op {
            // d(Σ aᵢ) = Σ daᵢ
            ReduceOp::Sum => {
                let args = ctx.args(l).to_vec();
                let dargs: Vec<ExprId> = args.iter().map(|&a| diff(ctx, a, wrt, memo)).collect();
                ctx.reduce(ReduceOp::Sum, dargs)
            }
            // d(Π aᵢ) = Σᵢ daᵢ · Πⱼ≠ᵢ aⱼ  (generalized product rule)
            ReduceOp::Product => {
                let args = ctx.args(l).to_vec();
                let mut terms = Vec::with_capacity(args.len());
                for i in 0..args.len() {
                    let dai = diff(ctx, args[i], wrt, memo);
                    if ctx.is_zero(dai) {
                        continue;
                    }
                    let others: Vec<ExprId> = args
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, &a)| a)
                        .collect();
                    let prod = ctx.reduce(ReduceOp::Product, others);
                    terms.push(ctx.mul(dai, prod));
                }
                ctx.reduce(ReduceOp::Sum, terms)
            }
            // Subgradient: the derivative of the (first) extremal argument.
            ReduceOp::Min | ReduceOp::Max => {
                let args = ctx.args(l).to_vec();
                let cmp = if op == ReduceOp::Max {
                    CmpOp::Gt
                } else {
                    CmpOp::Lt
                };
                let mut m = args[0];
                let mut dm = diff(ctx, args[0], wrt, memo);
                for &a in &args[1..] {
                    let da = diff(ctx, a, wrt, memo);
                    let cond = ctx.cmp(cmp, a, m);
                    dm = ctx.select(cond, da, dm);
                    m = ctx.reduce(op, vec![m, a]);
                }
                dm
            }
        },
        // d(Σ aᵢbᵢ) = Σ (daᵢ·bᵢ + aᵢ·dbᵢ)
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let mut terms = Vec::with_capacity(a.len());
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let dai = diff(ctx, ai, wrt, memo);
                let dbi = diff(ctx, bi, wrt, memo);
                let t1 = ctx.mul(dai, bi);
                let t2 = ctx.mul(ai, dbi);
                terms.push(ctx.add(t1, t2));
            }
            ctx.reduce(ReduceOp::Sum, terms)
        }
        // Chain rule through a call: d/dx f_out(a) = Σ_i (∂f_out/∂p_i)(a) · da_i,
        // each partial a call into the function's derivative output.
        Node::Call(o, l) => {
            let args = ctx.args(l).to_vec();
            let (f, out) = ctx.output(o);
            let mut acc = ctx.zero();
            for (i, &arg) in args.iter().enumerate() {
                let dai = diff(ctx, arg, wrt, memo);
                if ctx.is_zero(dai) {
                    continue;
                }
                let k = ctx.derivative_output(f, out, i as u32);
                let partial = ctx.call(f, k, &args);
                let term = ctx.mul(partial, dai);
                acc = ctx.add(acc, term);
            }
            acc
        }
    };
    memo.set(expr, d);
    d
}

/// Jacobian matrix: `jac[i][j] = d(residuals[i]) / d(wrt[j])`.
pub fn jacobian<K: Field>(
    ctx: &mut Context<K>,
    residuals: &[ExprId],
    wrt: &[SymbolId],
) -> Vec<Vec<ExprId>> {
    residuals
        .iter()
        .map(|&r| wrt.iter().map(|&s| differentiate(ctx, r, s)).collect())
        .collect()
}

/// Reverse-mode symbolic gradient: `d(f)/d(wrt[j])` for every `j`, built in ONE
/// adjoint sweep over the reachable sub-DAG instead of one forward sweep per
/// symbol -- the right shape for a scalar objective over many leaves (a
/// parameter gradient). The local rules (domain-guard mirroring, `Select` /
/// `Min` / `Max` subgradients) are shared with [`differentiate`], so both modes
/// return the same values everywhere; only the graph shape of the result
/// differs. The result is an ordinary expression in the same context, so it can
/// be differentiated again (see [`hessian`]).
pub fn gradient<K: Field>(ctx: &mut Context<K>, f: ExprId, wrt: &[SymbolId]) -> Vec<ExprId> {
    // Reachable sub-DAG of f. Ascending ExprId is a topological order (a
    // hash-consed node has a larger id than its children), so iterating the
    // sorted set in REVERSE visits every node after all of its parents.
    let mut reach: std::collections::BTreeSet<ExprId> = std::collections::BTreeSet::new();
    let mut stack = vec![f];
    while let Some(e) = stack.pop() {
        if !reach.insert(e) {
            continue;
        }
        stack.extend_from_slice(&ctx.operands(e));
    }

    // Adjoint accumulation: per node a term list, folded into one fused
    // Reduce(Sum) when the node is visited (all parents seen by then).
    let mut adj: HashMap<ExprId, Vec<ExprId>> = HashMap::default();
    let one = ctx.one();
    adj.insert(f, vec![one]);
    let push = |adj: &mut HashMap<ExprId, Vec<ExprId>>, child: ExprId, term: ExprId| {
        adj.entry(child).or_default().push(term);
    };
    let mut sym_adj: HashMap<SymbolId, ExprId> = HashMap::default();
    for &e in reach.iter().rev() {
        let terms = match adj.remove(&e) {
            Some(t) => t,
            None => continue, // unreachable from f's value path (e.g. below a Cmp)
        };
        let a_bar = ctx.reduce(ReduceOp::Sum, terms);
        if ctx.is_zero(a_bar) {
            continue;
        }
        match *ctx.node(e) {
            Node::Const(_) => {}
            Node::Symbol(s) => {
                sym_adj.insert(s, a_bar);
            }
            Node::Add(x, y) => {
                push(&mut adj, x, a_bar);
                push(&mut adj, y, a_bar);
            }
            Node::Mul(x, y) => {
                let tx = ctx.mul(a_bar, y);
                let ty = ctx.mul(a_bar, x);
                push(&mut adj, x, tx);
                push(&mut adj, y, ty);
            }
            Node::Neg(x) => {
                let t = ctx.neg(a_bar);
                push(&mut adj, x, t);
            }
            Node::Pow(x, n) => {
                // d/dx x^n = n * x^(n-1)
                let coeff = ctx.konst_int(n);
                let p = ctx.pow_i(x, n - 1);
                let cp = ctx.mul(coeff, p);
                let t = ctx.mul(a_bar, cp);
                push(&mut adj, x, t);
            }
            Node::Unary(op, x) => {
                let factor = unary_factor(ctx, op, x);
                let t = ctx.mul(a_bar, factor);
                push(&mut adj, x, t);
            }
            // Piecewise-constant: no value path into the operands.
            Node::Cmp(..) => {}
            // Subgradient: the adjoint flows into the taken branch only
            // (matching the forward rule d = select(c, dt, de)).
            Node::Select(c, t, e2) => {
                let zero = ctx.zero();
                let tt = ctx.select(c, a_bar, zero);
                let te = ctx.select(c, zero, a_bar);
                push(&mut adj, t, tt);
                push(&mut adj, e2, te);
            }
            Node::Reduce(op, l) => match op {
                ReduceOp::Sum => {
                    for &x in ctx.args(l) {
                        push(&mut adj, x, a_bar);
                    }
                }
                // d(Π aᵢ)/daᵢ = Πⱼ≠ᵢ aⱼ, via prefix/suffix products (O(k) nodes).
                ReduceOp::Product => {
                    let args = ctx.args(l).to_vec();
                    let k = args.len();
                    let mut prefix = Vec::with_capacity(k);
                    let mut acc = ctx.one();
                    for &x in &args {
                        prefix.push(acc);
                        acc = ctx.mul(acc, x);
                    }
                    let mut suffix = ctx.one();
                    for i in (0..k).rev() {
                        let others = ctx.mul(prefix[i], suffix);
                        let t = ctx.mul(a_bar, others);
                        push(&mut adj, args[i], t);
                        suffix = ctx.mul(suffix, args[i]);
                    }
                }
                // Subgradient of the (first) extremal argument, exactly the
                // forward rule's select chain written as 0/1 coefficients:
                // coef_i = cond_i * Π_{k>i} (1 - cond_k), cond_0 = 1, with
                // cond_k = (args[k] <op-cmp> running extremum of args[..k]).
                ReduceOp::Min | ReduceOp::Max => {
                    let args = ctx.args(l).to_vec();
                    let cmp = if op == ReduceOp::Max {
                        CmpOp::Gt
                    } else {
                        CmpOp::Lt
                    };
                    let n = args.len();
                    let mut m = args[0];
                    let mut conds = Vec::with_capacity(n.saturating_sub(1));
                    for &x in &args[1..] {
                        let c = ctx.cmp(cmp, x, m);
                        conds.push(c);
                        m = ctx.reduce(op, vec![m, x]);
                    }
                    let one = ctx.one();
                    let mut tail = one;
                    for i in (0..n).rev() {
                        let c_i = if i == 0 { one } else { conds[i - 1] };
                        let coef = ctx.mul(c_i, tail);
                        let t = ctx.mul(a_bar, coef);
                        push(&mut adj, args[i], t);
                        if i > 0 {
                            let not_c = ctx.sub(one, conds[i - 1]);
                            tail = ctx.mul(tail, not_c);
                        }
                    }
                }
            },
            Node::Dot(l) => {
                let (xs, ys) = ctx.dot_args(l);
                let (xs, ys) = (xs.to_vec(), ys.to_vec());
                for (&x, &y) in xs.iter().zip(ys.iter()) {
                    let tx = ctx.mul(a_bar, y);
                    let ty = ctx.mul(a_bar, x);
                    push(&mut adj, x, tx);
                    push(&mut adj, y, ty);
                }
            }
            // Chain rule through a call: the same derivative outputs as the
            // forward mode.
            Node::Call(o, l) => {
                let args = ctx.args(l).to_vec();
                let (f, out) = ctx.output(o);
                for (i, &arg) in args.iter().enumerate() {
                    let k = ctx.derivative_output(f, out, i as u32);
                    let partial = ctx.call(f, k, &args);
                    let t = ctx.mul(a_bar, partial);
                    push(&mut adj, arg, t);
                }
            }
        }
    }

    wrt.iter()
        .map(|s| sym_adj.get(s).copied().unwrap_or_else(|| ctx.zero()))
        .collect()
}

/// Symbolic Hessian `hess[i][j] = d²f / d(wrt[i]) d(wrt[j])`, built
/// forward-over-reverse: one reverse sweep for the gradient, then one forward
/// sweep per column. Like every derivative here it is an ordinary expression,
/// so third and higher orders are just repeated application.
pub fn hessian<K: Field>(ctx: &mut Context<K>, f: ExprId, wrt: &[SymbolId]) -> Vec<Vec<ExprId>> {
    let grad = gradient(ctx, f, wrt);
    grad.iter()
        .map(|&g| wrt.iter().map(|&s| differentiate(ctx, g, s)).collect())
        .collect()
}

/// Structural sparsity pattern of a Jacobian: `true` where the entry is not the
/// constant zero.
pub fn sparsity<K: Field>(ctx: &Context<K>, jac: &[Vec<ExprId>]) -> Vec<Vec<bool>> {
    jac.iter()
        .map(|row| row.iter().map(|&e| !ctx.is_zero(e)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;
    use num_complex::Complex64;
    use std::collections::HashMap;

    fn sid<K: Field>(ctx: &mut Context<K>, name: &str) -> SymbolId {
        let id = ctx.sym(name);
        match ctx.node(id) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        }
    }

    #[test]
    fn derivative_matches_finite_difference() {
        let mut ctx: Context = Context::new();
        // f = Is*(exp(v/Vt) - 1) + g*v^2  (a diode current plus a quadratic term)
        let v = ctx.sym("v");
        let vt = ctx.sym("Vt");
        let is = ctx.sym("Is");
        let g = ctx.sym("g");
        let arg = ctx.div(v, vt);
        let e = ctx.exp(arg);
        let one = ctx.one();
        let em1 = ctx.sub(e, one);
        let diode = ctx.mul(is, em1);
        let v2 = ctx.pow_i(v, 2);
        let gv2 = ctx.mul(g, v2);
        let f = ctx.add(diode, gv2);

        let v_id = sid(&mut ctx, "v");
        let df = differentiate(&mut ctx, f, v_id);

        // Derivative wrt an absent symbol is structurally zero.
        let w_id = sid(&mut ctx, "w");
        let dfw = differentiate(&mut ctx, f, w_id);
        assert!(ctx.is_zero(dfw));

        let vt_id = sid(&mut ctx, "Vt");
        let is_id = sid(&mut ctx, "Is");
        let g_id = sid(&mut ctx, "g");

        let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
        env.insert(vt_id, Complex64::new(0.05, 0.0));
        env.insert(is_id, Complex64::new(1e-12, 0.0));
        env.insert(g_id, Complex64::new(0.3, 0.0));

        let h = 1e-6;
        for &x0 in &[-0.1, 0.0, 0.1, 0.3] {
            env.insert(v_id, Complex64::new(x0, 0.0));
            let analytic = eval(&ctx, df, &env).re;
            env.insert(v_id, Complex64::new(x0 + h, 0.0));
            let fp = eval(&ctx, f, &env).re;
            env.insert(v_id, Complex64::new(x0 - h, 0.0));
            let fm = eval(&ctx, f, &env).re;
            let numeric = (fp - fm) / (2.0 * h);
            assert!(
                (analytic - numeric).abs() <= 1e-4 * (1.0 + analytic.abs()),
                "x0={x0}: analytic={analytic} numeric={numeric}"
            );
        }
    }

    #[test]
    fn exp_derivative_shares_primal_and_matches_limexp_tail() {
        use crate::node::EXP_LIMIT;
        let mut ctx: Context = Context::new();
        let x = ctx.sym("x");
        let e = ctx.exp(x);
        let xid = sid(&mut ctx, "x");
        let de = differentiate(&mut ctx, e, xid);
        // The derivative reuses the primal exp node (no second exp in the DAG).
        let n_exp = (0..ctx.len())
            .filter(|&i| matches!(ctx.node(ExprId(i as u32)), Node::Unary(UnaryOp::Exp, _)))
            .count();
        assert_eq!(n_exp, 1);
        let mut env: HashMap<SymbolId, f64> = HashMap::new();
        for &xv in &[-3.0, 0.0, 10.0, EXP_LIMIT, EXP_LIMIT + 5.0, 500.0] {
            env.insert(xid, xv);
            let v = crate::eval::eval_real(&ctx, &env, &[de])[0];
            let want = if xv <= EXP_LIMIT {
                xv.exp()
            } else {
                EXP_LIMIT.exp()
            };
            assert_eq!(v.to_bits(), want.to_bits(), "x={xv}");
        }
    }

    #[test]
    fn select_and_opaque_autodiff() {
        use crate::node::CmpOp;
        let mut ctx: Context = Context::new();
        // f = select(x > 0, x*x, -x);  df/dx = select(x>0, 2x, -1)
        let x = ctx.sym("x");
        let zero = ctx.zero();
        let cond = ctx.cmp(CmpOp::Gt, x, zero);
        let x2 = ctx.mul(x, x);
        let negx = ctx.neg(x);
        let f = ctx.select(cond, x2, negx);
        let xid = sid(&mut ctx, "x");
        let df = differentiate(&mut ctx, f, xid);

        let eval_at = |ctx: &Context, e: ExprId, xv: f64| {
            let mut env = HashMap::new();
            env.insert(xid, Complex64::new(xv, 0.0));
            eval(ctx, e, &env).re
        };
        assert!((eval_at(&ctx, df, 3.0) - 6.0).abs() < 1e-9); // 2*3
        assert!((eval_at(&ctx, df, -2.0) + 1.0).abs() < 1e-9); // -1

        // Chain rule through a call: g = f(2*x) with f(p) = p^3; dg/dx must
        // reference the derivative output of f (3p^2, a call) times 2:
        // at x = 1, 3*4*2 = 24.
        let two = ctx.konst_int(2);
        let two_x = ctx.mul(two, x);
        let p = ctx.sym("p");
        let ps = sid(&mut ctx, "p");
        let p3 = ctx.pow_i(p, 3);
        let f = ctx.define_func("cube", vec![ps], vec![p3]);
        let g = ctx.call(f, 0, &[two_x]);
        let dg = differentiate(&mut ctx, g, xid);
        assert!(!ctx.is_zero(dg));
        assert!(crate::to_string(&ctx, dg).contains("cube#1"));
        assert!((eval_at(&ctx, dg, 1.0) - 24.0).abs() < 1e-9);
    }

    #[test]
    fn gradient_matches_forward_mode() {
        // A device-like expression exercising every node kind the reverse sweep
        // handles: exp/ln guards, select subgradients, powers, reduce, dot.
        let mut ctx: Context = Context::new();
        let x = ctx.sym("x");
        let y = ctx.sym("y");
        let z = ctx.sym("z");
        let arg = ctx.div(x, y);
        let e = ctx.exp(arg);
        let x2 = ctx.pow_i(x, 3);
        let ln = ctx.ln(y);
        let zero = ctx.zero();
        let cond = ctx.cmp(CmpOp::Gt, z, zero);
        let t = ctx.mul(x2, ln);
        let nx = ctx.neg(x);
        let sel = ctx.select(cond, t, nx);
        let red = ctx.reduce(ReduceOp::Product, vec![x, y, z]);
        let mn = ctx.reduce(ReduceOp::Min, vec![x, y, z]);
        let dot = ctx.dot(vec![x, sel], vec![y, z]);
        let s1 = ctx.add(e, red);
        let s2 = ctx.add(mn, dot);
        let f = ctx.add(s1, s2);

        let (xs, ys, zs) = (sid(&mut ctx, "x"), sid(&mut ctx, "y"), sid(&mut ctx, "z"));
        let grad = gradient(&mut ctx, f, &[xs, ys, zs]);
        let fwd: Vec<ExprId> = [xs, ys, zs]
            .iter()
            .map(|&s| differentiate(&mut ctx, f, s))
            .collect();

        let pts = [
            (0.7, 1.3, 0.4),
            (-0.5, 2.0, -1.1),
            (1.9, 0.3, 2.5),
            (-2.0, -0.7, 0.9),
        ];
        for &(xv, yv, zv) in &pts {
            let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
            env.insert(xs, Complex64::new(xv, 0.0));
            env.insert(ys, Complex64::new(yv, 0.0));
            env.insert(zs, Complex64::new(zv, 0.0));
            for (g, d) in grad.iter().zip(&fwd) {
                let gv = eval(&ctx, *g, &env).re;
                let dv = eval(&ctx, *d, &env).re;
                assert!(
                    (gv - dv).abs() <= 1e-12 * (1.0 + dv.abs()),
                    "at ({xv},{yv},{zv}): reverse={gv} forward={dv}"
                );
            }
        }
    }

    #[test]
    fn gradient_handles_calls_and_absent_symbols() {
        let mut ctx: Context = Context::new();
        let x = ctx.sym("x");
        let two = ctx.konst_int(2);
        let two_x = ctx.mul(two, x);
        let p = ctx.sym("p");
        let ps = sid(&mut ctx, "p");
        let p3 = ctx.pow_i(p, 3);
        let f = ctx.define_func("cube", vec![ps], vec![p3]);
        let g = ctx.call(f, 0, &[two_x]);
        let xs = sid(&mut ctx, "x");
        let ws = sid(&mut ctx, "w");
        let grad = gradient(&mut ctx, g, &[xs, ws]);
        assert!(!ctx.is_zero(grad[0]));
        assert!(crate::to_string(&ctx, grad[0]).contains("cube#1"));
        // Absent symbol: structurally zero.
        assert!(ctx.is_zero(grad[1]));
    }

    #[test]
    fn hessian_is_symmetric_and_correct() {
        // f = exp(x*y) + x^3*y  ->  d2f/dxdy = exp(xy)*(1 + xy) + 3x^2 (both orders).
        let mut ctx: Context = Context::new();
        let x = ctx.sym("x");
        let y = ctx.sym("y");
        let xy = ctx.mul(x, y);
        let e = ctx.exp(xy);
        let x3 = ctx.pow_i(x, 3);
        let x3y = ctx.mul(x3, y);
        let f = ctx.add(e, x3y);
        let (xs, ys) = (sid(&mut ctx, "x"), sid(&mut ctx, "y"));
        let h = hessian(&mut ctx, f, &[xs, ys]);

        let mut env: HashMap<SymbolId, Complex64> = HashMap::new();
        env.insert(xs, Complex64::new(0.6, 0.0));
        env.insert(ys, Complex64::new(-0.8, 0.0));
        let (xv, yv) = (0.6_f64, -0.8_f64);
        let want_xy = (xv * yv).exp() * (1.0 + xv * yv) + 3.0 * xv * xv;
        let h01 = eval(&ctx, h[0][1], &env).re;
        let h10 = eval(&ctx, h[1][0], &env).re;
        assert!((h01 - want_xy).abs() <= 1e-12 * (1.0 + want_xy.abs()));
        assert!((h10 - want_xy).abs() <= 1e-12 * (1.0 + want_xy.abs()));
        // Third order by repeated application: d3f/dx3 = 6y + y^3*exp(xy).
        let gx = gradient(&mut ctx, f, &[xs])[0];
        let gxx = differentiate(&mut ctx, gx, xs);
        let gxxx = differentiate(&mut ctx, gxx, xs);
        let want3 = 6.0 * yv + yv.powi(3) * (xv * yv).exp();
        let got3 = eval(&ctx, gxxx, &env).re;
        assert!((got3 - want3).abs() <= 1e-12 * (1.0 + want3.abs()));
    }

    #[test]
    fn jacobian_and_sparsity() {
        let mut ctx: Context = Context::new();
        // r0 = a*x + b*y ; r1 = x  (so dr1/dy = 0)
        let x = ctx.sym("x");
        let y = ctx.sym("y");
        let a = ctx.sym("a");
        let b = ctx.sym("b");
        let ax = ctx.mul(a, x);
        let by = ctx.mul(b, y);
        let r0 = ctx.add(ax, by);
        let r1 = x;

        let x_id = sid(&mut ctx, "x");
        let y_id = sid(&mut ctx, "y");
        let jac = jacobian(&mut ctx, &[r0, r1], &[x_id, y_id]);
        let sp = sparsity(&ctx, &jac);
        assert_eq!(sp, vec![vec![true, true], vec![true, false]]);
    }
}
