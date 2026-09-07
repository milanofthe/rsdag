//! Structural rewrites of the DAG: substituting one symbol for another, the
//! primitive behind graph transformations like node merging (shorting two
//! nodes replaces one node-voltage symbol with the other's everywhere).

use crate::field::Field;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Context, Memo};
use crate::node::{ExprId, Node, SymbolId};

/// Replace every occurrence of symbol `from` with symbol `to` in `expr`,
/// rebuilding through the smart constructors (so the result is re-folded and
/// hash-consed). Memoised over shared subexpressions.
pub fn substitute<K: Field>(
    ctx: &mut Context<K>,
    expr: ExprId,
    from: SymbolId,
    to: SymbolId,
) -> ExprId {
    let te = ctx.symbol_expr(to);
    substitute_expr(ctx, expr, from, te)
}

/// Replace every occurrence of symbol `from` with the expression `to` in `expr`
/// (the symbol-to-expression generalisation of [`substitute`], the primitive
/// behind node elimination: solving a node's KCL for its voltage and inlining
/// that expression everywhere). Rebuilt through the smart constructors and
/// memoised over shared subexpressions.
pub fn substitute_expr<K: Field>(
    ctx: &mut Context<K>,
    expr: ExprId,
    from: SymbolId,
    to: ExprId,
) -> ExprId {
    let mut memo = ctx.take_memo();
    let r = subst_inner(ctx, expr, &|s| (s == from).then_some(to), &mut memo);
    ctx.put_memo(memo);
    r
}

/// Replace every symbol present in `map` with its mapped expression in a single
/// pass (the multi-symbol generalisation of [`substitute_expr`]). Symbols absent
/// from the map are left untouched. Rebuilt through the smart constructors and
/// memoised over shared subexpressions. The substitution is simultaneous (a
/// single traversal), so a target expression that itself mentions a mapped
/// symbol is not re-substituted.
pub fn substitute_many<K: Field>(
    ctx: &mut Context<K>,
    expr: ExprId,
    map: &HashMap<SymbolId, ExprId>,
) -> ExprId {
    let mut memo = ctx.take_memo();
    let r = subst_inner(ctx, expr, &|s| map.get(&s).copied(), &mut memo);
    ctx.put_memo(memo);
    r
}

/// [`substitute_many`] over a whole forest of roots with ONE shared memo: a
/// subexpression reachable from several roots is rebuilt once, not once per
/// root. This is the instantiation primitive for anything with many outputs
/// hanging off one shared core (a device template's terminal currents, noise
/// densities and operating-point variables; a stamp's derivatives w.r.t. each
/// of its ports) -- substituting those root by root re-walks and re-interns the
/// shared core per root, which measured as the dominant cost of Verilog-A
/// instance cloning. Returns the substituted roots in input order.
pub fn substitute_many_all<K: Field>(
    ctx: &mut Context<K>,
    exprs: &[ExprId],
    map: &HashMap<SymbolId, ExprId>,
) -> Vec<ExprId> {
    let mut memo = ctx.take_memo();
    let resolve = |s: SymbolId| map.get(&s).copied();
    let out: Vec<ExprId> = exprs
        .iter()
        .map(|&e| subst_inner(ctx, e, &resolve, &mut memo))
        .collect();
    ctx.put_memo(memo);
    out
}

/// The shared substitution traverser: rebuild `expr` through the smart
/// constructors, replacing every symbol for which `resolve` returns `Some`,
/// memoised over shared subexpressions. Single-symbol and multi-symbol
/// substitution differ only in their `resolve` closure, so the per-node walk
/// lives here once (a new `Node` variant is then handled in exactly one place).
fn subst_inner<K: Field, F: Fn(SymbolId) -> Option<ExprId>>(
    ctx: &mut Context<K>,
    expr: ExprId,
    resolve: &F,
    memo: &mut Memo,
) -> ExprId {
    if let Some(r) = memo.get(expr) {
        return r;
    }
    let node = *ctx.node(expr);
    let r = match node {
        Node::Const(_) => expr,
        Node::Symbol(s) => resolve(s).unwrap_or(expr),
        Node::Add(a, b) => {
            let a = subst_inner(ctx, a, resolve, memo);
            let b = subst_inner(ctx, b, resolve, memo);
            ctx.add(a, b)
        }
        Node::Mul(a, b) => {
            let a = subst_inner(ctx, a, resolve, memo);
            let b = subst_inner(ctx, b, resolve, memo);
            ctx.mul(a, b)
        }
        Node::Neg(a) => {
            let a = subst_inner(ctx, a, resolve, memo);
            ctx.neg(a)
        }
        Node::Pow(a, n) => {
            let a = subst_inner(ctx, a, resolve, memo);
            ctx.pow_i(a, n)
        }
        Node::Unary(op, a) => {
            let a = subst_inner(ctx, a, resolve, memo);
            ctx.unary(op, a)
        }
        Node::Cmp(op, a, b) => {
            let a = subst_inner(ctx, a, resolve, memo);
            let b = subst_inner(ctx, b, resolve, memo);
            ctx.cmp(op, a, b)
        }
        Node::Select(c, t, e) => {
            let c = subst_inner(ctx, c, resolve, memo);
            let t = subst_inner(ctx, t, resolve, memo);
            let e = subst_inner(ctx, e, resolve, memo);
            ctx.select(c, t, e)
        }
        Node::Reduce(op, l) => {
            let args = ctx.args(l).to_vec();
            let na: Vec<ExprId> = args
                .iter()
                .map(|&a| subst_inner(ctx, a, resolve, memo))
                .collect();
            ctx.reduce(op, na)
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            let (a, b) = (a.to_vec(), b.to_vec());
            let na: Vec<ExprId> = a
                .iter()
                .map(|&e| subst_inner(ctx, e, resolve, memo))
                .collect();
            let nb: Vec<ExprId> = b
                .iter()
                .map(|&e| subst_inner(ctx, e, resolve, memo))
                .collect();
            ctx.dot(na, nb)
        }
        Node::Call(o, l) => {
            let args = ctx.args(l).to_vec();
            let na: Vec<ExprId> = args
                .iter()
                .map(|&a| subst_inner(ctx, a, resolve, memo))
                .collect();
            ctx.call_output(o, &na)
        }
    };
    memo.set(expr, r);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    fn sid<K: Field>(ctx: &mut Context<K>, name: &str) -> SymbolId {
        let e = ctx.sym(name);
        match ctx.node(e) {
            Node::Symbol(s) => *s,
            _ => unreachable!(),
        }
    }

    #[test]
    fn substitute_many_is_simultaneous() {
        // Swap x<->y in x - y: simultaneous, so the result is y - x (not 0).
        let mut ctx: Context = Context::new();
        let x = ctx.sym("x");
        let y = ctx.sym("y");
        let f = ctx.sub(x, y);
        let (xs, ys) = (sid(&mut ctx, "x"), sid(&mut ctx, "y"));
        let mut map: HashMap<SymbolId, ExprId> = HashMap::default();
        map.insert(xs, y);
        map.insert(ys, x);
        let g = substitute_many(&mut ctx, f, &map);
        let want = ctx.sub(y, x);
        assert_eq!(g, want);
    }
}
