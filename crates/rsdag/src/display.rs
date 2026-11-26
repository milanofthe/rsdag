use crate::field::Field;
use crate::graph::Graph;
use crate::node::{ExprId, Node};

use crate::constants::{INFIX_MAX, INFIX_OP_BYTES};

/// Render an expression (raw, unsimplified): infix while that stays short,
/// else a listing `%k = ...` of the nodes it reaches, in dependency order,
/// the last line the root.
///
/// Good enough for inspection and tests. Pretty/canonical rendering of
/// `H(s)` as a collected rational function is a job for the rewrite layer.
pub fn to_string<K: Field>(ctx: &Graph<K>, id: ExprId) -> String {
    let cone = ctx.cone(&[id], false);
    let mut len: Vec<usize> = Vec::with_capacity(cone.nodes.len());
    for &e in cone.nodes.iter() {
        let ops = ctx.operands(e);
        let own = if ops.is_empty() {
            leaf(ctx, e).len()
        } else {
            INFIX_OP_BYTES
        };
        let total = (ops.iter()).fold(own, |acc, &c| acc.saturating_add(len[cone.pos(c)]));
        len.push(total.min(INFIX_MAX + 1));
    }
    let mut out = String::new();
    if len[cone.pos(id)] <= INFIX_MAX {
        infix(ctx, id, &mut out);
        return out;
    }
    let mut operand = |c: ExprId, _: bool, out: &mut String| {
        if ctx.operands(c).is_empty() {
            out.push_str(&leaf(ctx, c));
        } else {
            out.push_str(&format!("%{}", c.0));
        }
    };
    let mut in_id_order = cone.nodes;
    in_id_order.sort_unstable();
    for &e in in_id_order.iter().filter(|&&e| !ctx.operands(e).is_empty()) {
        out.push_str(&format!("%{} = ", e.0));
        write_node(ctx, e, &mut out, &mut operand);
        out.push('\n');
    }
    operand(id, false, &mut out);
    out
}

fn leaf<K: Field>(ctx: &Graph<K>, e: ExprId) -> String {
    match *ctx.node(e) {
        Node::Const(c) => ctx.const_val(c).render(),
        Node::Symbol(s) => ctx.symbol_name(s).to_string(),
        _ => unreachable!("a leaf"),
    }
}

fn infix<K: Field>(ctx: &Graph<K>, id: ExprId, out: &mut String) {
    if ctx.operands(id).is_empty() {
        out.push_str(&leaf(ctx, id));
        return;
    }
    write_node(ctx, id, out, &mut |c, factor, out| {
        let needs_parens = factor && matches!(ctx.node(c), Node::Add(..) | Node::Neg(..));
        if needs_parens {
            out.push('(');
        }
        infix(ctx, c, out);
        if needs_parens {
            out.push(')');
        }
    });
}

fn write_node<K: Field>(
    ctx: &Graph<K>,
    id: ExprId,
    out: &mut String,
    operand: &mut dyn FnMut(ExprId, bool, &mut String),
) {
    let mut list = |name: &str, args: &[ExprId], out: &mut String| {
        out.push_str(name);
        out.push('(');
        for (i, &a) in args.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            operand(a, false, out);
        }
        out.push(')');
    };
    match *ctx.node(id) {
        Node::Const(_) | Node::Symbol(_) => out.push_str(&leaf(ctx, id)),
        Node::Add(a, b) => {
            out.push('(');
            operand(a, false, out);
            out.push_str(" + ");
            operand(b, false, out);
            out.push(')');
        }
        Node::Mul(a, b) => {
            operand(a, true, out);
            out.push('*');
            operand(b, true, out);
        }
        Node::Neg(a) => {
            out.push('-');
            operand(a, true, out);
        }
        Node::Pow(a, n) => {
            operand(a, true, out);
            out.push_str(&format!("^{n}"));
        }
        Node::Unary(op, a) => list(op.name(), &[a], out),
        Node::Binary(op, a, b) => list(op.name(), &[a, b], out),
        Node::Cmp(op, a, b) => {
            out.push('(');
            operand(a, false, out);
            out.push_str(&format!(" {} ", op.symbol()));
            operand(b, false, out);
            out.push(')');
        }
        Node::Select(c, t, e) => list("select", &[c, t, e], out),
        Node::Call(o, l) => {
            let (f, k) = ctx.output(o);
            list(
                &format!("{}#{k}", ctx.func(f).name()),
                &ctx.full_args(o, l),
                out,
            )
        }
        Node::Reduce(op, l) => list(op.name(), ctx.args(l), out),
        Node::Solve(l, i) => {
            let (n, _, _) = ctx.solve_args(l);
            out.push_str(&format!("solve{n}[{i}]"));
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            out.push_str("dot(");
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                if i > 0 {
                    out.push_str(" + ");
                }
                operand(x, true, out);
                out.push('*');
                operand(y, true, out);
            }
            out.push(')');
        }
    }
}
