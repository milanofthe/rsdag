use crate::field::Field;
use crate::graph::Context;
use crate::node::{CmpOp, ExprId, Node, ReduceOp, UnaryOp};

/// Render an expression to an infix string (raw, unsimplified).
///
/// Good enough for inspection and tests. Pretty/canonical rendering of
/// `H(s)` as a collected rational function is a job for the rewrite layer.
pub fn to_string<K: Field>(ctx: &Context<K>, id: ExprId) -> String {
    let mut out = String::new();
    write_expr(ctx, id, &mut out);
    out
}

fn write_expr<K: Field>(ctx: &Context<K>, id: ExprId, out: &mut String) {
    match ctx.node(id) {
        Node::Const(c) => {
            out.push_str(&ctx.const_val(*c).render());
        }
        Node::Symbol(s) => out.push_str(ctx.symbol_name(*s)),
        Node::Add(a, b) => {
            out.push('(');
            write_expr(ctx, *a, out);
            out.push_str(" + ");
            write_expr(ctx, *b, out);
            out.push(')');
        }
        Node::Mul(a, b) => {
            write_factor(ctx, *a, out);
            out.push('*');
            write_factor(ctx, *b, out);
        }
        Node::Neg(a) => {
            out.push('-');
            write_factor(ctx, *a, out);
        }
        Node::Pow(a, n) => {
            write_factor(ctx, *a, out);
            out.push_str(&format!("^{n}"));
        }
        Node::Unary(op, a) => {
            out.push_str(unary_name(*op));
            out.push('(');
            write_expr(ctx, *a, out);
            out.push(')');
        }
        Node::Cmp(op, a, b) => {
            out.push('(');
            write_expr(ctx, *a, out);
            out.push_str(match op {
                CmpOp::Gt => " > ",
                CmpOp::Ge => " >= ",
                CmpOp::Lt => " < ",
                CmpOp::Le => " <= ",
                CmpOp::Eq => " == ",
                CmpOp::Ne => " != ",
            });
            write_expr(ctx, *b, out);
            out.push(')');
        }
        Node::Select(c, t, e) => {
            out.push_str("select(");
            write_expr(ctx, *c, out);
            out.push_str(", ");
            write_expr(ctx, *t, out);
            out.push_str(", ");
            write_expr(ctx, *e, out);
            out.push(')');
        }
        Node::Call(o, l) => {
            let (f, k) = ctx.output(*o);
            out.push_str(&format!("{}#{k}", ctx.func(f).name));
            out.push('(');
            for (i, &a) in ctx.args(*l).iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_expr(ctx, a, out);
            }
            out.push(')');
        }
        Node::Reduce(op, l) => {
            let args = ctx.args(*l);
            out.push_str(match op {
                ReduceOp::Sum => "sum",
                ReduceOp::Product => "prod",
                ReduceOp::Min => "min",
                ReduceOp::Max => "max",
            });
            out.push('(');
            for (i, &a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_expr(ctx, a, out);
            }
            out.push(')');
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(*l);
            out.push_str("dot(");
            for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                if i > 0 {
                    out.push_str(" + ");
                }
                write_factor(ctx, x, out);
                out.push('*');
                write_factor(ctx, y, out);
            }
            out.push(')');
        }
    }
}

fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Exp => "exp",
        UnaryOp::Ln => "ln",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Sinh => "sinh",
        UnaryOp::Cosh => "cosh",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Atan => "atan",
        UnaryOp::Floor => "floor",
    }
}

/// Wrap sums in parens when they appear as a factor.
fn write_factor<K: Field>(ctx: &Context<K>, id: ExprId, out: &mut String) {
    match ctx.node(id) {
        Node::Add(..) | Node::Neg(..) => {
            out.push('(');
            write_expr(ctx, id, out);
            out.push(')');
        }
        _ => write_expr(ctx, id, out),
    }
}
