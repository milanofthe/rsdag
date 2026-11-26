mod common;
use common::constants::{CENTRAL_DIFFERENCE_STEP, FINITE_DIFFERENCE};
use rsdag::eval::eval;
use rsdag::node::UNARY_OPS;
use rsdag::*;
use std::collections::HashMap;

fn domain_point(op: UnaryOp) -> f64 {
    let inside_unit_interval = matches!(
        op,
        UnaryOp::Tan | UnaryOp::Asin | UnaryOp::Acos | UnaryOp::Atanh
    );
    if inside_unit_interval {
        0.5
    } else {
        1.5
    }
}

fn check_derivatives(
    name: &str,
    ctx: &mut Graph<BigRational>,
    f: ExprId,
    point: &[(SymbolId, f64)],
) {
    let wrt: Vec<SymbolId> = point.iter().map(|&(s, _)| s).collect();
    let reverse = gradient(ctx, f, &wrt);
    let forward: Vec<ExprId> = wrt.iter().map(|&s| differentiate(ctx, f, s)).collect();
    let mut env: HashMap<SymbolId, f64> = point.iter().copied().collect();
    let h = CENTRAL_DIFFERENCE_STEP;
    for (k, &(s, x)) in point.iter().enumerate() {
        env.insert(s, x + h);
        let up = eval(ctx, &[f], &env)[0];
        env.insert(s, x - h);
        let down = eval(ctx, &[f], &env)[0];
        env.insert(s, x);
        let central = (up - down) / (2.0 * h);
        for (mode, d) in [("forward", forward[k]), ("reverse", reverse[k])] {
            let analytic = eval(ctx, &[d], &env)[0];
            assert!(
                (analytic - central).abs() <= FINITE_DIFFERENCE * (1.0 + analytic.abs()),
                "{name} {mode} d/d{k}: analytic {analytic}, central difference {central}"
            );
        }
    }
}

#[test]
fn every_smooth_unary_op_derivative_matches_a_central_difference() {
    let smooth = UNARY_OPS
        .iter()
        .map(|spec| spec.op)
        .filter(|&op| op.is_smooth());
    for op in smooth {
        let mut ctx: Graph<BigRational> = Graph::new();
        let x = ctx.sym("x");
        let f = ctx.unary(op, x);
        let xs = ctx.symbol_id("x");
        let name = format!("{op:?}");
        check_derivatives(&name, &mut ctx, f, &[(xs, domain_point(op))]);
    }
}

#[test]
fn binary_op_and_power_partials_match_a_central_difference() {
    for op in [BinOp::Powf, BinOp::Mod, BinOp::Atan2, BinOp::Hypot] {
        let mut ctx: Graph<BigRational> = Graph::new();
        let a = ctx.sym("a");
        let b = ctx.sym("b");
        let f = ctx.binary(op, a, b);
        let point = [(ctx.symbol_id("a"), 1.5), (ctx.symbol_id("b"), 0.7)];
        check_derivatives(&format!("{op:?}"), &mut ctx, f, &point);
    }
    let mut ctx: Graph<BigRational> = Graph::new();
    let x = ctx.sym("x");
    let f = ctx.pow_i(x, 3);
    let point = [(ctx.symbol_id("x"), 1.5)];
    check_derivatives("pow_i", &mut ctx, f, &point);
}
