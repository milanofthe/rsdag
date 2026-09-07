//! SANE symbolic core: a hash-consed expression DAG over free symbols and
//! exact rational constants. This is the shared substrate for all analysis
//! modes (linear-symbolic, DAE extraction, nonlinear transient).

pub mod autodiff;
pub mod display;
pub mod eval;
pub mod extern_fn;
pub mod field;
pub mod func;
pub mod graph;
pub mod node;
pub mod role;
pub mod simplify;
pub mod tape;
// Expression substitution is exposed only through the curated `substitute*`
// re-exports below, not as a module path.
pub(crate) mod transform;

pub use autodiff::{differentiate, gradient, hessian, jacobian, sparsity, time_derivative};
pub use display::to_string;
pub use eval::{eval, eval_named, eval_real, eval_real_all};
pub use extern_fn::ExternBundle;
pub use field::{ratio_powi, Field, F64};
pub use func::{CompiledBody, FuncId, Function, FunctionBody, Output, OutputId};
pub use graph::Graph;
pub use node::{
    binary_f64, unary_f64, ArgList, BinOp, CmpOp, ConstId, ExprId, Node, Operands, ReduceOp,
    SymbolId, UnaryOp,
};
pub use role::{OutputRole, ParamRole};
pub use simplify::rebuild;
pub use tape::{SchedulePolicy, SpecializedTape, Tape, TapeVisitor};
pub use transform::{substitute, substitute_expr, substitute_many, substitute_many_all};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_select_jacobian_blocks() {
        // A one-state system: residual r = x' - (-k x + u), output y = 2 x.
        let mut g: Graph = Graph::new();
        let (x, xd, u, k) = (g.sym("x"), g.sym("xd"), g.sym("u"), g.sym("k"));
        let kx = g.mul(k, x);
        let rhs = g.sub(u, kx);
        let r = g.sub(xd, rhs);
        let two = g.konst_int(2);
        let y = g.mul(two, x);
        let f = g.close("sys", vec![r, y]);
        let params = g.func(f).params.clone();
        assert_eq!(params.len(), 4);
        for (i, s) in params.iter().enumerate() {
            let role = match g.symbol_name(*s) {
                "x" => ParamRole::State { id: 0 },
                "xd" => ParamRole::StateDot { id: 0 },
                "u" => ParamRole::Input { port: 0, elem: 0 },
                _ => ParamRole::Param,
            };
            g.set_param_role(f, i as u32, role);
        }
        g.set_output_role(f, 0, OutputRole::Residual { id: 0 });
        g.set_output_role(f, 1, OutputRole::Output { port: 0, elem: 0 });
        let jx = g.jacobian_by_role(
            f,
            |o| matches!(o, OutputRole::Residual { .. }),
            |p| matches!(p, ParamRole::State { .. }),
        );
        assert_eq!(jx.len(), 1);
        let jy = g.jacobian_by_role(
            f,
            |o| matches!(o, OutputRole::Output { .. }),
            |p| matches!(p, ParamRole::State { .. }),
        );
        assert_eq!(jy.len(), 1);
        let (of, wrt, k) = jy[0];
        assert_eq!(
            g.func(f).output_roles[k as usize],
            OutputRole::Derivative { of, wrt }
        );
        match g.func(f).outputs[k as usize] {
            Output::Expr(e) => assert_eq!(g.const_f64(e), Some(2.0)),
            _ => panic!("expected an expression"),
        }
    }

    #[test]
    fn f64_field_builds_folds_and_evaluates() {
        let mut g: Graph<F64> = Graph::new();
        let x = g.sym("x");
        let half = g.konst_f64(0.5);
        let quarter = g.konst_f64(0.25);
        let c = g.mul(half, quarter); // folds in f64
        assert_eq!(g.const_f64(c), Some(0.125));
        let e = g.add(x, c);
        let tape = Tape::compile(&g, &[e], &[SymbolId(0)]);
        let (mut work, mut out) = (Vec::new(), Vec::new());
        tape.eval(&[2.0], &mut work, &mut out);
        assert_eq!(out[0], 2.125);
        assert_eq!(to_string(&g, e), "(x + 0.125)");
    }
    use num_complex::Complex64;

    #[test]
    fn hash_consing_shares_identical_subexpressions() {
        let mut ctx: Graph = Graph::new();
        let a = ctx.sym("a");
        let b = ctx.sym("b");
        let s1 = ctx.add(a, b);
        let s2 = ctx.add(b, a); // commutative -> same node after ordering
        assert_eq!(s1, s2);

        // (a+b)*(a+b): the inner sum is interned exactly once.
        let before = ctx.len();
        let _prod = ctx.mul(s1, s2);
        // Only the product node is new.
        assert_eq!(ctx.len(), before + 1);
    }

    #[test]
    fn folds_constants_and_identities() {
        let mut ctx: Graph = Graph::new();
        let a = ctx.sym("a");
        let zero = ctx.zero();
        let one = ctx.one();

        assert_eq!(ctx.add(a, zero), a);
        assert_eq!(ctx.mul(a, one), a);
        let az = ctx.mul(a, zero);
        assert!(ctx.is_zero(az));

        let two = ctx.konst_int(2);
        let three = ctx.konst_int(3);
        let six = ctx.mul(two, three);
        let q = ctx.div(six, six);
        assert!(ctx.is_one(q));
    }

    #[test]
    fn evaluates_admittance_like_expression() {
        // Y = 1/R + s*C, a capacitor-in-parallel-with-resistor admittance.
        let mut ctx: Graph = Graph::new();
        let r = ctx.sym("R");
        let c = ctx.sym("C");
        let s = ctx.sym("s");
        let g = ctx.recip(r);
        let sc = ctx.mul(s, c);
        let y = ctx.add(g, sc);

        // R=1k, C=1u, omega=1000 rad/s  ->  Y = 1e-3 + j*1e-3
        let val = eval_named(
            &mut ctx,
            y,
            &[
                ("R", Complex64::new(1000.0, 0.0)),
                ("C", Complex64::new(1e-6, 0.0)),
                ("s", Complex64::new(0.0, 1000.0)),
            ],
        );
        assert!((val.re - 1e-3).abs() < 1e-12);
        assert!((val.im - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn unary_folding_and_eval() {
        let mut ctx: Graph = Graph::new();
        // exp(0) = 1, ln(1) = 0 fold structurally.
        let z = ctx.zero();
        let o = ctx.one();
        let e0 = ctx.exp(z);
        let l1 = ctx.ln(o);
        assert!(ctx.is_one(e0));
        assert!(ctx.is_zero(l1));

        // exp(ln(x)) evaluates to x; diode-like exp(v/Vt) evaluates correctly.
        let v = ctx.sym("v");
        let vt = ctx.sym("Vt");
        let arg = ctx.div(v, vt);
        let e = ctx.exp(arg);
        let got = eval_named(
            &mut ctx,
            e,
            &[
                ("v", Complex64::new(0.5, 0.0)),
                ("Vt", Complex64::new(0.025, 0.0)),
            ],
        );
        let want = (0.5_f64 / 0.025).exp();
        assert!((got.re - want).abs() <= want * 1e-12);
    }
}
