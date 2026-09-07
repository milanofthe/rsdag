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
pub mod tape;
// Expression substitution is exposed only through the curated `substitute*`
// re-exports below, not as a module path.
pub(crate) mod transform;

pub use autodiff::{differentiate, gradient, hessian, jacobian, sparsity, time_derivative};
pub use display::to_string;
pub use eval::{eval, eval_named, eval_real, eval_real_all};
pub use extern_fn::ExternBundle;
pub use field::{ratio_powi, Field, F64};
pub use func::{CompiledBody, Func, FuncId, FunctionBody, Output, OutputId};
pub use graph::Context;
pub use node::{ArgList, CmpOp, ConstId, ExprId, Node, Operands, ReduceOp, SymbolId, UnaryOp};
pub use tape::{SchedulePolicy, SpecializedTape, Tape, TapeVisitor};
pub use transform::{substitute, substitute_expr, substitute_many, substitute_many_all};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_field_builds_folds_and_evaluates() {
        let mut g: Context<F64> = Context::new();
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
        let mut ctx: Context = Context::new();
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
        let mut ctx: Context = Context::new();
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
        let mut ctx: Context = Context::new();
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
        let mut ctx: Context = Context::new();
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
