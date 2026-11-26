mod common;
use common::constants::AGAINST_LIBM;
use rsdag::*;

#[test]
fn charge_form_rows_select_by_role() {
    // One row in charge form: i(x) = x/r, q(x) = c x (an RC node), so
    // dF/dx' = dq/dx is the derivative of the charge output.
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "rc");
    let x = s.param("x", ParamRole::State { id: 0 });
    let r = s.param("r", ParamRole::Param);
    let c = s.param("c", ParamRole::Param);
    let i = s.div(x, r);
    let q = s.mul(c, x);
    let f = s.close([
        (i, OutputRole::Residual { id: 0 }),
        (q, OutputRole::Charge { id: 0 }),
    ]);
    assert_eq!(
        g.func(f)
            .exprs_with_role(|o| matches!(o, OutputRole::Charge { .. })),
        vec![(q, OutputRole::Charge { id: 0 })]
    );
    let x = match g.node(x) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    assert_eq!(differentiate(&mut g, q, x), c);
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
    tape.eval(&[2.0f64], &mut work, &mut out);
    assert_eq!(out[0], 2.125);
    assert_eq!(to_string(&g, e), "(x + 0.125)");
}

use rsdag::BigRational;

#[test]
fn hash_consing_shares_identical_subexpressions() {
    let mut ctx: Graph<BigRational> = Graph::new();
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
    let mut ctx: Graph<BigRational> = Graph::new();
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
    assert_eq!(q, ctx.one());
}

#[test]
fn unary_folding_and_eval() {
    let mut ctx: Graph<BigRational> = Graph::new();
    // exp(0) = 1, ln(1) = 0 fold structurally.
    let z = ctx.zero();
    let o = ctx.one();
    let e0 = ctx.exp(z);
    let l1 = ctx.ln(o);
    assert_eq!(e0, ctx.one());
    assert!(ctx.is_zero(l1));

    // exp(ln(x)) evaluates to x; diode-like exp(v/Vt) evaluates correctly.
    let v = ctx.sym("v");
    let vt = ctx.sym("Vt");
    let arg = ctx.div(v, vt);
    let e = ctx.exp(arg);
    let sym = |e| match ctx.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let env: std::collections::HashMap<SymbolId, f64> =
        [(sym(v), 0.5), (sym(vt), 0.025)].into_iter().collect();
    let got = eval(&ctx, &[e], &env)[0];
    let want = (0.5_f64 / 0.025).exp();
    assert!((got - want).abs() <= AGAINST_LIBM * want);
}

/// Printing is linear in the graph: a small expression is infix, a large
/// or deeply shared one a listing of its nodes, however deep.
#[test]
fn printing_a_shared_or_deep_dag_stays_linear() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let mut e = x;
    for _ in 0..40 {
        e = g.mul(e, e);
    }
    let text = rsdag::to_string(&g, e);
    assert!(text.lines().count() == 41, "{text}");
    let y = g.sym("y");
    let small = g.add(x, y);
    assert_eq!(rsdag::to_string(&g, small), "(x + y)");
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            let mut acc = x;
            for _ in 0..100_000 {
                acc = g.sin(acc);
            }
            assert!(rsdag::to_string(&g, acc).lines().count() > 100_000);
        })
        .unwrap()
        .join()
        .unwrap();
}

/// A rule substitution renames every free symbol it maps and keeps the
/// rest; a pin substitutes and specializes the calls it makes constant.
#[test]
fn substitute_with_a_rule_and_pin_to_constants() {
    let mut g: Graph<F64> = Graph::new();
    let (a, b) = (g.sym("a"), g.sym("b"));
    let f = g.mul(a, b);
    let out = substitute_with(&mut g, &[f], |g, s| {
        let name = format!("X.{}", g.symbol_name(s));
        (name != "X.b").then(|| g.sym(&name))
    });
    let xa = g.sym("X.a");
    assert_eq!(out[0], g.mul(xa, b));
    // a function of (p, q) called with q pinned to 2
    let mut s = Scope::new(&mut g, "f");
    let pe = s.param("p", ParamRole::Free);
    let qe = s.param("q", ParamRole::Free);
    let body = s.mul(pe, qe);
    let func = s.close([(body, OutputRole::Plain)]);
    let call = g.call(func, 0, &[a, b]);
    let two = g.konst_f64(2.0);
    let sb = g.as_symbol(b).unwrap();
    let map = [(sb, two)].into_iter().collect();
    let pinned = pin(&mut g, &[call], &map)[0];
    let sa = g.as_symbol(a).unwrap();
    let env = [(sa, 3.0)].into_iter().collect();
    assert_eq!(eval(&g, &[pinned], &env), vec![6.0]);
    assert!(!g.free_symbols(pinned).contains(&sb));
}

/// A signature without some roles keeps the others in order; a function
/// lists its expression outputs of a role.
#[test]
fn signature_without_and_outputs_of_a_role() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.symbol_id("x");
    let mut s = Scope::new(&mut g, "rows");
    let xe = s.param(x, ParamRole::State { id: 0 });
    let ne = s.param("n", ParamRole::Noise { id: 0 });
    let i = s.add(xe, ne);
    let f = s.close([(i, OutputRole::Residual { id: 0 }), (xe, OutputRole::Plain)]);
    let sig = Signature::of(g.func(f)).without(|r| matches!(r, ParamRole::Noise { .. }));
    assert_eq!(sig.syms, vec![x]);
    let rows = g
        .func(f)
        .exprs_with_role(|r| matches!(r, OutputRole::Residual { .. }));
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, i);
}

/// A function's parameters: the ones asked for, in order, with their roles,
/// then the other symbols its outputs read by symbol, `Free`; a global is
/// none of them.
#[test]
fn params_in_order_then_the_rest_and_no_globals() {
    let mut g: Graph<F64> = Graph::new();
    let (a, b, c, d) = (g.sym("a"), g.sym("b"), g.sym("ns.c"), g.sym("glob"));
    let ab = g.mul(a, b);
    let cd = g.add(c, d);
    let out = g.add(ab, cd);
    let [sa, sb, sc, sd] = ["a", "b", "ns.c", "glob"].map(|n| g.symbol_id(n));
    let mut s = Scope::new(&mut g, "f");
    s.param(sb, ParamRole::Free);
    s.param(sa, ParamRole::Param);
    s.global(sd);
    let f = s.close([(out, OutputRole::Plain)]);
    assert_eq!(g.func(f).params(), &[sb, sa, sc]);
    assert_eq!(
        g.func(f).param_roles(),
        &[ParamRole::Free, ParamRole::Param, ParamRole::Free]
    );
}

/// A partial application binds what the rule binds and passes the rest in
/// order: a call through it is the call with every argument.
#[test]
fn partial_binds_and_passes_in_order() {
    let mut g: Graph<F64> = Graph::new();
    let [p, q, r] = ["p", "q", "r"].map(|n| g.symbol_id(n));
    let mut s = Scope::new(&mut g, "f");
    let [pe, qe, re] = [p, q, r].map(|p| s.param(p, ParamRole::Free));
    let pq = s.mul(pe, qe);
    let body = s.sub(pq, re);
    let f = s.close([(body, OutputRole::Plain)]);
    let (two, x, y) = (g.konst_f64(2.0), g.sym("x"), g.sym("y"));
    let (b, args) = g.partial(f, |_, s| {
        if s == q {
            Arg::Bind(two)
        } else if s == p {
            Arg::Pass(x)
        } else {
            Arg::Pass(y)
        }
    });
    assert_eq!(args, vec![x, y]);
    let call = g.calls_bound(b, &[0], &args)[0];
    let env = [
        (g.as_symbol(x).unwrap(), 3.0),
        (g.as_symbol(y).unwrap(), 1.0),
    ]
    .into_iter()
    .collect();
    assert_eq!(eval(&g, &[call], &env), vec![5.0]);
}

/// A call of an output no argument moves is the output's value.
#[test]
fn a_constant_output_calls_as_its_value() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "f");
    let pe = s.param("p", ParamRole::Free);
    let three = s.konst_f64(3.0);
    let f = s.close([(pe, OutputRole::Plain), (three, OutputRole::Plain)]);
    let x = g.sym("x");
    let calls = g.calls(f, &[0, 1], &[x]);
    assert!(matches!(g.node(calls[0]), Node::Call(..)));
    assert_eq!(calls[1], three);
}

/// A symbol only a called function reads is that function's: no parameter
/// of a function over the call.
#[test]
fn a_called_bodys_symbols_are_not_the_callers_params() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "inner");
    let pe = s.param("p", ParamRole::Free);
    let glob = s.global("glob");
    let body = s.mul(pe, glob);
    let inner = s.close([(body, OutputRole::Plain)]);
    let x = g.sym("x");
    let call = g.call(inner, 0, &[x]);
    let sx = g.as_symbol(x).unwrap();
    let outer = Scope::new(&mut g, "outer").close([(call, OutputRole::Plain)]);
    assert_eq!(g.func(outer).params(), &[sx]);
}

/// Symbols by name and by node, and the terms of a sum.
#[test]
fn symbols_and_summands() {
    let mut g: Graph<F64> = Graph::new();
    assert_eq!(g.find_symbol("a"), None);
    let a = g.symbol_id("a");
    assert_eq!(g.find_symbol("a"), Some(a));
    let (ea, b) = (g.symbol_expr(a), g.sym("b"));
    let s = g.reduce(ReduceOp::Sum, vec![ea, b]);
    assert_eq!(g.summands(s).len(), 2);
    let zero = g.zero();
    assert!(g.summands(zero).is_empty());
    assert_eq!(g.summands(ea), vec![ea]);
    assert_eq!(g.as_symbol(s), None);
    assert_eq!(g.as_symbol(ea), Some(a));
}

/// `Tape::reads` lists the inputs the ops of a range read, each once and
/// ascending: those of every operand, the calls' per instance included.
#[test]
fn reads_are_the_inputs_the_operands_name() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "dev");
    let v = s.param("v", ParamRole::State { id: 0 });
    let p = s.param("p", ParamRole::Param);
    let e = s.exp(p);
    let o = s.mul(v, e);
    let f = s.close([(o, OutputRole::Plain)]);
    let (mut syms, mut pure) = (Vec::new(), Vec::new());
    let mut input = |g: &mut Graph<F64>, name: &str, is_pure: bool| {
        let e = g.sym(name);
        let Node::Symbol(s) = *g.node(e) else {
            unreachable!()
        };
        syms.push(s);
        pure.push(is_pure);
        e
    };
    // the instances' inputs listed out of order, a parameter two share
    let shared = input(&mut g, "p_shared", true);
    let mut roots = Vec::new();
    for i in (0..5).rev() {
        let x = input(&mut g, &format!("x{i}"), false);
        let q = match i % 2 {
            0 => shared,
            _ => input(&mut g, &format!("p{i}"), true),
        };
        roots.push(g.call(f, 0, &[x, q]));
    }
    let y = input(&mut g, "y", false);
    let t = g.sin(y);
    roots.push(g.add(roots[0], t));
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let n = tape.n_ops();
    for range in [0..tape.prolog_len(), tape.prolog_len()..n, 0..n] {
        let mut want: Vec<u32> = Vec::new();
        for i in range.clone() {
            tape.for_each_operand(i, |k| want.extend(rsdag::tape::input_index(k)));
        }
        want.sort_unstable();
        want.dedup();
        assert_eq!(tape.reads(range.clone()), want, "{range:?}");
    }
}
