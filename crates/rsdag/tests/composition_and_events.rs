//! Two structural questions a block simulator asks of this representation:
//! can several independently built graphs be combined into one, and do
//! events fit without a mechanism of their own.
//!
//! Both are answered here by construction rather than by assertion in a
//! doc: the test builds two blocks separately, splices them into one system
//! whose state derivative is a single graph, and drives an event through a
//! guard output, its time derivative and a state-write effect.

use rsdag::Program;
use rustc_hash::FxHashMap as HashMap;

use rsdag::synth::{build, inputs, Spec};
use rsdag::{
    differentiate, substitute, ExprId, Graph, OutputRole, ParamRole, Scope, SymbolId, Tape, F64,
};

fn first_order_lag(name: &str) -> (Graph<F64>, rsdag::Module<F64>) {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, name);
    let x = s.param("x", ParamRole::State { id: 0 });
    let u = s.param("u", ParamRole::Input { port: 0, elem: 0 });
    let tau = s.param("tau", ParamRole::Param);
    let d = s.sub(u, x);
    let dx = s.div(d, tau);
    s.close([
        (dx, OutputRole::StateDeriv { id: 0 }),
        (x, OutputRole::Output { port: 0, elem: 0 }),
    ]);
    let module = g.to_module();
    (g, module)
}

fn saturating_gain_as_lowered_clamp(name: &str) -> (Graph<F64>, rsdag::Module<F64>) {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, name);
    let u = s.param("u", ParamRole::Input { port: 0, elem: 0 });
    let k = s.param("k", ParamRole::Param);
    let v = s.mul(k, u);
    let hi = s.konst_f64(1.0);
    let lo = s.konst_f64(-1.0);
    let over = s.cmp(rsdag::CmpOp::Gt, v, hi);
    let capped = s.select(over, hi, v);
    let under = s.cmp(rsdag::CmpOp::Lt, capped, lo);
    let y = s.select(under, lo, capped);
    s.close([(y, OutputRole::Output { port: 0, elem: 0 })]);
    let module = g.to_module();
    (g, module)
}

#[test]
fn separate_blocks_compose_into_one_graph() {
    const X: f64 = 0.5;
    const TAU: f64 = 2.0;
    const GAIN: f64 = 10.0;
    const U_PAST_THE_CAP: f64 = 0.4;
    const CAP: f64 = 1.0;
    let (_, lag_mod) = first_order_lag("lag");
    let (_, sat_mod) = saturating_gain_as_lowered_clamp("sat");

    let mut sys: Graph<F64> = Graph::new();
    let lag_map = sys.load_module(&lag_mod).unwrap();
    let sat_map = sys.load_module(&sat_mod).unwrap();

    let (lag_dx, lag_u) = {
        let f = sys.func(lag_map.funcs[0]);
        let dx = match f.outputs()[0] {
            rsdag::Output::Expr(e) => e,
            _ => panic!("the lag's state derivative is an expression"),
        };
        (dx, f.params()[1])
    };
    let sat_y = match sys.func(sat_map.funcs[0]).outputs()[0] {
        rsdag::Output::Expr(e) => e,
        _ => panic!("the saturation's output is an expression"),
    };

    let mut lag_input_from_saturation_output: HashMap<SymbolId, ExprId> = HashMap::default();
    lag_input_from_saturation_output.insert(lag_u, sat_y);
    let dx = substitute(&mut sys, &[lag_dx], &lag_input_from_saturation_output)[0];

    let free = sys.free_symbols(dx);
    let names: Vec<&str> = free.iter().map(|&s| sys.symbol_name(s)).collect();
    assert!(
        names.contains(&"x") && names.contains(&"tau") && names.contains(&"k"),
        "the composed derivative does not read the system symbols x, tau and k"
    );

    let order: Vec<SymbolId> = free.iter().copied().collect();
    let tape = Tape::compile(&sys, &[dx], &order);
    let row: Vec<f64> = order
        .iter()
        .map(|&s| match sys.symbol_name(s) {
            "x" => X,
            "tau" => TAU,
            "k" => GAIN,
            _ => U_PAST_THE_CAP,
        })
        .collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&row, &mut w, &mut o);
    assert_eq!(o[0], (CAP - X) / TAU, "dx/dt");

    let x_sym = *free.iter().find(|&&s| sys.symbol_name(s) == "x").unwrap();
    let ddx = differentiate(&mut sys, dx, x_sym);
    let tape = Tape::compile(&sys, &[ddx], &order);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&row, &mut w, &mut o);
    assert_eq!(o[0], -1.0 / TAU, "d(dx/dt)/dx");
}

#[test]
fn an_event_is_a_guard_output_and_a_state_write() {
    let mut g: Graph<F64> = Graph::new();

    let mut s = Scope::new(&mut g, "bounce");
    let h = s.param("h", ParamRole::State { id: 0 });
    let v = s.param("v", ParamRole::State { id: 1 });
    let guard = h;
    let dh = v;
    let gravity = s.konst_f64(-9.81);
    let guard_role = OutputRole::Guard {
        id: 0,
        dir: rsdag::Crossing::Falling,
    };
    let f = s.close([
        (guard, guard_role),
        (dh, OutputRole::StateDeriv { id: 0 }),
        (gravity, OutputRole::StateDeriv { id: 1 }),
    ]);
    let params = g.func(f).params().to_vec();

    let guards = g
        .func(f)
        .exprs_with_role(|r| matches!(r, OutputRole::Guard { .. }));
    assert_eq!(guards.len(), 1, "the role does not select the guard");

    let h_sym = params[0];
    let v_sym = params[1];
    let dgdh = rsdag::differentiate(&mut g, guard, h_sym);
    let v = g.symbol_expr(v_sym);
    let dgdt = g.mul(dgdh, v);

    let mut e = Scope::new(&mut g, "bounce_effect");
    let ve = e.param("v", ParamRole::State { id: 1 });
    let restitution = e.konst_f64(-0.8);
    let v_new = e.mul(restitution, ve);
    let eff = e.close([(v_new, OutputRole::StateWrite { id: 1 })]);
    assert_eq!(
        g.func(eff)
            .exprs_with_role(|r| matches!(r, OutputRole::StateWrite { .. }))
            .len(),
        1
    );

    let v_new_expr = match g.func(eff).outputs()[0] {
        rsdag::Output::Expr(x) => x,
        _ => unreachable!(),
    };
    let tape = Tape::compile(&g, &[guard, dgdt, gravity, v_new_expr], &[h_sym, v_sym]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[0.25f64, -3.0], &mut w, &mut o);
    assert_eq!(o[0], 0.25, "guard value");
    assert_eq!(o[1], -3.0, "guard rate is v");
    assert_eq!(o[2], -9.81, "gravity");
    assert_eq!(o[3], -0.8 * -3.0, "reflected velocity");
}

#[test]
fn a_region_flip_is_reported_by_the_specialization() {
    let (g, _) = saturating_gain_as_lowered_clamp("sat");
    let f = rsdag::FuncId(0);
    let y = match g.func(f).outputs()[0] {
        rsdag::Output::Expr(e) => e,
        _ => unreachable!(),
    };
    let params = g.func(f).params().to_vec();
    let tape = Tape::compile(&g, &[y], &params);

    let traced_in_the_linear_region = [0.05f64, 10.0];
    let still_in_the_linear_region = [0.06f64, 10.0];
    let past_the_cap = [0.5f64, 10.0];
    let (mut w, mut o, mut choices) = (Vec::new(), Vec::new(), Vec::new());
    choices.clear();
    tape.eval_with(&traced_in_the_linear_region, &mut w, &mut o, &mut choices);
    let spec = tape.specialize(&choices, &vec![true; tape.n_selects()]);
    assert!(spec.n_ops() < tape.n_ops(), "the specialization is shorter");

    let (mut sw, mut so) = (Vec::new(), Vec::new());
    assert!(
        spec.eval_checked(&still_in_the_linear_region, &mut sw, &mut so),
        "a guard fails inside the traced region"
    );
    tape.eval(&still_in_the_linear_region, &mut w, &mut o);
    assert_eq!(
        so[0].to_bits(),
        o[0].to_bits(),
        "the specialization differs from the full tape inside the traced region"
    );

    assert!(
        !spec.eval_checked(&past_the_cap, &mut sw, &mut so),
        "a crossing into the saturated region must be reported"
    );
}

#[test]
fn loading_a_block_twice_shares_its_structure() {
    let mut spec = Spec::new(11).steps(120).params(3);
    let mut src: Graph<F64> = Graph::new();
    let (roots, syms) = build(&mut src, &mut spec);
    let module = src.to_module();

    let mut sys: Graph<F64> = Graph::new();
    let first = sys.load_module(&module).unwrap();
    let after_first = sys.len();
    let second = sys.load_module(&module).unwrap();
    assert_eq!(sys.len(), after_first, "the second instance added nodes");
    assert_eq!(first.exprs, second.exprs);

    let row = inputs(&mut spec.rng(), syms.len());
    let order: Vec<SymbolId> = syms.iter().map(|s| first.symbols[s.0 as usize]).collect();
    let mapped: Vec<ExprId> = roots.iter().map(|r| first.exprs[r.0 as usize]).collect();
    let (mut w1, mut o1) = (Vec::new(), Vec::new());
    Tape::compile(&src, &roots, &syms).eval(&row, &mut w1, &mut o1);
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    Tape::compile(&sys, &mapped, &order).eval(&row, &mut w2, &mut o2);
    assert!(
        o1.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits()),
        "the shared instance evaluates differently from the original"
    );
}
