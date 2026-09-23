//! Two structural questions a block simulator asks of this representation:
//! can several independently built graphs be combined into one, and do
//! events fit without a mechanism of their own.
//!
//! Both are answered here by construction rather than by assertion in a
//! doc: the test builds two blocks separately, splices them into one system
//! whose state derivative is a single graph, and drives an event through a
//! guard output, its time derivative and a state-write effect.

use rustc_hash::FxHashMap as HashMap;

use rsdag::synth::{build, inputs, Spec};
use rsdag::{
    differentiate, substitute, ExprId, Graph, OutputRole, ParamRole, Scope, SymbolId, Tape, F64,
};

/// A first-order lag: `y = x`, `dx/dt = (u - x) / tau`. Built in its own
/// graph, as a block library would.
fn lag(name: &str) -> (Graph<F64>, rsdag::Module<F64>) {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, name);
    let x = s.param_with_role("x", ParamRole::State { id: 0 });
    let u = s.param_with_role("u", ParamRole::Input { port: 0, elem: 0 });
    let tau = s.param_with_role("tau", ParamRole::Param);
    let d = s.sub(u, x);
    let dx = s.div(d, tau);
    s.close_with_roles(vec![
        (OutputRole::StateDeriv { id: 0 }, dx),
        (OutputRole::Output { port: 0, elem: 0 }, x),
    ]);
    let module = g.to_module();
    (g, module)
}

/// A saturating gain: `y = clamp(k * u, -1, 1)`, written with the selects a
/// clamp lowers into.
fn saturation(name: &str) -> (Graph<F64>, rsdag::Module<F64>) {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, name);
    let u = s.param_with_role("u", ParamRole::Input { port: 0, elem: 0 });
    let k = s.param_with_role("k", ParamRole::Param);
    let v = s.mul(k, u);
    let hi = s.konst_f64(1.0);
    let lo = s.konst_f64(-1.0);
    let over = s.cmp(rsdag::CmpOp::Gt, v, hi);
    let capped = s.select(over, hi, v);
    let under = s.cmp(rsdag::CmpOp::Lt, capped, lo);
    let y = s.select(under, lo, capped);
    s.close_with_roles(vec![(OutputRole::Output { port: 0, elem: 0 }, y)]);
    let module = g.to_module();
    (g, module)
}

/// Two blocks built apart, loaded into one graph and spliced: the
/// saturation's output feeds the lag's input, and the result is one
/// expression for `dx/dt` over the system's own symbols.
#[test]
fn separate_blocks_compose_into_one_graph() {
    let (_, lag_mod) = lag("lag");
    let (_, sat_mod) = saturation("sat");

    // One arena for the system. Loading re-interns, so shared structure
    // between the blocks is shared here, and each block's ids are mapped.
    let mut sys: Graph<F64> = Graph::new();
    let lag_map = sys.load_module(&lag_mod).unwrap();
    let sat_map = sys.load_module(&sat_mod).unwrap();

    let (lag_dx, lag_u) = {
        let f = sys.func(lag_map.funcs[0]);
        let dx = match f.outputs[0] {
            rsdag::Output::Expr(e) => e,
            _ => panic!("the lag's state derivative is an expression"),
        };
        (dx, f.params[1])
    };
    let sat_y = match sys.func(sat_map.funcs[0]).outputs[0] {
        rsdag::Output::Expr(e) => e,
        _ => panic!("the saturation's output is an expression"),
    };

    // Splice: the lag's `u` becomes the saturation's output. Everything the
    // saturation reads (its own `u`, `k`) stays a system input.
    let mut wire: HashMap<SymbolId, ExprId> = HashMap::default();
    wire.insert(lag_u, sat_y);
    let dx = substitute(&mut sys, &[lag_dx], &wire)[0];

    // The composed derivative reads exactly the surviving system symbols.
    let free = sys.free_symbols(dx);
    let names: Vec<&str> = free.iter().map(|&s| sys.symbol_name(s)).collect();
    assert!(names.contains(&"x") && names.contains(&"tau") && names.contains(&"k"));

    // Evaluate the composed system: k*u saturates at 1, so with tau = 2 and
    // x = 0.5 the derivative is (1 - 0.5) / 2.
    let order: Vec<SymbolId> = free.iter().copied().collect();
    let tape = Tape::compile(&sys, &[dx], &order);
    let row: Vec<f64> = order
        .iter()
        .map(|&s| match sys.symbol_name(s) {
            "x" => 0.5,
            "tau" => 2.0,
            "k" => 10.0,
            _ => 0.4, // the saturation's own input, driven past the cap
        })
        .collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&row, &mut w, &mut o);
    assert!((o[0] - 0.25).abs() < 1e-15, "dx/dt = {}", o[0]);

    // And the composition differentiates like anything else: below the cap
    // the chain rule carries the gain through.
    let x_sym = *free.iter().find(|&&s| sys.symbol_name(s) == "x").unwrap();
    let ddx = differentiate(&mut sys, dx, x_sym);
    let tape = Tape::compile(&sys, &[ddx], &order);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&row, &mut w, &mut o);
    assert!((o[0] + 0.5).abs() < 1e-15, "d(dx/dt)/dx = {}", o[0]);
}

/// An event needs no mechanism of its own: the crossing function is a
/// `Guard` output, its derivative comes from `differentiate`, and the effect
/// is a function with `StateWrite` outputs.
#[test]
fn an_event_is_a_guard_output_and_a_state_write() {
    let mut g: Graph<F64> = Graph::new();

    // Crossing: the ball hits the floor when its height reaches zero.
    let mut s = Scope::new(&mut g, "bounce");
    let h = s.param_with_role("h", ParamRole::State { id: 0 });
    let v = s.param_with_role("v", ParamRole::State { id: 1 });
    let guard = h;
    let dh = v;
    let gravity = s.konst_f64(-9.81);
    let f = s.close_with_roles(vec![
        (
            OutputRole::Guard {
                id: 0,
                dir: rsdag::Crossing::Falling,
            },
            guard,
        ),
        (OutputRole::StateDeriv { id: 0 }, dh),
        (OutputRole::StateDeriv { id: 1 }, gravity),
    ]);
    let params = g.func(f).params.clone();

    // The roles select the event's parts without any lookup by name.
    let guards = g
        .func(f)
        .outputs_with_role(|r| matches!(r, OutputRole::Guard { .. }));
    assert_eq!(guards.len(), 1);

    // A derivative-based locator wants dg/dt along the flow, which is the
    // graph's own `time_derivative`: dh/dt = v.
    let mut deriv_of: HashMap<SymbolId, ExprId> = HashMap::default();
    let h_sym = params[0];
    let v_sym = params[1];
    deriv_of.insert(h_sym, g.symbol_expr(v_sym));
    let dgdt = rsdag::time_derivative(&mut g, guard, &deriv_of);

    // The effect: a second function writing the reflected velocity.
    let mut e = Scope::new(&mut g, "bounce_effect");
    let ve = e.param_with_role("v", ParamRole::State { id: 1 });
    let restitution = e.konst_f64(-0.8);
    let v_new = e.mul(restitution, ve);
    let eff = e.close_with_roles(vec![(OutputRole::StateWrite { id: 1 }, v_new)]);
    assert_eq!(
        g.func(eff)
            .outputs_with_role(|r| matches!(r, OutputRole::StateWrite { .. }))
            .len(),
        1
    );

    // Everything evaluates in one tape: guard, its rate, the derivatives and
    // the effect.
    let v_new_expr = match g.func(eff).outputs[0] {
        rsdag::Output::Expr(x) => x,
        _ => unreachable!(),
    };
    let tape = Tape::compile(&g, &[guard, dgdt, gravity, v_new_expr], &[h_sym, v_sym]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[0.25f64, -3.0], &mut w, &mut o);
    assert_eq!(o[0], 0.25, "guard value");
    assert_eq!(o[1], -3.0, "guard rate is v");
    assert_eq!(o[2], -9.81, "gravity");
    assert!((o[3] - 2.4).abs() < 1e-15, "reflected velocity {}", o[3]);
}

/// A `Select` is a discontinuity the tape can watch on its own: specializing
/// against a traced region gives a shorter program plus guards that report
/// when the region no longer holds. That report is the event.
#[test]
fn a_region_flip_is_reported_by_the_specialization() {
    let (g, _) = saturation("sat");
    let f = rsdag::FuncId(0);
    let y = match g.func(f).outputs[0] {
        rsdag::Output::Expr(e) => e,
        _ => unreachable!(),
    };
    let params = g.func(f).params.clone();
    let tape = Tape::compile(&g, &[y], &params);

    // Trace inside the linear region and specialize there.
    let (mut w, mut o, mut choices) = (Vec::new(), Vec::new(), Vec::new());
    choices.clear();
    tape.eval_with(&[0.05f64, 10.0], &mut w, &mut o, &mut choices);
    let spec = tape.specialize(&choices, &vec![true; tape.n_selects()]);
    assert!(spec.n_ops() < tape.n_ops(), "the specialization is shorter");

    // Still inside: the guards hold and the value matches the full tape.
    let (mut sw, mut so) = (Vec::new(), Vec::new());
    assert!(spec.eval_checked(&[0.06, 10.0], &mut sw, &mut so));
    tape.eval(&[0.06f64, 10.0], &mut w, &mut o);
    assert_eq!(so[0].to_bits(), o[0].to_bits());

    // Past the cap: the guard reports the flip instead of returning a wrong
    // value, which is exactly the signal an integrator needs.
    assert!(
        !spec.eval_checked(&[0.5, 10.0], &mut sw, &mut so),
        "a crossing into the saturated region must be reported"
    );
}

/// Composition is not a special case: a module loaded twice shares its
/// structure, so two instances of one block cost one copy of the shared
/// subexpressions.
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

    // The shared instance still evaluates like the original.
    let row = inputs(&mut spec.rng(), syms.len());
    let order: Vec<SymbolId> = syms.iter().map(|s| first.symbols[s.0 as usize]).collect();
    let mapped: Vec<ExprId> = roots.iter().map(|r| first.exprs[r.0 as usize]).collect();
    let (mut w1, mut o1) = (Vec::new(), Vec::new());
    Tape::compile(&src, &roots, &syms).eval(&row, &mut w1, &mut o1);
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    Tape::compile(&sys, &mapped, &order).eval(&row, &mut w2, &mut o2);
    assert!(o1.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits()));
}
