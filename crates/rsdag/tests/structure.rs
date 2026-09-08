//! What a block simulator asks about the *shape* of a system rather than
//! about its values: does hierarchy survive, what does an incremental change
//! cost, does the interface order matter, and where is an algebraic loop.

use rsdag::synth::{build, Spec};
use rsdag::{
    substitute_expr, ExprId, FuncId, Graph, Node, Output, OutputRole, ParamRole, Scope, SymbolId,
    Tape, F64,
};

fn expr_of(g: &Graph<F64>, f: FuncId, out: usize) -> ExprId {
    match g.func(f).outputs[out] {
        Output::Expr(e) => e,
        _ => panic!("output {out} of function {f:?} is not an expression"),
    }
}

/// Whether a call survives anywhere under `e`.
fn contains_call(g: &Graph<F64>, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if matches!(g.node(id), Node::Call(..)) {
            return true;
        }
        stack.extend_from_slice(&g.operands(id));
    }
    false
}

fn sym_of(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// A gain block `y = k * u`, as a function with roles.
fn gain(g: &mut Graph<F64>, name: &str) -> FuncId {
    let mut s = Scope::new(g, name);
    let u = s.param_with_role("u", ParamRole::Input { port: 0, elem: 0 });
    let k = s.param_with_role("k", ParamRole::Param);
    let y = s.mul(k, u);
    s.close_with_roles(vec![(OutputRole::Output { port: 0, elem: 0 }, y)])
}

/// Hierarchy is a call graph: a subsystem is a function whose body calls
/// other functions, to any depth, and each instance carries its own
/// parameter values as call arguments.
#[test]
fn subsystems_nest_and_instances_keep_their_parameters() {
    let mut g: Graph<F64> = Graph::new();
    let inner = gain(&mut g, "gain");

    // A subsystem with two instances of the same block in series, each with
    // its own gain: the parameters are arguments, so one definition serves
    // both instances.
    let mut s = Scope::new(&mut g, "chain");
    let u = s.param_with_role("u", ParamRole::Input { port: 0, elem: 0 });
    let k1 = s.param_with_role("k1", ParamRole::Param);
    let k2 = s.param_with_role("k2", ParamRole::Param);
    let mid = s.call(inner, 0, &[u, k1]);
    let out = s.call(inner, 0, &[mid, k2]);
    let chain = s.close_with_roles(vec![(OutputRole::Output { port: 0, elem: 0 }, out)]);

    // A third level: the whole subsystem instantiated inside another one.
    let mut s = Scope::new(&mut g, "system");
    let x = s.param_with_role("x", ParamRole::Input { port: 0, elem: 0 });
    let a = s.konst_f64(2.0);
    let b = s.konst_f64(3.0);
    let y = s.call(chain, 0, &[x, a, b]);
    let sys = s.close_with_roles(vec![(OutputRole::Output { port: 0, elem: 0 }, y)]);

    // The hierarchy is intact in the graph: the system's output is a call.
    assert!(matches!(g.node(expr_of(&g, sys, 0)), Node::Call(..)));

    let params = g.func(sys).params.clone();
    let tape = Tape::compile(&g, &[expr_of(&g, sys, 0)], &params);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[5.0], &mut w, &mut o);
    assert_eq!(o[0], 30.0, "2 * 3 * 5 through two levels of calls");

    // `inline_outputs` opens one level: the system's call becomes the
    // chain's body, which still calls the gain.
    let arg = g.symbol_expr(params[0]);
    let one_level = g.inline_outputs(sys, &[0], &[arg])[0];
    assert!(
        contains_call(&g, one_level),
        "one level of inlining leaves the inner calls"
    );

    // `inline_all` goes to the bottom, which is the static fusion a build
    // does before compiling: no call survives and the value is the same.
    let flat = g.inline_all(&[one_level])[0];
    assert!(!contains_call(&g, flat), "fully inlined");
    let tape = Tape::compile(&g, &[flat], &params);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[5.0], &mut w, &mut o);
    assert_eq!(o[0], 30.0);
}

/// An incremental change costs the cone it touches and nothing else: the
/// graph is hash-consed, so everything the change does not reach is shared
/// with the version before it.
#[test]
fn an_incremental_change_costs_only_its_cone() {
    let mut g: Graph<F64> = Graph::new();
    let mut spec = Spec::new(5).steps(400).params(6);
    let (roots, syms) = build(&mut g, &mut spec);
    let before = g.len();

    // Replace one input by an expression: only the nodes above that input
    // are rebuilt.
    let extra = g.sym("extra");
    let two = g.konst_f64(2.0);
    let replacement = g.mul(two, extra);
    let after_building_the_replacement = g.len();
    let changed = substitute_expr(&mut g, roots[0], syms[0], replacement);
    let grew = g.len() - after_building_the_replacement;

    let cone = g.free_symbols(changed).len();
    assert!(cone > 0);
    assert!(
        grew < before / 2,
        "a one-symbol substitution rebuilt {grew} nodes of {before}"
    );
    // The untouched roots are literally the same ids: nothing was copied.
    let again = substitute_expr(&mut g, roots[0], syms[0], replacement);
    assert_eq!(again, changed, "the same substitution is the same node");
}

/// The interface order is a compile-time choice, not a property of the
/// graph: the same roots compiled against a permuted symbol list give the
/// same values through the permuted inputs.
#[test]
fn permuting_the_interface_is_free() {
    let mut g: Graph<F64> = Graph::new();
    let mut spec = Spec::new(17).steps(200).params(5).outputs(3);
    let (roots, syms) = build(&mut g, &mut spec);
    let row: Vec<f64> = (0..syms.len()).map(|k| 0.1 + 0.3 * k as f64).collect();

    let tape = Tape::compile(&g, &roots, &syms);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&row, &mut w, &mut o);

    // Reverse the input order and feed the values in the same order.
    let perm: Vec<SymbolId> = syms.iter().rev().copied().collect();
    let prow: Vec<f64> = row.iter().rev().copied().collect();
    let ptape = Tape::compile(&g, &roots, &perm);
    let (mut pw, mut po) = (Vec::new(), Vec::new());
    ptape.eval(&prow, &mut pw, &mut po);
    assert!(
        o.iter().zip(&po).all(|(a, b)| a.to_bits() == b.to_bits()),
        "a permuted interface changed the values"
    );

    // Outputs permute the same way, and a subset is just a shorter root list.
    let rev_roots: Vec<ExprId> = roots.iter().rev().copied().collect();
    let rtape = Tape::compile(&g, &rev_roots, &syms);
    let (mut rw, mut ro) = (Vec::new(), Vec::new());
    rtape.eval(&row, &mut rw, &mut ro);
    assert!(o
        .iter()
        .rev()
        .zip(&ro)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// Direct feedthrough per block, and the algebraic loop that falls out of it
/// when two blocks feed each other.
#[test]
fn feedthrough_shows_where_an_algebraic_loop_is() {
    let mut g: Graph<F64> = Graph::new();

    // A block with feedthrough: y = k * u reads its input directly.
    let direct = gain(&mut g, "direct");
    let ft = g.feedthrough(direct);
    assert_eq!(ft[0], vec![true, true], "y reads u and k");

    // A block without: y = x, dx/dt = u. The output reads the state only,
    // so it breaks a loop.
    let mut s = Scope::new(&mut g, "integrator");
    let x = s.param_with_role("x", ParamRole::State { id: 0 });
    let u = s.param_with_role("u", ParamRole::Input { port: 0, elem: 0 });
    let integ = s.close_with_roles(vec![
        (OutputRole::Output { port: 0, elem: 0 }, x),
        (OutputRole::StateDeriv { id: 0 }, u),
    ]);
    let ft = g.feedthrough(integ);
    assert_eq!(
        ft[0],
        vec![true, false],
        "the output reads the state, not u"
    );
    assert_eq!(ft[1], vec![false, true], "the derivative reads u");

    // Two blocks wired into a cycle. The loop is algebraic exactly when
    // every hop has feedthrough, which is a walk over these rows.
    let feeds = |f: FuncId, out: usize| -> bool {
        // Does output `out` read the block's *input* parameter?
        g.func(f).params.iter().enumerate().any(|(k, _)| {
            matches!(g.func(f).param_roles[k], ParamRole::Input { .. }) && g.feedthrough(f)[out][k]
        })
    };
    assert!(feeds(direct, 0), "gain -> gain is an algebraic loop");
    assert!(
        !feeds(integ, 0),
        "an integrator in the cycle breaks the loop"
    );

    // The same question on a composed expression: after splicing the gain
    // into itself, its output still depends on the outer input, which is
    // what a solver would have to iterate on.
    let k = g.sym("k");
    let uu = g.sym("u");
    let once = g.inline_outputs(direct, &[0], &[uu, k])[0];
    let u_sym = sym_of(&g, uu);
    let twice = substitute_expr(&mut g, once, u_sym, once);
    assert!(g.free_symbols(twice).contains(&u_sym));
}
