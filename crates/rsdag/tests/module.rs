use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::*;
use rsdag::{Tape, F64};

#[test]
fn a_module_round_trips_bit_exactly() {
    for seed in 0..24u64 {
        let mut spec = Spec::new(seed)
            .steps(60 + 10 * (seed as usize % 6))
            .params(4);
        spec = match seed % 3 {
            0 => spec.vocab(Vocabulary::Ring).max_list(20),
            1 => spec.vocab(Vocabulary::Elementary),
            _ => spec.vocab(Vocabulary::Full),
        };
        let mut g: Graph<F64> = Graph::new();
        let (roots, syms) = build(&mut g, &mut spec);
        let f = Scope::new(&mut g, "f").close(roots.iter().map(|&e| (e, OutputRole::Plain)));

        let module = g.to_module();
        let (loaded, map) = Graph::from_module(&module).unwrap();
        assert_eq!(map.funcs[f.0 as usize], f, "seed {seed}: function ids");

        let roots2: Vec<ExprId> = roots.iter().map(|e| map.exprs[e.0 as usize]).collect();
        let syms2: Vec<SymbolId> = syms.iter().map(|s| map.symbols[s.0 as usize]).collect();
        let row = inputs(&mut spec.rng(), syms.len());
        let (mut w, mut o) = (Vec::new(), Vec::new());
        Tape::compile(&g, &roots, &syms).eval(&row, &mut w, &mut o);
        let (mut w2, mut o2) = (Vec::new(), Vec::new());
        Tape::compile(&loaded, &roots2, &syms2).eval(&row, &mut w2, &mut o2);
        assert!(
            o.iter()
                .zip(&o2)
                .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
            "seed {seed}: {o:?} vs {o2:?}"
        );
        assert_eq!(
            loaded.to_module(),
            module,
            "seed {seed}: second module differs"
        );
    }
}

#[test]
fn loading_into_a_populated_graph_shares_nodes() {
    let mut a: Graph<F64> = Graph::new();
    let mut spec = Spec::new(9).steps(80).params(3);
    let (roots, syms) = build(&mut a, &mut spec);
    let module = a.to_module();

    let (mut b, first) = Graph::from_module(&module).unwrap();
    let before = b.len();
    let map = b.load_module(&module).unwrap();
    assert_eq!(b.len(), before, "reloading a module added nodes");
    assert_eq!(map.exprs, first.exprs, "reloading a module moved the ids");
    for (k, r) in roots.iter().enumerate() {
        assert_eq!(map.exprs[r.0 as usize], *r, "root {k} moved");
    }
    for (k, s) in syms.iter().enumerate() {
        assert_eq!(map.symbols[s.0 as usize], *s, "symbol {k} moved");
    }
    let _ = &a;
}

#[cfg(feature = "serde")]
#[test]
fn a_module_survives_json_with_its_constants_bit_exact() {
    let mut g: Graph<F64> = Graph::new();
    let mut spec = Spec::new(3).steps(120).params(4).vocab(Vocabulary::Full);
    let (mut roots, syms) = build(&mut g, &mut spec);
    let shortest_decimal_parses_back_to_its_neighbour = 1.106704f64.next_up();
    let values_a_text_format_mangles = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        shortest_decimal_parses_back_to_its_neighbour,
    ];
    for v in values_a_text_format_mangles {
        let k = g.konst_f64(v);
        roots.push(k);
    }
    Scope::new(&mut g, "f").close(roots.iter().map(|&e| (e, OutputRole::Plain)));
    let module = g.to_module();
    let text = serde_json::to_string(&module).expect("serialize");
    let back: Module<F64> = serde_json::from_str(&text).expect("deserialize");
    assert_eq!(back, module);

    let (loaded, map) = Graph::from_module(&back).unwrap();
    let roots2: Vec<ExprId> = roots.iter().map(|e| map.exprs[e.0 as usize]).collect();
    let syms2: Vec<SymbolId> = syms.iter().map(|s| map.symbols[s.0 as usize]).collect();
    let row = inputs(&mut spec.rng(), syms.len());
    let (mut w, mut o) = (Vec::new(), Vec::new());
    Tape::compile(&g, &roots, &syms).eval(&row, &mut w, &mut o);
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    Tape::compile(&loaded, &roots2, &syms2).eval(&row, &mut w2, &mut o2);
    assert!(o.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn calls_and_nested_functions_round_trip() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = rsdag::Scope::new(&mut g, "gain");
    let u = s.param("u", ParamRole::Free);
    let k = s.param("k", ParamRole::Free);
    let y = s.mul(k, u);
    let gain = s.close([(y, OutputRole::Plain)]);

    let mut s = rsdag::Scope::new(&mut g, "chain");
    let x = s.param("x", ParamRole::Free);
    let a = s.konst_f64(2.0);
    let b = s.konst_f64(3.0);
    let mid = s.call(gain, 0, &[x, a]);
    let out = s.call(gain, 0, &[mid, b]);
    let e = s.exp(out);
    let chain = s.close([(e, OutputRole::Plain)]);
    let root = match g.func(chain).outputs()[0] {
        rsdag::Output::Expr(e) => e,
        _ => unreachable!(),
    };
    let params = g.func(chain).params().to_vec();

    let module = g.to_module();
    let (loaded, map) = Graph::from_module(&module).unwrap();
    assert_eq!(map.funcs.len(), 2);
    let root2 = map.exprs[root.0 as usize];
    let params2: Vec<SymbolId> = params.iter().map(|p| map.symbols[p.0 as usize]).collect();

    let (mut w, mut o) = (Vec::new(), Vec::new());
    Tape::compile(&g, &[root], &params).eval(&[0.5f64], &mut w, &mut o);
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    Tape::compile(&loaded, &[root2], &params2).eval(&[0.5f64], &mut w2, &mut o2);
    assert_eq!(
        o[0].to_bits(),
        o2[0].to_bits(),
        "exp(2 * 3 * 0.5) through two calls"
    );
    assert_eq!(loaded.to_module(), module);
}

struct Pair;
impl rsdag::ExternBundle for Pair {
    fn n_outputs(&self) -> usize {
        2
    }
    fn call_into(&self, args: &[f64], _work: &mut [f64], out: &mut [f64]) {
        out[0] = args[0] + args[1];
        out[1] = args[0] * args[1];
    }
}

#[test]
fn an_extern_function_round_trips_with_its_body_resolved_by_name() {
    use std::sync::Arc;
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let y = g.sym("y");
    let f = g.define_extern_func(
        "pair",
        &[ParamRole::Free; 2],
        Arc::new(Pair),
        [
            rsdag::Output::Slot(0),
            rsdag::Output::Slot(1),
            rsdag::Output::Zero,
        ]
        .map(|o| (o, OutputRole::Plain)),
    );
    let sum = g.call(f, 0, &[x, y]);
    let prod = g.call(f, 1, &[x, y]);
    let zero = g.call(f, 2, &[x, y]);
    let root = {
        let t = g.add(sum, prod);
        g.add(t, zero)
    };
    let syms: Vec<SymbolId> = [x, y]
        .iter()
        .map(|&e| match g.node(e) {
            rsdag::Node::Symbol(s) => *s,
            _ => unreachable!(),
        })
        .collect();

    let module = g.to_module();
    assert_eq!(module.funcs[0].extern_body.as_deref(), Some("pair"));

    let mut loaded: Graph<F64> = Graph::new();
    let map = loaded
        .load_module_with(&module, |name| {
            assert_eq!(name, "pair");
            Some(Arc::new(Pair) as Arc<dyn rsdag::ExternBundle>)
        })
        .unwrap();
    let root2 = map.exprs[root.0 as usize];
    let syms2: Vec<SymbolId> = syms.iter().map(|s| map.symbols[s.0 as usize]).collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    Tape::compile(&loaded, &[root2], &syms2).eval(&[2.0f64, 5.0], &mut w, &mut o);
    assert_eq!(o[0], 7.0 + 10.0 + 0.0);
    assert_eq!(loaded.to_module(), module);
}

#[test]
fn derivatives_demanded_after_a_call_round_trip() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let y = g.sym("y");
    let body = {
        let s = g.sin(x);
        g.mul(s, y)
    };
    let f = Scope::new(&mut g, "f").close([(body, OutputRole::Plain)]);
    let (a, b) = (g.sym("a"), g.sym("b"));
    let c = g.call(f, 0, &[a, b]);
    let syms = [SymbolId(2), SymbolId(3)];
    let da = differentiate(&mut g, c, syms[0]);
    let db = differentiate(&mut g, c, syms[1]);
    let n_out = g.func(f).outputs().len();
    assert_eq!(n_out, 3, "f and its two partials");

    let module = g.to_module();
    let (mut loaded, map) = Graph::from_module(&module).unwrap();
    assert_eq!(loaded.to_module(), module);
    let f2 = map.funcs[f.0 as usize];
    let a2 = map.symbols[syms[0].0 as usize];
    differentiate(&mut loaded, map.exprs[c.0 as usize], a2);
    assert_eq!(
        loaded.func(f2).outputs().len(),
        n_out,
        "no output derived twice"
    );

    let roots2: Vec<ExprId> = [da, db].iter().map(|e| map.exprs[e.0 as usize]).collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    Tape::compile(&g, &[da, db], &syms).eval(&[0.3f64, 1.7], &mut w, &mut o);
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    Tape::compile(&loaded, &roots2, &syms).eval(&[0.3f64, 1.7], &mut w2, &mut o2);
    assert_eq!(o, o2);
}

#[test]
fn a_declared_extern_derivative_survives_a_reload() {
    use std::sync::Arc;
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let y = g.sym("y");
    let f = g.define_extern_func(
        "pair",
        &[ParamRole::Free; 2],
        Arc::new(Pair),
        [(rsdag::Output::Slot(0), OutputRole::Plain)],
    );
    let slot_a_times_b_as_a_stand_in = rsdag::Output::Slot(1);
    let k = g.declare_derivative(f, 0, 0, slot_a_times_b_as_a_stand_in);
    let sum = g.call(f, 0, &[x, y]);

    let module = g.to_module();
    let mut loaded: Graph<F64> = Graph::new();
    let map = loaded
        .load_module_with(&module, |_| {
            Some(Arc::new(Pair) as Arc<dyn rsdag::ExternBundle>)
        })
        .unwrap();
    let f2 = map.funcs[f.0 as usize];
    let role = loaded.func(f2).output_roles()[k as usize];
    assert_eq!(role, OutputRole::Derivative { of: 0, wrt: 0 });
    let syms = [SymbolId(0), SymbolId(1)];
    let d = differentiate(&mut loaded, map.exprs[sum.0 as usize], syms[0]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    Tape::compile(&loaded, &[d], &syms).eval(&[2.0f64, 5.0], &mut w, &mut o);
    assert_eq!(o[0], 10.0);
}

#[test]
fn loading_a_module_with_externs_without_bodies_says_so() {
    use std::sync::Arc;
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let f = g.define_extern_func(
        "pair",
        &[ParamRole::Free; 2],
        Arc::new(Pair),
        [(rsdag::Output::Slot(0), OutputRole::Plain)],
    );
    let _ = g.call(f, 0, &[x, x]);
    let module = g.to_module();
    let err = Graph::from_module(&module).err();
    assert_eq!(err, Some(rsdag::ModuleError::MissingExtern("pair".into())));
}

#[test]
fn a_broken_module_is_refused_not_panicked_on() {
    use rsdag::{ModuleError, Node};
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let y = g.sym("y");
    let s = g.add(x, y);
    let t = g.sin(s);
    let d = g.dot(vec![x, y], vec![y, t]);
    let _ = Scope::new(&mut g, "f").close([(d, OutputRole::Plain)]);
    let good = g.to_module();
    assert!(Graph::from_module(&good).is_ok());

    let mut breaks: Vec<(&str, rsdag::Module<F64>)> = Vec::new();
    let mut m = good.clone();
    m.version += 1;
    breaks.push(("version", m));
    let mut m = good.clone();
    let last = m.nodes.len() - 1;
    let i = m
        .nodes
        .iter()
        .position(|n| matches!(n, Node::Add(..)))
        .unwrap();
    m.nodes[i] = Node::Add(rsdag::ExprId(last as u32), rsdag::ExprId(0));
    breaks.push(("forward operand", m));
    let mut m = good.clone();
    let i = m
        .nodes
        .iter()
        .position(|n| matches!(n, Node::Symbol(_)))
        .unwrap();
    m.nodes[i] = Node::Symbol(rsdag::SymbolId(99));
    breaks.push(("symbol", m));
    let mut m = good.clone();
    m.arg_pool.pop();
    breaks.push(("operand list", m));
    let mut m = good.clone();
    m.funcs[0].outputs[0] = rsdag::Output::Expr(rsdag::ExprId(10_000));
    breaks.push(("function output", m));
    let mut m = good.clone();
    let i = m
        .nodes
        .iter()
        .position(|n| matches!(n, Node::Dot(_)))
        .unwrap();
    if let Node::Dot(l) = m.nodes[i] {
        m.nodes[i] = Node::Dot(rsdag::node::ArgList {
            start: l.start,
            len: l.len - 1,
        });
    }
    breaks.push(("dot halves", m));

    for (what, m) in breaks {
        let mut into: Graph<F64> = Graph::new();
        let _ = into.sym("keep");
        let before = into.len();
        let err: Result<_, ModuleError> = into.load_module(&m);
        assert!(err.is_err(), "{what}: loaded");
        assert_eq!(into.len(), before, "{what}: the graph changed");
    }
}
