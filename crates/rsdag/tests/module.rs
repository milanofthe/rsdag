use rsdag::synth::{build, inputs, Spec, Vocabulary};
use rsdag::*;
use rsdag::{Tape, F64};

/// A module is the program: reloading one and evaluating it must give
/// the same bits, over programs nobody wrote by hand.
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
        let f = g.close("f", roots.clone());

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
        // A module of the reloaded graph is the module it was built
        // from: loading is not lossy and not order-dependent.
        assert_eq!(
            loaded.to_module(),
            module,
            "seed {seed}: second module differs"
        );
    }
}

/// Loading into a graph that already holds the same subexpressions
/// shares them instead of duplicating: the ids move, the values do not.
#[test]
fn loading_into_a_populated_graph_shares_nodes() {
    let mut a: Graph<F64> = Graph::new();
    let mut spec = Spec::new(9).steps(80).params(3);
    let (roots, syms) = build(&mut a, &mut spec);
    let module = a.to_module();

    let (mut b, first) = Graph::from_module(&module).unwrap();
    let before = b.len();
    // Loading the same module a second time must not add a node: every
    // one of them is already interned, so the ids come back unchanged.
    let map = b.load_module(&module).unwrap();
    assert_eq!(b.len(), before, "reloading a module added nodes");
    assert_eq!(map.exprs, first.exprs);
    for (k, r) in roots.iter().enumerate() {
        assert_eq!(map.exprs[r.0 as usize], *r, "root {k} moved");
    }
    for (k, s) in syms.iter().enumerate() {
        assert_eq!(map.symbols[s.0 as usize], *s, "symbol {k} moved");
    }
    let _ = &a;
}

/// A text format loses `f64` bits unless the constants are written as
/// bit patterns, and cannot hold an infinity at all; both are checked
/// here because a module that comes back one ulp away is a different
/// program.
#[cfg(feature = "serde")]
#[test]
fn a_module_survives_json() {
    let mut g: Graph<F64> = Graph::new();
    let mut spec = Spec::new(3).steps(120).params(4).vocab(Vocabulary::Full);
    let (mut roots, syms) = build(&mut g, &mut spec);
    // Values a text format mangles: an infinity, a NaN, and a double
    // whose shortest decimal parses back to its neighbour.
    for v in [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.1067040000000001,
    ] {
        let k = g.konst_f64(v);
        roots.push(k);
    }
    g.close("f", roots.clone());
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

/// A module with functions and calls to them, nested: the loader has to
/// define a function before it rebuilds the first call to it, and the
/// function's outputs are nodes it has already rebuilt by then.
#[test]
fn calls_and_nested_functions_round_trip() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = rsdag::Scope::new(&mut g, "gain");
    let u = s.param("u");
    let k = s.param("k");
    let y = s.mul(k, u);
    let gain = s.close(vec![y]);

    let mut s = rsdag::Scope::new(&mut g, "chain");
    let x = s.param("x");
    let a = s.konst_f64(2.0);
    let b = s.konst_f64(3.0);
    let mid = s.call(gain, 0, &[x, a]);
    let out = s.call(gain, 0, &[mid, b]);
    let e = s.exp(out);
    let chain = s.close(vec![e]);
    let root = match g.func(chain).outputs[0] {
        rsdag::Output::Expr(e) => e,
        _ => unreachable!(),
    };
    let params = g.func(chain).params.clone();

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

/// A body of `[a + b, a * b]` over `[a, b]`.
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

/// An extern function keeps its name and its slot layout in the module; the
/// body comes back through the loader's resolver.
#[test]
fn an_extern_function_round_trips_with_its_body_resolved_by_name() {
    use std::sync::Arc;
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let y = g.sym("y");
    let f = g.define_extern_func(
        "pair",
        2,
        Arc::new(Pair),
        vec![
            rsdag::Output::Slot(0),
            rsdag::Output::Slot(1),
            rsdag::Output::Zero,
        ],
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
fn loading_a_module_with_externs_without_bodies_says_so() {
    use std::sync::Arc;
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let f = g.define_extern_func("pair", 2, Arc::new(Pair), vec![rsdag::Output::Slot(0)]);
    let _ = g.call(f, 0, &[x, x]);
    let module = g.to_module();
    let err = Graph::from_module(&module).err();
    assert_eq!(err, Some(rsdag::ModuleError::MissingExtern("pair".into())));
}

/// A module is outside data: every way of breaking one is reported, none
/// panics, and the graph it was loaded into is left as it was.
#[test]
fn a_broken_module_is_refused_not_panicked_on() {
    use rsdag::{ModuleError, Node};
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let y = g.sym("y");
    let s = g.add(x, y);
    let t = g.sin(s);
    let d = g.dot(vec![x, y], vec![y, t]);
    let _ = g.close("f", vec![d]);
    let good = g.to_module();
    assert!(good.validate().is_ok());

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
