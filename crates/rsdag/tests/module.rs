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
        let (loaded, map) = Graph::from_module(&module);
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

    let (mut b, first) = Graph::from_module(&module);
    let before = b.len();
    // Loading the same module a second time must not add a node: every
    // one of them is already interned, so the ids come back unchanged.
    let map = b.load_module(&module);
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

    let (loaded, map) = Graph::from_module(&back);
    let roots2: Vec<ExprId> = roots.iter().map(|e| map.exprs[e.0 as usize]).collect();
    let syms2: Vec<SymbolId> = syms.iter().map(|s| map.symbols[s.0 as usize]).collect();
    let row = inputs(&mut spec.rng(), syms.len());
    let (mut w, mut o) = (Vec::new(), Vec::new());
    Tape::compile(&g, &roots, &syms).eval(&row, &mut w, &mut o);
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    Tape::compile(&loaded, &roots2, &syms2).eval(&row, &mut w2, &mut o2);
    assert!(o.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits()));
}
