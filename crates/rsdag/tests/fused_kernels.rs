//! Product kernels fold their consumers' subtraction or addition in: the
//! program shrinks and stays bit-identical to the arena.

use rsdag::{Graph, Node, SymbolId, Tape, F64};

fn syms(g: &mut Graph<F64>, names: &[String]) -> (Vec<rsdag::ExprId>, Vec<SymbolId>) {
    let mut es = Vec::new();
    let mut ss = Vec::new();
    for n in names {
        let e = g.sym(n);
        if let Node::Symbol(s) = g.node(e) {
            ss.push(*s);
        }
        es.push(e);
    }
    (es, ss)
}

#[test]
fn a_gemv_folds_the_subtractions_of_its_rows() {
    let (m, n) = (10usize, 4usize);
    let mut g: Graph<F64> = Graph::new();
    let names: Vec<String> = (0..m * n)
        .map(|k| format!("a{k}"))
        .chain((0..n).map(|k| format!("x{k}")))
        .chain((0..m).map(|k| format!("c{k}")))
        .collect();
    let (es, ss) = syms(&mut g, &names);
    let x: Vec<_> = es[m * n..m * n + n].to_vec();
    let roots: Vec<_> = (0..m)
        .map(|i| {
            let d = g.dot(es[i * n..(i + 1) * n].to_vec(), x.clone());
            g.sub(es[m * n + n + i], d)
        })
        .collect();
    let tape = Tape::compile(&g, &roots, &ss);
    let d = tape.dump();
    assert!(d.contains(" acc "), "{d}");
    assert!(!d.contains("Sub("), "{d}");
    let vals: Vec<f64> = (0..names.len()).map(|k| (k as f64 * 0.37).sin()).collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    for i in 0..m {
        let mut acc = 0.0;
        for j in 0..n {
            acc += vals[i * n + j] * vals[m * n + j];
        }
        let want = vals[m * n + n + i] - acc;
        assert!((o[i] - want).abs() < 1e-12, "{} vs {want}", o[i]);
    }
}

/// `m` rows against `x`, each consumed as `neg(dot)`: the fold gives the
/// negation's bits, `-0.0` for a zero product included.
#[test]
fn a_negated_gemv_keeps_the_sign_of_zero() {
    let (m, n) = (9usize, 3usize);
    let mut g: Graph<F64> = Graph::new();
    let names: Vec<String> = (0..m * n)
        .map(|k| format!("a{k}"))
        .chain((0..n).map(|k| format!("x{k}")))
        .collect();
    let (es, ss) = syms(&mut g, &names);
    let x: Vec<_> = es[m * n..].to_vec();
    let roots: Vec<_> = (0..m)
        .map(|i| {
            let d = g.dot(es[i * n..(i + 1) * n].to_vec(), x.clone());
            g.neg(d)
        })
        .collect();
    let tape = Tape::compile(&g, &roots, &ss);
    let d = tape.dump();
    assert!(d.contains("neg"), "{d}");
    assert!(!d.contains("Neg("), "{d}");
    let mut vals = vec![0.0; names.len()]; // every product is +0.0
    vals[0] = 1.0;
    let (mut w, mut o): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    for v in &o {
        assert!(v.is_sign_negative() && *v == 0.0, "{v}");
    }
}

/// The complex product pattern: rows `ar`, `ai` against `br`, `bi` in one
/// kernel, `re = rr - ii` and `im = ri + ir` folded from its own outputs.
#[test]
fn a_kernel_folds_its_own_outputs() {
    let (m, n) = (6usize, 4usize);
    let mut g: Graph<F64> = Graph::new();
    let names: Vec<String> = (0..2 * m * n)
        .map(|k| format!("a{k}"))
        .chain((0..2 * n).map(|k| format!("b{k}")))
        .collect();
    let (es, ss) = syms(&mut g, &names);
    let br: Vec<_> = es[2 * m * n..2 * m * n + n].to_vec();
    let bi: Vec<_> = es[2 * m * n + n..].to_vec();
    let mut roots = Vec::new();
    for i in 0..m {
        let ar = es[i * n..(i + 1) * n].to_vec();
        let ai = es[(m + i) * n..(m + i + 1) * n].to_vec();
        let rr = g.dot(ar.clone(), br.clone());
        let ii = g.dot(ai.clone(), bi.clone());
        let ri = g.dot(ar, bi.clone());
        let ir = g.dot(ai, br.clone());
        roots.push(g.sub(rr, ii));
        roots.push(g.add(ri, ir));
    }
    let tape = Tape::compile(&g, &roots, &ss);
    let d = tape.dump();
    assert!(d.contains("-#") && d.contains("+#"), "{d}");
    assert!(!d.contains("Sub(") && !d.contains("Add("), "{d}");
    let vals: Vec<f64> = (0..names.len()).map(|k| (k as f64 * 0.61).cos()).collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let (brv, biv) = (&vals[2 * m * n..2 * m * n + n], &vals[2 * m * n + n..]);
    for i in 0..m {
        let ar = &vals[i * n..(i + 1) * n];
        let ai = &vals[(m + i) * n..(m + i + 1) * n];
        let re = dot(ar, brv) - dot(ai, biv);
        let im = dot(ar, biv) + dot(ai, brv);
        assert!((o[2 * i] - re).abs() < 1e-12 && (o[2 * i + 1] - im).abs() < 1e-12);
    }
}
