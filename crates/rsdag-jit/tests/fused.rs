//! The accumulating kernels natively: a running block from which products
//! are subtracted one after another (each product one kernel whose
//! accumulator is the previous result, read in place) evaluates to the same
//! bits as the interpreter, whole and split into prolog and main.

use rsdag::{ExprId, Graph, Node, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn sym(g: &mut Graph<F64>, name: &str, syms: &mut Vec<SymbolId>) -> ExprId {
    let e = g.sym(name);
    if let Node::Symbol(s) = g.node(e) {
        syms.push(*s);
    }
    e
}

#[test]
fn chained_block_updates_run_natively_with_folded_kernels() {
    // Small blocks run as inline dots, the eight-block's products on the
    // host.
    for b in [3usize, 4, 8] {
        chained_block_updates(b);
    }
}

fn chained_block_updates(b: usize) {
    let updates = 3usize;
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    let matrix = |g: &mut Graph<F64>, name: &str, syms: &mut Vec<SymbolId>| -> Vec<ExprId> {
        (0..b * b)
            .map(|k| sym(g, &format!("{name}{k}"), syms))
            .collect()
    };
    // Parameter-pure: the running block and the factors, `L` row-major and
    // `U` column-major so both operands of a product are runs; main: a
    // vector.
    let mut acc = matrix(&mut g, "c", &mut syms);
    let factors: Vec<(Vec<ExprId>, Vec<ExprId>)> = (0..updates)
        .map(|u| {
            let l = matrix(&mut g, &format!("l{u}_"), &mut syms);
            let r = matrix(&mut g, &format!("u{u}_"), &mut syms);
            (l, r)
        })
        .collect();
    let n_pure = syms.len();
    let v: Vec<ExprId> = (0..b)
        .map(|k| sym(&mut g, &format!("v{k}"), &mut syms))
        .collect();
    // acc -= L_u U_u, entry by entry, one product after the other.
    for (l, r) in &factors {
        acc = (0..b * b)
            .map(|q| {
                let (i, j) = (q / b, q % b);
                let row: Vec<ExprId> = (0..b).map(|k| l[i * b + k]).collect();
                let col: Vec<ExprId> = (0..b).map(|k| r[j * b + k]).collect();
                let d = g.dot(row, col);
                g.sub(acc[q], d)
            })
            .collect();
    }
    // y = v - acc v, the main part.
    let roots: Vec<ExprId> = (0..b)
        .map(|i| {
            let row: Vec<ExprId> = (0..b).map(|k| acc[i * b + k]).collect();
            let d = g.dot(row, v.clone());
            g.sub(v[i], d)
        })
        .collect();
    let vals: Vec<f64> = (0..syms.len())
        .map(|k| ((k * 7 + 3) % 11) as f64 * 0.125 - 0.6)
        .collect();
    let mut pure = vec![true; n_pure];
    pure.resize(syms.len(), false);
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let d = tape.dump();
    assert!(d.contains(" acc "), "{d}");
    let native = NativeTape::compile(&tape).expect("compile");
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    let (mut wn, mut on) = (Vec::new(), Vec::new());
    native.eval(&vals, &mut wn, &mut on);
    assert_eq!(o.len(), on.len());
    for (k, (a, b)) in o.iter().zip(&on).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {k}: {a} vs {b}");
    }
    // The prolog and main split evaluate the same.
    native.eval_prolog(&vals, &mut wn);
    native.eval_main(&vals, &mut wn, &mut on);
    for (a, b) in o.iter().zip(&on) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// The complex product pattern, small enough to run as inline dots:
/// `re = rr - ii` and `im = ri + ir` folded from the kernel's own outputs,
/// a negated product beside them.
#[test]
fn a_small_kernel_folding_its_own_outputs_runs_natively() {
    let (m, n) = (6usize, 4usize);
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    let es: Vec<ExprId> = (0..2 * m * n + 2 * n)
        .map(|k| sym(&mut g, &format!("s{k}"), &mut syms))
        .collect();
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
        roots.push(g.neg(rr));
    }
    let tape = Tape::compile(&g, &roots, &syms);
    let d = tape.dump();
    assert!(d.contains("-#") && d.contains("+#"), "{d}");
    let native = NativeTape::compile(&tape).expect("compile");
    let vals: Vec<f64> = (0..syms.len()).map(|k| (k as f64 * 0.61).cos()).collect();
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    let (mut wn, mut on) = (Vec::new(), Vec::new());
    native.eval(&vals, &mut wn, &mut on);
    for (k, (a, b)) in o.iter().zip(&on).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {k}: {a} vs {b}");
    }
}
