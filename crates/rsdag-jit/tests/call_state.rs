//! Stateful calls natively: the caller's prolog runs the body's prolog per
//! instance, the main phase reads the state, bit for bit the interpreter;
//! and since both backends lay out the state the same way, a prolog run by
//! one serves the main phase of the other.

use rsdag::{ExprId, FuncId, Graph, Node, ParamRole, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

fn device(g: &mut Graph<F64>) -> FuncId {
    let (v, vs) = sym(g, "dev.v");
    let (p, ps) = sym(g, "dev.p");
    let mut k = p;
    for i in 0..30 {
        let c = g.konst_f64(1.0 + 0.01 * i as f64);
        let e = g.exp(k);
        let t = g.mul(c, e);
        k = g.ln(t);
    }
    let tv = g.tanh(v);
    let o0 = g.mul(tv, k);
    let v2 = g.mul(v, v);
    let o1 = g.add(v2, k);
    let f = g.define_func("dev", vec![vs, ps], vec![o0, o1]);
    g.set_param_role(f, 0, ParamRole::State { id: 0 });
    g.set_param_role(f, 1, ParamRole::Param);
    f
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn stateful_calls_run_natively_and_share_the_state_layout() {
    for n in [1usize, 12] {
        let mut g: Graph<F64> = Graph::new();
        let f = device(&mut g);
        let mut syms = Vec::new();
        let mut roots = Vec::new();
        for i in 0..n {
            let (x, xs) = sym(&mut g, &format!("x{i}"));
            let (q, qs) = sym(&mut g, &format!("q{i}"));
            syms.push(xs);
            syms.push(qs);
            let a = g.call(f, 0, &[x, q]);
            let b = g.call(f, 1, &[x, q]);
            roots.push(g.add(a, b));
        }
        let pure: Vec<bool> = (0..syms.len()).map(|k| k % 2 == 1).collect();
        let tape = Tape::compile_split(&g, &roots, &syms, &pure);
        let native = NativeTape::compile(&tape).expect("native");
        let ins = |t: f64| -> Vec<f64> {
            (0..syms.len())
                .map(|k| {
                    if k % 2 == 1 {
                        0.2 + 0.01 * k as f64
                    } else {
                        t + 0.1 * k as f64
                    }
                })
                .collect()
        };
        let (mut w, mut want) = (Vec::new(), Vec::new());
        let (mut nw, mut got) = (Vec::new(), Vec::new());
        native.eval_prolog(&ins(0.0), &mut nw);
        for t in [0.0, 0.5, -1.25] {
            tape.eval(&ins(t), &mut w, &mut want);
            native.eval_main(&ins(t), &mut nw, &mut got);
            assert!(same(&got, &want), "{n}: native at t = {t}");
        }
        // The interpreter's prolog under the native main phase, and back.
        let s = tape.state_len();
        let mut iw = vec![0.0; tape.work_len()];
        tape.eval_prolog_into(&ins(0.0), &mut iw);
        let mut nw2 = vec![f64::NAN; nw.len()];
        nw2[..s].copy_from_slice(&iw[..s]);
        native.eval_main(&ins(0.75), &mut nw2, &mut got);
        tape.eval(&ins(0.75), &mut w, &mut want);
        assert!(same(&got, &want), "{n}: interpreter prolog, native main");
        let mut iw2 = vec![f64::NAN; tape.work_len()];
        iw2[..s].copy_from_slice(&nw[..s]);
        let mut o = vec![0.0; tape.out_len()];
        tape.eval_main_into(&ins(0.75), &mut iw2, &mut o);
        assert!(same(&o, &want), "{n}: native prolog, interpreter main");
    }
}
