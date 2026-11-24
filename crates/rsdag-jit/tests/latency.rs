//! Calls under a latency, interpreted and natively: an instance whose
//! arguments stay within the tolerance of those it computed at keeps its
//! outputs, one that moves past it computes, a prolog voids what was kept;
//! with a zero tolerance the outputs are those of every instance computed.

use rsdag::{
    ExprId, FuncId, Graph, Latency, Node, OutputRole, ParamRole, Program, Scope, SymbolId, Tape,
    F64,
};
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
    let e = g.exp(p);
    let tv = g.tanh(v);
    let o0 = g.mul(tv, e);
    let v2 = g.mul(v, v);
    let o1 = g.add(v2, e);
    let mut s = Scope::new(g, "dev");
    s.param(vs, ParamRole::State { id: 0 });
    s.param(ps, ParamRole::Param);
    s.close([(o0, OutputRole::Plain), (o1, OutputRole::Plain)])
}

const N: usize = 12;

/// `N` instances, each output a sum of its instance's two outputs; inputs
/// `x0, p0, x1, p1, ...`, the `p` parameter-pure.
fn tape() -> Tape {
    let mut g: Graph<F64> = Graph::new();
    let f = device(&mut g);
    let (mut syms, mut roots) = (Vec::new(), Vec::new());
    for i in 0..N {
        let (x, xs) = sym(&mut g, &format!("x{i}"));
        let (q, qs) = sym(&mut g, &format!("q{i}"));
        syms.extend([xs, qs]);
        let a = g.call(f, 0, &[x, q]);
        let b = g.call(f, 1, &[x, q]);
        roots.push(g.add(a, b));
    }
    let pure: Vec<bool> = (0..syms.len()).map(|k| k % 2 == 1).collect();
    Tape::compile_split(&g, &roots, &syms, &pure)
}

fn ins(x: &[f64], p: f64) -> Vec<f64> {
    x.iter().flat_map(|&x| [x, p]).collect()
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The outputs of `prog` over one work buffer: the main phase at each
/// parameter and state of `steps`, a prolog first where the parameter
/// changes.
fn run(prog: &dyn Program, steps: &[(f64, &[f64])]) -> Vec<Vec<f64>> {
    let mut w = vec![0.0; prog.work_len()];
    let mut bound = f64::NAN;
    steps
        .iter()
        .map(|&(p, x)| {
            if p != bound {
                prog.eval_prolog_into(&ins(x, p), &mut w);
                bound = p;
            }
            let mut o = vec![0.0; prog.out_len()];
            prog.eval_main_into(&ins(x, p), &mut w, &mut o);
            o
        })
        .collect()
}

fn forms(latency: Latency) -> (Tape, Vec<Box<dyn Program>>) {
    let reference = tape();
    let mut t = tape();
    t.set_latency(Some(latency));
    let native = NativeTape::compile(&t).expect("native");
    (reference, vec![Box::new(t), Box::new(native)])
}

#[test]
fn a_zero_tolerance_keeps_the_outputs_of_every_instance_computed() {
    let (reference, forms) = forms(Latency {
        atol: 0.0,
        rtol: 0.0,
    });
    let x0: Vec<f64> = (0..N).map(|i| 0.1 * i as f64 - 0.4).collect();
    let mut x1 = x0.clone();
    x1[3] += 1e-12;
    let mut x2 = x1.clone();
    x2[7] = -x2[7];
    let steps: Vec<(f64, &[f64])> = [&x0, &x0, &x1, &x2, &x2, &x1]
        .iter()
        .enumerate()
        .map(|(k, x)| (if k < 5 { 0.3 } else { 0.7 }, x.as_slice()))
        .collect();
    let want = run(&reference, &steps);
    for prog in &forms {
        let got = run(&**prog, &steps);
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(same(g, w), "state {k}");
        }
    }
}

#[test]
fn an_instance_keeps_its_outputs_within_the_tolerance_and_computes_past_it() {
    let (reference, forms) = forms(Latency {
        atol: 1e-3,
        rtol: 0.0,
    });
    let x0: Vec<f64> = (0..N).map(|i| 0.1 * i as f64 - 0.4).collect();
    let (mut near, mut far) = (x0.clone(), x0.clone());
    near[5] += 5e-4;
    far[5] += 5e-3;
    let mut far_near = far.clone();
    far_near[5] -= 5e-4;
    // the first run after a prolog computes, the second keeps the outputs
    // of an instance that did not move
    let steps = [
        (0.3, &x0[..]),
        (0.3, &x0),
        (0.3, &near),
        (0.3, &far),
        (0.3, &far_near),
        (0.7, &near),
    ];
    let want = run(&reference, &steps);
    assert!(want[2][5] != want[0][5]);
    for prog in &forms {
        let got = run(&**prog, &steps);
        assert!(same(&got[1], &want[1]), "still");
        // within it: instance 5 keeps what it computed at x0
        assert!(same(&got[2], &want[0]), "near");
        // past it: computed anew, and kept from there
        assert!(same(&got[3], &want[3]), "far");
        assert!(same(&got[4], &want[3]), "near far");
        // a prolog at other parameters voids what was kept
        assert!(same(&got[5], &want[5]), "prolog");
    }
}

#[test]
fn instances_moving_fast_compute_and_settle_to_be_kept_again() {
    let (reference, forms) = forms(Latency {
        atol: 1e-3,
        rtol: 0.0,
    });
    // every instance moving by far more than the tolerance for a while
    // (the call skips its checks), then still
    let states: Vec<Vec<f64>> = (0..30)
        .map(|k| {
            let at = k.min(12) as f64;
            (0..N).map(|i| 0.1 * i as f64 + at).collect()
        })
        .collect();
    let steps: Vec<(f64, &[f64])> = states.iter().map(|x| (0.3, x.as_slice())).collect();
    let want = run(&reference, &steps);
    for prog in &forms {
        let got = run(&**prog, &steps);
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(same(g, w), "state {k}");
        }
    }
}
