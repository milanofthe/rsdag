//! Natively compiled programs run their stages of independent calls on an
//! installed pool, bit for bit as serially and as the interpreter: prologs,
//! main phases over the instances' states, and calls without state.

use std::sync::Arc;

use rsdag::parallel::{self, Parallel};
use rsdag::{ExprId, FuncId, Graph, Node, ParamRole, ReduceOp, Scope, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

const N: usize = 96;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// A two-terminal device with `Param`-role parameters and enough work per
/// instance for a stage to be worth a pool.
fn device(g: &mut Graph<F64>, name: &str, k: f64) -> FuncId {
    let mut s = Scope::new(g, name);
    let a = s.param_with_role("a", ParamRole::State { id: 0 });
    let b = s.param_with_role("b", ParamRole::State { id: 1 });
    let p0 = s.param_with_role("p0", ParamRole::Param);
    let p1 = s.param_with_role("p1", ParamRole::Param);
    let mut pe = p0;
    for i in 0..20 {
        let c = s.konst_f64(k + 0.01 * i as f64);
        let m = s.mul(pe, c);
        let e = s.exp(m);
        pe = s.ln(e);
    }
    let d = s.sub(a, b);
    let x = s.mul(p1, d);
    let t = s.tanh(x);
    let i = s.mul(pe, t);
    let ab = s.mul(a, b);
    let q = s.sin(ab);
    let q = s.mul(pe, q);
    s.close(vec![i, q])
}

fn ring() -> (Graph<F64>, Vec<ExprId>, Vec<SymbolId>, Vec<bool>) {
    let mut g: Graph<F64> = Graph::new();
    let fs = [
        device(&mut g, "na", 1.0),
        device(&mut g, "nb", 0.5),
        device(&mut g, "nc", 0.25),
    ];
    let v: Vec<(ExprId, SymbolId)> = (0..N).map(|k| sym(&mut g, &format!("v{k}"))).collect();
    let p: Vec<(ExprId, SymbolId)> = (0..2 * N).map(|k| sym(&mut g, &format!("p{k}"))).collect();
    let mut node: Vec<Vec<ExprId>> = vec![Vec::new(); N];
    for k in 0..N {
        let (a, b) = (v[k].0, v[(k + 1) % N].0);
        let out = g.calls(fs[k % 3], &[0, 1], &[a, b, p[2 * k].0, p[2 * k + 1].0]);
        node[k].push(out[0]);
        node[k].push(out[1]);
        let ni = g.neg(out[0]);
        node[(k + 1) % N].push(ni);
    }
    let roots = node
        .into_iter()
        .map(|terms| g.reduce(ReduceOp::Sum, terms))
        .collect();
    let syms = v.iter().chain(&p).map(|&(_, s)| s).collect();
    let pure = (0..3 * N).map(|k| k >= N).collect();
    (g, roots, syms, pure)
}

fn pool(threads: usize) -> Parallel {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    Parallel {
        pool: Arc::new(pool),
        min_ops: 0,
    }
}

fn inputs() -> Vec<f64> {
    (0..3 * N)
        .map(|k| 0.2 + 0.013 * k as f64 - (k % 5) as f64 * 0.07)
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_split_native_program_on_a_pool_is_the_serial_one() {
    let (g, roots, syms, pure) = ring();
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    assert!(!tape.stages().is_empty());
    let native = NativeTape::compile(&tape).expect("native code");
    let x = inputs();
    let (mut w, mut out) = (Vec::new(), Vec::new());
    tape.eval_prolog(&x, &mut w);
    tape.eval_main(&x, &mut w, &mut out);
    let (mut wn, mut on) = (Vec::new(), Vec::new());
    native.eval_prolog(&x, &mut wn);
    native.eval_main(&x, &mut wn, &mut on);
    assert_eq!(bits(&out), bits(&on), "native serial");
    for threads in [2, 5, 8] {
        let (mut wp, mut op) = (Vec::new(), Vec::new());
        parallel::install(pool(threads), || {
            native.eval_prolog(&x, &mut wp);
            for _ in 0..3 {
                native.eval_main(&x, &mut wp, &mut op);
            }
        });
        assert_eq!(bits(&out), bits(&op), "{threads} threads");
        let n = tape.state_len();
        assert_eq!(bits(&w[..n]), bits(&wp[..n]), "{threads} threads");
    }
}

#[test]
fn native_calls_without_state_on_a_pool_are_the_serial_ones() {
    let (g, roots, syms, _) = ring();
    let tape = Tape::compile(&g, &roots, &syms);
    let native = NativeTape::compile(&tape).expect("native code");
    let x = inputs();
    let (mut w, mut serial) = (Vec::new(), Vec::new());
    tape.eval(&x, &mut w, &mut serial);
    let (mut wn, mut par) = (Vec::new(), Vec::new());
    parallel::install(pool(4), || native.eval(&x, &mut wn, &mut par));
    assert_eq!(bits(&serial), bits(&par));
}
