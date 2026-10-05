//! A program's independent calls run as stages on an installed pool, and
//! compute exactly what the serial loop computes: prologs, main phases over
//! the instances' states, and calls without state.

use std::sync::Arc;

use rsdag::parallel::{self, Parallel};
use rsdag::{ExprId, FuncId, Graph, Node, ParamRole, ReduceOp, Scope, SymbolId, Tape, F64};

const N: usize = 64;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    match *g.node(e) {
        Node::Symbol(s) => (e, s),
        _ => unreachable!(),
    }
}

/// A two-terminal device: the current `p0 tanh(p1 (a - b))` and a charge
/// `p0 exp(-a b)`, its parameters with the `Param` role.
fn device(g: &mut Graph<F64>, name: &str, k: f64) -> FuncId {
    let mut s = Scope::new(g, name);
    let a = s.param_with_role("a", ParamRole::State { id: 0 });
    let b = s.param_with_role("b", ParamRole::State { id: 1 });
    let p0 = s.param_with_role("p0", ParamRole::Param);
    let p1 = s.param_with_role("p1", ParamRole::Param);
    let kk = s.konst_f64(k);
    let p1k = s.mul(p1, kk);
    let e = s.exp(p1k);
    let pe = s.mul(p0, e);
    let d = s.sub(a, b);
    let x = s.mul(p1k, d);
    let t = s.tanh(x);
    let i = s.mul(pe, t);
    let ab = s.mul(a, b);
    let nab = s.neg(ab);
    let q = s.exp(nab);
    let q = s.mul(pe, q);
    s.close(vec![i, q])
}

/// A ring of `N` nodes with two kinds of device between neighbours: the
/// node equations, and the symbols (the states, then the parameters).
fn ring() -> (Graph<F64>, Vec<ExprId>, Vec<SymbolId>, Vec<bool>) {
    let mut g: Graph<F64> = Graph::new();
    let (fa, fb) = (device(&mut g, "na", 1.0), device(&mut g, "nb", 0.5));
    let v: Vec<(ExprId, SymbolId)> = (0..N).map(|k| sym(&mut g, &format!("v{k}"))).collect();
    let p: Vec<(ExprId, SymbolId)> = (0..2 * N).map(|k| sym(&mut g, &format!("p{k}"))).collect();
    let mut node: Vec<Vec<ExprId>> = vec![Vec::new(); N];
    for k in 0..N {
        let f = if k % 2 == 0 { fa } else { fb };
        let (a, b) = (v[k].0, v[(k + 1) % N].0);
        let out = g.calls(f, &[0, 1], &[a, b, p[2 * k].0, p[2 * k + 1].0]);
        let (i, q) = (out[0], out[1]);
        node[k].push(i);
        node[k].push(q);
        let ni = g.neg(i);
        node[(k + 1) % N].push(ni);
    }
    let roots = node
        .into_iter()
        .map(|terms| g.reduce(ReduceOp::Sum, terms))
        .collect();
    let syms: Vec<SymbolId> = v.iter().chain(&p).map(|&(_, s)| s).collect();
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
        .map(|k| 0.3 + 0.01 * k as f64 - (k % 7) as f64 * 0.05)
        .collect()
}

fn eval(tape: &Tape, x: &[f64]) -> Vec<f64> {
    let (mut w, mut out) = (Vec::new(), Vec::new());
    tape.eval(x, &mut w, &mut out);
    out
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn stages_hold_the_independent_calls() {
    let (g, roots, syms, pure) = ring();
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let st = tape.stages();
    assert!(!st.is_empty(), "the device calls form stages");
    let calls: u32 = st.iter().map(|s| s.instances).sum();
    // every instance once in the prolog and once in the main phase
    assert_eq!(calls, 2 * N as u32);
}

#[test]
fn a_split_program_on_a_pool_is_the_serial_one() {
    let (g, roots, syms, pure) = ring();
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let x = inputs();
    let (mut w, mut out) = (Vec::new(), Vec::new());
    tape.eval_prolog(&x, &mut w);
    tape.eval_main(&x, &mut w, &mut out);
    for threads in [2, 3, 8] {
        let (mut wp, mut op) = (Vec::new(), Vec::new());
        parallel::install(pool(threads), || {
            tape.eval_prolog(&x, &mut wp);
            tape.eval_main(&x, &mut wp, &mut op);
        });
        assert_eq!(bits(&out), bits(&op), "{threads} threads");
        // the states the prologs left, too
        let n = tape.state_len();
        assert_eq!(bits(&w[..n]), bits(&wp[..n]), "{threads} threads");
    }
}

#[test]
fn calls_without_state_on_a_pool_are_the_serial_ones() {
    let (g, roots, syms, _) = ring();
    let tape = Tape::compile(&g, &roots, &syms);
    assert!(!tape.stages().is_empty());
    let x = inputs();
    let serial = eval(&tape, &x);
    let par = parallel::install(pool(4), || eval(&tape, &x));
    assert_eq!(bits(&serial), bits(&par));
}

#[test]
fn small_stages_stay_serial_and_nothing_is_installed_outside() {
    let (g, roots, syms, _) = ring();
    let tape = Tape::compile(&g, &roots, &syms);
    let x = inputs();
    let mut p = pool(4);
    p.min_ops = usize::MAX;
    let r = parallel::install(p, || {
        assert!(parallel::current().is_some());
        eval(&tape, &x)
    });
    assert!(parallel::current().is_none());
    assert_eq!(bits(&r), bits(&eval(&tape, &x)));
}
