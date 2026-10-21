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
    Parallel {
        pool: Arc::new(parallel::Workers::new(threads)),
        dispatch: rsdag::parallel::Dispatch::free(threads),
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
    p.dispatch = parallel::Dispatch::never();
    let r = parallel::install(p, || {
        assert!(parallel::current().is_some());
        eval(&tape, &x)
    });
    assert!(parallel::current().is_none());
    assert_eq!(bits(&r), bits(&eval(&tape, &x)));
}

#[test]
fn workers_run_every_piece_once_awake_or_woken() {
    use rsdag::parallel::{Pool, Workers};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let w = Workers::with_spin(4, std::time::Duration::from_micros(50));
    let hits: Vec<AtomicUsize> = (0..500).map(|_| AtomicUsize::new(0)).collect();
    for round in 0..200 {
        w.run(500, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        // now and then long enough for the workers to fall asleep
        if round % 50 == 0 {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 200));
}

#[test]
fn a_panic_in_a_piece_reaches_the_caller() {
    use rsdag::parallel::{Pool, Workers};
    let w = Workers::new(3);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.run(64, &|i| assert!(i != 37, "piece 37"));
    }));
    assert!(r.is_err());
    // the pool goes on
    let n = std::sync::atomic::AtomicUsize::new(0);
    w.run(64, &|_| {
        n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(n.into_inner(), 64);
}

#[test]
fn sections_nest_on_the_workers() {
    // A run whose pieces run, join and scope again, the scope's tasks
    // starting tasks: every leaf once, with the pool's threads only.
    use rsdag::parallel::{Pool, Spawn, Workers};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let w = Workers::new(4);
    let hits: Vec<AtomicUsize> = (0..16 * 32 * 2).map(|_| AtomicUsize::new(0)).collect();
    let tasks = AtomicUsize::new(0);
    for _ in 0..20 {
        w.run(16, &|i| {
            w.run(32, &|j| {
                let k = (i * 32 + j) * 2;
                let (h0, h1) = (&hits[k], &hits[k + 1]);
                w.join(
                    &mut || {
                        h0.fetch_add(1, Ordering::Relaxed);
                    },
                    &mut || {
                        h1.fetch_add(1, Ordering::Relaxed);
                    },
                );
            });
            w.scope(&mut |s: &dyn Spawn<'_>| {
                for _ in 0..3 {
                    let tasks = &tasks;
                    s.spawn(Box::new(move |s: &dyn Spawn<'_>| {
                        tasks.fetch_add(1, Ordering::Relaxed);
                        s.spawn(Box::new(move |_: &dyn Spawn<'_>| {
                            tasks.fetch_add(1, Ordering::Relaxed);
                        }));
                    }));
                }
            });
        });
    }
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 20));
    assert_eq!(tasks.into_inner(), 20 * 16 * 6);
}

#[test]
fn threads_outside_the_pool_share_it() {
    // Four threads outside the pool run sections on it at once.
    use rsdag::parallel::{Pool, Workers};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let w = Workers::new(3);
    let n = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                for _ in 0..200 {
                    w.run(17, &|_| {
                        w.run(5, &|_| {
                            n.fetch_add(1, Ordering::Relaxed);
                        });
                    });
                }
            });
        }
    });
    assert_eq!(n.into_inner(), 4 * 200 * 17 * 5);
}

#[test]
fn a_panic_in_a_task_or_a_side_reaches_its_section() {
    use rsdag::parallel::{Pool, Spawn, Workers};
    let w = Workers::new(3);
    let side = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.join(&mut || {}, &mut || panic!("side"));
    }));
    assert!(side.is_err());
    let task = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.scope(&mut |s: &dyn Spawn<'_>| {
            s.spawn(Box::new(|_: &dyn Spawn<'_>| panic!("task")));
        });
    }));
    assert!(task.is_err());
    let n = std::sync::atomic::AtomicUsize::new(0);
    w.run(8, &|_| {
        n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(n.into_inner(), 8);
}

#[test]
fn a_section_goes_as_wide_as_its_work_is_worth() {
    use parallel::Dispatch;
    // free: every section of two pieces or more on as many threads as pieces
    assert_eq!(Dispatch::free(8).width(1.0, 3), 3);
    assert_eq!(Dispatch::free(8).width(1.0, 100), 8);
    assert_eq!(Dispatch::free(8).width(1.0, 1), 1);
    assert_eq!(Dispatch::never().width(1e12, 100), 1);
    // 1 us per section plus 1 us per helper: a microsecond of work stays
    // serial, a millisecond goes to all eight threads, and in between some
    let d = Dispatch::new((0..=8).map(|k| 1e3 * k as f64).collect());
    assert_eq!(d.width(1e3, 64), 1);
    assert_eq!(d.width(1e6, 64), 8);
    let mid = d.width(2e4, 64);
    assert!(mid > 1 && mid < 8, "{mid}");
    // nothing to gain: serial
    assert_eq!(d.width(f64::NAN, 64), 1);
}

#[test]
fn workers_measure_what_a_section_costs_them() {
    use parallel::{Pool, Workers};
    let w = Workers::new(4);
    let d = w.dispatch();
    assert_eq!(d.threads(), 4);
    let costs: Vec<f64> = (2..=4).map(|k| d.cost(k)).collect();
    assert!(
        costs.iter().all(|c| c.is_finite() && *c >= 0.0),
        "{costs:?}"
    );
    assert!(costs.windows(2).all(|c| c[0] <= c[1]), "{costs:?}");
    assert_eq!(Workers::new(1).dispatch().width(1e12, 100), 1);
}

#[test]
fn a_rate_learns_what_an_op_costs() {
    let r = parallel::Rate::new(10.0);
    assert_eq!(r.ns_per_op(), 10.0);
    for _ in 0..64 {
        r.learn(1e4, 1e4);
    }
    assert!((r.ns_per_op() - 1.0).abs() < 1e-3, "{}", r.ns_per_op());
    // a run of no work is never sampled, any other now and then
    assert!(!(0..64).any(|_| r.sample(0.0)));
    assert!((0..64).any(|_| r.sample(1.0)));
}
