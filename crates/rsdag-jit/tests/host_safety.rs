//! The boundary between emitted code and the host: a bundle that panics
//! raises its panic in the caller instead of aborting the process, and a
//! bundle whose output count changed after compile is refused instead of
//! being handed a write past its block.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rsdag::{ExprId, ExternBundle, Graph, Node, Output, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

/// `[a + b]`, panicking when `a` is negative; reports `n_out` outputs.
struct Touchy {
    n_out: Arc<AtomicUsize>,
}

impl ExternBundle for Touchy {
    fn n_outputs(&self) -> usize {
        self.n_out.load(Ordering::Relaxed)
    }
    fn call_into(&self, args: &[f64], _work: &mut [f64], out: &mut [f64]) {
        assert!(args[0] >= 0.0, "negative argument");
        out[0] = args[0] + args[1];
    }
}

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// `calls` instances of the bundle over distinct arguments, summed; many
/// instances lower to one batched call.
fn program(calls: usize) -> (NativeTape, Arc<AtomicUsize>, usize) {
    let n_out = Arc::new(AtomicUsize::new(1));
    let mut g: Graph<F64> = Graph::new();
    let (x, xs) = sym(&mut g, "x");
    let f = g.define_extern_func(
        "touchy",
        2,
        Arc::new(Touchy {
            n_out: n_out.clone(),
        }),
        vec![Output::Slot(0)],
    );
    let mut terms = Vec::new();
    for k in 0..calls {
        let c = g.konst_f64(k as f64);
        terms.push(g.call(f, 0, &[x, c]));
    }
    let root = g.reduce(rsdag::ReduceOp::Sum, terms);
    let tape = Tape::compile(&g, &[root], &[xs]);
    (NativeTape::compile(&tape).expect("native"), n_out, calls)
}

#[test]
fn a_panicking_bundle_raises_in_the_caller() {
    for calls in [1, 40] {
        let (native, _, n) = program(calls);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        native.eval(&[1.0], &mut w, &mut o);
        assert_eq!(o[0], (0..n).map(|k| 1.0 + k as f64).sum::<f64>());

        let caught = catch_unwind(AssertUnwindSafe(|| native.eval(&[-1.0], &mut w, &mut o)));
        let msg = caught.expect_err("the panic reaches the caller");
        let msg = msg
            .downcast_ref::<String>()
            .map(String::as_str)
            .or(msg.downcast_ref::<&str>().copied());
        assert!(
            msg.is_some_and(|m| m.contains("negative argument")),
            "{calls} calls: {msg:?}"
        );

        // Nothing is left pending: the next evaluation is an ordinary one.
        native.eval(&[2.0], &mut w, &mut o);
        assert_eq!(o[0], (0..n).map(|k| 2.0 + k as f64).sum::<f64>());
    }
}

#[test]
fn a_bundle_that_grew_its_outputs_is_refused() {
    for calls in [1, 40] {
        let (native, n_out, _) = program(calls);
        n_out.store(1 << 20, Ordering::Relaxed);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        let caught = catch_unwind(AssertUnwindSafe(|| native.eval(&[1.0], &mut w, &mut o)));
        let msg = caught.expect_err("refused");
        let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(
            msg.contains("changed since compile"),
            "{calls} calls: {msg}"
        );
    }
}
