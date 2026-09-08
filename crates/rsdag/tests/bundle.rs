//! Extern-function semantics: several outputs of one extern function must
//! each read the right slot, and the body must run exactly once per eval no
//! matter how many outputs share the call.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rsdag::{ExternBundle, Graph, Output, Tape};

/// Outputs `[a+b, a*b, a-b]` from `[a, b]`, counting how often it runs.
struct TriBundle {
    calls: Arc<AtomicUsize>,
}

impl ExternBundle for TriBundle {
    fn n_outputs(&self) -> usize {
        3
    }
    fn call(&self, args: &[f64], out: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (a, b) = (args[0], args[1]);
        out[0] = a + b;
        out[1] = a * b;
        out[2] = a - b;
    }
}

#[test]
fn bundle_scatters_and_runs_once() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let y = ctx.sym("y");
    let xs = match ctx.node(x) {
        rsdag::Node::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let ys = match ctx.node(y) {
        rsdag::Node::Symbol(s) => *s,
        _ => unreachable!(),
    };

    let f = ctx.define_extern_func(
        "tri",
        2,
        Arc::new(TriBundle {
            calls: calls.clone(),
        }),
        vec![Output::Slot(0), Output::Slot(1), Output::Slot(2)],
    );
    // Three outputs of the same function over the same arguments.
    let o_sum = ctx.call(f, 0, &[x, y]);
    let o_prod = ctx.call(f, 1, &[x, y]);
    let o_diff = ctx.call(f, 2, &[x, y]);

    // Combine all three so the root depends on every slot.
    let s1 = ctx.add(o_sum, o_prod);
    let root = ctx.add(s1, o_diff);

    let tape = Tape::compile(&ctx, &[o_sum, o_prod, o_diff, root], &[xs, ys]);
    let mut work = Vec::new();
    let mut out = Vec::new();
    tape.eval(&[3.0, 4.0], &mut work, &mut out);

    assert_eq!(out[0], 7.0, "a+b");
    assert_eq!(out[1], 12.0, "a*b");
    assert_eq!(out[2], -1.0, "a-b");
    assert_eq!(out[3], 7.0 + 12.0 - 1.0, "root");
    // One eval over a single (bundle, args) group: the body runs exactly once.
    assert_eq!(calls.load(Ordering::Relaxed), 1, "bundle ran once per eval");

    // Second eval reuses the compiled tape; one more call, fresh values.
    tape.eval(&[1.0, 2.0], &mut work, &mut out);
    assert_eq!(out[1], 2.0);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}
