//! Everything the crate does on one path, built for `wasm32-unknown-unknown`
//! by CI, which then checks the module imports nothing: a browser supplies
//! no host functions to a plain wasm module, so any import (a clock, a
//! random source) fails instantiation. (`egraph` is native only.)
//!
//!     cargo build --release -p rsdag --example wasm_probe --target wasm32-unknown-unknown --features exact,complex,serde
//!     python3 scripts/wasm_imports.py target/wasm32-unknown-unknown/release/examples/wasm_probe.wasm

use std::hint::black_box;

use rsdag::{sparse_jacobian, Graph, Node, Tape, F64};

fn main() {
    let mut g: Graph<F64> = Graph::new();
    let x = g.sym("x");
    let Node::Symbol(xs) = *g.node(x) else {
        unreachable!()
    };
    let e = g.exp(x);
    let s = g.sin(e);
    let r = g.add(s, x);
    let jac = sparse_jacobian(&mut g, &[r], &[xs]);
    let tape = Tape::compile(&g, &[r, jac[0][0].1], &[xs]);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[black_box(0.5)], &mut w, &mut o);
    black_box(&o);
    #[cfg(feature = "complex")]
    {
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(
            &[num_complex::Complex64::new(black_box(0.5), 1.0)],
            &mut w,
            &mut o,
        );
        black_box(&o);
    }
    #[cfg(feature = "serde")]
    black_box(g.to_module());
}
