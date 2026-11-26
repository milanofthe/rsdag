#![allow(dead_code)]

pub mod constants;

use std::collections::HashMap;
use std::sync::Arc;

use rsdag::parallel::{self, Lent};
use rsdag::Program;
use rsdag::{ExprId, Graph, SymbolId, Tape, F64};

pub fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let s = g.symbol_id(name);
    (g.symbol_expr(s), s)
}

pub fn symbol_of(g: &Graph<F64>, e: ExprId) -> SymbolId {
    g.as_symbol(e).expect("a symbol")
}

pub fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

pub fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| same_bits(x, y))
}

pub fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

pub fn identical(a: &[f64], b: &[f64]) -> bool {
    bits(a) == bits(b)
}

pub fn pool(threads: usize) -> Lent {
    Lent {
        pool: Arc::new(parallel::Workers::new(threads)),
        dispatch: parallel::Dispatch::free(threads),
    }
}

pub fn sine_series(g: &mut Graph<F64>, n: usize) -> (ExprId, Vec<SymbolId>) {
    let (p, x) = (g.sym("p"), g.sym("x"));
    let mut acc = g.zero();
    let mut xp = g.one();
    for k in 0..n {
        let kk = g.konst_f64(k as f64);
        let pk = g.mul(p, kk);
        let s = g.sin(pk);
        let t = g.mul(s, xp);
        acc = g.add(acc, t);
        xp = g.mul(xp, x);
    }
    (acc, vec![symbol_of(g, p), symbol_of(g, x)])
}

pub fn assert_matches_eval(
    g: &Graph<F64>,
    roots: &[ExprId],
    syms: &[SymbolId],
    tape: &Tape,
    inputs: &[f64],
) {
    let env: HashMap<SymbolId, f64> = syms.iter().copied().zip(inputs.iter().copied()).collect();
    let want = rsdag::eval(g, roots, &env);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(inputs, &mut w, &mut o);
    for (k, (a, b)) in want.iter().zip(&o).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {k}: {a} vs {b}");
    }
}
