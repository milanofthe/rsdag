#![allow(dead_code)]

pub mod constants;

use std::collections::HashMap;
use std::sync::Arc;

use rsdag::parallel::{self, Lent};
use rsdag::Program;
use rsdag::{ExprId, FuncId, Graph, OutputRole, ParamRole, Scope, SymbolId, Tape, F64};

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

/// `f(v, p)`, `v` a state and `p` a parameter, whose parameter passes
/// through `links` rounds of `ln(c * exp(k))` that a split caller runs in
/// its prolog.
pub fn exp_ln_device(g: &mut Graph<F64>, links: usize) -> FuncId {
    let (v, vs) = sym(g, "dev.v");
    let (p, ps) = sym(g, "dev.p");
    let mut k = p;
    for i in 0..links {
        let c = g.konst_f64(1.0 + 0.01 * i as f64);
        let e = g.exp(k);
        let t = g.mul(c, e);
        k = g.ln(t);
    }
    let tv = g.tanh(v);
    let o0 = g.mul(tv, k);
    let v2 = g.mul(v, v);
    let o1 = g.add(v2, k);
    let mut s = Scope::new(g, "dev");
    s.param(vs, ParamRole::State { id: 0 });
    s.param(ps, ParamRole::Param);
    s.close([(o0, OutputRole::Plain), (o1, OutputRole::Plain)])
}

pub struct Circuit {
    pub g: Graph<F64>,
    pub roots: Vec<ExprId>,
    pub syms: Vec<SymbolId>,
    /// Per input whether it is a parameter.
    pub pure: Vec<bool>,
}

/// `n` instances of the two-output function that `device` builds, instance
/// `i` called on the state `x{i}` and on one parameter per name in `params`
/// suffixed by `i`, both outputs summed.
pub fn circuit(
    n: usize,
    device: impl FnOnce(&mut Graph<F64>) -> FuncId,
    params: &[&str],
) -> Circuit {
    let mut g: Graph<F64> = Graph::new();
    let f = device(&mut g);
    let (mut syms, mut roots, mut pure) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (x, xs) = sym(&mut g, &format!("x{i}"));
        let mut args = vec![x];
        syms.push(xs);
        pure.push(false);
        for name in params {
            let (p, ps) = sym(&mut g, &format!("{name}{i}"));
            args.push(p);
            syms.push(ps);
            pure.push(true);
        }
        let a = g.call(f, 0, &args);
        let b = g.call(f, 1, &args);
        roots.push(g.add(a, b));
    }
    Circuit {
        g,
        roots,
        syms,
        pure,
    }
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
