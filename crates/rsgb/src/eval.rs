use std::collections::HashMap;

use crate::field::Field;
use num_complex::Complex64;

use crate::func::{CompiledBody, FuncId, Output};
use crate::graph::Context;
use crate::node::ArgList;
use crate::node::{
    cmp_bool, dot_slice, reduce_slice, unary_f64, ExprId, Node, ReduceOp, SymbolId, UnaryOp,
};

/// Evaluate a set of expression roots over the reals, given numeric symbol
/// bindings. Single forward sweep over the arena (ascending `ExprId` is already
/// topological), so every shared subexpression is computed exactly once -- this
/// is the in-Rust numeric evaluator behind the Python interface (residual /
/// Jacobian per Newton step). Calls evaluate their function once per distinct
/// argument list (see [`FuncEval`]).
pub fn eval_real<K: Field>(
    ctx: &Context<K>,
    env: &HashMap<SymbolId, f64>,
    roots: &[ExprId],
) -> Vec<f64> {
    let w = eval_real_all(ctx, env);
    roots.iter().map(|&r| w[r.0 as usize]).collect()
}

/// Evaluate the real value of *every* node in `ctx` (topological order) and
/// return the full work vector. Indexed by `ExprId`. Used by diagnostics that
/// need to locate the first non-finite node: since nodes are interned bottom-up,
/// the lowest-index non-finite entry is the origin of a NaN/Inf (all its
/// operands have smaller indices and are therefore finite).
pub fn eval_real_all<K: Field>(ctx: &Context<K>, env: &HashMap<SymbolId, f64>) -> Vec<f64> {
    let n = ctx.len();
    let mut w = vec![0.0_f64; n];
    let mut fe = FuncEval::new();
    let g = |w: &[f64], id: ExprId| w[id.0 as usize];
    for i in 0..n {
        w[i] = match ctx.node(ExprId(i as u32)) {
            Node::Const(c) => ctx.const_val(*c).to_f64(),
            Node::Symbol(s) => env.get(s).copied().unwrap_or(f64::NAN),
            Node::Add(a, b) => g(&w, *a) + g(&w, *b),
            Node::Mul(a, b) => g(&w, *a) * g(&w, *b),
            Node::Neg(a) => -g(&w, *a),
            Node::Pow(a, k) => g(&w, *a).powi(*k as i32),
            Node::Unary(op, a) => unary_f64(*op, g(&w, *a)),
            Node::Cmp(op, a, b) => {
                if cmp_bool(*op, g(&w, *a), g(&w, *b)) {
                    1.0
                } else {
                    0.0
                }
            }
            Node::Select(c, t, e) => {
                if g(&w, *c) != 0.0 {
                    g(&w, *t)
                } else {
                    g(&w, *e)
                }
            }
            Node::Reduce(op, l) => {
                let vals: Vec<f64> = ctx.args(*l).iter().map(|&a| g(&w, a)).collect();
                reduce_slice(*op, &vals)
            }
            Node::Dot(l) => {
                let (a, b) = ctx.dot_args(*l);
                let va: Vec<f64> = a.iter().map(|&x| g(&w, x)).collect();
                let vb: Vec<f64> = b.iter().map(|&y| g(&w, y)).collect();
                dot_slice(&va, &vb)
            }
            Node::Call(o, l) => {
                let vals: Vec<f64> = ctx.args(*l).iter().map(|&a| g(&w, a)).collect();
                let (f, out) = ctx.output(*o);
                fe.output(ctx, f, out, *l, &vals)
            }
        };
    }
    w
}

/// Per-sweep function evaluation: a body per function (the registered one,
/// else an interpreted one built here) and the outputs of every distinct
/// `(function, argument list)` already evaluated.
pub struct FuncEval {
    bodies: rustc_hash::FxHashMap<FuncId, CompiledBody>,
    vals: rustc_hash::FxHashMap<(FuncId, ArgList), Vec<f64>>,
}

impl FuncEval {
    pub fn new() -> Self {
        Self {
            bodies: Default::default(),
            vals: Default::default(),
        }
    }

    /// Output `out` of `f` at the argument values `args` (whose interned list
    /// `l` keys the memo).
    pub fn output<K: Field>(
        &mut self,
        ctx: &Context<K>,
        f: FuncId,
        out: u32,
        l: ArgList,
        args: &[f64],
    ) -> f64 {
        let func = ctx.func(f);
        if matches!(func.outputs[out as usize], Output::Zero) {
            return 0.0;
        }
        let body = self
            .bodies
            .entry(f)
            .or_insert_with(|| match func.evaluator() {
                Some((b, slot_of)) if slot_of.get(out as usize).copied().flatten().is_some() => {
                    CompiledBody {
                        bundle: b.clone(),
                        slot_of,
                    }
                }
                _ => func
                    .interpreted_body(ctx)
                    .expect("extern function output without a slot"),
            });
        let Some(slot) = body.slot_of.get(out as usize).copied().flatten() else {
            // The cached body predates this output (a derivative demanded
            // later): rebuild the interpreted body over every output.
            let fresh = func.interpreted_body(ctx).expect("symbolic function");
            self.vals.retain(|k, _| k.0 != f);
            *body = fresh;
            let slot = body.slot_of[out as usize].expect("interpreted body has every output");
            return self.eval_slot(f, l, args, slot);
        };
        self.eval_slot(f, l, args, slot)
    }

    fn eval_slot(&mut self, f: FuncId, l: ArgList, args: &[f64], slot: u32) -> f64 {
        let body = &self.bodies[&f];
        let vals = self.vals.entry((f, l)).or_insert_with(|| {
            let mut out = vec![0.0; body.bundle.n_outputs()];
            body.bundle.call(args, &mut out);
            out
        });
        vals[slot as usize]
    }
}

impl Default for FuncEval {
    fn default() -> Self {
        Self::new()
    }
}

/// Numerically evaluate an expression over the complex field.
///
/// Every free symbol must be bound in `env` (component values as real
/// `Complex64`, the Laplace variable `s` as `j*omega` for AC analysis).
/// This is the bridge from the symbolic layer to numeric results (Bode,
/// verification); the fast batched evaluator (tape) comes later.
pub fn eval<K: Field>(
    ctx: &Context<K>,
    id: ExprId,
    env: &HashMap<SymbolId, Complex64>,
) -> Complex64 {
    // Memoize per node: the expression is a hash-consed DAG (a node is reachable
    // by many paths -- e.g. an `H(s)` from symbolic LU), so naive recursion is
    // worst-case exponential. The cache makes it linear in the reachable nodes
    // while preserving `Select` short-circuiting (an untaken branch is never
    // visited, hence never cached).
    let mut memo: HashMap<ExprId, Complex64> = HashMap::new();
    eval_memo(ctx, id, env, &mut memo)
}

fn eval_memo<K: Field>(
    ctx: &Context<K>,
    id: ExprId,
    env: &HashMap<SymbolId, Complex64>,
    memo: &mut HashMap<ExprId, Complex64>,
) -> Complex64 {
    if let Some(&v) = memo.get(&id) {
        return v;
    }
    let v = eval_node(ctx, id, env, memo);
    memo.insert(id, v);
    v
}

fn eval_node<K: Field>(
    ctx: &Context<K>,
    id: ExprId,
    env: &HashMap<SymbolId, Complex64>,
    memo: &mut HashMap<ExprId, Complex64>,
) -> Complex64 {
    match ctx.node(id) {
        Node::Const(c) => Complex64::new(ctx.const_val(*c).to_f64(), 0.0),
        Node::Symbol(s) => *env
            .get(s)
            .unwrap_or_else(|| panic!("unbound symbol '{}'", ctx.symbol_name(*s))),
        Node::Add(a, b) => eval_memo(ctx, *a, env, memo) + eval_memo(ctx, *b, env, memo),
        Node::Mul(a, b) => eval_memo(ctx, *a, env, memo) * eval_memo(ctx, *b, env, memo),
        Node::Neg(a) => -eval_memo(ctx, *a, env, memo),
        Node::Pow(a, n) => eval_memo(ctx, *a, env, memo).powi(*n as i32),
        Node::Unary(op, a) => {
            let x = eval_memo(ctx, *a, env, memo);
            match op {
                UnaryOp::Exp => x.exp(),
                UnaryOp::Ln => x.ln(),
                UnaryOp::Sqrt => x.sqrt(),
                UnaryOp::Sin => x.sin(),
                UnaryOp::Cos => x.cos(),
                UnaryOp::Sinh => x.sinh(),
                UnaryOp::Cosh => x.cosh(),
                UnaryOp::Tanh => x.tanh(),
                UnaryOp::Atan => x.atan(),
                UnaryOp::Floor => Complex64::new(x.re.floor(), 0.0),
            }
        }
        Node::Cmp(op, a, b) => {
            // Compare real parts; result is the indicator 1.0 / 0.0.
            let (x, y) = (
                eval_memo(ctx, *a, env, memo).re,
                eval_memo(ctx, *b, env, memo).re,
            );
            Complex64::new(if cmp_bool(*op, x, y) { 1.0 } else { 0.0 }, 0.0)
        }
        Node::Select(c, t, e) => {
            // Short-circuit so an untaken (possibly opaque) branch isn't evaluated.
            if eval_memo(ctx, *c, env, memo).re != 0.0 {
                eval_memo(ctx, *t, env, memo)
            } else {
                eval_memo(ctx, *e, env, memo)
            }
        }
        Node::Reduce(op, l) => {
            // Sum / Product compose over the complex field; Min / Max compare
            // real parts (consistent with `Cmp`), since they only arise in
            // time-domain region/source logic, not the AC transfer.
            let vals: Vec<Complex64> = ctx
                .args(*l)
                .iter()
                .map(|&a| eval_memo(ctx, a, env, memo))
                .collect();
            let it = vals.into_iter();
            match op {
                ReduceOp::Sum => it.fold(Complex64::new(0.0, 0.0), |acc, z| acc + z),
                ReduceOp::Product => it.fold(Complex64::new(1.0, 0.0), |acc, z| acc * z),
                ReduceOp::Min => it
                    .reduce(|acc, z| if z.re < acc.re { z } else { acc })
                    .unwrap_or(Complex64::new(0.0, 0.0)),
                ReduceOp::Max => it
                    .reduce(|acc, z| if z.re > acc.re { z } else { acc })
                    .unwrap_or(Complex64::new(0.0, 0.0)),
            }
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(*l);
            let mut acc = Complex64::new(0.0, 0.0);
            for (&x, &y) in a.iter().zip(b.iter()) {
                acc += eval_memo(ctx, x, env, memo) * eval_memo(ctx, y, env, memo);
            }
            acc
        }
        // A bundled opaque with purely real arguments evaluates through its
        // (real-valued) body -- the symbolic-eval convenience path then works
        // on batched instances too. A genuinely complex argument has no real
        // body to call, so that stays a loud error.
        Node::Call(o, l) => {
            let vals: Vec<Complex64> = ctx
                .args(*l)
                .iter()
                .map(|&a| eval_memo(ctx, a, env, memo))
                .collect();
            let (f, out) = ctx.output(*o);
            if vals.iter().all(|v| v.im == 0.0) {
                let re: Vec<f64> = vals.iter().map(|v| v.re).collect();
                Complex64::new(FuncEval::new().output(ctx, f, out, *l, &re), 0.0)
            } else {
                panic!(
                    "cannot evaluate a call of '{}' over complex arguments",
                    ctx.func(f).name
                )
            }
        }
    }
}

/// Convenience wrapper: bind symbols by name and evaluate.
pub fn eval_named<K: Field>(
    ctx: &mut Context<K>,
    id: ExprId,
    values: &[(&str, Complex64)],
) -> Complex64 {
    let env: HashMap<SymbolId, Complex64> = values
        .iter()
        .map(|(name, v)| {
            let sid_node = ctx.sym(name);
            match ctx.node(sid_node) {
                Node::Symbol(sid) => (*sid, *v),
                _ => unreachable!("sym() always yields a Symbol node"),
            }
        })
        .collect();
    eval(ctx, id, &env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Context;

    #[test]
    fn complex_eval_is_linear_on_shared_dag() {
        // Balanced doubling DAG: x_{k+1} = x_k + x_k, one hash-consed node that
        // references x_k twice. 40 levels => 2^40 naive recursions but only 40
        // distinct nodes. Completing quickly proves `eval` memoizes (is linear in
        // reachable nodes, not exponential in paths).
        let mut ctx: Context = Context::new();
        let s = ctx.sym("s");
        let mut e = s;
        for _ in 0..40 {
            e = ctx.add(e, e);
        }
        let sid = match ctx.node(s) {
            Node::Symbol(id) => *id,
            _ => unreachable!(),
        };
        let mut env = HashMap::new();
        env.insert(sid, Complex64::new(1.0, 0.0));
        let v = eval(&ctx, e, &env);
        assert!((v.re - 2f64.powi(40)).abs() < 1.0, "got {}", v.re);
    }
}
