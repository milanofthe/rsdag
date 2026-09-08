use std::collections::HashMap;

use crate::field::Field;
use num_complex::Complex64;

use crate::func::{CompiledBody, FuncId, Output};
use crate::graph::Graph;
use crate::node::ArgList;
use crate::node::{ExprId, Node, SymbolId};
use crate::scalar::{dot_slice_t, reduce_slice_t, Scalar};

/// The value of one node from its operands, for any execution scalar.
///
/// The single per-node semantics of the arena: the real sweep, the complex
/// evaluator and (through [`crate::Tape::eval_typed`]) the tape all take
/// their arithmetic from [`Scalar`] and their fold orders from
/// [`reduce_slice_t`] and [`dot_slice_t`], so a value cannot depend on which
/// of them computed it.
///
/// `get` reads an operand: an array lookup in the sweep, a memoised
/// recursion in the lazy evaluator. `Select` calls it only for the branch it
/// takes, so a lazy caller does not evaluate the other one.
fn node_value<T: Scalar, K: Field>(
    ctx: &Graph<K>,
    node: &Node,
    mut get: impl FnMut(ExprId) -> T,
    sym: &mut impl FnMut(SymbolId) -> T,
    call: &mut impl FnMut(FuncId, u32, ArgList, &[T]) -> T,
) -> T {
    match *node {
        Node::Const(c) => T::from_f64(ctx.const_val(c).to_f64()),
        Node::Symbol(s) => sym(s),
        Node::Add(a, b) => get(a).add(get(b)),
        Node::Mul(a, b) => get(a).mul(get(b)),
        Node::Neg(a) => get(a).neg(),
        Node::Pow(a, k) => get(a).powi(k as i32),
        Node::Unary(op, a) => T::unary(op, get(a)),
        Node::Binary(op, a, b) => T::binary(op, get(a), get(b)),
        Node::Cmp(op, a, b) => T::cmp(op, get(a), get(b)),
        Node::Select(c, t, e) => {
            if get(c).is_true() {
                get(t)
            } else {
                get(e)
            }
        }
        Node::Reduce(op, l) => {
            let vals: Vec<T> = ctx.args(l).iter().map(|&a| get(a)).collect();
            reduce_slice_t(op, &vals)
        }
        Node::Dot(l) => {
            let (a, b) = ctx.dot_args(l);
            let va: Vec<T> = a.iter().map(|&x| get(x)).collect();
            let vb: Vec<T> = b.iter().map(|&y| get(y)).collect();
            dot_slice_t(&va, &vb)
        }
        Node::Call(o, l) => {
            let vals: Vec<T> = ctx.args(l).iter().map(|&a| get(a)).collect();
            let (f, out) = ctx.output(o);
            call(f, out, l, &vals)
        }
    }
}

/// Evaluate a set of expression roots over the reals, given numeric symbol
/// bindings. Single forward sweep over the arena (ascending `ExprId` is already
/// topological), so every shared subexpression is computed exactly once -- this
/// is the in-Rust numeric evaluator behind the Python interface (residual /
/// Jacobian per Newton step). Calls evaluate their function once per distinct
/// argument list (see [`FuncEval`]).
pub fn eval_real<K: Field>(
    ctx: &Graph<K>,
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
pub fn eval_real_all<K: Field>(ctx: &Graph<K>, env: &HashMap<SymbolId, f64>) -> Vec<f64> {
    let n = ctx.len();
    let mut w = vec![0.0_f64; n];
    let mut fe = FuncEval::new();
    for i in 0..n {
        let node = *ctx.node(ExprId(i as u32));
        // An unbound symbol is NaN here rather than an error: a diagnostic
        // sweep over a partially bound graph must still produce a value.
        let mut sym = |s: SymbolId| env.get(&s).copied().unwrap_or(f64::NAN);
        let mut call =
            |f: FuncId, out: u32, l: ArgList, args: &[f64]| fe.output(ctx, f, out, l, args);
        w[i] = node_value(ctx, &node, |e| w[e.0 as usize], &mut sym, &mut call);
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
        ctx: &Graph<K>,
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
pub fn eval<K: Field>(ctx: &Graph<K>, id: ExprId, env: &HashMap<SymbolId, Complex64>) -> Complex64 {
    // Memoize per node: the expression is a hash-consed DAG (a node is reachable
    // by many paths -- e.g. an `H(s)` from symbolic LU), so naive recursion is
    // worst-case exponential. The cache makes it linear in the reachable nodes
    // while preserving `Select` short-circuiting (an untaken branch is never
    // visited, hence never cached).
    let mut memo: HashMap<ExprId, Complex64> = HashMap::new();
    eval_memo(ctx, id, env, &mut memo)
}

fn eval_memo<K: Field>(
    ctx: &Graph<K>,
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
    ctx: &Graph<K>,
    id: ExprId,
    env: &HashMap<SymbolId, Complex64>,
    memo: &mut HashMap<ExprId, Complex64>,
) -> Complex64 {
    let node = *ctx.node(id);
    // Unlike the sweep, an unbound symbol is an error: a transfer function
    // evaluated with a missing component value is a caller mistake, not a
    // NaN to propagate.
    let mut sym = |s: SymbolId| {
        *env.get(&s)
            .unwrap_or_else(|| panic!("unbound symbol '{}'", ctx.symbol_name(s)))
    };
    // A call with purely real arguments evaluates through its real body, so
    // the convenience path works on bundled opaques too; a genuinely complex
    // argument has no real body to call.
    let mut call = |f: FuncId, out: u32, l: ArgList, args: &[Complex64]| {
        if args.iter().all(|v| v.im == 0.0) {
            let re: Vec<f64> = args.iter().map(|v| v.re).collect();
            Complex64::new(FuncEval::new().output(ctx, f, out, l, &re), 0.0)
        } else {
            panic!(
                "cannot evaluate a call of '{}' over complex arguments",
                ctx.func(f).name
            )
        }
    };
    node_value(
        ctx,
        &node,
        |e| eval_memo(ctx, e, env, memo),
        &mut sym,
        &mut call,
    )
}

/// Convenience wrapper: bind symbols by name and evaluate.
pub fn eval_named<K: Field>(
    ctx: &mut Graph<K>,
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
