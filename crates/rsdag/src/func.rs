//! Functions in the graph: a sub-DAG over formal parameters, applied through
//! [`Node::Call`](crate::node::Node::Call) nodes.
//!
//! A compact model instantiated a thousand times is one function and a
//! thousand calls. The function's outputs are its terminal currents, internal
//! residual rows, noise densities, operating-point variables -- whatever the
//! frontend lowers once over the formal leaves; a call binds those leaves to
//! one instance's expressions. Everything the engine does with a call is a
//! property of this one construct:
//!
//! - **differentiation** is the chain rule over *derivative outputs*
//!   (`d out_j / d param_i`), which the context differentiates from the body
//!   on first demand and memoises, so the Jacobian of a circuit references
//!   derivative calls into the same function -- no marker names, no parsing;
//! - **inlining** substitutes the arguments into the output expression, which
//!   is how a single-instance module keeps a fully symbolic fragment and how
//!   the differential reference (every instance as its own graph) is produced;
//! - **evaluation** runs the body once per distinct argument list and reads
//!   the outputs, in the arena evaluator through a per-function tape and in a
//!   compiled tape through its body (its native forms
//!   and lane batching); a batch of calls into one function is what the SIMD
//!   lanes evaluate.
//!
//! An *extern* function has no symbolic body: its outputs, including the
//! derivative outputs it can supply, are slots of a numeric
//! [`crate::extern_fn::ExternBundle`] (a compiled OSDI model).
//! A derivative an extern cannot supply is the zero output.

use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::extern_fn::ExternBundle;
use crate::node::{ExprId, SymbolId};
use crate::role::{OutputRole, ParamRole};
use crate::tape::Program as _;

/// Index of a function in a [`Graph`](crate::graph::Graph).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct FuncId(pub u32);

/// An interned `(function, output index)` pair -- what a `Call` node names.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct OutputId(pub u32);

/// One output of a function.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Output {
    /// A symbolic expression over the function's parameters.
    Expr(ExprId),
    /// Output slot `k` of an extern body.
    Slot(u32),
    /// Identically zero (a derivative the body does not carry).
    Zero,
}

#[derive(Clone)]
pub(crate) struct Body {
    pub(crate) bundle: Arc<dyn ExternBundle>,
    pub(crate) slot_of: Vec<Option<u32>>,
}

/// A function of a [`Graph`](crate::graph::Graph), read through its
/// accessors and changed only through the graph, so the derivative index
/// always says what the output roles say.
pub struct Function {
    name: String,
    params: Vec<SymbolId>,
    param_roles: Vec<ParamRole>,
    outputs: Vec<Output>,
    output_roles: Vec<OutputRole>,
    extern_body: Option<Arc<dyn ExternBundle>>,
    interpreted: std::sync::Mutex<Vec<Body>>,
    lowered: std::sync::Mutex<Option<Arc<crate::tape::Lowered>>>,
    deriv_index: HashMap<(u32, u32), u32>,
    pub(crate) deps: std::sync::Mutex<Option<Arc<crate::graph::Deps>>>,
    support: std::sync::Mutex<Vec<Option<Arc<[u32]>>>>,
    reads: std::sync::Mutex<HashMap<ReadsKey, Arc<[Arc<[u32]>]>>>,
    pub(crate) globals: std::sync::OnceLock<Globals>,
    composite: std::sync::OnceLock<bool>,
}

pub(crate) type ReadsKey = (crate::graph::Through, Box<[u32]>);

pub(crate) struct InterpretedBody {
    pub(crate) tape: crate::tape::Tape,
    n_out: usize,
    pure: Vec<bool>,
    pub(crate) backends: crate::extern_fn::BackendCache,
    rate: crate::parallel::Rate,
}

impl InterpretedBody {
    pub(crate) fn new(tape: crate::tape::Tape, n_out: usize, pure: Vec<bool>) -> InterpretedBody {
        InterpretedBody {
            tape,
            n_out,
            pure,
            backends: Default::default(),
            rate: crate::parallel::Rate::new(crate::constants::INTERPRETED_NS_PER_OP),
        }
    }
}

impl ExternBundle for InterpretedBody {
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    fn rate(&self) -> &crate::parallel::Rate {
        &self.rate
    }
    fn batch_lanes(&self) -> usize {
        self.tape.batch_lanes()
    }
    fn call_into(&self, args: &[f64], _work: &mut [f64], out: &mut [f64]) {
        let at = crate::Instances::first(1, args.len(), 0);
        self.tape.eval_batch(args, out, self.n_out, &at);
    }
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        let at = crate::Instances::first(n_groups, n_args, 0);
        self.tape.eval_batch(args, out, self.n_out, &at);
    }
    fn state_len(&self) -> usize {
        self.tape.state_len()
    }
    fn pure_args(&self) -> &[bool] {
        &self.pure
    }
    fn prolog_into(&self, pure: &[f64], _work: &mut [f64], state: &mut [f64]) {
        let at = crate::Instances::first(1, pure.len(), self.tape.state_len());
        self.prolog_batch(pure, state, &at);
    }
    fn prolog_batch(&self, pure: &[f64], states: &mut [f64], at: &crate::Instances) {
        self.tape.prolog_batch(&self.pure, pure, states, at);
    }
    fn main_into(&self, args: &[f64], state: &[f64], _work: &mut [f64], out: &mut [f64]) {
        let at = crate::Instances::first(1, args.len(), self.tape.state_len());
        self.main_batch(args, state, out, &at);
    }
    fn main_batch(&self, args: &[f64], states: &[f64], out: &mut [f64], at: &crate::Instances) {
        self.tape.main_batch(args, states, out, self.n_out, at);
    }
    fn body(&self) -> Option<&crate::tape::Tape> {
        Some(&self.tape)
    }
    fn backend_cache(&self) -> Option<&crate::extern_fn::BackendCache> {
        Some(&self.backends)
    }
}

pub(crate) struct Globals {
    pub(crate) exprs: Arc<[ExprId]>,
    pub(crate) symbols: Arc<[SymbolId]>,
}

impl Function {
    pub(crate) fn new(
        name: &str,
        params: Vec<SymbolId>,
        param_roles: Vec<ParamRole>,
        extern_body: Option<Arc<dyn ExternBundle>>,
    ) -> Function {
        Function {
            name: name.to_string(),
            param_roles,
            params,
            outputs: Vec::new(),
            output_roles: Vec::new(),
            extern_body,
            interpreted: Default::default(),
            lowered: Default::default(),
            deriv_index: HashMap::default(),
            deps: Default::default(),
            support: Default::default(),
            reads: Default::default(),
            globals: Default::default(),
            composite: Default::default(),
        }
    }

    pub(crate) fn push_output(&mut self, output: Output, role: OutputRole) -> u32 {
        let may_read_new_globals = !matches!(role, OutputRole::Derivative { .. });
        if may_read_new_globals {
            self.globals = Default::default();
            self.composite = Default::default();
            self.deps = Default::default();
        }
        let k = self.outputs.len() as u32;
        if let OutputRole::Derivative { of, wrt } = role {
            self.deriv_index.insert((of, wrt), k);
        }
        self.outputs.push(output);
        self.output_roles.push(role);
        k
    }

    pub(crate) fn derivative(&self, out: u32, param: u32) -> Option<u32> {
        self.deriv_index.get(&(out, param)).copied()
    }

    pub(crate) fn globals_in<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
    ) -> &Globals {
        if let Some(g) = self.globals.get() {
            return g;
        }
        let exprs = self.exprs();
        let params: rustc_hash::FxHashSet<SymbolId> = self.params.iter().copied().collect();
        let mut globals: Vec<(ExprId, SymbolId)> = ctx
            .cone(&exprs, true)
            .nodes
            .iter()
            .filter_map(|&e| match *ctx.node(e) {
                crate::node::Node::Symbol(s) if !params.contains(&s) => Some((e, s)),
                _ => None,
            })
            .collect();
        globals.sort_unstable();
        self.globals.get_or_init(|| Globals {
            exprs: globals.iter().map(|&(e, _)| e).collect(),
            symbols: globals.iter().map(|&(_, s)| s).collect(),
        })
    }

    pub(crate) fn composite_in<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
    ) -> bool {
        *self.composite.get_or_init(|| {
            let exprs: Vec<ExprId> = self
                .outputs
                .iter()
                .zip(&self.output_roles)
                .filter_map(|(o, r)| match (*o, r) {
                    (_, OutputRole::Derivative { .. }) => None,
                    (Output::Expr(e), _) => Some(e),
                    _ => None,
                })
                .collect();
            !ctx.free_calls_in(&exprs).is_empty()
        })
    }

    pub(crate) fn cached_reads(&self, key: &ReadsKey) -> Option<Arc<[Arc<[u32]>]>> {
        self.reads.lock().unwrap().get(key).cloned()
    }

    pub(crate) fn cached_support(&self, out: u32) -> Option<Arc<[u32]>> {
        self.support
            .lock()
            .unwrap()
            .get(out as usize)
            .cloned()
            .flatten()
    }

    pub(crate) fn cache_support(&self, out: u32, support: Arc<[u32]>) {
        let mut cache = self.support.lock().unwrap();
        if cache.len() <= out as usize {
            cache.resize(out as usize + 1, None);
        }
        cache[out as usize] = Some(support);
    }

    pub(crate) fn cache_reads(&self, key: ReadsKey, reads: Arc<[Arc<[u32]>]>) {
        self.reads.lock().unwrap().insert(key, reads);
    }

    pub(crate) fn exprs(&self) -> Vec<ExprId> {
        (self.outputs.iter())
            .filter_map(|o| match *o {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn outputs_over(&self, exprs: Vec<ExprId>) -> Vec<Output> {
        let mut outputs = self.outputs.clone();
        let is_expr = |o: &Output| matches!(o, Output::Expr(_));
        assert_eq!(outputs.iter().filter(|o| is_expr(o)).count(), exprs.len());
        let slots = outputs.iter_mut().filter(|o| is_expr(o));
        for (o, e) in slots.zip(exprs) {
            *o = Output::Expr(e);
        }
        outputs
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    /// Formal leaves in argument order.
    pub fn params(&self) -> &[SymbolId] {
        &self.params
    }
    /// One role per parameter.
    pub fn param_roles(&self) -> &[ParamRole] {
        &self.param_roles
    }
    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }
    /// One role per output (derivative outputs are tagged `Derivative`).
    pub fn output_roles(&self) -> &[OutputRole] {
        &self.output_roles
    }
    pub(crate) fn extern_body(&self) -> Option<&Arc<dyn ExternBundle>> {
        self.extern_body.as_ref()
    }

    pub(crate) fn body<K: crate::field::Field>(&self, ctx: &crate::graph::Graph<K>) -> Body {
        let all: Vec<u32> = (0..self.outputs.len() as u32).collect();
        self.body_for(ctx, &all)
    }

    pub(crate) fn body_for<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
        needed: &[u32],
    ) -> Body {
        if self.extern_body.is_none() {
            let built = self.interpreted.lock().unwrap();
            let covering = built.iter().filter(|c| self.covers(c, needed));
            if let Some(c) = covering.min_by_key(|c| c.bundle.n_outputs()) {
                return c.clone();
            }
        }
        self.build_body(ctx, needed)
    }

    fn build_body<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
        needed: &[u32],
    ) -> Body {
        if let Some(b) = &self.extern_body {
            return Body {
                bundle: b.clone(),
                slot_of: self
                    .outputs
                    .iter()
                    .map(|o| match o {
                        Output::Slot(k) => Some(*k),
                        _ => None,
                    })
                    .collect(),
            };
        }
        let mut roots = Vec::new();
        let mut slot_of = vec![None; self.outputs.len()];
        for &k in needed {
            if let (Output::Expr(e), None) = (self.outputs[k as usize], slot_of[k as usize]) {
                slot_of[k as usize] = Some(roots.len() as u32);
                roots.push(e);
            }
        }
        let globals = &self.globals_in(ctx).symbols;
        let params_then_globals: Vec<crate::node::SymbolId> = self
            .params
            .iter()
            .copied()
            .chain(globals.iter().copied())
            .collect();
        let pure: Vec<bool> = self
            .param_roles
            .iter()
            .map(|r| matches!(r, ParamRole::Param))
            .chain(globals.iter().map(|_| true))
            .collect();
        let pure = if pure.iter().any(|&p| p) {
            pure
        } else {
            Vec::new()
        };
        let tape = self.view(ctx, &roots, &params_then_globals, &pure);
        let specializes =
            (!pure.is_empty() && tape.has_param_selects(&pure)) || tape.n_main_selects() > 0;
        let full = InterpretedBody::new(tape, roots.len(), pure);
        let bundle: Arc<dyn ExternBundle> = match specializes {
            true => Arc::new(crate::variant::VariantBody::new(full)),
            false => Arc::new(full),
        };
        let body = Body { bundle, slot_of };
        self.interpreted.lock().unwrap().push(body.clone());
        body
    }

    fn view<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
        roots: &[ExprId],
        inputs: &[crate::node::SymbolId],
        pure: &[bool],
    ) -> crate::tape::Tape {
        let split = (!pure.is_empty()).then_some(pure);
        let held = self.lowered.lock().unwrap().clone();
        let lowered = match held {
            Some(l) if l.serves(roots, inputs, split) => l,
            held => {
                let mut all: Vec<ExprId> = (held.iter())
                    .filter(|l| l.signature(inputs, split))
                    .flat_map(|l| l.roots().to_vec())
                    .collect();
                all.extend_from_slice(roots);
                let l = Arc::new(crate::tape::Lowered::new(ctx, &all, inputs, split));
                *self.lowered.lock().unwrap() = Some(l.clone());
                l
            }
        };
        lowered.tape(ctx, roots)
    }

    pub(crate) fn body_exact<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
        needed: &[u32],
    ) -> Body {
        if self.extern_body.is_none() {
            let mut exprs: Vec<u32> = (needed.iter().copied())
                .filter(|&k| matches!(self.outputs[k as usize], Output::Expr(_)))
                .collect();
            exprs.sort_unstable();
            exprs.dedup();
            let built = self.interpreted.lock().unwrap();
            let exact = (built.iter())
                .find(|c| self.covers(c, needed) && c.bundle.n_outputs() == exprs.len());
            if let Some(c) = exact {
                return c.clone();
            }
        }
        self.build_body(ctx, needed)
    }

    pub(crate) fn slot_outputs(&self, bundle: &Arc<dyn ExternBundle>) -> Option<Vec<u32>> {
        let built = self.interpreted.lock().unwrap();
        let body = built.iter().find(|c| Arc::ptr_eq(&c.bundle, bundle))?;
        let mut outs = vec![u32::MAX; body.bundle.n_outputs()];
        for (k, s) in body.slot_of.iter().enumerate() {
            if let Some(s) = s {
                outs[*s as usize] = k as u32;
            }
        }
        Some(outs)
    }

    fn covers(&self, body: &Body, needed: &[u32]) -> bool {
        needed.iter().all(|&k| {
            !matches!(self.outputs[k as usize], Output::Expr(_))
                || body.slot_of.get(k as usize).is_some_and(|s| s.is_some())
        })
    }

    /// The expression outputs carrying `role`, and their roles, in output
    /// order.
    pub fn exprs_with_role(&self, role: impl Fn(&OutputRole) -> bool) -> Vec<(ExprId, OutputRole)> {
        (self.outputs.iter().zip(&self.output_roles))
            .filter(|(_, r)| role(r))
            .filter_map(|(o, r)| match *o {
                Output::Expr(e) => Some((e, *r)),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn is_extern(&self) -> bool {
        self.extern_body.is_some()
    }
}
