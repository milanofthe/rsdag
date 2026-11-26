//! Functions, calls and the system layer over them: defining a function,
//! calling it, binding its parameters, roles, derivative outputs,
//! specialization, inlining.
//!
//! A child module of `graph` so it reaches the arena's private fields; a
//! second `impl Graph` block rather than a second type, because a call is a
//! node like any other and the functions live in the same arena.

use super::*;
use crate::transform::{Call, Contexts, Rule};

impl<K: Field> Graph<K> {
    pub(crate) fn push_function(&mut self, func: Function) -> FuncId {
        self.funcs.push(func);
        FuncId(self.funcs.len() as u32 - 1)
    }

    pub(crate) fn push_output(&mut self, f: FuncId, output: Output, role: OutputRole) -> u32 {
        self.funcs[f.0 as usize].push_output(output, role)
    }

    pub(crate) fn define(
        &mut self,
        name: &str,
        params: Vec<SymbolId>,
        roles: &[ParamRole],
        body: Option<Arc<dyn ExternBundle>>,
        outputs: impl IntoIterator<Item = (Output, OutputRole)>,
    ) -> FuncId {
        debug_assert_eq!(params.len(), roles.len(), "one role per parameter");
        let key = FuncKey {
            params: params.into(),
            roles: roles.into(),
            outputs_at_definition: outputs.into_iter().collect(),
            body_address: body.as_ref().map(|b| Arc::as_ptr(b) as *const () as usize),
        };
        if let Some(&f) = self.defined.get(&key) {
            return f;
        }
        let func = Function::new(name, key.params.to_vec(), key.roles.to_vec(), body);
        let f = self.push_function(func);
        for &(o, r) in key.outputs_at_definition.iter() {
            self.push_output(f, o, r);
        }
        self.defined.insert(key, f);
        f
    }

    /// Inline every call reachable from `roots`, to the bottom.
    ///
    /// The static fusion a consumer does before it compiles: a hierarchy of
    /// blocks or subcircuits becomes one expression per root, so the
    /// scheduler, the slot allocator and common-subexpression elimination
    /// see across the instance boundaries that the calls were hiding. The
    /// price is size -- an instance that was one `Call` becomes a copy of
    /// its body -- which is why it is a choice and not the default.
    ///
    /// Calls into an extern body cannot be inlined (there is no expression
    /// to inline) and are left as they are.
    pub fn inline_all(&mut self, roots: &[ExprId]) -> Vec<ExprId> {
        self.inline_where(roots, &mut |_, _| true)
    }

    /// Every call into a function whose body calls further functions,
    /// inlined, to the bottom; the calls of leaf functions (a device model's
    /// body) stay calls: the hierarchy flattened down to its leaves, as an
    /// expression. A compiled program needs no such rewrite: the tape
    /// compiler takes a composite function as a template and batches the
    /// leaf calls of every instance itself (see [`Tape::compile`](crate::Tape::compile)).
    pub fn inline_composite(&mut self, roots: &[ExprId]) -> Vec<ExprId> {
        let mut composite: HashMap<FuncId, bool> = HashMap::default();
        self.inline_where(roots, &mut |g, f| {
            *composite
                .entry(f)
                .or_insert_with(|| !g.free_calls_in(&g.funcs[f.0 as usize].exprs()).is_empty())
        })
    }

    fn inline_where(
        &mut self,
        roots: &[ExprId],
        which: &mut dyn FnMut(&Self, FuncId) -> bool,
    ) -> Vec<ExprId> {
        let _s = rsprobe::scope!("inline");
        let nodes = self.len();
        let mut rule = Inline {
            g: self,
            which,
            bodies: &mut HashMap::default(),
            contexts: Contexts::default(),
        };
        let out = crate::transform::rewrite(&mut rule, roots);
        rsprobe::add!("nodes", self.len() - nodes);
        out
    }

    /// Every call under `roots` that passes constants, redirected to a copy
    /// of its function specialized to them: the copy's outputs are the
    /// function's with those parameters replaced by the constants (and
    /// folded), it takes the other arguments only. The calls that pass the
    /// same constants in the same places share one copy, so their instances
    /// still run as one batch, and calls in a copy's body are specialized
    /// too. A ground terminal, or the derivatives of a DC analysis set to
    /// zero, take their share of a device body away before it is compiled.
    /// A parameter with the `Param` role stays an argument when constant:
    /// its work is the body's prolog, once per binding, and the instances
    /// keep sharing one body whatever their parameter values.
    ///
    /// The copy keeps the roles of the parameters it keeps; its outputs are
    /// `Plain` apart from their non-derivative roles, a derivative of it is
    /// derived anew. Calls into an extern body are left as they are.
    pub fn specialize_calls(&mut self, roots: &[ExprId]) -> Vec<ExprId> {
        let _s = rsprobe::scope!("specialize calls");
        let funcs = self.n_funcs();
        let out = specialize_calls_in(self, roots, &mut HashMap::default());
        rsprobe::add!("functions", self.n_funcs() - funcs);
        out
    }

    /// `f` with the globals `map` binds to something else than themselves
    /// substituted in its body, its nested calls' included: the function a
    /// call of `f` runs where `map` holds (a substitution, a caller binding
    /// a parameter some called body reads as a global). `None` when `map`
    /// binds none of them. One copy per function and binding; the copy
    /// keeps `f`'s parameters, roles and output indices, and the derivative
    /// roles with respect to its parameters.
    pub fn rebound(&mut self, f: FuncId, map: &HashMap<SymbolId, ExprId>) -> Option<FuncId> {
        let globals = self.globals(f);
        let mut binding: Vec<(SymbolId, ExprId)> = globals
            .iter()
            .filter_map(|&e| match *self.node(e) {
                Node::Symbol(s) => map.get(&s).filter(|&&v| v != e).map(|&v| (s, v)),
                _ => None,
            })
            .collect();
        if binding.is_empty() {
            return None;
        }
        binding.sort_unstable();
        if let Some(&copy) = self.rebound.get(&(f, binding.clone())) {
            return Some(copy);
        }
        let func = self.func(f);
        let (name, params) = (func.name().to_string(), func.params().to_vec());
        let roles = func.param_roles().to_vec();
        let (exprs, out_roles) = (func.exprs(), func.output_roles().to_vec());
        let bound: HashMap<SymbolId, ExprId> = binding.iter().copied().collect();
        let done = crate::transform::substitute(self, &exprs, &bound);
        let outputs = self.func(f).outputs_over(done);
        let n_params = params.len();
        let derivative_wrt_global = |r: &OutputRole| matches!(r, OutputRole::Derivative { wrt, .. } if *wrt as usize >= n_params);
        let out_roles = out_roles.into_iter().map(|r| {
            if derivative_wrt_global(&r) {
                OutputRole::Plain
            } else {
                r
            }
        });
        let copy = self.define(
            &name,
            params,
            &roles,
            None,
            outputs.into_iter().zip(out_roles),
        );
        self.rebound.insert((f, binding), copy);
        Some(copy)
    }

    pub(crate) fn output_support(&self, f: FuncId, out: u32) -> Arc<[u32]> {
        let func = &self.funcs[f.0 as usize];
        if let Some(s) = func.cached_support(out) {
            return s;
        }
        let n = func.params().len() + self.globals(f).len();
        let all: Vec<u32> = (0..n as u32).collect();
        let reads = self.reads(f, Through::Carries, &all);
        for (k, r) in reads.iter().enumerate() {
            func.cache_support(k as u32, r.clone());
        }
        reads[out as usize].clone()
    }

    /// Per root, whether its value reads any of `syms`, a comparison's
    /// operands and a selector's condition included: which entries of a
    /// Jacobian vary with the state, say. One pass over the roots' cone; a
    /// call reads them if an argument that does goes into its output's
    /// value, which the called function answers once per pattern of such
    /// arguments.
    pub fn depends_on(&self, roots: &[ExprId], syms: &[SymbolId]) -> Vec<bool> {
        let wanted: FxHashSet<SymbolId> = syms.iter().copied().collect();
        let flow = self.flow(roots, Through::Reads, |n| match *n {
            Node::Symbol(s) => wanted.contains(&s),
            _ => false,
        });
        roots.iter().map(|&r| *flow.get(r)).collect()
    }

    /// Define an extern function over one argument per role of `roles`, its
    /// outputs the given slots of `body` (or [`Output::Zero`]) with their
    /// roles. Derivative outputs the body carries are declared with
    /// [`declare_derivative`](Self::declare_derivative).
    pub fn define_extern_func(
        &mut self,
        name: &str,
        roles: &[ParamRole],
        body: Arc<dyn ExternBundle>,
        outputs: impl IntoIterator<Item = (Output, OutputRole)>,
    ) -> FuncId {
        let params: Vec<SymbolId> = (0..roles.len())
            .map(|i| self.symbol_id(&format!("{name}.${i}")))
            .collect();
        self.define(name, params, roles, Some(body), outputs)
    }

    /// Declare `d outputs[out] / d params[param]` of an extern function as
    /// `deriv` (a slot of its body, or zero). Undeclared derivatives are zero.
    pub fn declare_derivative(&mut self, f: FuncId, out: u32, param: u32, deriv: Output) -> u32 {
        self.push_output(
            f,
            deriv,
            OutputRole::Derivative {
                of: out,
                wrt: param,
            },
        )
    }

    pub fn func(&self, f: FuncId) -> &Function {
        &self.funcs[f.0 as usize]
    }

    pub(crate) fn n_symbols(&self) -> usize {
        self.symbol_names.len()
    }

    pub fn n_funcs(&self) -> usize {
        self.funcs.len()
    }

    /// The `(function, output index)` an output id names.
    #[inline]
    pub fn output(&self, o: OutputId) -> (FuncId, u32) {
        self.outputs[o.0 as usize]
    }

    pub(crate) fn output_in(&mut self, f: FuncId, out: u32, ctx: u32) -> OutputId {
        if let Some(&o) = self.output_dedup.get(&(f, out, ctx)) {
            return o;
        }
        let o = OutputId(self.outputs.len() as u32);
        self.outputs.push((f, out));
        self.output_ctx.push(ctx);
        self.output_dedup.insert((f, out, ctx), o);
        o
    }

    /// The context a call of output `o` runs in (see [`bind`](Self::bind)):
    /// its function's bound parameters, ascending, and their expressions.
    pub fn context(&self, o: OutputId) -> Option<(&[u32], &[ExprId])> {
        match self.output_ctx[o.0 as usize] {
            NO_CONTEXT => None,
            c => {
                let c = &self.contexts[c as usize];
                Some((&c.at, self.args(c.exprs)))
            }
        }
    }

    pub(crate) fn context_list(&self, c: u32) -> ArgList {
        self.contexts[c as usize].exprs
    }

    pub(crate) fn context_of(&self, o: OutputId) -> u32 {
        self.output_ctx[o.0 as usize]
    }

    /// `f` partially applied: each parameter `arg` binds bound (see
    /// [`bind`](Self::bind)), and the arguments a call through the binding
    /// passes, the parameters `arg` passes, in their order.
    pub fn partial(
        &mut self,
        f: FuncId,
        mut arg: impl FnMut(&mut Self, SymbolId) -> Arg,
    ) -> (Bound, Vec<ExprId>) {
        let params = self.func(f).params().to_vec();
        let (mut bound, mut args) = (Vec::new(), Vec::new());
        for (k, s) in params.into_iter().enumerate() {
            match arg(self, s) {
                Arg::Bind(e) => bound.push((k as u32, e)),
                Arg::Pass(e) => args.push(e),
            }
        }
        (self.bind(f, &bound), args)
    }

    /// `f` with the parameters at the given positions bound to expressions:
    /// a call through it passes the others, in their order, and runs `f`'s
    /// body with the bound ones taken from the binding. A model card is a
    /// binding of its device function: a call carries an instance's own
    /// arguments, the card is the binding's, once. The body, its
    /// derivatives and the positions of its parameters stay `f`'s; a bound
    /// call is in every respect the call of `f` with all its arguments.
    pub fn bind(&mut self, f: FuncId, bound: &[(u32, ExprId)]) -> Bound {
        let mut pairs: Vec<(u32, ExprId)> = bound.to_vec();
        pairs.sort_unstable();
        pairs.dedup_by_key(|p| p.0);
        if pairs.is_empty() {
            return Bound {
                func: f,
                ctx: NO_CONTEXT,
            };
        }
        let at: Box<[u32]> = pairs.iter().map(|&(k, _)| k).collect();
        let exprs: Vec<ExprId> = pairs.iter().map(|&(_, e)| e).collect();
        let exprs = self.intern_args(&exprs);
        let key = (f, at.clone(), exprs);
        if let Some(&ctx) = self.context_dedup.get(&key) {
            return Bound { func: f, ctx };
        }
        let n = self.funcs[f.0 as usize].params().len();
        let mut slot = vec![0u32; n];
        let (mut next, mut j) = (0u32, 0usize);
        for (p, s) in slot.iter_mut().enumerate() {
            if at.get(j) == Some(&(p as u32)) {
                *s = BOUND | j as u32;
                j += 1;
            } else {
                *s = next;
                next += 1;
            }
        }
        let ctx = self.contexts.len() as u32;
        self.contexts.push(Context {
            f,
            at,
            exprs,
            slot: slot.into(),
        });
        self.context_dedup.insert(key, ctx);
        Bound { func: f, ctx }
    }

    pub(crate) fn context_over(&mut self, c: u32, f: FuncId, exprs: &[ExprId]) -> u32 {
        if c == NO_CONTEXT {
            return NO_CONTEXT;
        }
        let ctx = &self.contexts[c as usize];
        if ctx.f == f && self.args(ctx.exprs) == exprs {
            return c;
        }
        let pairs: Vec<(u32, ExprId)> = ctx.at.iter().copied().zip(exprs.iter().copied()).collect();
        self.bind(f, &pairs).ctx
    }

    fn bound_arity(&self, b: Bound) -> usize {
        let n = self.funcs[b.func.0 as usize].params().len();
        match b.ctx {
            NO_CONTEXT => n,
            c => n - self.contexts[c as usize].at.len(),
        }
    }

    /// Calls of the outputs `outs` through `b`, over one list of the
    /// parameters it leaves.
    pub fn calls_bound(&mut self, b: Bound, outs: &[u32], args: &[ExprId]) -> Vec<ExprId> {
        debug_assert_eq!(args.len(), self.bound_arity(b), "call arity");
        let l = self.intern_args(args);
        outs.iter()
            .map(|&out| self.call_list_in(b.func, out, b.ctx, l))
            .collect()
    }

    pub(crate) fn full_args(&self, o: OutputId, l: ArgList) -> std::borrow::Cow<'_, [ExprId]> {
        let bound = self.context(o).map_or(&[][..], |(_, b)| b);
        self.in_param_order(self.context_of(o), self.args(l), bound)
    }

    pub(crate) fn in_param_order<'a, T: Copy>(
        &self,
        ctx: u32,
        args: &'a [T],
        bound: &[T],
    ) -> std::borrow::Cow<'a, [T]> {
        match ctx {
            NO_CONTEXT => std::borrow::Cow::Borrowed(args),
            c => {
                let c = &self.contexts[c as usize];
                std::borrow::Cow::Owned((0..c.slot.len()).map(|p| c.pick(p, args, bound)).collect())
            }
        }
    }

    pub(crate) fn calls_over(
        &mut self,
        f: FuncId,
        ctx: u32,
        call: &crate::transform::Call<ExprId>,
        outs: &[(ExprId, u32)],
        vals: &mut Vec<ExprId>,
    ) {
        let l = self.intern_args(call.args);
        for &(_, k) in outs {
            vals.push(self.call_list_in(f, k, ctx, l));
        }
    }

    /// Output `out` of `f` applied to `args` (one argument per parameter). A
    /// zero output folds to the constant zero.
    pub fn call(&mut self, f: FuncId, out: u32, args: &[ExprId]) -> ExprId {
        let func = &self.funcs[f.0 as usize];
        debug_assert_eq!(args.len(), func.params().len(), "call arity");
        if matches!(func.outputs()[out as usize], Output::Zero) {
            return self.zero;
        }
        let l = self.intern_args(args);
        self.call_list(f, out, l)
    }

    /// Outputs `outs` of `f` called over one argument list: the calls of one
    /// instance. The list is interned once, so an instance of a wide body
    /// costs its width once, not once per output.
    pub fn calls(&mut self, f: FuncId, outs: &[u32], args: &[ExprId]) -> Vec<ExprId> {
        debug_assert_eq!(
            args.len(),
            self.funcs[f.0 as usize].params().len(),
            "call arity"
        );
        let l = self.intern_args(args);
        outs.iter().map(|&out| self.call_list(f, out, l)).collect()
    }

    pub(crate) fn call_list(&mut self, f: FuncId, out: u32, l: ArgList) -> ExprId {
        self.call_list_in(f, out, NO_CONTEXT, l)
    }

    pub(crate) fn call_list_in(&mut self, f: FuncId, out: u32, ctx: u32, l: ArgList) -> ExprId {
        match self.funcs[f.0 as usize].outputs()[out as usize] {
            Output::Zero => return self.zero,
            Output::Expr(e) if matches!(self.node(e), Node::Const(_)) => return e,
            _ => {}
        }
        let o = self.output_in(f, out, ctx);
        if ctx != NO_CONTEXT && !self.list_shape.contains_key(&(ctx, l)) {
            let shape_in_param_order = self.full_args(o, l).iter().fold(0, |h, &a| {
                crate::node::shape::mix(h, self.shape[a.0 as usize])
            });
            self.list_shape.insert((ctx, l), shape_in_param_order);
        }
        self.intern(Node::Call(o, l))
    }

    pub(crate) fn derivative_output(&mut self, f: FuncId, out: u32, param: u32) -> u32 {
        if let Some(k) = self.funcs[f.0 as usize].derivative(out, param) {
            return k;
        }
        self.derive_every_dependent_output(f, out, param);
        if let Some(k) = self.funcs[f.0 as usize].derivative(out, param) {
            return k;
        }
        let output = self.funcs[f.0 as usize].outputs()[out as usize];
        let d = match output {
            Output::Expr(e) => {
                let wrt = self.operand_symbol(f, param);
                let de = crate::autodiff::differentiate(self, e, wrt);
                self.as_output(de)
            }
            Output::Slot(_) | Output::Zero => Output::Zero,
        };
        self.push_output(
            f,
            d,
            OutputRole::Derivative {
                of: out,
                wrt: param,
            },
        )
    }

    pub(crate) fn derivative_outputs(&mut self, f: FuncId, out: u32, params: &[u32]) -> Vec<u32> {
        let func = &self.funcs[f.0 as usize];
        if let Output::Expr(e) = func.outputs()[out as usize] {
            let missing: Vec<u32> = (params.iter().copied())
                .filter(|&p| self.is_model_param(f, p))
                .filter(|&p| func.derivative(out, p).is_none())
                .collect();
            if missing.len() >= crate::autodiff::REVERSE_MIN_TOUCHED {
                let wrt: Vec<SymbolId> =
                    missing.iter().map(|&p| self.operand_symbol(f, p)).collect();
                let grad = crate::autodiff::gradient(self, e, &wrt);
                for (&p, d) in missing.iter().zip(grad) {
                    let d = self.as_output(d);
                    self.push_output(f, d, OutputRole::Derivative { of: out, wrt: p });
                }
            }
        }
        params
            .iter()
            .map(|&p| self.derivative_output(f, out, p))
            .collect()
    }

    pub(crate) fn vjp_function(&mut self, f: FuncId, outs: &[u32], params: &[u32]) -> FuncId {
        let key = (f, Box::<[u32]>::from(outs), Box::<[u32]>::from(params));
        if let Some(&g) = self.vjps.get(&key) {
            return g;
        }
        let func = &self.funcs[f.0 as usize];
        let name = format!("{}#vjp{}", func.name(), self.vjps.len());
        let mut formal = func.params().to_vec();
        let mut roles = func.param_roles().to_vec();
        let exprs: Vec<Option<ExprId>> = (outs.iter())
            .map(|&o| match func.outputs()[o as usize] {
                Output::Expr(e) => Some(e),
                Output::Slot(_) | Output::Zero => None,
            })
            .collect();
        let mut cotangents = Vec::with_capacity(outs.len());
        let mut terms = Vec::with_capacity(outs.len());
        for (j, e) in exprs.into_iter().enumerate() {
            let s = self.symbol_id(&format!("{name}$c{j}"));
            let c = self.symbol_expr(s);
            cotangents.push(s);
            if let Some(e) = e {
                terms.push(self.mul(c, e));
            }
        }
        let cotangent_weighted_sum = self.reduce(ReduceOp::Sum, terms);
        let wrt: Vec<SymbolId> = params.iter().map(|&p| self.operand_symbol(f, p)).collect();
        let grads = crate::autodiff::gradient(self, cotangent_weighted_sum, &wrt);
        roles.resize(formal.len() + cotangents.len(), ParamRole::Free);
        formal.extend(cotangents);
        let outputs = grads
            .into_iter()
            .map(|e| (Output::Expr(e), OutputRole::Plain));
        let g = self.define(&name, formal, &roles, None, outputs);
        self.vjps.insert(key, g);
        g
    }

    fn is_model_param(&self, f: FuncId, p: u32) -> bool {
        let roles = self.funcs[f.0 as usize].param_roles();
        (roles.get(p as usize)).is_none_or(|r| matches!(r, ParamRole::Param))
    }

    fn as_output(&self, e: ExprId) -> Output {
        if self.is_zero(e) {
            Output::Zero
        } else {
            Output::Expr(e)
        }
    }

    fn derive_every_dependent_output(&mut self, f: FuncId, out: u32, param: u32) {
        let func = &self.funcs[f.0 as usize];
        let second_derivative = matches!(
            func.output_roles()[out as usize],
            OutputRole::Derivative { .. }
        );
        if second_derivative || self.is_model_param(f, param) {
            return;
        }
        let deps = self.deps(f);
        let Some(&j) = deps.column_of_param.get(&param) else {
            return;
        };
        let func = &self.funcs[f.0 as usize];
        let roots: Vec<(u32, ExprId)> = (deps.outputs_reading_column[j as usize].iter())
            .filter(|&&o| func.derivative(o, param).is_none())
            .filter_map(|&o| match func.outputs()[o as usize] {
                Output::Expr(e) => Some((o, e)),
                _ => None,
            })
            .collect();
        if roots.is_empty() {
            return;
        }
        let wrt = self.operand_symbol(f, param);
        let exprs: Vec<ExprId> = roots.iter().map(|&(_, e)| e).collect();
        let mut memo = NodeTable::new(self.len());
        let moves = |c: ExprId| {
            (deps.columns_read_by_node.get(&c)).is_some_and(|s| s.binary_search(&j).is_ok())
        };
        let mut calls = crate::autodiff::CallOperands::new(&moves);
        let ds = crate::autodiff::forward(self, &exprs, wrt, &mut memo, &moves, &mut calls);
        for (&(o, _), d) in roots.iter().zip(ds) {
            let d = self.as_output(d);
            self.push_output(f, d, OutputRole::Derivative { of: o, wrt: param });
        }
    }

    fn deps(&self, f: FuncId) -> Arc<Deps> {
        let func = &self.funcs[f.0 as usize];
        if let Some(d) = &*func.deps.lock().unwrap() {
            return d.clone();
        }
        let params: Vec<u32> = (0..func.params().len() as u32)
            .filter(|&p| !matches!(func.param_roles()[p as usize], ParamRole::Param))
            .collect();
        let col: HashMap<SymbolId, u32> = (params.iter().enumerate())
            .map(|(c, &p)| (func.params()[p as usize], c as u32))
            .collect();
        let outs: Vec<(u32, ExprId)> = (func.outputs().iter().enumerate())
            .filter(|&(k, _)| !matches!(func.output_roles()[k], OutputRole::Derivative { .. }))
            .filter_map(|(k, o)| match *o {
                Output::Expr(e) => Some((k as u32, e)),
                _ => None,
            })
            .collect();
        let roots: Vec<ExprId> = outs.iter().map(|&(_, e)| e).collect();
        let n = params.len();
        let flow = self.flow(&roots, Through::Carries, |node| match *node {
            Node::Symbol(s) => col.get(&s).map_or(Set::bottom(), |&c| Set::one(c, n)),
            _ => Set::bottom(),
        });
        let mut by_col = vec![Vec::new(); n];
        for &(k, e) in &outs {
            for c in flow.get(e).iter() {
                by_col[c as usize].push(k);
            }
        }
        let deps = Arc::new(Deps {
            column_of_param: (params.iter().enumerate())
                .map(|(c, &p)| (p, c as u32))
                .collect(),
            columns_read_by_node: (flow.iter())
                .filter(|(_, v)| !v.is_bottom())
                .map(|(e, v)| (e, v.iter().collect()))
                .collect(),
            outputs_reading_column: by_col,
        });
        *func.deps.lock().unwrap() = Some(deps.clone());
        deps
    }

    /// Every `(function, output)` called anywhere in `exprs` (one pass over
    /// the forest). A solver uses it to compile exactly the outputs a set of
    /// roots reads.
    pub fn free_calls_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<OutputId> {
        self.cone(exprs, false)
            .nodes
            .iter()
            .filter_map(|&e| match *self.node(e) {
                Node::Call(o, _) => Some(o),
                _ => None,
            })
            .collect()
    }

    /// The free symbols of `expr` (see
    /// [`free_symbols_in`](Self::free_symbols_in)).
    pub fn free_symbols(&self, expr: ExprId) -> std::collections::BTreeSet<SymbolId> {
        self.free_symbols_in(&[expr])
    }

    /// The symbols `exprs` read: what they mention and what the bodies they
    /// call read as globals (see [`globals`](Self::globals)), the inputs a
    /// program over them takes. One walk over the forest, each shared node
    /// once.
    pub fn free_symbols_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        self.symbols_of(&self.cone(exprs, true).nodes)
    }

    /// The symbols `exprs` mention, the bodies they call not looked into: the
    /// parameters a function closed over them takes (see
    /// [`Scope::close`](crate::Scope::close)).
    pub fn mentioned_symbols_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        self.symbols_of(&self.cone(exprs, false).nodes)
    }

    fn symbols_of(&self, nodes: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        nodes.iter().filter_map(|&e| self.as_symbol(e)).collect()
    }
}

type Specialized = HashMap<(FuncId, Vec<(u32, ExprId)>), FuncId>;

fn specialize_calls_in<K: Field>(
    g: &mut Graph<K>,
    roots: &[ExprId],
    made: &mut Specialized,
) -> Vec<ExprId> {
    let mut rule = Specialize {
        g,
        made,
        kept_contexts: HashMap::default(),
        contexts: Contexts::default(),
    };
    crate::transform::rewrite(&mut rule, roots)
}

struct Specialize<'a, K: Field> {
    g: &'a mut Graph<K>,
    made: &'a mut Specialized,
    kept_contexts: HashMap<(FuncId, u32), u32>,
    contexts: Contexts,
}

impl<K: Field> Rule<K> for Specialize<'_, K> {
    type V = ExprId;
    fn source(&self) -> &Graph<K> {
        self.g
    }
    fn node(&mut self, _: ExprId, node: Node, ops: &[ExprId]) -> ExprId {
        self.g.build(node, ops)
    }
    fn calls(&mut self, call: &Call<ExprId>, outs: &[(ExprId, u32)], vals: &mut Vec<ExprId>) {
        let g = &mut *self.g;
        let ctx = self.contexts.of(g, call, call.f);
        let full = call.params(g);
        match specialize_instance(g, call.f, ctx, &full, self.made, &mut self.kept_contexts) {
            Some((copy, ctx, rest)) => {
                for &(_, k) in outs {
                    vals.push(g.call_list_in(copy, k, ctx, rest));
                }
            }
            None => g.calls_over(call.f, ctx, call, outs, vals),
        }
    }
}

fn specialize_instance<K: Field>(
    g: &mut Graph<K>,
    f: FuncId,
    ctx: u32,
    full: &[ExprId],
    made: &mut Specialized,
    kept_contexts: &mut HashMap<(FuncId, u32), u32>,
) -> Option<(FuncId, u32, ArgList)> {
    let roles = g.func(f).param_roles();
    let has_param_role = |k: usize| matches!(roles.get(k), Some(ParamRole::Param));
    let consts: Vec<(u32, ExprId)> = full
        .iter()
        .enumerate()
        .filter(|&(k, &a)| g.const_of(a).is_some() && !has_param_role(k))
        .map(|(k, &a)| (k as u32, a))
        .collect();
    if consts.is_empty() || g.func(f).is_extern() {
        return None;
    }
    let key = (f, consts);
    let copy = match made.get(&key) {
        Some(&c) => c,
        None => {
            let c = specialize_function(g, f, &key.1, made);
            made.insert(key.clone(), c);
            c
        }
    };
    debug_assert!(
        key.1.windows(2).all(|w| w[0].0 < w[1].0),
        "the constants are in parameter order, which the merge below needs"
    );
    let mut bound_at = vec![false; full.len()];
    if ctx != NO_CONTEXT {
        g.contexts[ctx as usize]
            .at
            .iter()
            .for_each(|&p| bound_at[p as usize] = true);
    }
    let mut consts_at = key.1.iter().map(|&(p, _)| p as usize).peekable();
    let (mut rest, mut rebound): (Vec<ExprId>, Vec<(u32, ExprId)>) = (Vec::new(), Vec::new());
    let known = kept_contexts.get(&(copy, ctx)).copied();
    let mut j = 0u32;
    for (k, &a) in full.iter().enumerate() {
        if consts_at.next_if_eq(&k).is_some() {
            continue;
        }
        if !bound_at[k] {
            rest.push(a);
        } else if known.is_none() {
            rebound.push((j, a));
        }
        j += 1;
    }
    let ctx = match known {
        Some(c) => c,
        None => {
            let c = g.bind(copy, &rebound).ctx;
            kept_contexts.insert((copy, ctx), c);
            c
        }
    };
    Some((copy, ctx, g.intern_args(&rest)))
}

fn specialize_function<K: Field>(
    g: &mut Graph<K>,
    f: FuncId,
    consts: &[(u32, ExprId)],
    made: &mut Specialized,
) -> FuncId {
    let func = g.func(f);
    let (name, params) = (func.name().to_string(), func.params().to_vec());
    let (roles, out_roles) = (func.param_roles().to_vec(), func.output_roles().to_vec());
    let bound: HashMap<SymbolId, ExprId> = consts
        .iter()
        .map(|&(k, c)| (params[k as usize], c))
        .collect();
    let exprs = func.exprs();
    let folded = crate::transform::substitute(g, &exprs, &bound);
    let folded = specialize_calls_in(g, &folded, made);
    let kept: Vec<usize> = (0..params.len())
        .filter(|&k| !bound.contains_key(&params[k]))
        .collect();
    let outputs = (g.func(f).outputs_over(folded).into_iter()).map(|o| match o {
        Output::Expr(e) if !g.is_zero(e) => o,
        _ => Output::Zero,
    });
    let plain_roles = out_roles.into_iter().map(|r| match r {
        OutputRole::Derivative { .. } => OutputRole::Plain,
        r => r,
    });
    let outputs: Vec<(Output, OutputRole)> = outputs.zip(plain_roles).collect();
    let kept_roles: Vec<ParamRole> = kept.iter().map(|&k| roles[k]).collect();
    let params = kept.iter().map(|&k| params[k]).collect();
    g.define(&name, params, &kept_roles, None, outputs)
}

struct Inline<'a, K: Field> {
    g: &'a mut Graph<K>,
    which: &'a mut dyn FnMut(&Graph<K>, FuncId) -> bool,
    bodies: &'a mut HashMap<FuncId, Vec<ExprId>>,
    contexts: Contexts,
}

impl<K: Field> Rule<K> for Inline<'_, K> {
    type V = ExprId;
    fn source(&self) -> &Graph<K> {
        self.g
    }
    fn node(&mut self, _: ExprId, node: Node, ops: &[ExprId]) -> ExprId {
        self.g.build(node, ops)
    }
    fn calls(&mut self, call: &Call<ExprId>, outs: &[(ExprId, u32)], vals: &mut Vec<ExprId>) {
        let f = call.f;
        if self.g.func(f).is_extern() || !(self.which)(self.g, f) {
            let ctx = self.contexts.of(self.g, call, f);
            return self.g.calls_over(f, ctx, call, outs, vals);
        }
        if !self.bodies.contains_key(&f) {
            let exprs = self.g.func(f).exprs();
            let mut inner = Inline {
                g: &mut *self.g,
                which: &mut *self.which,
                bodies: &mut *self.bodies,
                contexts: Contexts::default(),
            };
            let inlined = crate::transform::rewrite(&mut inner, &exprs);
            let zero = self.g.zero();
            let outputs = (self.g.func(f).outputs_over(inlined).into_iter())
                .map(|o| match o {
                    Output::Expr(e) => e,
                    _ => zero,
                })
                .collect();
            self.bodies.insert(f, outputs);
        }
        let g = &mut *self.g;
        let bind: HashMap<SymbolId, ExprId> = (g.func(f).params().iter().copied())
            .zip(call.params(g).iter().copied())
            .collect();
        let bodies: Vec<ExprId> = outs
            .iter()
            .map(|&(_, k)| self.bodies[&f][k as usize])
            .collect();
        vals.extend(crate::transform::substitute(g, &bodies, &bind));
    }
}

pub(crate) struct Deps {
    column_of_param: HashMap<u32, u32>,
    columns_read_by_node: HashMap<ExprId, Box<[u32]>>,
    outputs_reading_column: Vec<Vec<u32>>,
}
