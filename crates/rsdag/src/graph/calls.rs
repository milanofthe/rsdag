//! Functions, calls and the system layer over them: defining a function,
//! calling it, roles, role-selected Jacobians, feedthrough, inlining.
//!
//! A child module of `graph` so it reaches the arena's private fields; a
//! second `impl Graph` block rather than a second type, because a call is a
//! node like any other and the functions live in the same arena.

use super::*;

impl<K: Field> Graph<K> {
    // --- functions and calls ---------------------------------------------

    /// Append a function; the ids stay in definition order.
    pub(crate) fn push_function(&mut self, func: Function) -> FuncId {
        self.funcs.push(func);
        FuncId(self.funcs.len() as u32 - 1)
    }

    /// Append an output with its role to `f`; returns its index.
    pub(crate) fn push_output(&mut self, f: FuncId, output: Output, role: OutputRole) -> u32 {
        self.funcs[f.0 as usize].push_output(output, role)
    }

    /// Define a symbolic function: `outputs` are expressions over the formal
    /// `params` (free symbols of the outputs not listed in `params` are shared
    /// globals every call sees unchanged).
    pub fn define_func(
        &mut self,
        name: &str,
        params: Vec<SymbolId>,
        outputs: Vec<ExprId>,
    ) -> FuncId {
        let f = self.push_function(Function::new(name, params, None));
        for e in outputs {
            self.push_output(f, Output::Expr(e), OutputRole::Plain);
        }
        f
    }

    /// Close an open graph over `outputs` into a function: every free symbol
    /// the outputs depend on becomes a parameter, in symbol order. The
    /// `Scope` idiom: build with named symbols, then close.
    pub fn close(&mut self, name: &str, outputs: Vec<ExprId>) -> FuncId {
        let params: Vec<SymbolId> = self.free_symbols_in(&outputs).into_iter().collect();
        self.define_func(name, params, outputs)
    }

    /// Set the role of parameter `param` of function `f`.
    pub fn set_param_role(&mut self, f: FuncId, param: u32, role: ParamRole) {
        self.funcs[f.0 as usize].set_param_role(param, role);
    }

    /// Set the role of output `out` of function `f`.
    pub fn set_output_role(&mut self, f: FuncId, out: u32, role: OutputRole) {
        self.funcs[f.0 as usize].set_output_role(out, role);
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
        self.inline_with(roots, &mut HashMap::default())
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
        specialize_calls_in(self, roots, &mut HashMap::default())
    }

    /// [`inline_all`](Self::inline_all) with `bodies` holding each output
    /// already inlined over its function's parameters, so a function is
    /// inlined once however many calls it has, and the recursion is only as
    /// deep as the call hierarchy.
    fn inline_with(
        &mut self,
        roots: &[ExprId],
        bodies: &mut HashMap<OutputId, ExprId>,
    ) -> Vec<ExprId> {
        // per instance (function, argument list): its parameters bound
        let mut binds: HashMap<(FuncId, ArgList), HashMap<SymbolId, ExprId>> = HashMap::default();
        crate::transform::rewrite(self, roots, |g, _, node, ops| {
            let (Node::Call(o, _), Some(l)) = (node, ops.list) else {
                return g.rebuild(node, ops);
            };
            let (f, k) = g.output(o);
            let e = match g.funcs[f.0 as usize].outputs()[k as usize] {
                // An extern body stays a call, over inlined arguments.
                Output::Slot(_) => return g.rebuild(node, ops),
                Output::Zero => return g.zero,
                Output::Expr(e) => e,
            };
            let body = match bodies.get(&o) {
                Some(&b) => b,
                None => {
                    let b = g.inline_with(&[e], bodies)[0];
                    bodies.insert(o, b);
                    b
                }
            };
            let map = binds.entry((f, l)).or_insert_with(|| g.bind(f, ops.ops));
            crate::transform::substitute(g, &[body], map)[0]
        })
    }

    /// The parameters of `f` bound to the arguments of a call.
    fn bind(&self, f: FuncId, args: &[ExprId]) -> HashMap<SymbolId, ExprId> {
        let params = self.funcs[f.0 as usize].params();
        params.iter().copied().zip(args.iter().copied()).collect()
    }

    /// The parameters output `out` of `f` can have a nonzero derivative in,
    /// by index, ascending: its [`support_in`](Self::support_in) among the
    /// function's parameters. Structural, read off the graph. The outputs of
    /// a function share their body, so the ones not known yet are found in
    /// one pass over it, and each is kept. An extern output is taken to read
    /// every parameter, a zero one none.
    pub fn output_support(&self, f: FuncId, out: u32) -> Arc<[u32]> {
        self.output_reach(Reach::Derivative, f, out)
    }

    /// The parameters output `out` of `f` reads, by index, ascending: the
    /// ones its value structurally depends on, a comparison's operands and a
    /// selector's condition included, and through a call only the arguments
    /// the called output reads. Computed like
    /// [`output_support`](Self::output_support), once per function.
    pub fn output_reads(&self, f: FuncId, out: u32) -> Arc<[u32]> {
        self.output_reach(Reach::Value, f, out)
    }

    /// Per root, whether its value reads any of `syms`, through calls only
    /// what the called outputs read: one pass over the roots' shared cone.
    /// Which entries of a Jacobian vary with the state, say, without a walk
    /// per entry over the calls' argument lists.
    pub fn depends_on(&self, roots: &[ExprId], syms: &[SymbolId]) -> Vec<bool> {
        let wanted: FxHashSet<SymbolId> = syms.iter().copied().collect();
        let mut ops = Vec::new();
        let cone = self.cone_of(Reach::Value, roots, &mut ops);
        let mut dep: HashMap<ExprId, bool> = HashMap::default();
        dep.reserve(cone.len());
        for &e in &cone {
            let d = match *self.node(e) {
                Node::Const(_) => false,
                Node::Symbol(s) => wanted.contains(&s),
                _ => {
                    reaching(self, Reach::Value, e, &mut ops);
                    ops.iter().any(|c| dep[c])
                }
            };
            dep.insert(e, d);
        }
        roots.iter().map(|r| dep[r]).collect()
    }

    /// The nodes under `roots` through the operands that `reach` follows,
    /// in ascending id order: every operand before its consumers.
    fn cone_of(&self, reach: Reach, roots: &[ExprId], ops: &mut Vec<ExprId>) -> Vec<ExprId> {
        let mut seen: FxHashSet<ExprId> = FxHashSet::default();
        let mut stack: Vec<ExprId> = roots.to_vec();
        let mut cone: Vec<ExprId> = Vec::new();
        while let Some(e) = stack.pop() {
            if seen.insert(e) {
                cone.push(e);
                reaching(self, reach, e, ops);
                stack.extend_from_slice(ops);
            }
        }
        cone.sort_unstable();
        cone
    }

    fn output_reach(&self, reach: Reach, f: FuncId, out: u32) -> Arc<[u32]> {
        let func = &self.funcs[f.0 as usize];
        if let Some(s) = func.cached_support(reach, out) {
            return s;
        }
        let pending: Vec<u32> = (0..func.outputs().len() as u32)
            .filter(|&k| func.cached_support(reach, k).is_none())
            .collect();
        let exprs: Vec<ExprId> = pending
            .iter()
            .filter_map(|&k| match func.outputs()[k as usize] {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let index: HashMap<SymbolId, u32> = func
            .params()
            .iter()
            .enumerate()
            .map(|(k, &s)| (s, k as u32))
            .collect();
        let mut found = self.param_supports(reach, &exprs, &index).into_iter();
        for &k in &pending {
            let support: Arc<[u32]> = match func.outputs()[k as usize] {
                Output::Zero => Arc::from([]),
                Output::Slot(_) => (0..func.params().len() as u32).collect(),
                Output::Expr(_) => found.next().expect("one per expression"),
            };
            func.cache_support(reach, k, support);
        }
        func.cached_support(reach, out).expect("just found")
    }

    /// Per root, the parameters (numbered by `index`) it reaches: one
    /// bottom-up pass over the roots' shared cone, each node's set the union
    /// of the sets of the operands `reach` follows (a node with one
    /// contributing operand shares that operand's set).
    fn param_supports(
        &self,
        reach: Reach,
        roots: &[ExprId],
        index: &HashMap<SymbolId, u32>,
    ) -> Vec<Arc<[u32]>> {
        let mut ops = Vec::new();
        let cone = self.cone_of(reach, roots, &mut ops);
        // Every node's set is a span of one arena; a node with a single
        // contributing operand shares its span.
        let mut arena: Vec<u32> = Vec::new();
        let mut at: HashMap<ExprId, (u32, u32)> = HashMap::default();
        at.reserve(cone.len());
        let (mut merged, mut scratch): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        for &e in &cone {
            let span = match *self.node(e) {
                Node::Const(_) => (0, 0),
                Node::Symbol(s) => match index.get(&s) {
                    Some(&k) => {
                        arena.push(k);
                        (arena.len() as u32 - 1, 1)
                    }
                    None => (0, 0),
                },
                _ => {
                    reaching(self, reach, e, &mut ops);
                    let mut parts = ops.iter().map(|c| at[c]).filter(|&(_, n)| n > 0);
                    match (parts.next(), parts.clone().next()) {
                        (None, _) => (0, 0),
                        (Some(one), None) => one,
                        (Some(first), Some(_)) => {
                            // sorted unions, one linear merge per operand
                            let slice = |(s, n): (u32, u32)| s as usize..(s + n) as usize;
                            merged.clear();
                            merged.extend_from_slice(&arena[slice(first)]);
                            for part in parts {
                                union_into(&mut merged, &arena[slice(part)], &mut scratch);
                            }
                            let start = arena.len() as u32;
                            arena.extend_from_slice(&merged);
                            (start, merged.len() as u32)
                        }
                    }
                }
            };
            at.insert(e, span);
        }
        roots
            .iter()
            .map(|r| {
                let (s, n) = at[r];
                Arc::from(&arena[s as usize..(s + n) as usize])
            })
            .collect()
    }

    /// The symbols `exprs` can have a nonzero derivative in: their
    /// [`free_symbols_in`](Self::free_symbols_in) through the operands that
    /// carry a derivative (not a comparison's, not a selector's condition),
    /// and through a call only the arguments its output's
    /// [`output_support`](Self::output_support) names. The sparsity of every
    /// derivative of `exprs`, nested calls included.
    pub fn support_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        let mut set = std::collections::BTreeSet::new();
        let mut visited = FxHashSet::default();
        let mut stack: Vec<ExprId> = exprs.to_vec();
        let mut ops = Vec::new();
        while let Some(e) = stack.pop() {
            if !visited.insert(e) {
                continue;
            }
            match *self.node(e) {
                Node::Const(_) => {}
                Node::Symbol(s) => {
                    set.insert(s);
                }
                _ => {
                    crate::autodiff::carrying(self, e, &mut ops);
                    stack.extend_from_slice(&ops);
                }
            }
        }
        set
    }

    /// Which outputs of `f` structurally read which of its parameters:
    /// `feedthrough(f)[out][param]`.
    ///
    /// Structural, not numeric: it asks whether the parameter occurs in the
    /// output's expression at all, so it costs one walk per output and needs
    /// no differentiation. That is the question a block scheduler asks --
    /// direct feedthrough decides the evaluation order, and a cycle among
    /// the blocks that have it is an algebraic loop.
    ///
    /// An extern output is opaque and is reported as reading every
    /// parameter, which is the safe direction: it can cost an ordering
    /// constraint, never a missed loop. A zero output reads nothing.
    pub fn feedthrough(&self, f: FuncId) -> Vec<Vec<bool>> {
        let func = &self.funcs[f.0 as usize];
        let index: HashMap<SymbolId, usize> = func
            .params()
            .iter()
            .enumerate()
            .map(|(k, &s)| (s, k))
            .collect();
        func.outputs()
            .iter()
            .map(|out| {
                let mut row = vec![false; func.params().len()];
                match *out {
                    Output::Zero => {}
                    Output::Slot(_) => row.iter_mut().for_each(|r| *r = true),
                    Output::Expr(e) => {
                        for s in self.free_symbols(e) {
                            if let Some(&k) = index.get(&s) {
                                row[k] = true;
                            }
                        }
                    }
                }
                row
            })
            .collect()
    }

    /// Jacobian of the outputs of `f` with a role against its parameters with
    /// a role: `(output index, param index, derivative output index)` for
    /// every structurally nonzero pair, differentiating on demand.
    pub fn jacobian_by_role(
        &mut self,
        f: FuncId,
        out_role: impl Fn(&OutputRole) -> bool,
        param_role: impl Fn(&ParamRole) -> bool,
    ) -> Vec<(u32, u32, u32)> {
        let outs = self.funcs[f.0 as usize].outputs_with_role(out_role);
        let pars = self.funcs[f.0 as usize].params_with_role(param_role);
        let mut entries = Vec::new();
        for &o in &outs {
            for (&p, k) in pars.iter().zip(self.derivative_outputs(f, o, &pars)) {
                if !matches!(self.funcs[f.0 as usize].outputs()[k as usize], Output::Zero) {
                    entries.push((o, p, k));
                }
            }
        }
        entries
    }

    /// Define an extern function over `arity` arguments whose outputs are the
    /// given slots of `body` (or [`Output::Zero`]). Derivative outputs the body
    /// carries are declared with [`declare_derivative`](Self::declare_derivative).
    pub fn define_extern_func(
        &mut self,
        name: &str,
        arity: usize,
        body: Arc<dyn ExternBundle>,
        outputs: Vec<Output>,
    ) -> FuncId {
        let params: Vec<SymbolId> = (0..arity)
            .map(|i| {
                let e = self.sym(&format!("{name}.${i}"));
                match *self.node(e) {
                    Node::Symbol(s) => s,
                    _ => unreachable!("sym yields a symbol"),
                }
            })
            .collect();
        self.define_extern_func_with_params(name, params, body, outputs)
    }

    /// Define an extern function over the given formal parameters (the
    /// symbols already exist), see [`define_extern_func`](Self::define_extern_func).
    pub fn define_extern_func_with_params(
        &mut self,
        name: &str,
        params: Vec<SymbolId>,
        body: Arc<dyn ExternBundle>,
        outputs: Vec<Output>,
    ) -> FuncId {
        let f = self.push_function(Function::new(name, params, Some(body)));
        for o in outputs {
            self.push_output(f, o, OutputRole::Plain);
        }
        f
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

    /// Number of symbols.
    pub fn n_symbols(&self) -> usize {
        self.symbol_names.len()
    }

    /// Register a body a consumer compiled for the symbolic function `f`:
    /// from now on a tape calls `body` (see [`Function::compiled`]). A tape
    /// compiled before keeps the body it was compiled with.
    pub fn set_func_body(&mut self, f: FuncId, body: crate::func::Body) {
        assert!(
            !self.funcs[f.0 as usize].is_extern(),
            "an extern function is its own body"
        );
        // Several bodies may serve one function (a residual-only one and
        // one with the partials); a program takes the smallest that covers
        // the outputs it calls.
        let bodies = self.funcs[f.0 as usize].compiled_mut();
        let ptr = Arc::as_ptr(&body.bundle) as *const () as usize;
        if !bodies
            .iter()
            .any(|b| Arc::as_ptr(&b.bundle) as *const () as usize == ptr)
        {
            bodies.push(body);
        }
    }

    pub fn n_funcs(&self) -> usize {
        self.funcs.len()
    }

    /// The `(function, output index)` an output id names.
    #[inline]
    pub fn output(&self, o: OutputId) -> (FuncId, u32) {
        self.outputs[o.0 as usize]
    }

    /// The interned id of output `out` of `f`.
    pub fn output_id(&mut self, f: FuncId, out: u32) -> OutputId {
        if let Some(&o) = self.output_dedup.get(&(f, out)) {
            return o;
        }
        let o = OutputId(self.outputs.len() as u32);
        self.outputs.push((f, out));
        self.output_dedup.insert((f, out), o);
        o
    }

    /// The expression of a symbolic output, `None` for a slot or zero output.
    pub fn output_expr(&self, f: FuncId, out: u32) -> Option<ExprId> {
        match self.funcs[f.0 as usize].outputs()[out as usize] {
            Output::Expr(e) => Some(e),
            _ => None,
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

    /// [`call`](Self::call) over an argument list already interned.
    pub(crate) fn call_list(&mut self, f: FuncId, out: u32, l: ArgList) -> ExprId {
        if matches!(
            self.funcs[f.0 as usize].outputs()[out as usize],
            Output::Zero
        ) {
            return self.zero;
        }
        let o = self.output_id(f, out);
        self.intern(Node::Call(o, l))
    }

    /// [`call`](Self::call) by output id.
    pub fn call_output(&mut self, o: OutputId, args: &[ExprId]) -> ExprId {
        let (f, out) = self.output(o);
        self.call(f, out, args)
    }

    /// The index of the derivative output `d outputs[out] / d params[param]`,
    /// differentiating the body on first demand (symbolic functions) or
    /// looking up the declared slot (extern functions; zero if undeclared).
    pub fn derivative_output(&mut self, f: FuncId, out: u32, param: u32) -> u32 {
        let func = &self.funcs[f.0 as usize];
        if let Some(k) = func.derivative(out, param) {
            return k;
        }
        let output = func.outputs()[out as usize];
        let d = match output {
            Output::Expr(e) => {
                let wrt = func.params()[param as usize];
                let de = crate::autodiff::differentiate(self, e, wrt);
                if self.is_zero(de) {
                    Output::Zero
                } else {
                    Output::Expr(de)
                }
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

    /// [`derivative_output`](Self::derivative_output) for several parameters
    /// of one output. The missing derivatives of a symbolic function are
    /// derived in one reverse sweep over the body when they are
    /// [`REVERSE_MIN_TOUCHED`](crate::autodiff::REVERSE_MIN_TOUCHED) or more
    /// (a device's parameters), in one forward sweep each otherwise.
    pub fn derivative_outputs(&mut self, f: FuncId, out: u32, params: &[u32]) -> Vec<u32> {
        let func = &self.funcs[f.0 as usize];
        if let Output::Expr(e) = func.outputs()[out as usize] {
            let missing: Vec<u32> = params
                .iter()
                .copied()
                .filter(|&p| func.derivative(out, p).is_none())
                .collect();
            if missing.len() >= crate::autodiff::REVERSE_MIN_TOUCHED {
                let wrt: Vec<SymbolId> =
                    missing.iter().map(|&p| func.params()[p as usize]).collect();
                let grad = crate::autodiff::gradient(self, e, &wrt);
                for (&p, d) in missing.iter().zip(grad) {
                    let d = if self.is_zero(d) {
                        Output::Zero
                    } else {
                        Output::Expr(d)
                    };
                    self.push_output(f, d, OutputRole::Derivative { of: out, wrt: p });
                }
            }
        }
        params
            .iter()
            .map(|&p| self.derivative_output(f, out, p))
            .collect()
    }

    /// Inline a call: the output expression with the parameters replaced by
    /// `args`. `None` for an extern (slot) output, which has no body to inline.
    pub fn inline_call(&mut self, f: FuncId, out: u32, args: &[ExprId]) -> Option<ExprId> {
        match self.funcs[f.0 as usize].outputs()[out as usize] {
            Output::Slot(_) => None,
            _ => Some(self.inline_outputs(f, &[out], args)[0]),
        }
    }

    /// Inline several outputs of a symbolic function at once, with one shared
    /// substitution pass (the outputs of a device template share its core, so
    /// per-output substitution would rebuild that core per output).
    pub fn inline_outputs(&mut self, f: FuncId, outs: &[u32], args: &[ExprId]) -> Vec<ExprId> {
        let map = self.bind(f, args);
        let func = &self.funcs[f.0 as usize];
        let exprs: Vec<ExprId> = outs
            .iter()
            .map(|&o| match func.outputs()[o as usize] {
                Output::Expr(e) => e,
                Output::Zero => self.zero,
                Output::Slot(_) => panic!("cannot inline an extern function output"),
            })
            .collect();
        crate::transform::substitute(self, &exprs, &map)
    }

    /// Every `(function, output)` called anywhere in `exprs` (one pass over
    /// the forest). A solver uses it to compile exactly the outputs a set of
    /// roots reads.
    pub fn free_calls_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<OutputId> {
        let mut set = std::collections::BTreeSet::new();
        let mut visited = FxHashSet::default();
        let mut stack: Vec<ExprId> = exprs.to_vec();
        while let Some(e) = stack.pop() {
            if !visited.insert(e) {
                continue;
            }
            if let Node::Call(o, _) = *self.node(e) {
                set.insert(o);
            }
            stack.extend_from_slice(&self.operands(e));
        }
        set
    }

    /// The set of free symbols reachable from `expr`.
    ///
    /// Memoised over shared subexpressions (a `visited` set): in a hash-consed
    /// DAG a node may be reachable by exponentially many paths, so without this
    /// the traversal is super-linear in the node count. With it, each node is
    /// visited once -- O(nodes reachable from `expr`).
    pub fn free_symbols(&self, expr: ExprId) -> std::collections::BTreeSet<SymbolId> {
        self.free_symbols_in(&[expr])
    }

    /// Union of the free symbols across many expressions, sharing one `visited`
    /// set so a subexpression hash-consed into several of them is traversed once
    /// (a single pass over the forest, not one per expression).
    pub fn free_symbols_in(&self, exprs: &[ExprId]) -> std::collections::BTreeSet<SymbolId> {
        let mut set = std::collections::BTreeSet::new();
        let mut visited = FxHashSet::default();
        // the calls of one instance share their list: walked once
        let mut lists: FxHashSet<ArgList> = FxHashSet::default();
        let mut stack: Vec<ExprId> = exprs.to_vec();
        while let Some(e) = stack.pop() {
            if !visited.insert(e) {
                continue;
            }
            match *self.node(e) {
                Node::Const(_) => {}
                Node::Symbol(s) => {
                    set.insert(s);
                }
                Node::Call(_, l) => {
                    if lists.insert(l) {
                        stack.extend_from_slice(self.args(l));
                    }
                }
                _ => stack.extend_from_slice(&self.operands(e)),
            }
        }
        set
    }
}

/// The specialized copies made so far: `(function, constant arguments by
/// position)` to the copy.
type Specialized = HashMap<(FuncId, Vec<(u32, ExprId)>), FuncId>;

fn specialize_calls_in<K: Field>(
    g: &mut Graph<K>,
    roots: &[ExprId],
    made: &mut Specialized,
) -> Vec<ExprId> {
    // per instance (function, argument list): the copy it calls over the
    // arguments that are not constant, or none
    let mut instances: HashMap<(FuncId, ArgList), Option<(FuncId, ArgList)>> = HashMap::default();
    crate::transform::rewrite(g, roots, |g, _e, node, ops| {
        let (Node::Call(o, _), Some(l)) = (node, ops.list) else {
            return g.rebuild(node, ops);
        };
        let (f, out) = g.output(o);
        let target = match instances.get(&(f, l)) {
            Some(&t) => t,
            None => {
                let t = specialize_instance(g, f, ops.ops, made);
                instances.insert((f, l), t);
                t
            }
        };
        match target {
            Some((copy, rest)) => g.call_list(copy, out, rest),
            None => g.rebuild(node, ops),
        }
    })
}

/// The copy of `f` a call over `args` runs, and the arguments it keeps
/// interned; `None` when the call passes no constant to specialize on.
fn specialize_instance<K: Field>(
    g: &mut Graph<K>,
    f: FuncId,
    args: &[ExprId],
    made: &mut Specialized,
) -> Option<(FuncId, ArgList)> {
    // A parameter stays an argument even when constant: its work is the
    // body's prolog, and specializing on it would split the instances of
    // one function into one copy per value.
    let roles = g.func(f).param_roles();
    let consts: Vec<(u32, ExprId)> = args
        .iter()
        .enumerate()
        .filter(|&(k, &a)| {
            g.const_of(a).is_some() && !matches!(roles.get(k), Some(ParamRole::Param))
        })
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
    // `key.1` is in argument order: one merge, not a search per argument.
    let mut bound = key.1.iter().map(|&(p, _)| p as usize).peekable();
    let rest: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(k, _)| bound.next_if_eq(&k).is_none())
        .map(|(_, &a)| a)
        .collect();
    Some((copy, g.intern_args(&rest)))
}

/// The copy of `f` with the parameters `consts` names bound to their
/// constants.
fn specialize_function<K: Field>(
    g: &mut Graph<K>,
    f: FuncId,
    consts: &[(u32, ExprId)],
    made: &mut Specialized,
) -> FuncId {
    let func = g.func(f);
    let name = func.name().to_string();
    let params = func.params().to_vec();
    let roles = func.param_roles().to_vec();
    let outputs = func.outputs().to_vec();
    let out_roles = func.output_roles().to_vec();
    let bound: HashMap<SymbolId, ExprId> = consts
        .iter()
        .map(|&(k, c)| (params[k as usize], c))
        .collect();
    let exprs: Vec<ExprId> = outputs
        .iter()
        .filter_map(|o| match *o {
            Output::Expr(e) => Some(e),
            _ => None,
        })
        .collect();
    let folded = crate::transform::substitute(g, &exprs, &bound);
    let folded = specialize_calls_in(g, &folded, made);
    let kept: Vec<usize> = (0..params.len())
        .filter(|&k| !bound.contains_key(&params[k]))
        .collect();
    let copy = g.push_function(Function::new(
        &name,
        kept.iter().map(|&k| params[k]).collect(),
        None,
    ));
    for (j, &k) in kept.iter().enumerate() {
        g.set_param_role(copy, j as u32, roles[k]);
    }
    let mut next = folded.into_iter();
    for (o, role) in outputs.iter().zip(out_roles) {
        let out = match *o {
            Output::Expr(_) => {
                let e = next.next().expect("one folded output per expression");
                if g.is_zero(e) {
                    Output::Zero
                } else {
                    Output::Expr(e)
                }
            }
            _ => Output::Zero,
        };
        let role = match role {
            OutputRole::Derivative { .. } => OutputRole::Plain,
            r => r,
        };
        g.push_output(copy, out, role);
    }
    copy
}

/// `acc` becomes the sorted union of itself and `part`, both sorted.
fn union_into(acc: &mut Vec<u32>, part: &[u32], scratch: &mut Vec<u32>) {
    scratch.clear();
    scratch.reserve(acc.len() + part.len());
    let (mut i, mut j) = (0, 0);
    while i < acc.len() && j < part.len() {
        let (a, b) = (acc[i], part[j]);
        scratch.push(a.min(b));
        i += usize::from(a <= b);
        j += usize::from(b <= a);
    }
    scratch.extend_from_slice(&acc[i..]);
    scratch.extend_from_slice(&part[j..]);
    std::mem::swap(acc, scratch);
}

/// What an analysis of what a node reaches follows: the operands its
/// derivative reads, or the ones its value does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Reach {
    Derivative = 0,
    Value = 1,
}

/// The operands of `e` that `reach` follows, into `out`: for a derivative
/// see [`crate::autodiff::carrying`]; for the value every operand, and of a
/// call only the arguments the called output reads.
fn reaching<K: Field>(g: &Graph<K>, reach: Reach, e: ExprId, out: &mut Vec<ExprId>) {
    match (reach, *g.node(e)) {
        (Reach::Derivative, _) => crate::autodiff::carrying(g, e, out),
        (Reach::Value, Node::Call(o, l)) => {
            out.clear();
            let (f, k) = g.output(o);
            let args = g.args(l);
            out.extend(g.output_reads(f, k).iter().map(|&p| args[p as usize]));
        }
        (Reach::Value, _) => {
            out.clear();
            out.extend_from_slice(&g.operands(e));
        }
    }
}
