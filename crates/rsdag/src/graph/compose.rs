//! Composition: programs built apart, called as one. A function of another
//! graph is imported with the functions it calls; calls into it compose it
//! into whatever this graph builds, and a program over the composition is
//! compiled as one, its composite functions as templates (see
//! [`Tape::compile`](crate::Tape::compile)).

use super::*;

impl<K: Field> Graph<K> {
    /// Function `f` of graph `other` as a function of this graph, together
    /// with the functions it calls: its outputs rebuilt here (sharing what
    /// this graph already holds), its parameters, roles and derivative
    /// outputs kept. `imported` maps the functions of `other` already
    /// imported to theirs here, so a function called from several imports
    /// comes over once; pass the same map to import more from `other`.
    ///
    /// A symbolic function this graph already holds with the same
    /// parameters, roles and outputs is taken for it rather than a copy made.
    /// An extern function shares its body. A body a consumer registered for
    /// a symbolic function stays with `other`; here the function compiles
    /// its own.
    pub fn import(
        &mut self,
        other: &Graph<K>,
        f: FuncId,
        imported: &mut HashMap<FuncId, FuncId>,
    ) -> FuncId {
        if let Some(&g) = imported.get(&f) {
            return g;
        }
        let func = other.func(f);
        let params: Vec<SymbolId> = func
            .params()
            .iter()
            .map(|&s| self.symbol_id(other.symbol_name(s)))
            .collect();
        let built = self.import_exprs(other, &func.exprs(), imported);
        let outputs = func.outputs_over(built);
        // A symbolic function this graph already holds, the same over the
        // same parameters, is the one to call: its instances, imported along
        // different paths, then batch together.
        let same = (0..self.funcs.len()).map(|k| FuncId(k as u32)).find(|&g| {
            let h = &self.funcs[g.0 as usize];
            func.extern_body().is_none()
                && h.extern_body().is_none()
                && h.params() == &params[..]
                && h.param_roles() == func.param_roles()
                && h.outputs() == &outputs[..]
                && h.output_roles() == func.output_roles()
        });
        if let Some(g) = same {
            imported.insert(f, g);
            return g;
        }
        let outputs = outputs.into_iter().zip(func.output_roles().iter().copied());
        let body = func.extern_body().cloned();
        let new = self.define(func.name(), params, func.param_roles(), body, outputs);
        imported.insert(f, new);
        new
    }

    /// Expressions of graph `other` rebuilt here: their symbols by name,
    /// their calls into the functions of `other` imported (see
    /// [`import`](Self::import)).
    pub fn import_exprs(
        &mut self,
        other: &Graph<K>,
        roots: &[ExprId],
        imported: &mut HashMap<FuncId, FuncId>,
    ) -> Vec<ExprId> {
        let mut rule = Import {
            g: self,
            other,
            imported,
        };
        crate::transform::rewrite(&mut rule, roots)
    }
}

/// [`Graph::import_exprs`]: `other`'s nodes rebuilt in `g`.
struct Import<'a, K: Field> {
    g: &'a mut Graph<K>,
    other: &'a Graph<K>,
    imported: &'a mut HashMap<FuncId, FuncId>,
}

impl<K: Field> crate::transform::Rule<K> for Import<'_, K> {
    type V = ExprId;
    fn source(&self) -> &Graph<K> {
        self.other
    }
    fn node(&mut self, _: ExprId, node: Node, ops: &[ExprId]) -> ExprId {
        match node {
            Node::Const(c) => self.g.konst(self.other.const_val(c).clone()),
            Node::Symbol(s) => self.g.sym(self.other.symbol_name(s)),
            _ => self.g.build(node, ops),
        }
    }
    fn calls(
        &mut self,
        call: &crate::transform::Call<ExprId>,
        outs: &[(ExprId, u32)],
        vals: &mut Vec<ExprId>,
    ) {
        let f = self.g.import(self.other, call.f, self.imported);
        // a bound call: its context over the imported expressions
        let ctx = match call.ctx {
            NO_CONTEXT => NO_CONTEXT,
            c => {
                let at = &self.other.contexts[c as usize].at;
                let pairs: Vec<(u32, ExprId)> =
                    at.iter().copied().zip(call.bound.iter().copied()).collect();
                self.g.bind(f, &pairs).ctx
            }
        };
        let l = self.g.intern_args(call.args);
        for &(_, k) in outs {
            vals.push(self.g.call_list_in(f, k, ctx, l));
        }
    }
}
