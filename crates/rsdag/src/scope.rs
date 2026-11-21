//! Building a function incrementally.
//!
//! A [`Scope`] is an open function: it hands out parameters by name as the
//! caller needs them, records their order and their roles, and closes into a
//! [`FuncId`], so that the order of a function's parameters is the order
//! they were asked for rather than an accident of symbol ids.
//!
//! A scope derefs to the graph it builds in, so the ordinary constructors
//! work on it directly:
//!
//! ```
//! use rsdag::{Graph, ParamRole, Scope, F64};
//!
//! let mut g: Graph<F64> = Graph::new();
//! let mut s = Scope::new(&mut g, "rc");
//! let v = s.param_with_role("v", ParamRole::State { id: 0 });
//! let r = s.param_with_role("r", ParamRole::Free);
//! let i = s.div(v, r);                 // a graph constructor, on the scope
//! let f = s.close(vec![i]);
//! # let _ = f;
//! ```

use std::ops::{Deref, DerefMut};

use crate::field::{Field, F64};
use crate::func::{FuncId, Output};
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};
use crate::role::{OutputRole, ParamRole};

/// An open function over a graph. See the module docs.
pub struct Scope<'g, K: Field = F64> {
    graph: &'g mut Graph<K>,
    name: String,
    params: Vec<SymbolId>,
    roles: Vec<ParamRole>,
    /// Symbols the body reads as globals, not parameters.
    globals: Vec<SymbolId>,
}

impl<'g, K: Field> Scope<'g, K> {
    pub fn new(graph: &'g mut Graph<K>, name: &str) -> Scope<'g, K> {
        Scope {
            graph,
            name: name.to_string(),
            params: Vec::new(),
            roles: Vec::new(),
            globals: Vec::new(),
        }
    }

    /// A parameter of this function, in call order, with the role the
    /// consumer's system layer needs (a state, an input port element, a
    /// mutable parameter, or [`ParamRole::Free`]). Asking twice for the same
    /// name returns the same parameter rather than a second one, with the
    /// role asked for last.
    pub fn param_with_role(&mut self, name: &str, role: ParamRole) -> ExprId {
        let s = self.graph.symbol_id(name);
        let e = self.graph.symbol_expr(s);
        match self.params.iter().position(|&p| p == s) {
            Some(k) => self.roles[k] = role,
            None => {
                self.params.push(s);
                self.roles.push(role);
            }
        }
        e
    }

    /// A global the body reads: a symbol every call sees the same (a model
    /// card's parameter), not a parameter (see [`Graph::globals`]).
    pub fn global(&mut self, name: &str) -> ExprId {
        let e = self.graph.sym(name);
        if let Node::Symbol(s) = *self.graph.node(e) {
            if !self.globals.contains(&s) {
                self.globals.push(s);
            }
        }
        e
    }

    /// Close the scope over `outputs`.
    ///
    /// Free symbols the outputs mention that were never asked for as
    /// parameters nor declared [`global`](Self::global) become trailing
    /// parameters, so closing over an expression built outside the scope is
    /// total rather than silently wrong. A called function's globals stay
    /// globals.
    pub fn close(mut self, outputs: Vec<ExprId>) -> FuncId {
        for s in self.graph.mentioned_symbols_in(&outputs) {
            if !self.params.contains(&s) && !self.globals.contains(&s) {
                self.params.push(s);
                self.roles.push(ParamRole::Free);
            }
        }
        let outputs = outputs
            .into_iter()
            .map(|e| (Output::Expr(e), OutputRole::Plain));
        (self.graph).define(&self.name, self.params, &self.roles, None, outputs)
    }
}

impl<K: Field> Deref for Scope<'_, K> {
    type Target = Graph<K>;
    fn deref(&self) -> &Graph<K> {
        self.graph
    }
}

impl<K: Field> DerefMut for Scope<'_, K> {
    fn deref_mut(&mut self) -> &mut Graph<K> {
        self.graph
    }
}
