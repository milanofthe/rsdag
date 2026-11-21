//! Defining a function: the one way a front end turns expressions into a
//! [`FuncId`].
//!
//! A [`Scope`] is an open function: it hands out parameters as the caller
//! asks for them, each with its role, records their order, and closes over
//! outputs that carry their roles too, so that the order of a function's
//! parameters is the order they were asked for rather than an accident of
//! symbol ids, and a consumer reads what each parameter and output stands
//! for off the function. Closing interns: the same parameters, roles and
//! outputs give the same function, whatever its name.
//!
//! A scope derefs to the graph it builds in, so the ordinary constructors
//! work on it directly:
//!
//! ```
//! use rsdag::{Graph, OutputRole, ParamRole, Scope, F64};
//!
//! let mut g: Graph<F64> = Graph::new();
//! let mut s = Scope::new(&mut g, "rc");
//! let v = s.param("v", ParamRole::State { id: 0 });
//! let r = s.param("r", ParamRole::Param);
//! let i = s.div(v, r);                 // a graph constructor, on the scope
//! let f = s.close([(i, OutputRole::Residual { id: 0 })]);
//! # let _ = f;
//! ```

use std::ops::{Deref, DerefMut};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::field::{Field, F64};
use crate::func::{FuncId, Output};
use crate::graph::Graph;
use crate::node::{ExprId, SymbolId};
use crate::role::{OutputRole, ParamRole};

/// A symbol a scope is asked for: by name, or a symbol of the graph.
pub trait Formal {
    fn symbol<K: Field>(self, graph: &mut Graph<K>) -> SymbolId;
}

impl Formal for &str {
    fn symbol<K: Field>(self, graph: &mut Graph<K>) -> SymbolId {
        graph.symbol_id(self)
    }
}

impl Formal for &String {
    fn symbol<K: Field>(self, graph: &mut Graph<K>) -> SymbolId {
        graph.symbol_id(self)
    }
}

impl Formal for SymbolId {
    fn symbol<K: Field>(self, _: &mut Graph<K>) -> SymbolId {
        self
    }
}

/// An open function over a graph. See the module docs.
pub struct Scope<'g, K: Field = F64> {
    graph: &'g mut Graph<K>,
    name: String,
    params: Vec<SymbolId>,
    roles: Vec<ParamRole>,
    /// Each parameter's place in `params`.
    index: HashMap<SymbolId, usize>,
    /// Symbols the body reads as globals, not parameters.
    globals: HashSet<SymbolId>,
}

impl<'g, K: Field> Scope<'g, K> {
    pub fn new(graph: &'g mut Graph<K>, name: &str) -> Scope<'g, K> {
        Scope {
            graph,
            name: name.to_string(),
            params: Vec::new(),
            roles: Vec::new(),
            index: HashMap::default(),
            globals: HashSet::default(),
        }
    }

    /// A parameter of this function, in call order, with the role the
    /// consumer's system layer needs (a state, an input port element, a
    /// tunable parameter, or [`ParamRole::Free`]). Asking twice for the same
    /// symbol returns the same parameter rather than a second one, with the
    /// role asked for last.
    pub fn param(&mut self, formal: impl Formal, role: ParamRole) -> ExprId {
        let s = formal.symbol(self.graph);
        match self.index.get(&s) {
            Some(&k) => self.roles[k] = role,
            None => {
                self.index.insert(s, self.params.len());
                self.params.push(s);
                self.roles.push(role);
            }
        }
        self.graph.symbol_expr(s)
    }

    /// A global the body reads: a symbol every call sees the same (a model
    /// card's parameter), not a parameter (see [`Graph::globals`]).
    pub fn global(&mut self, formal: impl Formal) -> ExprId {
        let s = formal.symbol(self.graph);
        self.globals.insert(s);
        self.graph.symbol_expr(s)
    }

    /// Close the scope over `outputs`, each with its role.
    ///
    /// Free symbols the outputs mention that were never asked for as
    /// parameters nor declared [`global`](Self::global) become trailing
    /// [`Free`](ParamRole::Free) parameters, by symbol, so closing over an
    /// expression built outside the scope is total rather than silently
    /// wrong. A called function's globals stay globals.
    pub fn close(mut self, outputs: impl IntoIterator<Item = (ExprId, OutputRole)>) -> FuncId {
        let (exprs, roles): (Vec<ExprId>, Vec<OutputRole>) = outputs.into_iter().unzip();
        for s in self.graph.mentioned_symbols_in(&exprs) {
            if !self.index.contains_key(&s) && !self.globals.contains(&s) {
                self.params.push(s);
                self.roles.push(ParamRole::Free);
            }
        }
        let outputs = exprs.into_iter().map(Output::Expr).zip(roles);
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
