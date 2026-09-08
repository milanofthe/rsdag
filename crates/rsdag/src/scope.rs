//! Building a function incrementally.
//!
//! A [`Scope`] is an open function: it hands out parameters by name as the
//! caller needs them, records their order and their roles, and closes into a
//! [`FuncId`]. It is the seam every frontend builds through -- a netlist
//! lowering, a Verilog-A lowering, the Python tracer -- so that the order of
//! a function's parameters is the order they were asked for rather than an
//! accident of symbol ids.
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
//! let r = s.param("r");
//! let i = s.div(v, r);                 // a graph constructor, on the scope
//! let f = s.close(vec![i]);
//! # let _ = f;
//! ```

use std::ops::{Deref, DerefMut};

use num_rational::BigRational;

use crate::field::Field;
use crate::func::FuncId;
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};
use crate::role::{OutputRole, ParamRole};

/// An open function over a graph. See the module docs.
pub struct Scope<'g, K: Field = BigRational> {
    graph: &'g mut Graph<K>,
    name: String,
    params: Vec<SymbolId>,
    roles: Vec<ParamRole>,
}

impl<'g, K: Field> Scope<'g, K> {
    pub fn new(graph: &'g mut Graph<K>, name: &str) -> Scope<'g, K> {
        Scope {
            graph,
            name: name.to_string(),
            params: Vec::new(),
            roles: Vec::new(),
        }
    }

    /// A parameter of this function, in call order. Asking twice for the
    /// same name returns the same parameter rather than a second one.
    pub fn param(&mut self, name: &str) -> ExprId {
        self.param_with_role(name, ParamRole::Free)
    }

    /// As [`Scope::param`], with the role the consumer's system layer needs
    /// (a state, an input port element, a mutable parameter).
    pub fn param_with_role(&mut self, name: &str, role: ParamRole) -> ExprId {
        let e = self.graph.sym(name);
        let s = match self.graph.node(e) {
            Node::Symbol(s) => *s,
            _ => unreachable!("sym returns a symbol node"),
        };
        match self.params.iter().position(|&p| p == s) {
            Some(k) => self.roles[k] = role,
            None => {
                self.params.push(s);
                self.roles.push(role);
            }
        }
        e
    }

    /// The parameters asked for so far, in order.
    pub fn params(&self) -> &[SymbolId] {
        &self.params
    }

    /// Close the scope over `outputs`.
    ///
    /// Free symbols the outputs depend on that were never asked for as
    /// parameters become trailing parameters, so closing over an expression
    /// built outside the scope is total rather than silently wrong.
    pub fn close(self, outputs: Vec<ExprId>) -> FuncId {
        self.close_with_roles(
            outputs
                .into_iter()
                .map(|e| (OutputRole::Plain, e))
                .collect(),
        )
    }

    /// As [`Scope::close`], giving each output its role.
    pub fn close_with_roles(mut self, outputs: Vec<(OutputRole, ExprId)>) -> FuncId {
        let exprs: Vec<ExprId> = outputs.iter().map(|&(_, e)| e).collect();
        for s in self.graph.free_symbols_in(&exprs) {
            if !self.params.contains(&s) {
                self.params.push(s);
                self.roles.push(ParamRole::Free);
            }
        }
        let f = self
            .graph
            .define_func(&self.name, self.params.clone(), exprs);
        for (k, role) in self.roles.iter().enumerate() {
            self.graph.set_param_role(f, k as u32, *role);
        }
        for (k, (role, _)) in outputs.iter().enumerate() {
            self.graph.set_output_role(f, k as u32, *role);
        }
        f
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OutputRole, Tape, F64};

    #[test]
    fn parameters_keep_the_order_they_were_asked_for() {
        let mut g: Graph<F64> = Graph::new();
        let mut s = Scope::new(&mut g, "f");
        // Asked for in the opposite order to their symbol ids.
        let z = s.param("z");
        let a = s.param("a");
        let e = s.sub(z, a);
        let f = s.close(vec![e]);
        let names: Vec<&str> = g.func(f).params.iter().map(|&p| g.symbol_name(p)).collect();
        assert_eq!(names, ["z", "a"]);
    }

    #[test]
    fn roles_survive_and_select_jacobian_blocks() {
        let mut g: Graph<F64> = Graph::new();
        let mut s = Scope::new(&mut g, "sys");
        let x = s.param_with_role("x", ParamRole::State { id: 0 });
        let k = s.param("k");
        let kx = s.mul(k, x);
        let f = s.close_with_roles(vec![(OutputRole::Residual { id: 0 }, kx)]);
        let blocks = g.jacobian_by_role(
            f,
            |o| matches!(o, OutputRole::Residual { .. }),
            |p| matches!(p, ParamRole::State { .. }),
        );
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn a_symbol_used_but_not_asked_for_still_becomes_a_parameter() {
        let mut g: Graph<F64> = Graph::new();
        let outside = g.sym("t");
        let mut s = Scope::new(&mut g, "f");
        let x = s.param("x");
        let e = s.add(x, outside);
        let f = s.close(vec![e]);
        let names: Vec<&str> = g.func(f).params.iter().map(|&p| g.symbol_name(p)).collect();
        assert_eq!(names, ["x", "t"]);
        let tape = Tape::compile(&g, &[e], &g.func(f).params.clone());
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&[2.0, 3.0], &mut w, &mut o);
        assert_eq!(o[0], 5.0);
    }

    #[test]
    fn asking_twice_for_a_name_gives_one_parameter() {
        let mut g: Graph<F64> = Graph::new();
        let mut s = Scope::new(&mut g, "f");
        let a = s.param("v");
        let b = s.param_with_role("v", ParamRole::State { id: 3 });
        assert_eq!(a, b);
        assert_eq!(s.params().len(), 1);
        let f = s.close(vec![a]);
        assert_eq!(g.func(f).param_roles[0], ParamRole::State { id: 3 });
    }
}
