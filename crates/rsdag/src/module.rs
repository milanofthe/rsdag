//! The interchange format: a module as data.
//!
//! A [`Graph`] carries the mathematics of a set of functions plus the caches
//! that make building them fast (the hash-cons tables, the traversal memo)
//! and, for a function a solver has compiled, a binding to native code. Only
//! the first of those three is the module; the caches rebuild on load and a
//! compiled body is re-bound by the consumer.
//!
//! [`Module`] is that first part, and it is what a consumer serializes to
//! cache an analysis, to hand a model to code generation, or to export it.
//! Round-tripping a module reproduces every value bit for bit, which is the
//! property the tests pin.
//!
//! ```
//! # use rsdag::{Graph, Tape, F64};
//! let mut g: Graph<F64> = Graph::new();
//! let x = g.sym("x");
//! let e = g.sin(x);
//! let f = g.close("f", vec![e]);
//! let module = g.to_module();          // plain data, `serde`-serializable
//! let (mut back, map) = Graph::from_module(&module);
//! assert_eq!(map.funcs[f.0 as usize], f);
//! # let _ = &mut back;
//! ```

use rustc_hash::FxHashMap as HashMap;

use crate::field::Field;
use crate::func::{FuncId, FunctionBody, Output, OutputId};
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};
use crate::role::{OutputRole, ParamRole};

/// A function as data: no body binding, no derivative memo.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionData {
    pub name: String,
    pub params: Vec<SymbolId>,
    pub param_roles: Vec<ParamRole>,
    pub outputs: Vec<Output>,
    pub output_roles: Vec<OutputRole>,
    /// An extern function names the body it expects; the consumer re-binds
    /// it by name after loading (see [`Graph::bind_extern`]).
    pub extern_body: Option<String>,
}

/// A module as data: the nodes, the constants, the operand pool, the symbol
/// names and the functions. Everything else in a [`Graph`] is a cache.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(serialize = "K: serde::Serialize")))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "K: serde::de::DeserializeOwned"))
)]
pub struct Module<K> {
    /// Format version, so a stored module can be rejected rather than
    /// misread when the node vocabulary changes.
    pub version: u32,
    pub nodes: Vec<Node>,
    pub consts: Vec<K>,
    pub arg_pool: Vec<ExprId>,
    pub symbols: Vec<String>,
    pub funcs: Vec<FunctionData>,
    /// The `(function, output)` pairs the `Call` nodes name, in id order.
    pub call_outputs: Vec<(FuncId, u32)>,
}

/// The current [`Module::version`]. Bump it when the meaning of an existing
/// node changes; adding a variant at the end of an enum does not need it,
/// because an older reader fails on the unknown discriminant anyway.
pub const MODULE_VERSION: u32 = 1;

/// How the ids of a loaded module map onto the graph it was loaded into.
/// Loading into an empty graph is the identity, but loading into a graph
/// that already holds nodes is not, so the caller gets the mapping rather
/// than an assumption.
#[derive(Clone, Debug, Default)]
pub struct IdMap {
    pub exprs: Vec<ExprId>,
    pub symbols: Vec<SymbolId>,
    pub funcs: Vec<FuncId>,
}

impl<K: Field> Graph<K> {
    /// This graph as data, ready to serialize.
    ///
    /// A function with an extern body keeps its name and its shape; the body
    /// itself is a binding to compiled code and is not part of the module.
    pub fn to_module(&self) -> Module<K> {
        Module {
            version: MODULE_VERSION,
            nodes: self.nodes_slice().to_vec(),
            consts: self.consts_slice().to_vec(),
            arg_pool: self.arg_pool_slice().to_vec(),
            symbols: (0..self.n_symbols())
                .map(|i| self.symbol_name(SymbolId(i as u32)).to_string())
                .collect(),
            funcs: self
                .funcs_slice()
                .iter()
                .map(|f| FunctionData {
                    name: f.name.clone(),
                    params: f.params.clone(),
                    param_roles: f.param_roles.clone(),
                    outputs: f.outputs.clone(),
                    output_roles: f.output_roles.clone(),
                    extern_body: match &f.body {
                        FunctionBody::Symbolic => None,
                        FunctionBody::Extern(_) => Some(f.name.clone()),
                    },
                })
                .collect(),
            call_outputs: self.call_outputs_slice().to_vec(),
        }
    }

    /// Rebuild a graph from a module, re-interning every node so the
    /// hash-cons tables and the caches are consistent with it.
    ///
    /// Nodes are re-interned in id order, which is a topological order, so a
    /// node's operands are already mapped when it is built. Re-interning
    /// (rather than copying the arena) means a module loaded into a graph
    /// that already holds equal subexpressions shares them, exactly as if it
    /// had been built there.
    pub fn from_module(module: &Module<K>) -> (Graph<K>, IdMap) {
        let mut g = Graph::new();
        let map = g.load_module(module);
        (g, map)
    }

    /// Load a module into this graph, returning how its ids map onto it.
    pub fn load_module(&mut self, module: &Module<K>) -> IdMap {
        assert_eq!(
            module.version, MODULE_VERSION,
            "module format version {} cannot be read by this build (expects {MODULE_VERSION})",
            module.version
        );
        let mut map = IdMap {
            exprs: Vec::with_capacity(module.nodes.len()),
            symbols: Vec::with_capacity(module.symbols.len()),
            funcs: Vec::with_capacity(module.funcs.len()),
        };
        for name in &module.symbols {
            let e = self.sym(name);
            map.symbols.push(match self.node(e) {
                Node::Symbol(s) => *s,
                _ => unreachable!("sym returns a symbol node"),
            });
        }
        // Output ids are interned on demand by `call`, so the `Call` nodes
        // are remapped through the module's own table.
        let mut out_map: HashMap<OutputId, (FuncId, u32)> = HashMap::default();
        for (i, &(f, k)) in module.call_outputs.iter().enumerate() {
            out_map.insert(OutputId(i as u32), (f, k));
        }
        for (i, node) in module.nodes.iter().enumerate() {
            let pool = &module.arg_pool;
            let e = self.rebuild_node(
                node,
                crate::graph::Rebuild::Exact,
                |n| match *n {
                    Node::Const(c) => module.consts[c.0 as usize].clone(),
                    _ => unreachable!("only a constant asks for its value"),
                },
                |s| map.symbols[s.0 as usize],
                |x| map.exprs[x.0 as usize],
                |l| {
                    pool[l.start as usize..(l.start + l.len) as usize]
                        .iter()
                        .map(|&x| map.exprs[x.0 as usize])
                        .collect()
                },
                |o| {
                    let (f, k) = out_map[&o];
                    (map.funcs[f.0 as usize], k)
                },
            );
            debug_assert_eq!(map.exprs.len(), i);
            map.exprs.push(e);
        }
        for f in &module.funcs {
            let params: Vec<SymbolId> =
                f.params.iter().map(|s| map.symbols[s.0 as usize]).collect();
            let outputs: Vec<ExprId> = f
                .outputs
                .iter()
                .filter_map(|o| match o {
                    Output::Expr(e) => Some(map.exprs[e.0 as usize]),
                    _ => None,
                })
                .collect();
            let id = self.define_func(&f.name, params, outputs);
            for (k, r) in f.param_roles.iter().enumerate() {
                self.set_param_role(id, k as u32, *r);
            }
            for (k, r) in f.output_roles.iter().enumerate() {
                self.set_output_role(id, k as u32, *r);
            }
            map.funcs.push(id);
        }
        map
    }
}
