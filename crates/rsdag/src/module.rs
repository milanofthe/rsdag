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
            let e = self.rebuild_node(node, &map, &module.arg_pool, &module.consts, &out_map);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{build, inputs, Spec, Vocabulary};
    use crate::{Tape, F64};

    /// A module is the program: reloading one and evaluating it must give
    /// the same bits, over programs nobody wrote by hand.
    #[test]
    fn a_module_round_trips_bit_exactly() {
        for seed in 0..24u64 {
            let mut spec = Spec::new(seed)
                .steps(60 + 10 * (seed as usize % 6))
                .params(4);
            spec = match seed % 3 {
                0 => spec.vocab(Vocabulary::Ring).max_list(20),
                1 => spec.vocab(Vocabulary::Elementary),
                _ => spec.vocab(Vocabulary::Full),
            };
            let mut g: Graph<F64> = Graph::new();
            let (roots, syms) = build(&mut g, &mut spec);
            let f = g.close("f", roots.clone());

            let module = g.to_module();
            let (loaded, map) = Graph::from_module(&module);
            assert_eq!(map.funcs[f.0 as usize], f, "seed {seed}: function ids");

            let roots2: Vec<ExprId> = roots.iter().map(|e| map.exprs[e.0 as usize]).collect();
            let syms2: Vec<SymbolId> = syms.iter().map(|s| map.symbols[s.0 as usize]).collect();
            let row = inputs(&mut spec.rng(), syms.len());
            let (mut w, mut o) = (Vec::new(), Vec::new());
            Tape::compile(&g, &roots, &syms).eval(&row, &mut w, &mut o);
            let (mut w2, mut o2) = (Vec::new(), Vec::new());
            Tape::compile(&loaded, &roots2, &syms2).eval(&row, &mut w2, &mut o2);
            assert!(
                o.iter()
                    .zip(&o2)
                    .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
                "seed {seed}: {o:?} vs {o2:?}"
            );
            // A module of the reloaded graph is the module it was built
            // from: loading is not lossy and not order-dependent.
            assert_eq!(
                loaded.to_module(),
                module,
                "seed {seed}: second module differs"
            );
        }
    }

    /// Loading into a graph that already holds the same subexpressions
    /// shares them instead of duplicating: the ids move, the values do not.
    #[test]
    fn loading_into_a_populated_graph_shares_nodes() {
        let mut a: Graph<F64> = Graph::new();
        let mut spec = Spec::new(9).steps(80).params(3);
        let (roots, syms) = build(&mut a, &mut spec);
        let module = a.to_module();

        let (mut b, first) = Graph::from_module(&module);
        let before = b.len();
        // Loading the same module a second time must not add a node: every
        // one of them is already interned, so the ids come back unchanged.
        let map = b.load_module(&module);
        assert_eq!(b.len(), before, "reloading a module added nodes");
        assert_eq!(map.exprs, first.exprs);
        for (k, r) in roots.iter().enumerate() {
            assert_eq!(map.exprs[r.0 as usize], *r, "root {k} moved");
        }
        for (k, s) in syms.iter().enumerate() {
            assert_eq!(map.symbols[s.0 as usize], *s, "symbol {k} moved");
        }
        let _ = &a;
    }

    /// A text format loses `f64` bits unless the constants are written as
    /// bit patterns, and cannot hold an infinity at all; both are checked
    /// here because a module that comes back one ulp away is a different
    /// program.
    #[cfg(feature = "serde")]
    #[test]
    fn a_module_survives_json() {
        let mut g: Graph<F64> = Graph::new();
        let mut spec = Spec::new(3).steps(120).params(4).vocab(Vocabulary::Full);
        let (mut roots, syms) = build(&mut g, &mut spec);
        // Values a text format mangles: an infinity, a NaN, and a double
        // whose shortest decimal parses back to its neighbour.
        for v in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1.1067040000000001,
        ] {
            let k = g.konst_f64(v);
            roots.push(k);
        }
        g.close("f", roots.clone());
        let module = g.to_module();
        let text = serde_json::to_string(&module).expect("serialize");
        let back: Module<F64> = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, module);

        let (loaded, map) = Graph::from_module(&back);
        let roots2: Vec<ExprId> = roots.iter().map(|e| map.exprs[e.0 as usize]).collect();
        let syms2: Vec<SymbolId> = syms.iter().map(|s| map.symbols[s.0 as usize]).collect();
        let row = inputs(&mut spec.rng(), syms.len());
        let (mut w, mut o) = (Vec::new(), Vec::new());
        Tape::compile(&g, &roots, &syms).eval(&row, &mut w, &mut o);
        let (mut w2, mut o2) = (Vec::new(), Vec::new());
        Tape::compile(&loaded, &roots2, &syms2).eval(&row, &mut w2, &mut o2);
        assert!(o.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}
