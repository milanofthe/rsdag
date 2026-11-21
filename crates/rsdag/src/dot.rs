//! A graph's DAG as data for a consumer that lays it out and draws it.
//!
//! [`GraphView`] views the hash-consed DAG: a shared subexpression is one
//! node with several parents, a call one node; `inline` takes every call as
//! its body, once per instance, in nested frames: the complete graph.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::constants::{LABEL_EXACT_MAX, LABEL_FIXED_MAX, LABEL_FIXED_MIN, LABEL_INT_MAX};
use crate::field::Field;
use crate::func::Output;
use crate::graph::{Graph, NodeTable};
use crate::node::{ExprId, Node, ReduceOp, SymbolId};
use crate::transform::{rewrite, Call, Rule};

/// What a node is, which decides how it is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A varying input: a state, an unknown.
    Input,
    /// A solve-constant input: a parameter.
    Param,
    Const,
    /// Arithmetic and elementary functions.
    Op,
    /// A comparison or a select: where a program branches.
    Choice,
    /// A fused kernel: a reduction, a dot, a product or a dense solve.
    Kernel,
    /// A call of a function body.
    Call,
    /// A named result.
    Output,
}

/// How a graph's operators are labelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Notation {
    /// `*`, `neg`, `^-1`, `exp`, `>`, `select`, `sum`, `dot`.
    Ascii,
    /// `×`, `−(·)`, `(·)^-1`, `exp(·)`, `(·) > (·)`, `(·) ? (·) : (·)`,
    /// `Σ`, `<·,·>`.
    Math,
}

/// A number as a label: an integer as one, else four significant digits,
/// in exponent form beyond `1e5` or below `1e-3`.
fn number(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < LABEL_INT_MAX {
        return format!("{}", x as i64);
    }
    let a = x.abs();
    if !(LABEL_FIXED_MIN..LABEL_FIXED_MAX).contains(&a) {
        return format!("{x:.3e}");
    }
    let s = format!("{x:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The nodes `roots` reach, themselves included.
pub fn reachable<K: Field>(g: &Graph<K>, roots: &[ExprId]) -> FxHashSet<ExprId> {
    g.cone(roots, false).nodes.iter().copied().collect()
}

/// A graph's DAG under some roots (see the module docs).
pub struct GraphView<'g, K: Field> {
    g: &'g Graph<K>,
    notation: Notation,
    roots: Vec<(ExprId, String)>,
    params: FxHashSet<SymbolId>,
    inline: bool,
    labels: FxHashMap<ExprId, String>,
}

/// The graph a [`GraphView`] draws, as data: for a consumer that lays it out
/// and draws it itself. Node ids are the DOT names (`n{expr}`, `out{root}`).
#[derive(Clone, Debug, Default)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    /// `(from, to, label)` by node index, operand to consumer (a select's
    /// operands labelled `if`, `then`, `else`).
    pub edges: Vec<(usize, usize, &'static str)>,
    /// The frames' labels; a node names its frame by index.
    pub clusters: Vec<String>,
    /// The frame each frame sits in (nested instances, see
    /// [`GraphView::inline`]).
    pub parents: Vec<Option<usize>>,
}

/// One node of [`GraphData`].
#[derive(Clone, Debug)]
pub struct GraphNode {
    /// The DOT name.
    pub id: String,
    pub label: String,
    pub kind: Kind,
    /// The frame it sits in.
    pub cluster: Option<usize>,
}

impl<'g, K: Field> GraphView<'g, K> {
    pub fn new(g: &'g Graph<K>) -> Self {
        GraphView {
            g,
            notation: Notation::Ascii,
            roots: Vec::new(),
            params: FxHashSet::default(),
            inline: false,
            labels: FxHashMap::default(),
        }
    }

    /// Draw node `e` labelled `label` instead of its own.
    pub fn label(mut self, e: ExprId, label: &str) -> Self {
        self.labels.insert(e, label.to_string());
        self
    }

    /// How the operators are labelled ([`Notation::Ascii`] by default).
    pub fn notation(mut self, notation: Notation) -> Self {
        self.notation = notation;
        self
    }

    /// Draw `e` and what it reaches, as a result named `name` (no result
    /// node when `name` is empty).
    pub fn root(mut self, e: ExprId, name: &str) -> Self {
        self.roots.push((e, name.to_string()));
        self
    }

    /// Draw these symbols as parameters, the others as inputs.
    pub fn params(mut self, syms: &[SymbolId]) -> Self {
        self.params.extend(syms.iter().copied());
        self
    }

    /// Draw every call as what it computes: the called function's body,
    /// its parameters bound to the call's arguments, once per instance (the
    /// calls of one function over one argument list), in a frame inside its
    /// caller's frame, named by the call's label or else the function. The
    /// complete graph, its hierarchy as nested frames; instances stay apart
    /// however equal their bodies' expressions are.
    pub fn inline(mut self) -> Self {
        self.inline = true;
        self
    }

    /// The label and kind of node `e`.
    fn style(&self, e: ExprId) -> (String, Kind) {
        let (label, kind) = self.own_style(e);
        match self.labels.get(&e) {
            Some(l) => (l.clone(), kind),
            None => (label, kind),
        }
    }

    /// The node's own label and its kind.
    fn own_style(&self, e: ExprId) -> (String, Kind) {
        let g = self.g;
        let math = self.notation == Notation::Math;
        match *g.node(e) {
            Node::Const(c) => {
                let exact = g.const_val(c).render();
                let label = match g.const_f64(e) {
                    Some(x) if exact.len() > LABEL_EXACT_MAX || !exact.contains('/') => number(x),
                    _ => exact,
                };
                (label, Kind::Const)
            }
            Node::Symbol(s) => (
                g.symbol_name(s).to_string(),
                if self.params.contains(&s) {
                    Kind::Param
                } else {
                    Kind::Input
                },
            ),
            Node::Add(..) => ("+".into(), Kind::Op),
            Node::Mul(..) => (if math { "\u{00d7}" } else { "*" }.into(), Kind::Op),
            Node::Neg(_) => (
                if math { "\u{2212}(\u{00b7})" } else { "neg" }.into(),
                Kind::Op,
            ),
            Node::Pow(_, n) if math => (format!("(\u{00b7})^{n}"), Kind::Op),
            Node::Pow(_, n) => (format!("^{n}"), Kind::Op),
            Node::Unary(op, _) if math => (format!("{}(\u{00b7})", op.name()), Kind::Op),
            Node::Unary(op, _) => (op.name().into(), Kind::Op),
            Node::Binary(op, ..) => (op.name().into(), Kind::Op),
            Node::Cmp(op, ..) => {
                let c = op.symbol();
                let label = if math {
                    format!("(\u{00b7}) {c} (\u{00b7})")
                } else {
                    c.to_string()
                };
                (label, Kind::Choice)
            }
            Node::Select(..) if math => {
                ("(\u{00b7}) ? (\u{00b7}) : (\u{00b7})".into(), Kind::Choice)
            }
            Node::Select(..) => ("select".into(), Kind::Choice),
            Node::Reduce(op, _) => (
                match (op, math) {
                    (ReduceOp::Sum, true) => "\u{03a3}",
                    (ReduceOp::Product, true) => "\u{03a0}",
                    (op, _) => op.name(),
                }
                .into(),
                Kind::Kernel,
            ),
            Node::Dot(_) => (
                if math { "<\u{00b7},\u{00b7}>" } else { "dot" }.into(),
                Kind::Kernel,
            ),
            Node::Solve(_, i) => (format!("solve[{i}]"), Kind::Kernel),
            Node::Call(o, _) => {
                let (f, k) = g.output(o);
                let func = g.func(f);
                let name = if func.outputs().len() > 1 {
                    format!("{}#{k}", func.name())
                } else {
                    func.name().to_string()
                };
                (name, Kind::Call)
            }
        }
    }

    /// The graph this view draws (see [`GraphData`]).
    pub fn data(&self) -> GraphData {
        if self.inline {
            return Inliner::data(self);
        }
        let g = self.g;
        let seeds: Vec<ExprId> = self.roots.iter().map(|r| r.0).collect();
        // Nodes in id order: operands before the nodes that read them.
        let mut exprs = g.cone(&seeds, false).nodes.to_vec();
        exprs.sort_unstable();
        let mut data = GraphData::default();
        let mut at: NodeTable<u32> = NodeTable::new(g.len());
        for &e in &exprs {
            let (label, kind) = self.style(e);
            at.set(e, data.nodes.len() as u32);
            data.nodes.push(GraphNode {
                id: format!("n{}", e.0),
                label,
                kind,
                cluster: None,
            });
        }
        let pos = |e: &ExprId| at.get(*e).expect("drawn") as usize;
        for &e in &exprs {
            let select = matches!(g.node(e), Node::Select(..));
            for (k, a) in g.operands(e).iter().enumerate() {
                let label = if select {
                    ["if", "then", "else"][k]
                } else {
                    ""
                };
                data.edges.push((pos(a), pos(&e), label));
            }
        }
        for (i, (e, name)) in self.roots.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            data.nodes.push(GraphNode {
                id: format!("out{i}"),
                label: name.clone(),
                kind: Kind::Output,
                cluster: None,
            });
            data.edges.push((pos(e), data.nodes.len() - 1, ""));
        }
        data
    }
}

/// [`GraphView::inline`]: the drawn graph with every call of a body
/// replaced by the body, per instance. A rewrite into the drawn nodes (see
/// [`crate::transform::rewrite`]) in frames: 0 the top, an instance's frame
/// the index of its cluster plus one, its body rewritten there with its
/// parameters bound to the drawn operands of the call in the caller's frame.
struct Inliner<'v, 'g, K: Field> {
    view: &'v GraphView<'g, K>,
    data: GraphData,
    /// The free symbols drawn, at the top.
    top: FxHashMap<ExprId, u32>,
}

/// One frame of an [`Inliner`]: its index and the drawn operands its
/// parameters are bound to.
struct Frame<'a, 'v, 'g, K: Field> {
    inl: &'a mut Inliner<'v, 'g, K>,
    fr: usize,
    bound: FxHashMap<SymbolId, u32>,
}

impl<K: Field> Rule<K> for Frame<'_, '_, '_, K> {
    type V = u32;
    fn source(&self) -> &Graph<K> {
        self.inl.view.g
    }
    fn node(&mut self, e: ExprId, node: Node, ops: &[u32]) -> u32 {
        let Node::Symbol(s) = node else {
            return self.inl.draw(self.fr, e, ops);
        };
        if let Some(&n) = self.bound.get(&s) {
            return n;
        }
        if let Some(&n) = self.inl.top.get(&e) {
            return n;
        }
        let n = self.inl.draw(0, e, ops);
        self.inl.top.insert(e, n);
        n
    }
    fn calls(&mut self, call: &Call<u32>, outs: &[(ExprId, u32)], vals: &mut Vec<u32>) {
        let view = self.inl.view;
        let func = view.g.func(call.f);
        let ops = call.params(view.g);
        let body = |k: u32| match func.outputs()[k as usize] {
            Output::Expr(b) => Some(b),
            _ => None,
        };
        let exprs: Vec<ExprId> = outs.iter().filter_map(|&(_, k)| body(k)).collect();
        let mut drawn = Vec::new().into_iter();
        if !exprs.is_empty() {
            let label = (outs.iter().find_map(|(e, _)| view.labels.get(e)))
                .map_or_else(|| func.name().to_string(), |l| l.clone());
            self.inl.data.clusters.push(label);
            self.inl.data.parents.push(self.fr.checked_sub(1));
            let mut frame = Frame {
                fr: self.inl.data.clusters.len(),
                inl: &mut *self.inl,
                bound: (func.params().iter().copied())
                    .zip(ops.iter().copied())
                    .collect(),
            };
            drawn = rewrite(&mut frame, &exprs).into_iter();
        }
        // a call of no body is drawn as the call
        for &(e, k) in outs {
            vals.push(match body(k) {
                Some(_) => drawn.next().expect("one per body"),
                None => self.inl.draw(self.fr, e, &ops),
            });
        }
    }
}

impl<'v, 'g, K: Field> Inliner<'v, 'g, K> {
    /// Node `e` drawn in frame `fr`, read from the drawn `ops`.
    fn draw(&mut self, fr: usize, e: ExprId, ops: &[u32]) -> u32 {
        let (label, kind) = self.view.style(e);
        let n = self.data.nodes.len();
        self.data.nodes.push(GraphNode {
            id: match fr {
                0 => format!("n{}", e.0),
                _ => format!("n{}_{fr}", e.0),
            },
            label,
            kind,
            cluster: fr.checked_sub(1),
        });
        let select = matches!(self.view.g.node(e), Node::Select(..));
        for (k, &op) in ops.iter().enumerate() {
            let label = if select {
                ["if", "then", "else"][k]
            } else {
                ""
            };
            self.data.edges.push((op as usize, n, label));
        }
        n as u32
    }

    fn data(view: &'v GraphView<'g, K>) -> GraphData {
        let mut inl = Inliner {
            view,
            data: GraphData::default(),
            top: FxHashMap::default(),
        };
        let seeds: Vec<ExprId> = view.roots.iter().map(|r| r.0).collect();
        let mut top = Frame {
            inl: &mut inl,
            fr: 0,
            bound: FxHashMap::default(),
        };
        let drawn = rewrite(&mut top, &seeds);
        for (i, (_, name)) in view.roots.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            inl.data.nodes.push(GraphNode {
                id: format!("out{i}"),
                label: name.clone(),
                kind: Kind::Output,
                cluster: None,
            });
            let n = drawn[i] as usize;
            inl.data.edges.push((n, inl.data.nodes.len() - 1, ""));
        }
        inl.data
    }
}
