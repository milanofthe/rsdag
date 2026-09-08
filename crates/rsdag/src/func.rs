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
//!   compiled tape through the body the solver registers (its native forms
//!   and lane batching); a batch of calls into one function is what the SIMD
//!   lanes evaluate.
//!
//! An *extern* function has no symbolic body: its outputs, including the
//! derivative outputs it can supply, are slots of a numeric
//! [`ExternBundle`](crate::extern_fn::ExternBundle) (a compiled OSDI model).
//! A derivative an extern cannot supply is the zero output.

use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::extern_fn::ExternBundle;
use crate::node::{ExprId, SymbolId};
use crate::role::{OutputRole, ParamRole};

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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Output {
    /// A symbolic expression over the function's parameters.
    Expr(ExprId),
    /// Output slot `k` of an extern body.
    Slot(u32),
    /// Identically zero (a derivative the body does not carry).
    Zero,
}

/// How a function's outputs are computed.
pub enum FunctionBody {
    /// Outputs are expressions over the parameters; a solver may register a
    /// compiled body for them (see [`Function::compiled`]).
    Symbolic,
    /// Outputs are slots of a numeric bundle.
    Extern(Arc<dyn ExternBundle>),
}

/// A compiled body registered for a symbolic function: which output each
/// bundle slot carries, so a tape can pick outputs without the symbolic
/// expressions.
pub struct CompiledBody {
    pub bundle: Arc<dyn ExternBundle>,
    /// `slot_of[out]` is the bundle slot holding output `out`, `None` when
    /// the body was compiled without it.
    pub slot_of: Vec<Option<u32>>,
}

pub struct Function {
    pub name: String,
    /// Formal leaves in argument order.
    pub params: Vec<SymbolId>,
    /// One role per parameter (`Free` unless set).
    pub param_roles: Vec<ParamRole>,
    pub outputs: Vec<Output>,
    /// One role per output (`Plain` unless set; derivative outputs are
    /// tagged `Derivative`).
    pub output_roles: Vec<OutputRole>,
    pub body: FunctionBody,
    /// Derivative output `d outputs[out] / d params[param]`, by index.
    pub(crate) deriv_index: HashMap<(u32, u32), u32>,
    /// The solver's compiled body (symbolic functions only).
    pub compiled: Option<CompiledBody>,
}

/// A function body evaluated by the interpreter: the fallback every consumer
/// can build from the symbolic outputs alone, so a tape or an arena sweep is
/// total without a solver-registered body (which only upgrades this to native
/// code and lane batching).
pub struct InterpretedBody {
    tape: crate::tape::Tape,
    n_out: usize,
}

impl ExternBundle for InterpretedBody {
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    fn call(&self, args: &[f64], out: &mut [f64]) {
        let (mut work, mut o) = (Vec::new(), Vec::new());
        self.tape.eval(args, &mut work, &mut o);
        out.copy_from_slice(&o[..self.n_out]);
    }
}

impl Function {
    /// An evaluator for every symbolic output of the function, interpreted
    /// (see [`InterpretedBody`]); `None` for an extern function.
    pub fn interpreted_body<K: crate::field::Field>(
        &self,
        ctx: &crate::graph::Graph<K>,
    ) -> Option<CompiledBody> {
        if self.is_extern() {
            return None;
        }
        let mut roots = Vec::new();
        let mut slot_of = Vec::with_capacity(self.outputs.len());
        for o in &self.outputs {
            match o {
                Output::Expr(e) => {
                    slot_of.push(Some(roots.len() as u32));
                    roots.push(*e);
                }
                _ => slot_of.push(None),
            }
        }
        let tape = crate::tape::Tape::compile(ctx, &roots, &self.params);
        Some(CompiledBody {
            bundle: Arc::new(InterpretedBody {
                tape,
                n_out: roots.len(),
            }),
            slot_of,
        })
    }

    /// Indices of the parameters carrying `role`, in argument order.
    pub fn params_with_role(&self, role: impl Fn(&ParamRole) -> bool) -> Vec<u32> {
        (0..self.params.len() as u32)
            .filter(|&i| role(&self.param_roles[i as usize]))
            .collect()
    }

    /// Indices of the outputs carrying `role`, in output order.
    pub fn outputs_with_role(&self, role: impl Fn(&OutputRole) -> bool) -> Vec<u32> {
        (0..self.outputs.len() as u32)
            .filter(|&i| role(&self.output_roles[i as usize]))
            .collect()
    }

    pub fn is_extern(&self) -> bool {
        matches!(self.body, FunctionBody::Extern(_))
    }

    /// The bundle evaluating this function and the slot of each output:
    /// the extern body itself, or the compiled body a solver registered.
    pub fn evaluator(&self) -> Option<(&Arc<dyn ExternBundle>, Vec<Option<u32>>)> {
        match &self.body {
            FunctionBody::Extern(b) => Some((
                b,
                self.outputs
                    .iter()
                    .map(|o| match o {
                        Output::Slot(k) => Some(*k),
                        _ => None,
                    })
                    .collect(),
            )),
            FunctionBody::Symbolic => self
                .compiled
                .as_ref()
                .map(|c| (&c.bundle, c.slot_of.clone())),
        }
    }
}
