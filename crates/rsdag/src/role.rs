//! Roles: the system layer as a signature.
//!
//! A role is metadata on a parameter or an output of a [`Function`]; nothing
//! in the graph changes. A DAE is a function with `Free`, `State` and `Time`
//! parameters and `Residual` outputs, a block diagram block one with `Input`,
//! `State`, `Time`, `Memory` and `Param` parameters and `Output`,
//! `StateDeriv` and `MemoryWrite` outputs, an event a `Guard` output (with
//! the [`Crossing`] direction that counts) plus an effect function with
//! `StateWrite` outputs. Consumers keep their own
//! integrators and schedulers; the backend guarantees that a function with
//! roles can be evaluated, differentiated with respect to any role subset,
//! specialized and lowered.
//!
//! [`Function`]: crate::func::Function

/// What a parameter of a function stands for.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ParamRole {
    /// A symbolic unknown or a formal leaf without further meaning.
    Free,
    /// An element of an input port.
    Input { port: u32, elem: u32 },
    /// A tunable parameter (bound to a persistent buffer at lowering).
    Param,
    /// A continuous state.
    State { id: u32 },
    /// The time derivative of a continuous state (DAE conventions).
    StateDot { id: u32 },
    /// The independent variable.
    Time,
    /// A discrete memory slot element.
    Memory { slot: u32, offset: u32 },
}

/// The direction of a sign change that counts as a crossing: Verilog-A's
/// `@(cross(expr, dir))` argument, and what a switch declares about its
/// threshold.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Crossing {
    /// Either direction (Verilog-A `0`).
    #[default]
    Either,
    /// Negative to positive (`+1`).
    Rising,
    /// Positive to negative (`-1`).
    Falling,
}

impl Crossing {
    /// Whether a sign change from `before` to `after` crosses in this
    /// direction. Zero counts as the sign it is left with.
    pub fn crosses(self, before: f64, after: f64) -> bool {
        match self {
            Crossing::Either => before.signum() != after.signum(),
            Crossing::Rising => before < 0.0 && after >= 0.0,
            Crossing::Falling => before > 0.0 && after <= 0.0,
        }
    }
}

/// What an output of a function computes.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OutputRole {
    /// A value without further meaning (the default).
    Plain,
    /// An element of an output port.
    Output { port: u32, elem: u32 },
    /// `dx/dt` of a continuous state.
    StateDeriv { id: u32 },
    /// A residual `F(x, x', t) = 0`.
    Residual { id: u32 },
    /// A discrete state assignment (event effects).
    StateWrite { id: u32 },
    /// A memory slot assignment.
    MemoryWrite { slot: u32, offset: u32 },
    /// A guard `g(x, t)` whose sign change in `dir` is an event: the
    /// consumer's integrator lands a step on the crossing and runs the
    /// effect function (the [`StateWrite`](Self::StateWrite) outputs).
    Guard { id: u32, dir: Crossing },
    /// A derivative output `d outputs[of] / d params[wrt]` (memoised by
    /// [`Graph::derivative_output`](crate::graph::Graph::derivative_output)).
    Derivative { of: u32, wrt: u32 },
}
