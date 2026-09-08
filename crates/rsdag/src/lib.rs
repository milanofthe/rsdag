//! Rust symbolic graph backend: a hash-consed expression DAG over free
//! symbols and exact or floating constants, with symbolic differentiation, a
//! flat tape and its interpreter, and a symbolic layer. The shared substrate
//! of SANE and fastsim, so an optimization here lands in every consumer.

pub mod autodiff;
pub mod display;
pub mod eval;
pub mod extern_fn;
pub mod field;
pub mod func;
pub mod graph;
pub mod mathfn;
pub mod module;
pub mod node;
pub mod role;
pub mod scalar;
pub mod scope;
pub mod simplify;
pub mod symbolic;
#[cfg(any(test, feature = "synth"))]
pub mod synth;
pub mod tape;
// Expression substitution is exposed only through the curated `substitute*`
// re-exports below, not as a module path.
pub(crate) mod transform;

pub use autodiff::{differentiate, gradient, hessian, jacobian, sparsity, time_derivative};
pub use display::to_string;
pub use eval::{eval, eval_named, eval_real, eval_real_all};
pub use extern_fn::ExternBundle;
pub use field::{ratio_powi, Field, F64};
pub use func::{CompiledBody, FuncId, Function, FunctionBody, Output, OutputId};
pub use graph::Graph;
pub use mathfn::lower_call;
pub use module::{IdMap, Module, MODULE_VERSION};
pub use node::{
    binary_f64, unary_f64, ArgList, BinOp, CmpOp, ConstId, ExprId, Node, Operands, ReduceOp,
    SymbolId, UnaryOp,
};
pub use role::{OutputRole, ParamRole};
pub use scalar::Scalar;
pub use scope::Scope;
pub use simplify::rebuild;
pub use symbolic::{collect, determinant, rational_form, simplify_egraph};
pub use tape::{SchedulePolicy, SpecializedTape, Tape, TapeVisitor};
/// The execution form of a function (see the design: `Program<T>` is the
/// tape evaluated in a [`Scalar`] `T`; the storage is `f64`, the typed
/// evaluators convert once).
pub type Program = Tape;
pub use transform::{substitute, substitute_expr, substitute_many, substitute_many_all};
