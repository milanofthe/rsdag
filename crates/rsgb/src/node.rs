use crate::func::OutputId;

/// Index of an interned expression node inside a [`crate::Graph`].
///
/// Cheap to copy and compare; identical subexpressions share one `ExprId`
/// thanks to hash-consing, so structural equality is `O(1)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ExprId(pub u32);

/// Index of an interned exact rational constant in a [`crate::Graph`]'s
/// constant table. Two equal rationals share one id, so a constant node is a
/// 4-byte handle and comparing constants is an integer compare.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ConstId(pub u32);

/// An interned operand list: a `(start, len)` window into the context's shared
/// argument pool. Lists are deduplicated by content, so two structurally equal
/// variadic nodes carry the *same* `ArgList` and hash-cons to one node. Resolve
/// to a slice with [`crate::Graph::args`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ArgList {
    pub start: u32,
    pub len: u32,
}

impl ArgList {
    pub fn len(self) -> usize {
        self.len as usize
    }
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Index of a free symbol (component value, `gm`, the Laplace variable `s`, ...).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

/// Comparison operators; a `Cmp` node evaluates to `1.0` (true) or `0.0`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// Associative reduction over a variadic operand list. Folds a flat list of
/// terms in one node, shrinking the tape (KCL current sums become one `Reduce`
/// instead of an Add-tree) and exposing a vectorizable loop.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
}

impl ReduceOp {
    /// The identity element (value of an empty reduction).
    pub fn identity(self) -> f64 {
        match self {
            ReduceOp::Sum => 0.0,
            ReduceOp::Product => 1.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }
    }

    /// Combine an accumulator with the next element (left fold).
    pub fn combine(self, acc: f64, x: f64) -> f64 {
        match self {
            ReduceOp::Sum => acc + x,
            ReduceOp::Product => acc * x,
            ReduceOp::Min => acc.min(x),
            ReduceOp::Max => acc.max(x),
        }
    }
}

/// Operand lists at least this long reduce with a 4-lane multi-accumulator
/// (breaks the serial dependency chain, so the compiler auto-vectorizes);
/// shorter lists fold sequentially, identical to the plain loop. Sized so the
/// short common case (low-degree KCL) is untouched and only dense nodes pay.
pub const REDUCE_SIMD_MIN: usize = 16;

/// Canonical evaluation of a [`ReduceOp`] over a value slice. The single
/// reference used by both the arena sweep and the tape, so their results agree
/// bit-for-bit; the 4-lane path uses a fixed `(l0+l1)+(l2+l3)` merge.
pub fn reduce_slice(op: ReduceOp, xs: &[f64]) -> f64 {
    match op {
        ReduceOp::Sum if xs.len() >= REDUCE_SIMD_MIN => {
            let mut a = [0.0f64; 4];
            let ch = xs.len() / 4;
            for c in 0..ch {
                a[0] += xs[4 * c];
                a[1] += xs[4 * c + 1];
                a[2] += xs[4 * c + 2];
                a[3] += xs[4 * c + 3];
            }
            let mut acc = (a[0] + a[1]) + (a[2] + a[3]);
            for &x in &xs[ch * 4..] {
                acc += x;
            }
            acc
        }
        ReduceOp::Product if xs.len() >= REDUCE_SIMD_MIN => {
            let mut a = [1.0f64; 4];
            let ch = xs.len() / 4;
            for c in 0..ch {
                a[0] *= xs[4 * c];
                a[1] *= xs[4 * c + 1];
                a[2] *= xs[4 * c + 2];
                a[3] *= xs[4 * c + 3];
            }
            let mut acc = (a[0] * a[1]) * (a[2] * a[3]);
            for &x in &xs[ch * 4..] {
                acc *= x;
            }
            acc
        }
        _ => {
            let mut acc = op.identity();
            for &x in xs {
                acc = op.combine(acc, x);
            }
            acc
        }
    }
}

/// Canonical inner product `Σ a[i]*b[i]` (4-lane for long lists). Shared by the
/// arena sweep and the tape; the lists must be equal length.
pub fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    if a.len() >= REDUCE_SIMD_MIN {
        let mut acc = [0.0f64; 4];
        let ch = a.len() / 4;
        for c in 0..ch {
            acc[0] += a[4 * c] * b[4 * c];
            acc[1] += a[4 * c + 1] * b[4 * c + 1];
            acc[2] += a[4 * c + 2] * b[4 * c + 2];
            acc[3] += a[4 * c + 3] * b[4 * c + 3];
        }
        let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for k in ch * 4..a.len() {
            s += a[k] * b[k];
        }
        s
    } else {
        let mut s = 0.0;
        for (&x, &y) in a.iter().zip(b.iter()) {
            s += x * y;
        }
        s
    }
}

/// Transcendental / elementary unary functions, needed by nonlinear device
/// constitutive equations (diode `exp`, EKV/`tanh`, ...).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum UnaryOp {
    Exp,
    Ln,
    Sqrt,
    Sin,
    Cos,
    Sinh,
    Cosh,
    Tanh,
    Atan,
    Floor,
}

/// Argument above which `exp` is linearly extrapolated (a "limited exponential",
/// SPICE `limexp`). `exp` overflows f64 near 709; compact models routinely drive
/// the argument far past that on intermediate Newton iterates (out-of-range
/// internal-node guesses), producing `inf`/`NaN` that poison the whole solve.
/// Linearizing above this threshold keeps the value and its (constant) derivative
/// finite with ample headroom against overflow, while leaving every physical
/// evaluation -- exp arguments are O(10) at any real operating point -- bit-for-bit
/// unchanged. 80 is the classic SPICE `limexp` knee. See [`unary_f64`].
pub const EXP_LIMIT: f64 = 80.0;

/// Argument at or below which `ln` is clamped (so `ln(x<=0)` returns a finite,
/// modest `ln(LN_FLOOR)` instead of `-inf`/`NaN` on an out-of-range Newton
/// iterate). A *clamp*, not a linear extrapolation: a steep extrapolation slope
/// (`1/LN_FLOOR`) would blow intermediate values up to `~1e30` and wreck
/// conditioning. In-range arguments are untouched. See [`unary_f64`].
pub const LN_FLOOR: f64 = 1e-30;

/// Evaluate a unary op on a real argument. Single source of truth shared by the
/// arena evaluator ([`crate::eval`]) and the compiled tape ([`crate::tape`]).
///
/// `exp`/`ln`/`sqrt` are domain-guarded (limited exponential, log floor, and
/// `sqrt` of a negative argument clamped to 0): on an out-of-range intermediate
/// Newton iterate these return a finite, smoothly-extrapolated value instead of
/// `inf`/`NaN`, so a single out-of-range internal node cannot poison the residual
/// and Jacobian. Every in-range argument evaluates identically to the bare op.
/// The derivative rules in [`crate::autodiff`] mirror these guards exactly.
pub fn unary_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Exp => {
            if x > EXP_LIMIT {
                EXP_LIMIT.exp() * (1.0 + (x - EXP_LIMIT))
            } else {
                x.exp()
            }
        }
        UnaryOp::Ln => {
            if x > LN_FLOOR {
                x.ln()
            } else {
                LN_FLOOR.ln()
            }
        }
        UnaryOp::Sqrt => {
            if x > 0.0 {
                x.sqrt()
            } else {
                0.0
            }
        }
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Floor => x.floor(),
    }
}

/// Evaluate a [`CmpOp`] on two ordered arguments. Single source of truth shared
/// by the arena evaluator, the complex evaluator, the compiled tape (all `f64`)
/// and the constant-folding interner (exact `BigRational`), so a `Cmp` node's
/// `1.0`/`0.0` result agrees across every backend.
pub fn cmp_bool<T: PartialOrd>(op: CmpOp, x: T, y: T) -> bool {
    match op {
        CmpOp::Gt => x > y,
        CmpOp::Ge => x >= y,
        CmpOp::Lt => x < y,
        CmpOp::Le => x <= y,
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
    }
}

/// A node in the symbolic DAG.
///
/// Leaves are exact rational constants or free symbols. Inner nodes are the
/// algebraic operations that show up when stamping and solving circuit
/// equations, plus the elementary functions used by nonlinear device models.
///
/// A node is a 16-byte `Copy` value: every payload is a small integer handle
/// (an `ExprId`, a `ConstId` into the constant table, an `ArgList` into the
/// argument pool). Interning therefore hashes and stores 16 bytes, never a
/// heap allocation, and the arena is one dense array the caches like -- the
/// build passes (differentiation, substitution, tape compilation) are bound by
/// exactly this per-node cost.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Node {
    /// Exact rational constant, by id into the context's constant table (kept
    /// reduced, sign in the numerator). Resolve with [`crate::Graph::const_val`].
    Const(ConstId),
    /// Free symbol, referenced by id into the context symbol table.
    Symbol(SymbolId),
    /// Binary addition. Operands ordered by id to maximise sharing.
    Add(ExprId, ExprId),
    /// Binary multiplication. Operands ordered by id to maximise sharing.
    Mul(ExprId, ExprId),
    /// Unary negation.
    Neg(ExprId),
    /// Integer power (covers reciprocals via negative exponents).
    Pow(ExprId, i64),
    /// Elementary unary function.
    Unary(UnaryOp, ExprId),
    /// Comparison, yielding `1.0`/`0.0`. Used to build region conditions.
    Cmp(CmpOp, ExprId, ExprId),
    /// Conditional: `cond != 0 ? then : else_`. For region-based device models.
    Select(ExprId, ExprId, ExprId),
    /// Associative reduction over a flat operand list (KCL sums, products).
    /// One fused node instead of a balanced/again-binary tree.
    Reduce(ReduceOp, ArgList),
    /// Inner product `Σ_i a[i]*b[i]` over two equal-length operand lists,
    /// stored as ONE list `[a_0..a_n, b_0..b_n]` (resolve the halves with
    /// [`crate::Graph::dot_args`]). The fused form of a sum of pairwise
    /// products (matrix-vector rows).
    Dot(ArgList),
    /// A call: output `OutputId` of a function (see [`crate::func`]) applied
    /// to the argument list. One compact model instantiated many times is one
    /// function and many calls; differentiation references the function's
    /// derivative outputs by the chain rule.
    Call(OutputId, ArgList),
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

/// The operands of a node, borrowed without allocation: inline for the
/// fixed-arity variants, a pool slice for the variadic ones. Derefs to
/// `&[ExprId]`.
pub enum Operands<'a> {
    Inline { buf: [ExprId; 3], n: u8 },
    Slice(&'a [ExprId]),
}

impl std::ops::Deref for Operands<'_> {
    type Target = [ExprId];
    fn deref(&self) -> &[ExprId] {
        match self {
            Operands::Inline { buf, n } => &buf[..*n as usize],
            Operands::Slice(s) => s,
        }
    }
}
