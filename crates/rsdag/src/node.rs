use crate::func::OutputId;

/// Index of an interned expression node inside a [`crate::Graph`].
///
/// Cheap to copy and compare; identical subexpressions share one `ExprId`
/// thanks to hash-consing, so structural equality is `O(1)`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ExprId(pub u32);

/// Index of an interned exact rational constant in a [`crate::Graph`]'s
/// constant table. Two equal rationals share one id, so a constant node is a
/// 4-byte handle and comparing constants is an integer compare.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ConstId(pub u32);

/// An interned operand list: a `(start, len)` window into the context's shared
/// argument pool. Lists are deduplicated by content, so two structurally equal
/// variadic nodes carry the *same* `ArgList` and hash-cons to one node. Resolve
/// to a slice with [`crate::Graph::args`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

/// Comparison operators; a `Cmp` node evaluates to `1.0` (true) or `0.0`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    // The floating-point extension (reference implementations from `libm`,
    // so the bits do not depend on the platform's C library).
    Tan,
    Log10,
    Log2,
    Log1p,
    Expm1,
    Cbrt,
    Abs,
    /// `-1`, `0`, `1` (`+-0.0` and NaN pass through, numpy's `sign`).
    Sign,
    Ceil,
    Round,
    Trunc,
    Asin,
    Acos,
    Asinh,
    Acosh,
    Atanh,
    Erf,
    Erfc,
    Lgamma,
    Tgamma,
    Digamma,
    Trigamma,
    /// Counter-based uniform noise in `[0, 1)` keyed by the argument's bits:
    /// a pure function, so it traces, replays and batches like any other op.
    RandUniform,
}

/// Binary operations beyond the ring (`Add`, `Mul`, `Neg`, `Pow` are their
/// own node kinds for the canonical ordering and the reduction fusion; `Sub`
/// and `Div` are `add(neg)` and `mul(recip)`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum BinOp {
    /// `a^b` for real exponents (`Pow` covers integer exponents).
    Powf,
    /// `fmod(a, b)`: the remainder with the sign of `a`.
    Mod,
    /// `atan2(a, b)`.
    Atan2,
    /// `sqrt(a^2 + b^2)` without overflow.
    Hypot,
}

/// Canonical evaluation of a [`BinOp`], the one reference for every backend.
pub fn binary_f64(op: BinOp, x: f64, y: f64) -> f64 {
    match op {
        BinOp::Powf => libm::pow(x, y),
        BinOp::Mod => libm::fmod(x, y),
        BinOp::Atan2 => libm::atan2(x, y),
        BinOp::Hypot => libm::hypot(x, y),
    }
}

/// Digamma `psi(x)`: recurrence into the asymptotic zone, then the series.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + libm::log(x)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)))
}

/// Trigamma `psi'(x)`: recurrence into the asymptotic zone, then the series.
pub fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result
        + inv
        + 0.5 * inv2
        + inv * inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)))
}

/// Counter-based uniform noise in `[0, 1)` from the bits of `key`
/// (splitmix64 finalizer; the top 53 bits become the mantissa).
pub fn rand_uniform(key: f64) -> f64 {
    let mut z = key.to_bits().wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
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
        UnaryOp::Tan => libm::tan(x),
        UnaryOp::Log10 => libm::log10(x),
        UnaryOp::Log2 => libm::log2(x),
        UnaryOp::Log1p => libm::log1p(x),
        UnaryOp::Expm1 => libm::expm1(x),
        UnaryOp::Cbrt => libm::cbrt(x),
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                x
            }
        }
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => libm::round(x),
        UnaryOp::Trunc => x.trunc(),
        UnaryOp::Asin => libm::asin(x),
        UnaryOp::Acos => libm::acos(x),
        UnaryOp::Asinh => libm::asinh(x),
        UnaryOp::Acosh => libm::acosh(x),
        UnaryOp::Atanh => libm::atanh(x),
        UnaryOp::Erf => libm::erf(x),
        UnaryOp::Erfc => libm::erfc(x),
        UnaryOp::Lgamma => libm::lgamma(x),
        UnaryOp::Tgamma => libm::tgamma(x),
        UnaryOp::Digamma => digamma(x),
        UnaryOp::Trigamma => trigamma(x),
        UnaryOp::RandUniform => rand_uniform(x),
    }
}

/// What a unary op *is*, as data rather than code: how it prints, what it is
/// called in generated C, and whether it is differentiable everywhere it is
/// defined.
///
/// The semantics live in [`unary_f64`] (and in [`crate::Scalar`] for the
/// other execution types), because a guard like the `exp` cap is code. What
/// is data lives here, so a new op is one row plus its semantics and its
/// lowerings, instead of an edit in the printer, the C backend, the JIT's
/// code mapping and the generator.
///
/// The aggregate ops (`Reduce`, `Dot`) are deliberately not in this table:
/// they carry an operand list rather than a fixed arity, they are how arrays
/// and matrices lower natively into the graph, and every backend emits them
/// as a loop rather than a call.
#[derive(Clone, Copy, Debug)]
pub struct UnarySpec {
    /// The variant this row describes; `UNARY_OPS[op as usize].op == op`.
    pub op: UnaryOp,
    /// Name in printed expressions and in the Python and C frontends.
    pub name: &'static str,
    /// The callee in generated C: a `libm` name where the semantics agree,
    /// an `rsdag_` helper where rsdag guards or defines the op itself.
    pub c_fn: &'static str,
    /// Differentiable everywhere it is defined. The rough ones (`floor`,
    /// `sign`, the roundings, the noise source) have a zero or undefined
    /// derivative and are excluded from smooth generated programs.
    pub smooth: bool,
}

/// The unary vocabulary, in the order of the enum: `UNARY_OPS[op as usize]`
/// is `op`'s row (checked by a test).
pub const UNARY_OPS: &[UnarySpec] = &[
    UnarySpec {
        op: UnaryOp::Exp,
        name: "exp",
        c_fn: "rsdag_exp",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Ln,
        name: "ln",
        c_fn: "rsdag_ln",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sqrt,
        name: "sqrt",
        c_fn: "rsdag_sqrt",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sin,
        name: "sin",
        c_fn: "sin",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cos,
        name: "cos",
        c_fn: "cos",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Sinh,
        name: "sinh",
        c_fn: "sinh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cosh,
        name: "cosh",
        c_fn: "cosh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Tanh,
        name: "tanh",
        c_fn: "tanh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Atan,
        name: "atan",
        c_fn: "atan",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Floor,
        name: "floor",
        c_fn: "floor",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Tan,
        name: "tan",
        c_fn: "tan",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log10,
        name: "log10",
        c_fn: "log10",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log2,
        name: "log2",
        c_fn: "log2",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Log1p,
        name: "log1p",
        c_fn: "log1p",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Expm1,
        name: "expm1",
        c_fn: "expm1",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Cbrt,
        name: "cbrt",
        c_fn: "cbrt",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Abs,
        name: "abs",
        c_fn: "fabs",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Sign,
        name: "sign",
        c_fn: "rsdag_sign",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Ceil,
        name: "ceil",
        c_fn: "ceil",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Round,
        name: "round",
        c_fn: "round",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Trunc,
        name: "trunc",
        c_fn: "trunc",
        smooth: false,
    },
    UnarySpec {
        op: UnaryOp::Asin,
        name: "asin",
        c_fn: "asin",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Acos,
        name: "acos",
        c_fn: "acos",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Asinh,
        name: "asinh",
        c_fn: "asinh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Acosh,
        name: "acosh",
        c_fn: "acosh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Atanh,
        name: "atanh",
        c_fn: "atanh",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Erf,
        name: "erf",
        c_fn: "erf",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Erfc,
        name: "erfc",
        c_fn: "erfc",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Lgamma,
        name: "lgamma",
        c_fn: "lgamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Tgamma,
        name: "tgamma",
        c_fn: "tgamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Digamma,
        name: "digamma",
        c_fn: "rsdag_digamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::Trigamma,
        name: "trigamma",
        c_fn: "rsdag_trigamma",
        smooth: true,
    },
    UnarySpec {
        op: UnaryOp::RandUniform,
        name: "rand_uniform",
        c_fn: "rsdag_rand_uniform",
        smooth: false,
    },
];

impl UnaryOp {
    /// This op's row of [`UNARY_OPS`].
    #[inline]
    pub fn spec(self) -> &'static UnarySpec {
        &UNARY_OPS[self as usize]
    }
    /// Name in printed expressions.
    #[inline]
    pub fn name(self) -> &'static str {
        self.spec().name
    }
    /// The callee to emit in generated C.
    #[inline]
    pub fn c_fn(self) -> &'static str {
        self.spec().c_fn
    }
    /// Differentiable everywhere it is defined.
    #[inline]
    pub fn is_smooth(self) -> bool {
        self.spec().smooth
    }
    /// A stable integer code, for backends that pass the op to a host
    /// routine as a value (the JIT's trampolines).
    #[inline]
    pub fn code(self) -> u32 {
        self as u32
    }
    /// The op for a [`UnaryOp::code`]; panics on an unknown code.
    #[inline]
    pub fn from_code(code: u32) -> UnaryOp {
        UNARY_OPS[code as usize].op
    }
}

/// As [`UnarySpec`], for the binary ops beyond the ring.
#[derive(Clone, Copy, Debug)]
pub struct BinarySpec {
    pub op: BinOp,
    pub name: &'static str,
    pub c_fn: &'static str,
    pub smooth: bool,
}

/// The binary vocabulary, in the order of the enum.
pub const BINARY_OPS: &[BinarySpec] = &[
    BinarySpec {
        op: BinOp::Powf,
        name: "powf",
        c_fn: "pow",
        smooth: true,
    },
    BinarySpec {
        op: BinOp::Mod,
        name: "mod",
        c_fn: "fmod",
        smooth: false,
    },
    BinarySpec {
        op: BinOp::Atan2,
        name: "atan2",
        c_fn: "atan2",
        smooth: true,
    },
    BinarySpec {
        op: BinOp::Hypot,
        name: "hypot",
        c_fn: "hypot",
        smooth: true,
    },
];

impl BinOp {
    #[inline]
    pub fn spec(self) -> &'static BinarySpec {
        &BINARY_OPS[self as usize]
    }
    #[inline]
    pub fn name(self) -> &'static str {
        self.spec().name
    }
    #[inline]
    pub fn c_fn(self) -> &'static str {
        self.spec().c_fn
    }
    #[inline]
    pub fn is_smooth(self) -> bool {
        self.spec().smooth
    }
    #[inline]
    pub fn code(self) -> u32 {
        self as u32
    }
    #[inline]
    pub fn from_code(code: u32) -> BinOp {
        BINARY_OPS[code as usize].op
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Binary function beyond the ring (`Powf`, `Mod`, `Atan2`, `Hypot`).
    Binary(BinOp, ExprId, ExprId),
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

#[cfg(test)]
mod op_table_tests {
    use super::*;

    /// The tables are indexed by the enum's discriminant, and `unary_f64`'s
    /// match is exhaustive, so a new variant without a row makes this fail
    /// rather than silently reading the wrong row.
    #[test]
    fn tables_line_up_with_the_enums() {
        for (i, spec) in UNARY_OPS.iter().enumerate() {
            assert_eq!(spec.op as usize, i, "row {i} describes {:?}", spec.op);
            assert_eq!(spec.op.spec().name, spec.name);
            assert_eq!(UnaryOp::from_code(spec.op.code()), spec.op);
            assert!(!spec.name.is_empty() && !spec.c_fn.is_empty());
        }
        for (i, spec) in BINARY_OPS.iter().enumerate() {
            assert_eq!(spec.op as usize, i);
            assert_eq!(BinOp::from_code(spec.op.code()), spec.op);
        }
        // Every variant has a row: the count is asserted against the last
        // variant's discriminant, which only holds if the table is complete.
        assert_eq!(UNARY_OPS.len(), UnaryOp::RandUniform as usize + 1);
        assert_eq!(BINARY_OPS.len(), BinOp::Hypot as usize + 1);
    }

    /// Names are what the printer emits and what a frontend parses back, so
    /// two ops must not share one.
    #[test]
    fn names_are_unique() {
        let mut names: Vec<&str> = UNARY_OPS.iter().map(|s| s.name).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n);
    }

    /// `smooth` is the property the generator and the derivative checks rely
    /// on: a rough op must not claim to be differentiable.
    #[test]
    fn roughness_is_recorded() {
        for op in [
            UnaryOp::Floor,
            UnaryOp::Ceil,
            UnaryOp::Round,
            UnaryOp::Trunc,
            UnaryOp::Sign,
            UnaryOp::Abs,
            UnaryOp::RandUniform,
        ] {
            assert!(!op.is_smooth(), "{op:?} is not differentiable");
        }
        for op in [UnaryOp::Exp, UnaryOp::Sin, UnaryOp::Erf, UnaryOp::Lgamma] {
            assert!(op.is_smooth(), "{op:?} is differentiable");
        }
    }
}
