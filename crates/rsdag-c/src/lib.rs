//! C source backend for rsdag programs.
//!
//! [`emit`] lowers a [`Tape`] to one C function
//! `void <name>(const double* in, double* work, double* out)` over the same
//! slot model the interpreter uses (`work` has [`Tape::n_slots`] entries),
//! preceded by the helper routines the ops need: the domain guards of the
//! reference math (`exp` limit, `ln` floor, `sqrt` clamp), `powi` as the
//! square-and-multiply of the Rust reference, the four-accumulator order of
//! the long reductions, and the special functions (`sign`, `digamma`,
//! `trigamma`, `rand_uniform`) as the same code. Everything in the ring and
//! those helpers is bit-identical to the interpreter; the transcendental
//! functions call the platform's `libm`, which may differ from the Rust
//! reference in the last bit. Compile the source with `-ffp-contract=off`:
//! a contracted multiply-add rounds once where the reference rounds twice. [`verify`] compiles the emitted source with the
//! host C compiler and compares it against the interpreter, so a consumer
//! can pin its own tolerance.
//!
//! Tapes with bundle calls (compiled function bodies) cannot be emitted:
//! their bodies are Rust closures.

use std::fmt::Write;
use std::sync::Arc;

use rsdag::node::REDUCE_SIMD_MIN;
use rsdag::{BinOp, CmpOp, ExternBundle, ReduceOp, Tape, TapeVisitor, UnaryOp};

pub mod verify;

/// Why a tape could not be emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CError {
    /// The tape calls a compiled bundle body.
    Bundle,
}

impl std::fmt::Display for CError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CError::Bundle => write!(f, "tapes with bundle calls have no C form"),
        }
    }
}
impl std::error::Error for CError {}

/// The helper routines every emitted function may call.
pub const PRELUDE: &str = r#"#include <math.h>
#include <stdint.h>
#include <string.h>

/* No fused multiply-add: the reference rounds a product and a sum
   separately (see the bit-exactness contract). Build with -ffp-contract=off
   as well; the pragma covers compilers that honour it. */
#pragma STDC FP_CONTRACT OFF

#define RSDAG_EXP_LIMIT 80.0
#define RSDAG_LN_FLOOR 1e-30

static double rsdag_exp(double x) {
    if (x > RSDAG_EXP_LIMIT) return exp(RSDAG_EXP_LIMIT) * (1.0 + (x - RSDAG_EXP_LIMIT));
    return exp(x);
}
static double rsdag_ln(double x) {
    if (x > RSDAG_LN_FLOOR) return log(x);
    return log(RSDAG_LN_FLOOR);
}
static double rsdag_sqrt(double x) { return x > 0.0 ? sqrt(x) : 0.0; }
static double rsdag_sign(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : x); }
static double rsdag_powi(double a, int32_t b) {
    /* compiler-rt __powidf2: the square-and-multiply of the Rust reference */
    const int recip = b < 0;
    double r = 1.0;
    while (1) {
        if (b & 1) r *= a;
        b /= 2;
        if (b == 0) break;
        a *= a;
    }
    return recip ? 1.0 / r : r;
}
static double rsdag_digamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0 / x; x += 1.0; }
    double inv = 1.0 / x;
    double inv2 = inv * inv;
    return result + log(x) - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)));
}
static double rsdag_trigamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result += 1.0 / (x * x); x += 1.0; }
    double inv = 1.0 / x;
    double inv2 = inv * inv;
    return result + inv + 0.5 * inv2
        + inv * inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)));
}
static double rsdag_rand_uniform(double key) {
    uint64_t z;
    memcpy(&z, &key, sizeof z);
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
"#;

/// Emit `tape` as the C function `name` (helpers included once at the top).
pub fn emit(tape: &Tape, name: &str) -> Result<String, CError> {
    let mut v = Emitter {
        body: String::new(),
        error: None,
    };
    tape.lower(&mut v);
    if let Some(e) = v.error {
        return Err(e);
    }
    let mut src = String::new();
    src.push_str(PRELUDE);
    let _ = writeln!(src);
    let _ = writeln!(
        src,
        "/* {} work slots, {} outputs */",
        tape.n_slots(),
        tape.n_outputs()
    );
    let _ = writeln!(
        src,
        "void {name}(const double* in, double* work, double* out) {{"
    );
    src.push_str(&v.body);
    for (k, &o) in tape.outputs().iter().enumerate() {
        let _ = writeln!(src, "    out[{k}] = work[{o}];");
    }
    src.push_str("}\n");
    Ok(src)
}

struct Emitter {
    body: String,
    error: Option<CError>,
}

/// A C literal of an `f64` that round-trips exactly.
pub fn lit_pub(v: f64) -> String {
    lit(v)
}

fn lit(v: f64) -> String {
    if v.is_nan() {
        "NAN".to_string()
    } else if v == f64::INFINITY {
        "INFINITY".to_string()
    } else if v == f64::NEG_INFINITY {
        "(-INFINITY)".to_string()
    } else {
        // Exact round trip: 17 significant digits.
        format!("{v:.17e}")
    }
}

fn unary_c(op: UnaryOp, x: &str) -> String {
    format!("{}({x})", op.c_fn())
}

impl Emitter {
    fn set(&mut self, dst: u32, expr: &str) {
        let _ = writeln!(self.body, "    work[{dst}] = {expr};");
    }
}

impl TapeVisitor for Emitter {
    fn constant(&mut self, dst: u32, v: f64) {
        self.set(dst, &lit(v));
    }
    fn input(&mut self, dst: u32, k: u32) {
        self.set(dst, &format!("in[{k}]"));
    }
    fn add(&mut self, dst: u32, a: u32, b: u32) {
        self.set(dst, &format!("work[{a}] + work[{b}]"));
    }
    fn mul(&mut self, dst: u32, a: u32, b: u32) {
        self.set(dst, &format!("work[{a}] * work[{b}]"));
    }
    fn mul_add(&mut self, dst: u32, a: u32, b: u32, c: u32) {
        // Two roundings, as the interpreter: a separate product first.
        let _ = writeln!(
            self.body,
            "    {{ const double p = work[{a}] * work[{b}]; work[{dst}] = p + work[{c}]; }}"
        );
    }
    fn sub(&mut self, dst: u32, a: u32, b: u32) {
        self.set(dst, &format!("work[{a}] - work[{b}]"));
    }
    fn neg(&mut self, dst: u32, a: u32) {
        self.set(dst, &format!("-work[{a}]"));
    }
    fn powi(&mut self, dst: u32, a: u32, n: i32) {
        self.set(dst, &format!("rsdag_powi(work[{a}], {n})"));
    }
    fn unary(&mut self, dst: u32, op: UnaryOp, a: u32) {
        let x = format!("work[{a}]");
        let e = unary_c(op, &x);
        self.set(dst, &e);
    }
    fn binary(&mut self, dst: u32, op: BinOp, a: u32, b: u32) {
        let e = format!("{}(work[{a}], work[{b}])", op.c_fn());
        self.set(dst, &e);
    }
    fn cmp(&mut self, dst: u32, op: CmpOp, a: u32, b: u32) {
        let c = match op {
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
        };
        self.set(dst, &format!("(work[{a}] {c} work[{b}]) ? 1.0 : 0.0"));
    }
    fn select(&mut self, dst: u32, c: u32, t: u32, e: u32) {
        self.set(dst, &format!("(work[{c}] != 0.0) ? work[{t}] : work[{e}]"));
    }
    fn reduce(&mut self, dst: u32, op: ReduceOp, args: &[u32]) {
        let long = args.len() >= REDUCE_SIMD_MIN && matches!(op, ReduceOp::Sum | ReduceOp::Product);
        if long {
            let (id, sym) = match op {
                ReduceOp::Sum => ("0.0", "+"),
                _ => ("1.0", "*"),
            };
            let ch = args.len() / 4;
            let _ = writeln!(
                self.body,
                "    {{ double a0 = {id}, a1 = {id}, a2 = {id}, a3 = {id};"
            );
            for c in 0..ch {
                for l in 0..4 {
                    let _ = writeln!(
                        self.body,
                        "      a{l} = a{l} {sym} work[{}];",
                        args[4 * c + l]
                    );
                }
            }
            let _ = writeln!(
                self.body,
                "      double acc = (a0 {sym} a1) {sym} (a2 {sym} a3);"
            );
            for &x in &args[ch * 4..] {
                let _ = writeln!(self.body, "      acc = acc {sym} work[{x}];");
            }
            let _ = writeln!(self.body, "      work[{dst}] = acc; }}");
            return;
        }
        let (start, step): (String, Box<dyn Fn(&str, u32) -> String>) = match op {
            ReduceOp::Sum => (
                "0.0".into(),
                Box::new(|acc, x| format!("{acc} + work[{x}]")),
            ),
            ReduceOp::Product => (
                "1.0".into(),
                Box::new(|acc, x| format!("{acc} * work[{x}]")),
            ),
            ReduceOp::Min => (
                "INFINITY".into(),
                Box::new(|acc, x| format!("fmin({acc}, work[{x}])")),
            ),
            ReduceOp::Max => (
                "(-INFINITY)".into(),
                Box::new(|acc, x| format!("fmax({acc}, work[{x}])")),
            ),
        };
        // Sequential left fold: the interpreter's `combine` order. `fmin`
        // and `fmax` ignore a NaN operand as the reference `min`/`max` do.
        let _ = writeln!(self.body, "    {{ double acc = {start};");
        for &x in args {
            let _ = writeln!(self.body, "      acc = {};", step("acc", x));
        }
        let _ = writeln!(self.body, "      work[{dst}] = acc; }}");
    }
    fn dot(&mut self, dst: u32, a: &[u32], b: &[u32]) {
        if a.len() >= REDUCE_SIMD_MIN {
            let ch = a.len() / 4;
            let _ = writeln!(
                self.body,
                "    {{ double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;"
            );
            for c in 0..ch {
                for l in 0..4 {
                    let _ = writeln!(
                        self.body,
                        "      a{l} = a{l} + work[{}] * work[{}];",
                        a[4 * c + l],
                        b[4 * c + l]
                    );
                }
            }
            let _ = writeln!(self.body, "      double acc = (a0 + a1) + (a2 + a3);");
            for k in ch * 4..a.len() {
                let _ = writeln!(
                    self.body,
                    "      acc = acc + work[{}] * work[{}];",
                    a[k], b[k]
                );
            }
            let _ = writeln!(self.body, "      work[{dst}] = acc; }}");
            return;
        }
        let _ = writeln!(self.body, "    {{ double acc = 0.0;");
        for k in 0..a.len() {
            let _ = writeln!(
                self.body,
                "      acc = acc + work[{}] * work[{}];",
                a[k], b[k]
            );
        }
        let _ = writeln!(self.body, "      work[{dst}] = acc; }}");
    }
    fn bundle_call(&mut self, _b: &Arc<dyn ExternBundle>, _args: &[u32], _scratch_base: u32) {
        self.error = Some(CError::Bundle);
    }
    fn bundle_batch(
        &mut self,
        _b: &Arc<dyn ExternBundle>,
        _args: &[u32],
        _n_groups: u32,
        _n_args: u32,
        _base0: u32,
    ) {
        self.error = Some(CError::Bundle);
    }
    fn bundle_pick(&mut self, _dst: u32, _idx: u32) {
        self.error = Some(CError::Bundle);
    }
}
