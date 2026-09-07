//! Shared generator of the extended op set for the parity fuzzers: every
//! extension op in a domain-safe form (arguments bounded or shifted so the
//! value is finite), the smooth ones first so an AD-vs-FD check can draw
//! from the prefix alone.

#![allow(dead_code)]

use rsgb::{BinOp, ExprId, Field, Graph, UnaryOp};

/// Number of smooth extension forms (the prefix of `ext_op`).
pub const EXT_SMOOTH: usize = 16;
/// Number of non-smooth extension forms (the suffix).
pub const EXT_ROUGH: usize = 7;

/// Build extension op `k` over `a` and `b` (`k < EXT_SMOOTH + EXT_ROUGH`).
pub fn ext_op<K: Field>(g: &mut Graph<K>, k: usize, a: ExprId, b: ExprId) -> ExprId {
    let one = g.one();
    let a2 = g.mul(a, a);
    let pos = g.add(a2, one); // a^2 + 1 >= 1
    let half = g.ratio(1, 2);
    let s = g.sin(a);
    let bounded = g.mul(half, s); // |.| <= 1/2
    match k {
        0 => {
            let t = g.tanh(a);
            g.unary(UnaryOp::Tan, t)
        }
        1 => g.unary(UnaryOp::Asinh, a),
        2 => g.unary(UnaryOp::Expm1, a),
        3 => g.unary(UnaryOp::Log1p, a2),
        4 => g.unary(UnaryOp::Erf, a),
        5 => g.unary(UnaryOp::Erfc, a),
        6 => g.binary(BinOp::Atan2, a, pos),
        7 => g.binary(BinOp::Hypot, pos, b),
        8 => {
            let t = g.tanh(b);
            g.binary(BinOp::Powf, pos, t)
        }
        9 => g.unary(UnaryOp::Cbrt, pos),
        10 => g.unary(UnaryOp::Lgamma, pos),
        11 => g.unary(UnaryOp::Digamma, pos),
        12 => g.unary(UnaryOp::Asin, bounded),
        13 => g.unary(UnaryOp::Atanh, bounded),
        14 => g.unary(UnaryOp::Log10, pos),
        15 => {
            let two = g.konst_int(2);
            let p2 = g.add(a2, two);
            g.unary(UnaryOp::Acosh, p2)
        }
        16 => g.unary(UnaryOp::Abs, a),
        17 => g.unary(UnaryOp::Sign, a),
        18 => g.unary(UnaryOp::Ceil, a),
        19 => g.unary(UnaryOp::Round, a),
        20 => g.unary(UnaryOp::Trunc, a),
        21 => {
            let m = g.ratio(3, 2);
            g.binary(BinOp::Mod, a, m)
        }
        _ => g.unary(UnaryOp::RandUniform, a),
    }
}

/// Draw an op index for the generators: the classic arms `0..classic`
/// (`classic_smooth` of them smooth) plus the extension forms, restricted to
/// the smooth ones when `smooth`. Returns `(index, is_extension)` where the
/// extension index is relative to `ext_op`.
pub fn draw_op(
    below: &mut dyn FnMut(usize) -> usize,
    classic: usize,
    classic_smooth: usize,
    smooth: bool,
    ext: bool,
) -> (usize, bool) {
    if !ext {
        return (below(if smooth { classic_smooth } else { classic }), false);
    }
    if smooth {
        let i = below(classic_smooth + EXT_SMOOTH);
        if i < classic_smooth {
            (i, false)
        } else {
            (i - classic_smooth, true)
        }
    } else {
        let i = below(classic + EXT_SMOOTH + EXT_ROUGH);
        if i < classic {
            (i, false)
        } else {
            (i - classic, true)
        }
    }
}
