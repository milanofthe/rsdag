use rsdag::node::{BINARY_OPS, UNARY_OPS};
use rsdag::*;

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
