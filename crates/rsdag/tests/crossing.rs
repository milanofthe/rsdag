//! The sign test of a guard, which decides when an integrator lands a step.

use rsdag::Crossing;

#[test]
fn the_direction_selects_the_sign_change() {
    assert!(Crossing::Rising.crosses(-1.0, 1.0));
    assert!(!Crossing::Rising.crosses(1.0, -1.0));
    assert!(Crossing::Falling.crosses(1.0, -1.0));
    assert!(!Crossing::Falling.crosses(-1.0, 1.0));
    assert!(Crossing::Either.crosses(-1.0, 1.0));
    assert!(Crossing::Either.crosses(1.0, -1.0));
}

#[test]
fn touching_the_surface_counts_from_either_side() {
    // A `Select` on `g > 0` flips between these two, so both are events; one
    // on `g >= 0` flips between the other two. The test covers all four
    // rather than miss a flip inside a step.
    assert!(Crossing::Rising.crosses(-1.0, 0.0));
    assert!(Crossing::Rising.crosses(0.0, 1.0));
    assert!(Crossing::Falling.crosses(1.0, 0.0));
    assert!(Crossing::Falling.crosses(0.0, -1.0));
}

#[test]
fn a_step_that_moves_nowhere_is_not_a_crossing() {
    for dir in [Crossing::Either, Crossing::Rising, Crossing::Falling] {
        assert!(!dir.crosses(0.0, 0.0), "{dir:?} fired on a flat zero");
        assert!(!dir.crosses(1.0, 1.0), "{dir:?} fired without a change");
    }
    // Moving within one side is never a crossing either.
    assert!(!Crossing::Either.crosses(-2.0, -1.0));
    assert!(!Crossing::Either.crosses(2.0, 1.0));
}
