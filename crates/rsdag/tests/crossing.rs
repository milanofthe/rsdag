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
    assert!(
        Crossing::Rising.crosses(-1.0, 0.0),
        "a Select on g >= 0 flips here"
    );
    assert!(
        Crossing::Rising.crosses(0.0, 1.0),
        "a Select on g > 0 flips here"
    );
    assert!(
        Crossing::Falling.crosses(1.0, 0.0),
        "a Select on g > 0 flips here"
    );
    assert!(
        Crossing::Falling.crosses(0.0, -1.0),
        "a Select on g >= 0 flips here"
    );
}

#[test]
fn a_step_that_moves_nowhere_is_not_a_crossing() {
    for dir in [Crossing::Either, Crossing::Rising, Crossing::Falling] {
        assert!(!dir.crosses(0.0, 0.0), "{dir:?} fired on a flat zero");
        assert!(!dir.crosses(1.0, 1.0), "{dir:?} fired without a change");
    }
    assert!(
        !Crossing::Either.crosses(-2.0, -1.0),
        "moving within the negative side crossed"
    );
    assert!(
        !Crossing::Either.crosses(2.0, 1.0),
        "moving within the positive side crossed"
    );
}
