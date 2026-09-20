//! The sign test of a guard, which decides when an integrator lands a step.

use rsdag::Crossing;

#[test]
fn a_step_starting_on_the_surface_is_not_a_crossing() {
    // Landing on a surface leaves g at zero; the next step must not fire
    // again, in any direction.
    for dir in [Crossing::Either, Crossing::Rising, Crossing::Falling] {
        assert!(!dir.crosses(0.0, 1.0), "{dir:?} fired leaving zero upward");
        assert!(
            !dir.crosses(0.0, -1.0),
            "{dir:?} fired leaving zero downward"
        );
        assert!(!dir.crosses(0.0, 0.0), "{dir:?} fired on a flat zero");
    }
}

#[test]
fn the_direction_selects_the_sign_change() {
    assert!(Crossing::Rising.crosses(-1.0, 1.0));
    assert!(!Crossing::Rising.crosses(1.0, -1.0));
    assert!(Crossing::Falling.crosses(1.0, -1.0));
    assert!(!Crossing::Falling.crosses(-1.0, 1.0));
    assert!(Crossing::Either.crosses(-1.0, 1.0));
    assert!(Crossing::Either.crosses(1.0, -1.0));
    // Reaching the surface counts, in the direction of approach.
    assert!(Crossing::Rising.crosses(-1.0, 0.0));
    assert!(Crossing::Falling.crosses(1.0, 0.0));
}
