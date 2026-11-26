/// Difference of two evaluations of one expression whose operations run in a
/// different order (reverse against forward derivatives, calls against the
/// inlined body, a fused kernel against a loop), relative to max(1, |value|).
/// To first order in u = 2^-53, each evaluation is within n u sum |t| of the
/// exact value, where n is the number of operations along a path and sum |t|
/// the sum of the magnitudes of the terms (Higham, Accuracy and Stability of
/// Numerical Algorithms, 2002, ch. 3). The expressions in the tests take at
/// most n = 40 operations along a path, and at their inputs of magnitude
/// below 2 the terms sum to at most 10 max(1, |value|). Two evaluations
/// differ by at most 2 * 40 * 10 u = 8.9e-14. A test may scale the bound by
/// 1 + |value| instead, which is larger.
pub const REORDERED: f64 = 2.0 * 40.0 * 10.0 * (f64::EPSILON / 2.0);

/// Relative difference between a value that rsdag computes with its own
/// exponential and at most one further rounded operation, and the same value
/// computed with the platform's `exp` and one rounded operation. rsdag's
/// `exp` is within 0.51 ulp (`rsdag::constants::exp_ln`), the platform's
/// within 1 ulp (the bound that glibc documents for `exp`), and each rounding
/// adds 0.5 ulp. The sum of 2.51 ulp is below 3 eps of the value.
pub const AGAINST_LIBM: f64 = 3.0 * f64::EPSILON;

/// Step h of the central differences (f(x + h) - f(x - h)) / 2h against which
/// the tests check derivatives.
pub const CENTRAL_DIFFERENCE_STEP: f64 = 1e-6;

/// Error of a central difference with the step h = 1e-6 of
/// [`CENTRAL_DIFFERENCE_STEP`] against f'(x), relative to 1 + |f'(x)|, for functions with |f| < 5,
/// |f'''| < 10 and |x| < 8 that take at most 300 operations (the largest is
/// the squared norm of a dense solve of order 6, |f| = 4.6). The truncation
/// error is h^2 |f'''| / 6 < 1.7e-11. Each evaluation is within 300 u |f| =
/// 1.7e-13 of f, which the division by 2h turns into 1.7e-7. The roundings of
/// x + h and x - h move the step by at most ulp(8) / 2h = 8.9e-10 of itself.
/// The sum is below 1.8e-7, and the bound is 2e-7.
pub const FINITE_DIFFERENCE: f64 = 2e-7;

/// Factor from the difference of two central differences to the truncation
/// error of the coarser one. The truncation error of a central difference
/// D(h) is c h^2 + O(h^4), so D(h) - D(h / 2) = 3/4 c h^2 to leading order,
/// and the error of D(h) is 4/3 |D(h) - D(h / 2)| (Richardson extrapolation of
/// order 2, 2^2 / (2^2 - 1)).
pub const RICHARDSON_CENTRAL: f64 = 4.0 / 3.0;

/// Multiple of the estimate [`RICHARDSON_CENTRAL`] |D(h) - D(h / 2)| within
/// which a derivative of a random expression must agree with D(h), where no
/// bound on f''' is known in advance. The estimate holds to leading order in
/// h, and the terms in h^4 and the round-off of both differences move it.
/// The factor 4 is a chosen safety factor on the estimate and not a derived
/// number. The round-off of D(h) is bounded separately by
/// [`central_round_off`].
pub const TRUNCATION_ESTIMATE_MULTIPLE: f64 = 4.0;

/// Bound on the round-off of the central difference (f(x+) - f(x-)) / (x+ -
/// x-) of an expression whose evaluation performs n = `roundings`
/// floating-point roundings, where `fp` and `fm` are the computed values
/// f(x+) and f(x-) at the rounded arguments x+ = x + h and x- = x - h, and
/// `step` is their difference x+ - x-, which is exact. The count n is the
/// number of roundings that the reference evaluation performs, not the
/// number of nodes: a node that performs several operations, such as a dot
/// product of m terms with its m multiplications and m - 1 additions,
/// contributes all of them, and an exact operation such as a negation
/// contributes none. The model is of first order in the unit round-off
/// u = 2^-53. Each rounding contributes a relative error of at most u, which
/// is at most one ulp of its result, and the errors add up in the function
/// value, so the expression is computed within n u |f| of its exact value.
/// The difference of the two values then carries at most
/// n u (|f(x+)| + |f(x-)|), and the division by the step carries it into the
/// quotient. An operation with a condition number above one amplifies the
/// error of its operand, which the model does not include, and for this
/// reason sinh(x0 + x0 x1) at the argument 4.88, where sinh has the
/// condition number 4.88, was measured at 1.86 ulp after three roundings
/// against 1.55 ulp in the model. The
/// roundings of the subtraction and the division add at most 2u relative to
/// the quotient, which the truncation term covers.
pub fn central_round_off(roundings: usize, fp: f64, fm: f64, step: f64) -> f64 {
    roundings as f64 * (f64::EPSILON / 2.0) * (fp.abs() + fm.abs()) / step
}

/// Error of `digamma` in units of the scale `1 + |psi(x)|` for x > 0 and
/// `1 + |psi(x)| + |psi(1 - x)|` for x <= 0, to first order in u = 2^-53.
/// The truncation of the asymptotic series is below 0.012 u relative to the
/// value (`rsdag::constants::POLYGAMMA_SHIFT`) and is neglected. For x > 0 the
/// value is a sum of at most m = 13 terms: at most ten recurrence terms
/// 1 / (x + k), ln x_N, 1 / 2x_N and the series at the shifted argument x_N.
/// The shift adds 1 exactly except where x + k enters a new binade, which
/// happens at most four times below 10, so x + k carries at most 2u
/// relative error, 1 / (x + k) at most 2.5u, and ln x_N with its libm error
/// of one ulp at most 2.9u relative to itself. The recursive summation of m
/// terms adds at most (m - 1) u times the sum S of their magnitudes (Higham,
/// Accuracy and Stability of Numerical Algorithms, 2002, eq. 4.4), so the
/// error is below (12 + 2.9) u S. Let A be the sum of the recurrence terms
/// and M = ln x_N + 1 / 2x_N + |series| < ln 11 + 0.051 the sum of the
/// magnitudes of the other terms. Then S = A + M = |A - M| + 2 min(A, M),
/// and |psi| differs from |A - M| by at most 2 (1 / 2x_N + |series|) <
/// 0.102, so S < |psi| + 4.898 + 0.102 = |psi| + 5 <= 5 (1 + |psi|), and the
/// error is below 74.5 u (1 + |psi|). For x <= 0 the reflection
/// psi(1 - x) - pi / tan(pi r) with the exact r = x - round(x) adds the
/// rounding of 1 - x (at most 2u,
/// since y psi'(y) <= 2 for y >= 1), the error of the cotangent term c (at
/// most 3.11 u |c| + 3.05 u from pi r, the libm tangent and the division)
/// and the final rounding 0.5 u |psi(x)|. With |c| <= |psi(x)| + |psi(1 - x)|
/// the sum is below (74.5 + 2 + 3.05) u + (74.5 + 3.11) u |psi(1 - x)| +
/// 3.61 u |psi(x)| < 80 u (1 + |psi(x)| + |psi(1 - x)|).
pub const DIGAMMA_ERROR: f64 = 80.0 * (f64::EPSILON / 2.0);

/// Error of `trigamma` in units of the scale `1 + |psi'(x)|` for x > 0 and
/// `1 + |psi'(x)| + |psi'(1 - x)|` for x <= 0, to first order in u = 2^-53.
/// The truncation of the asymptotic series is below 0.045 u relative to the
/// value and is neglected. For x > 0 every term of the sum is positive, so
/// S = psi'(x). Each recurrence term 1 / (x + k)^2 carries at most 5u (twice
/// the 2u of x + k and two roundings), and the summation of at most 13 terms
/// adds 12u, so the error is below 17 u psi'(x). For x <= 0 the reflection
/// pi^2 / sin^2(pi r) - psi'(1 - x) adds 2u psi'(1 - x) from the rounding of
/// 1 - x, at most 6.94 u relative to the first term t (pi^2 rounded, the
/// libm sine of pi r, the square and the division) and the final rounding
/// 0.5 u psi'(x). With t <= psi'(x) + psi'(1 - x) the sum is below
/// 26 u (1 + psi'(x) + psi'(1 - x)).
pub const TRIGAMMA_ERROR: f64 = 26.0 * (f64::EPSILON / 2.0);

/// `(x, psi(x), psi'(x))` from mpmath 1.3.0 at 40 digits (`mp.digamma(x)`,
/// `mp.polygamma(1, x)` at the double x), rounded to the nearest double.
/// The arguments cover both sides of the recurrence shift 10, the positive
/// zero of psi, a small argument and a pole approached to within a few ulps.
/// `psi(1)` is the negative Euler constant, which the standard library has.
pub const POLYGAMMA_MPMATH: [(f64, f64, f64); 16] = [
    (-99999.7, 9.230426796514083, 15.079403702621997),
    (-28.000000000000004, 281474976710659.4, 7.922816251426434e28),
    (-18.7556, -0.2962093295988112, 20.407003394699764),
    (-7.3, 4.33730730551005, 14.951383181433922),
    (-2.5, 1.103156640645243, 9.539246644989124),
    (-0.75, -2.894120200042932, 18.975106932284888),
    (-0.1, 9.245073050052948, 101.92253995947719),
    (-0.001, 999.4211381978913, 1000001.6473414317),
    (1e-8, -100000000.57721564, 1.0000000000000002e16),
    (0.3, -3.502524222200133, 12.245364546107732),
    (1.0, -std::f64::consts::EULER_GAMMA, 1.6449340668482264),
    (
        1.4616321449683622,
        -9.241265521729427e-17,
        0.9676722454476212,
    ),
    (2.5, 0.7031566406452432, 0.49035775610023485),
    (9.999999999999998, 2.251752589066721, 0.10516633568168576),
    (10.0, 2.251752589066721, 0.10516633568168575),
    (11.0, 2.351752589066721, 0.09516633568168574),
];

/// ln 2 - `LN_2` from mpmath 1.3.0 at 40 digits, rounded to the nearest
/// double, so that `LN_2 + LN2_LO` is ln 2 to 107 bits.
pub const LN2_LO: f64 = 2.319_046_813_846_299_6e-17;
