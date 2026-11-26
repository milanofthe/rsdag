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

/// Error of a central difference (f(x + h) - f(x - h)) / 2h with h = 1e-6
/// against f'(x), relative to 1 + |f'(x)|, for functions with |f| < 5,
/// |f'''| < 10 and |x| < 8 that take at most 300 operations (the largest is
/// the squared norm of a dense solve of order 6, |f| = 4.6). The truncation
/// error is h^2 |f'''| / 6 < 1.7e-11. Each evaluation is within 300 u |f| =
/// 1.7e-13 of f, which the division by 2h turns into 1.7e-7. The roundings of
/// x + h and x - h move the step by at most ulp(8) / 2h = 8.9e-10 of itself.
/// The sum is below 1.8e-7, and the bound is 2e-7.
pub const FINITE_DIFFERENCE: f64 = 2e-7;

/// Absolute error of `trigamma`. Its asymptotic series is evaluated at
/// x >= 6 and stops after the term in B_8 / x^9. The series alternates, so
/// the error is below the first omitted term B_10 / x^11 = (5 / 66) / 6^11 =
/// 2.09e-10. The recurrence and the reflection round with relative errors
/// near 1e-14 at the test points, which the bound covers through its factor
/// 1 + |psi'(x)|.
pub const TRIGAMMA_SERIES: f64 = 2.1e-10;

/// `(x, psi(x), psi'(x))` from mpmath 1.3.0 at 40 digits (`mp.digamma(x)`,
/// `mp.polygamma(1, x)` at the double x), rounded to the nearest double.
pub const POLYGAMMA_MPMATH: [(f64, f64, f64); 9] = [
    (-18.7556, -0.2962093295988112, 20.407003394699764),
    (-7.3, 4.33730730551005, 14.951383181433922),
    (-2.5, 1.103156640645243, 9.539246644989124),
    (-0.75, -2.894120200042932, 18.975106932284888),
    (-0.1, 9.245073050052948, 101.92253995947719),
    (-0.001, 999.4211381978913, 1000001.6473414317),
    (0.3, -3.502524222200133, 12.245364546107732),
    (2.5, 0.7031566406452432, 0.49035775610023485),
    (11.0, 2.351752589066721, 0.09516633568168574),
];

/// ln 2 - `LN_2` from mpmath 1.3.0 at 40 digits, rounded to the nearest
/// double, so that `LN_2 + LN2_LO` is ln 2 to 107 bits.
pub const LN2_LO: f64 = 2.319_046_813_846_299_6e-17;
