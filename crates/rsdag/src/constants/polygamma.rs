//! The constants of [`digamma`](crate::semantics::digamma) and
//! [`trigamma`](crate::semantics::trigamma). Part of the contract: every
//! backend evaluates both functions through these two routines, so the
//! constants decide the bits of every result.
//!
//! Both functions shift the argument with the recurrences
//! `psi(x) = psi(x + 1) - 1 / x` and `psi'(x) = psi'(x + 1) + 1 / x^2` to
//! `x >= POLYGAMMA_SHIFT` and sum the asymptotic expansions there (DLMF
//! 5.11.2 and its derivative)
//!
//! `psi(x) ~ ln x - 1 / (2x) - sum_k B_2k / (2k x^2k)`,
//!
//! `psi'(x) ~ 1 / x + 1 / (2x^2) + sum_k B_2k / x^(2k + 1)`,
//!
//! with the Bernoulli numbers `B_2k`. For real `x > 0` the remainder of
//! either expansion after any number of terms has the sign of the first
//! omitted term and is smaller in magnitude (DLMF 5.11(ii)). The shift and
//! the numbers of terms are chosen so that this term is below one eighth of
//! the unit round-off `u = 2^-53` relative to the value at `x =
//! POLYGAMMA_SHIFT`, where it is largest. Each coefficient is the quotient
//! of two integers, which the compiler rounds once to the nearest double.

/// The argument from which on the asymptotic expansions are summed.
/// Smaller arguments are shifted up by the recurrences, at most ten steps.
/// At this shift the series of `digamma` with the eight terms of
/// [`DIGAMMA_COEFFS`] stops before `|B_18| / (18 x^18) = (43867 / 14364) /
/// 10^18 = 3.05e-18`, which is `0.012 u` relative to `psi(10) = 2.2518`, and
/// the series of `trigamma` with the nine terms of [`TRIGAMMA_COEFFS`] stops
/// before `|B_20| / x^21 = (174611 / 330) / 10^21 = 5.29e-19`, which is
/// `0.045 u` relative to `psi'(10) = 0.10517`. The smallest shift that keeps
/// both terms below `u / 8` is 7, with 11 and 15 terms; shifts from 9 to 13
/// need between 6 and 10 terms, and 10 was taken
/// among them as the round number.
pub const POLYGAMMA_SHIFT: f64 = 10.0;

/// The coefficients `B_2k / (2k)` for `k = 1, ..., 8` of the asymptotic
/// expansion of `digamma` (DLMF 5.11.2), with the Bernoulli numbers
/// `B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66,
/// B_12 = -691/2730, B_14 = 7/6, B_16 = -3617/510` (DLMF Table 24.2.1).
pub const DIGAMMA_COEFFS: [f64; 8] = [
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
    1.0 / 12.0,
    -3617.0 / 8160.0,
];

/// The coefficients `B_2k` for `k = 1, ..., 9` of the asymptotic expansion of
/// `trigamma`, the derivative of DLMF 5.11.2: the numbers of
/// [`DIGAMMA_COEFFS`] times `2k`, and `B_18 = 43867/798` (DLMF Table
/// 24.2.1).
pub const TRIGAMMA_COEFFS: [f64; 9] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
    43867.0 / 798.0,
];
