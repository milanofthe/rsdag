use num_rational::BigRational;
use num_traits::{One, Zero};
use rsdag::*;

#[test]
fn rational_powi_matches_repeated_multiplication() {
    let b = BigRational::from_ratio(3, 2);
    let mut acc = <BigRational as One>::one();
    for _ in 0..7 {
        acc = acc.mul(&b);
    }
    assert_eq!(b.powi(7), Some(acc.clone()));
    assert_eq!(b.powi(-7), Some(acc.recip()));
    assert_eq!(<BigRational as Zero>::zero().powi(-1), None);
}

#[test]
fn f64_field_canonicalizes_zero_and_nan() {
    assert_eq!(F64::new(-0.0), F64::new(0.0));
    assert!(F64::new(-0.0).is_zero());
    assert_eq!(F64::new(f64::NAN), F64::new(-f64::NAN));
    assert_ne!(F64::new(1.0), F64::new(1.0 + f64::EPSILON));
    assert_eq!(
        F64::from_f64(f64::INFINITY).map(|v| v.get()),
        Some(f64::INFINITY)
    );
}
