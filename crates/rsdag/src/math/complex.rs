//! The complex elementary functions over the real kernels of [`super`] and
//! the `libm` crate, in the textbook identities, so a complex value is as
//! platform-independent as a real one (`num-complex` would take its real
//! functions from the platform library).

use num_complex::Complex64 as C;

use super::{cosh, exp, ln, sinh};

#[inline]
fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

/// `|z|`, without overflow in the squares.
#[inline]
pub fn norm(z: C) -> f64 {
    libm::hypot(z.re, z.im)
}

/// The principal argument.
#[inline]
pub fn arg(z: C) -> f64 {
    libm::atan2(z.im, z.re)
}

pub fn exp_c(z: C) -> C {
    let e = exp(z.re);
    c(e * libm::cos(z.im), e * libm::sin(z.im))
}

pub fn ln_c(z: C) -> C {
    c(ln(norm(z)), arg(z))
}

/// The principal square root, stable in both half planes.
pub fn sqrt_c(z: C) -> C {
    if z.re == 0.0 && z.im == 0.0 {
        return c(0.0, z.im);
    }
    let t = ((z.re.abs() + norm(z)) * 0.5).sqrt();
    if z.re >= 0.0 {
        c(t, z.im / (2.0 * t))
    } else {
        c(z.im.abs() / (2.0 * t), t.copysign(z.im))
    }
}

pub fn sin_c(z: C) -> C {
    c(libm::sin(z.re) * cosh(z.im), libm::cos(z.re) * sinh(z.im))
}

pub fn cos_c(z: C) -> C {
    c(libm::cos(z.re) * cosh(z.im), -libm::sin(z.re) * sinh(z.im))
}

pub fn sinh_c(z: C) -> C {
    c(sinh(z.re) * libm::cos(z.im), cosh(z.re) * libm::sin(z.im))
}

pub fn cosh_c(z: C) -> C {
    c(cosh(z.re) * libm::cos(z.im), sinh(z.re) * libm::sin(z.im))
}

pub fn tan_c(z: C) -> C {
    let (a, b) = (2.0 * z.re, 2.0 * z.im);
    let d = libm::cos(a) + cosh(b);
    c(libm::sin(a) / d, sinh(b) / d)
}

pub fn tanh_c(z: C) -> C {
    let (a, b) = (2.0 * z.re, 2.0 * z.im);
    let d = cosh(a) + libm::cos(b);
    c(sinh(a) / d, libm::sin(b) / d)
}

const I: C = C { re: 0.0, im: 1.0 };
const ONE: C = C { re: 1.0, im: 0.0 };

/// `(ln(1 + iz) - ln(1 - iz)) / 2i`.
pub fn atan_c(z: C) -> C {
    let iz = I * z;
    (ln_c(ONE + iz) - ln_c(ONE - iz)) / (2.0 * I)
}

/// `-i ln(iz + sqrt(1 - z^2))`.
pub fn asin_c(z: C) -> C {
    -I * ln_c(I * z + sqrt_c(ONE - z * z))
}

/// `-i ln(z + i sqrt(1 - z^2))`.
pub fn acos_c(z: C) -> C {
    -I * ln_c(z + I * sqrt_c(ONE - z * z))
}

/// `ln(z + sqrt(z^2 + 1))`.
pub fn asinh_c(z: C) -> C {
    ln_c(z + sqrt_c(z * z + ONE))
}

/// `2 ln(sqrt((z + 1)/2) + sqrt((z - 1)/2))`.
pub fn acosh_c(z: C) -> C {
    2.0 * ln_c(sqrt_c((z + ONE) * 0.5) + sqrt_c((z - ONE) * 0.5))
}

/// `(ln(1 + z) - ln(1 - z)) / 2`.
pub fn atanh_c(z: C) -> C {
    (ln_c(ONE + z) - ln_c(ONE - z)) * 0.5
}

/// `x^y = e^(y ln x)`, `0^y = 0` for `y` with a positive real part.
pub fn pow_c(x: C, y: C) -> C {
    if x.re == 0.0 && x.im == 0.0 {
        return if y.re > 0.0 && y.im == 0.0 {
            c(0.0, 0.0)
        } else {
            exp_c(y * ln_c(x))
        };
    }
    exp_c(y * ln_c(x))
}

/// `z^(1/3)` on the principal branch.
pub fn cbrt_c(z: C) -> C {
    pow_c(z, c(1.0 / 3.0, 0.0))
}
