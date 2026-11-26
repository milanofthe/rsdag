//! The contract constants, to the bit: the values every backend
//! reproduces (the domain guards, the exponential's and the logarithm's
//! constants and bit fields, the fold order, the integer powers, the
//! elimination's panels) and the module format. Changing one changes
//! results, so it takes an edit here in the same commit, where review sees
//! it.

use rsdag::constants::{LU_PANEL_LARGE, LU_PANEL_SMALL, LU_PANEL_SWITCH, LU_UNBLOCKED_MAX};
use rsdag::semantics::exp_ln::*;

fn bits(name: &str, v: f64, want: u64) {
    assert_eq!(
        v.to_bits(),
        want,
        "{name} = {v:e} ({:#018x}), the contract {:e} ({want:#018x})",
        v.to_bits(),
        f64::from_bits(want)
    );
}

#[test]
fn domain_guards() {
    bits("EXP_LIMIT", EXP_LIMIT, 0x4054_0000_0000_0000);
    bits("LN_FLOOR", LN_FLOOR, 0x39b4_484b_feeb_c2a0);
    bits("EXP_AT_LIMIT", EXP_AT_LIMIT, 0x4725_5779_b984_f3eb);
    bits("LN_AT_FLOOR", LN_AT_FLOOR, 0xc051_44f6_9ff9_ffc4);
    bits("EXP_UNDERFLOW", EXP_UNDERFLOW, 0xc087_4999_9999_999a);
}

#[test]
fn exp_and_ln_bit_fields() {
    assert_eq!(F64_EXP_BIAS, 1023);
    assert_eq!(F64_FRAC_BITS, 52);
    assert_eq!(EXP_TABLE_BITS, 7);
    assert_eq!(EXP_SCALE_BIAS, 64);
    assert_eq!(LN_TABLE_BITS, 7);
    assert_eq!(LN_OFF, 0x3fe6_0000_0000_0000);
}

#[test]
fn fold_order_and_integer_powers() {
    assert_eq!(rsdag::constants::FOLD_LANES, 4);
    bits(
        "POWF_AS_POWI_MAX",
        rsdag::constants::POWF_AS_POWI_MAX,
        0x4050_0000_0000_0000,
    );
}

#[test]
fn exp_and_ln_coefficients() {
    for (name, v, want) in [
        ("MAGIC", MAGIC, 0x4338_0000_0000_0000),
        ("EXP_INV_LN2_N", EXP_INV_LN2_N, 0x4067_1547_652b_82fe),
        ("EXP_LN2_HI_N", EXP_LN2_HI_N, 0x3f76_2e42_fefc_0000),
        ("EXP_LN2_LO_N", EXP_LN2_LO_N, 0xbd3c_610c_a86c_3899),
        ("EXP_POLY[0]", EXP_POLY[0], 0x3fdf_ffff_ffff_ff57),
        ("EXP_POLY[1]", EXP_POLY[1], 0x3fc5_5555_5555_5525),
        ("EXP_POLY[2]", EXP_POLY[2], 0x3fa5_5555_acf7_d28e),
        ("EXP_POLY[3]", EXP_POLY[3], 0x3f81_1111_4324_c640),
        ("LN_LN2_HI", LN_LN2_HI, 0x3fe6_2e42_fefa_3800),
        ("LN_LN2_LO", LN_LN2_LO, 0x3d2e_f357_93c7_6730),
        ("LN_POLY[0]", LN_POLY[0], 0x3fd5_5555_5555_5555),
        ("LN_POLY[1]", LN_POLY[1], 0xbfd0_0000_0000_0b45),
        ("LN_POLY[2]", LN_POLY[2], 0x3fc9_9999_9999_ae17),
        ("LN_POLY[3]", LN_POLY[3], 0xbfc5_5555_4a15_f694),
        ("LN_POLY[4]", LN_POLY[4], 0x3fc2_4924_880f_8532),
        ("LN_POLY[5]", LN_POLY[5], 0xbfc0_0167_311e_1591),
        ("LN_POLY[6]", LN_POLY[6], 0x3fbc_7454_3096_1a00),
    ] {
        bits(name, v, want);
    }
}

const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_over_words(bits: impl IntoIterator<Item = u64>) -> u64 {
    bits.into_iter().fold(FNV1A_64_OFFSET_BASIS, |h, b| {
        (h ^ b).wrapping_mul(FNV1A_64_PRIME)
    })
}

#[test]
fn exp_and_ln_tables() {
    assert_eq!(fnv1a_over_words(EXP_TABLE), 0xa1af_296a_f08c_1db8);
    assert_eq!(
        fnv1a_over_words(LN_TABLE.iter().flatten().map(|v| v.to_bits())),
        0xaead_42a1_6854_2e23
    );
}

#[test]
fn elimination_panels() {
    assert_eq!(LU_PANEL_SMALL, 16);
    assert_eq!(LU_PANEL_LARGE, 32);
    assert_eq!(LU_PANEL_SWITCH, 512);
    assert_eq!(LU_UNBLOCKED_MAX, 64);
}

#[test]
fn exact_powers_and_module_format() {
    assert_eq!(rsdag::field::EXACT_POWI_MAX, 1024);
    assert_eq!(rsdag::MODULE_VERSION, 2);
}
