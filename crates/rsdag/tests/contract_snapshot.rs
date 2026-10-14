//! The contract constants, to the bit: the values every backend
//! reproduces (the domain guards, the exponential's and the logarithm's
//! constants and bit fields, the fold order, the integer powers, the
//! elimination's panels) and the module format. Changing one changes
//! results, so it takes an edit here in the same commit, where review sees
//! it.

use rsdag::semantics::exp_ln::*;
use rsdag::semantics::{LU_PANEL_LARGE, LU_PANEL_SMALL, LU_PANEL_SWITCH, LU_UNBLOCKED_MAX};

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
    assert_eq!(F64_HIGH_WORD, 32);
    assert_eq!(F64_HIGH_FRAC_BITS, 20);
    assert_eq!(F64_LOW_WORD_MASK, 0xffff_ffff);
    assert_eq!(LN_HIGH_FRAC_MASK, 0x000f_ffff);
    assert_eq!(LN_SQRT2_CARRY, 0x9_5f64);
    assert_eq!(LN_HIGH_UNIT, 0x10_0000);
    assert_eq!(LN_HIGH_ONE, 0x3ff0_0000);
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
        ("INV_LN2", INV_LN2, 0x3ff7_1547_652b_82fe),
        ("LN2_HI", LN2_HI, 0x3fe6_2e42_fee0_0000),
        ("LN2_LO", LN2_LO, 0x3dea_39ef_3579_3c76),
        ("EP1", EP1, 0x3fc5_5555_5555_553e),
        ("EP2", EP2, 0xbf66_c16c_16be_bd93),
        ("EP3", EP3, 0x3f11_566a_af25_de2c),
        ("EP4", EP4, 0xbebb_bd41_c5d2_6bf1),
        ("EP5", EP5, 0x3e66_3769_72be_a4d0),
        ("LG1", LG1, 0x3fe5_5555_5555_5593),
        ("LG2", LG2, 0x3fd9_9999_9997_fa04),
        ("LG3", LG3, 0x3fd2_4924_9422_9359),
        ("LG4", LG4, 0x3fcc_71c5_1d8e_78af),
        ("LG5", LG5, 0x3fc7_4664_96cb_03de),
        ("LG6", LG6, 0x3fc3_9a09_d078_c69f),
        ("LG7", LG7, 0x3fc2_f112_df3e_5244),
    ] {
        bits(name, v, want);
    }
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
