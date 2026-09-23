//! The LU program natively: the prolog leaves the same state as the
//! interpreter's, so the guard reads the same, and the solution matches to
//! the bit.

use rsdag::symbolic::solve::{plan, LuProgram, Panels, Pattern};
use rsdag_jit::NativeTape;

fn lu(n: usize, num: &[Vec<f64>], panels: Option<Panels>) -> (LuProgram, Vec<f64>) {
    let mut entries = Vec::new();
    let mut values = Vec::new();
    let mut pattern: Pattern = vec![Vec::new(); n];
    for (i, row) in num.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                entries.push((i, j));
                values.push(v);
                pattern[i].push(j);
            }
        }
    }
    let p = plan(&pattern).unwrap();
    (LuProgram::build(n, entries, p, panels), values)
}

fn check(lu: &LuProgram, values: &[f64], want_ok: bool) {
    let nt = NativeTape::compile(lu.tape()).unwrap();
    let mut inputs = vec![0.0; lu.input_len()];
    lu.write_values(values, None, &mut inputs);
    let b: Vec<f64> = (0..lu.n()).map(|i| 1.0 + i as f64).collect();
    lu.write_rhs(&b, None, &mut inputs);
    let (mut w0, mut w1, mut o0, mut o1) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    lu.tape().eval_prolog(&inputs, &mut w0);
    nt.eval_prolog(&inputs, &mut w1);
    assert_eq!(lu.factored(&inputs, &w0), want_ok);
    assert_eq!(lu.factored(&inputs, &w1), want_ok);
    let s = lu.tape().state_len();
    assert_eq!(
        w0[..s].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        w1[..s].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    lu.tape().eval_main(&inputs, &mut w0, &mut o0);
    nt.eval_main(&inputs, &mut w1, &mut o1);
    assert_eq!(
        lu.solution(&o0)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        lu.solution(&o1)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn the_native_prolog_holds_the_guard_alike() {
    let n = 24;
    let num: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| match (i as isize - j as isize).abs() {
                    0 => 6.0 + i as f64 * 0.1,
                    1 | 5 => 1.0 / (1.0 + (i + j) as f64),
                    _ => 0.0,
                })
                .collect()
        })
        .collect();
    let (scalar, values) = lu(n, &num, None);
    check(&scalar, &values, true);
    let panels = Panels {
        min_n: 0,
        min_width: 1,
        min_share: 0.0,
    };
    let (supernodal, values) = lu(n, &num, Some(panels));
    assert!(supernodal.supernodal().is_some());
    check(&supernodal, &values, true);
    // A zero first pivot fails the scalar program's guard on both (a
    // panel's dense kernel pivots within the panel).
    let (scalar, mut values) = lu(n, &num, None);
    let k = scalar
        .entries()
        .iter()
        .position(|&(i, j)| i == 0 && j == 0)
        .unwrap();
    values[k] = 0.0;
    check(&scalar, &values, false);
}
