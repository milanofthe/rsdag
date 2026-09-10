//! `eval_many` is `eval` per instance on the rayon pool: the same outputs,
//! in instance order, for any number of instances including none.

use rsdag::synth::{cases, Spec, Vocabulary};
use rsdag_jit::NativeTape;

#[test]
fn eval_many_is_eval_per_instance() {
    for case in cases(0..40, |seed| {
        Spec::new(seed)
            .steps(10 + seed as usize % 40)
            .params(3)
            .outputs(2)
            .vocab(Vocabulary::Full)
    }) {
        let native = NativeTape::compile(&case.tape).expect("compile");
        let stride = case.rows[0].len();
        let flat: Vec<f64> = case.rows.iter().flatten().copied().collect();
        let mut many = Vec::new();
        native.eval_many(&flat, stride, &mut many);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        for (i, row) in case.rows.iter().enumerate() {
            native.eval(row, &mut w, &mut o);
            for (k, v) in o.iter().enumerate() {
                let got = many[i * o.len() + k];
                assert!(
                    got.to_bits() == v.to_bits() || (got.is_nan() && v.is_nan()),
                    "seed {}: instance {i} output {k}: {got} vs {v}",
                    case.seed
                );
            }
        }
        native.eval_many(&[], stride, &mut many);
        assert!(many.is_empty());
    }
}
