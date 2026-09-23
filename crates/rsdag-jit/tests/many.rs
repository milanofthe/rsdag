//! `eval_many` is `eval` per instance on the rayon pool: the same outputs,
//! in instance order, for any number of instances including none. The
//! serial `Program::eval_many_into` is the same on either backend.

use rsdag::synth::{cases, Spec, Vocabulary};
use rsdag::Program;
use rsdag_jit::NativeTape;

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

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

#[test]
fn eval_many_into_is_the_same_on_both_backends() {
    for case in cases(0..20, |seed| {
        Spec::new(seed)
            .steps(10 + seed as usize % 30)
            .params(3)
            .outputs(2)
            .vocab(Vocabulary::Full)
    }) {
        let native = NativeTape::compile(&case.tape).expect("compile");
        let stride = case.rows[0].len();
        let flat: Vec<f64> = case.rows.iter().flatten().copied().collect();
        let n_out = case.tape.out_len();
        let run = |p: &dyn Program| {
            let mut work = vec![0.0; p.work_len()];
            let mut out = vec![0.0; case.rows.len() * n_out];
            p.eval_many_into(&flat, stride, &mut work, &mut out);
            out
        };
        let (a, b) = (run(&case.tape), run(&native));
        assert!(same(&a, &b), "seed {}", case.seed);
        let mut many = Vec::new();
        native.eval_many(&flat, stride, &mut many);
        assert!(same(&a, &many), "seed {}", case.seed);
    }
}

#[test]
fn short_input_vectors_are_padded_with_nan() {
    let case = cases(3..4, |seed| Spec::new(seed).steps(20).params(3).outputs(2))
        .next()
        .unwrap();
    let native = NativeTape::compile(&case.tape).expect("compile");
    let stride = case.rows[0].len();
    assert!(stride >= 2);
    let short: Vec<f64> = case
        .rows
        .iter()
        .flat_map(|r| r[..stride - 1].iter().copied())
        .collect();
    let mut many = Vec::new();
    native.eval_many(&short, stride - 1, &mut many);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    for (i, row) in case.rows.iter().enumerate() {
        native.eval(&row[..stride - 1], &mut w, &mut o);
        assert!(
            same(&many[i * o.len()..(i + 1) * o.len()], &o),
            "instance {i}"
        );
    }
}
