use rsdag::node::{BINARY_OPS, UNARY_OPS};
use rsdag::*;

#[test]
fn tables_line_up_with_the_enums() {
    for (i, spec) in UNARY_OPS.iter().enumerate() {
        assert_eq!(spec.op as usize, i, "row {i} describes {:?}", spec.op);
        assert_eq!(spec.op.spec().name, spec.name);
        assert_eq!(UnaryOp::from_code(spec.op.code()), spec.op);
        assert!(!spec.name.is_empty());
    }
    for (i, spec) in BINARY_OPS.iter().enumerate() {
        assert_eq!(spec.op as usize, i);
        assert_eq!(BinOp::from_code(spec.op.code()), spec.op);
    }
    assert_eq!(
        UNARY_OPS.len(),
        UnaryOp::RandUniform as usize + 1,
        "a unary variant has no row"
    );
    assert_eq!(
        BINARY_OPS.len(),
        BinOp::RandStream as usize + 1,
        "a binary variant has no row"
    );
}

#[test]
fn names_are_unique() {
    let mut names: Vec<&str> = UNARY_OPS.iter().map(|s| s.name).collect();
    names.sort_unstable();
    let n = names.len();
    names.dedup();
    assert_eq!(
        names.len(),
        n,
        "two ops share the name that the printer emits"
    );
}

#[test]
fn roughness_is_recorded() {
    for op in [
        UnaryOp::Floor,
        UnaryOp::Ceil,
        UnaryOp::Round,
        UnaryOp::Trunc,
        UnaryOp::Sign,
        UnaryOp::Abs,
        UnaryOp::RandUniform,
    ] {
        assert!(!op.is_smooth(), "{op:?} is not differentiable");
    }
    assert!(!BinOp::RandStream.is_smooth() && !BinOp::Mod.is_smooth());
    for op in [UnaryOp::Exp, UnaryOp::Sin, UnaryOp::Erf, UnaryOp::Lgamma] {
        assert!(op.is_smooth(), "{op:?} is differentiable");
    }
}

#[test]
fn elementary_functions_by_name() {
    use rsdag::semantics::elementary_f64;
    let names = UNARY_OPS
        .iter()
        .map(|s| s.name)
        .chain(BINARY_OPS.iter().map(|s| s.name));
    for name in names.chain(["pow", "min", "max"]) {
        let f = Elementary::from_name(name).expect("a known name");
        let mut g: Graph = Graph::new();
        let args: Vec<ExprId> = (0..f.arity()).map(|k| g.sym(&format!("a{k}"))).collect();
        let e = g.elementary(f, &args);
        let at = [0.7, 0.3];
        let env = (0..f.arity())
            .map(|k| (g.as_symbol(args[k]).unwrap(), at[k]))
            .collect();
        let want = elementary_f64(f, &at[..f.arity()]);
        let got = eval(&g, &[e], &env)[0];
        assert!(
            got == want || (got.is_nan() && want.is_nan()),
            "{name}: {got} vs {want}"
        );
    }
    assert_eq!(
        Elementary::from_name("pow"),
        Some(Elementary::Binary(BinOp::Powf))
    );
    assert_eq!(
        Elementary::from_name("max").map(Elementary::name),
        Some("max")
    );
    assert_eq!(Elementary::from_name("nope"), None);
}

#[test]
fn rand_stream_draws_independent_uniforms() {
    use rsdag::semantics::rand_stream;
    let n = 20_000;
    let a: Vec<f64> = (0..n).map(|k| rand_stream(52924.0, k as f64)).collect();
    let b: Vec<f64> = (0..n).map(|k| rand_stream(52925.0, k as f64)).collect();
    assert_eq!(rand_stream(52924.0, 7.0), a[7]);
    assert!(a.iter().all(|&u| (0.0..1.0).contains(&u)));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mb) = (mean(&a), mean(&b));
    let var = a.iter().map(|u| (u - ma).powi(2)).sum::<f64>() / n as f64;
    let cov = (a.iter().zip(&b))
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / n as f64;
    let lag = (a.windows(2))
        .map(|w| (w[0] - ma) * (w[1] - ma))
        .sum::<f64>()
        / n as f64;
    let five_sigma_per_unit_deviation = 5.0 / (n as f64).sqrt();
    let s = five_sigma_per_unit_deviation;
    assert!((ma - 0.5).abs() < 0.29 * s, "mean {ma}");
    assert!((var - 1.0 / 12.0).abs() < 0.075 * s, "variance {var}");
    assert!(cov.abs() < s / 12.0, "streams correlate: {cov}");
    assert!(lag.abs() < s / 12.0, "counters correlate: {lag}");
    assert_ne!(
        rand_stream(3.0, 5.0),
        rand_stream(5.0, 3.0),
        "swapping key and counter gave the same draw"
    );
}
