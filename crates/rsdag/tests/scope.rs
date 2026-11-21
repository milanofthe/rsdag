use rsdag::*;
use rsdag::{Tape, F64};

#[test]
fn parameters_keep_the_order_they_were_asked_for() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "f");
    // Asked for in the opposite order to their symbol ids.
    let z = s.param("z", ParamRole::Free);
    let a = s.param("a", ParamRole::Free);
    let e = s.sub(z, a);
    let f = s.close([(e, OutputRole::Plain)]);
    let names: Vec<&str> = g
        .func(f)
        .params()
        .iter()
        .map(|&p| g.symbol_name(p))
        .collect();
    assert_eq!(names, ["z", "a"]);
}

#[test]
fn roles_survive_closing() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "sys");
    let x = s.param("x", ParamRole::State { id: 0 });
    let k = s.param("k", ParamRole::Free);
    let kx = s.mul(k, x);
    let f = s.close([(kx, OutputRole::Plain)]);
    assert!(g
        .func(f)
        .param_roles()
        .contains(&ParamRole::State { id: 0 }));
}

#[test]
fn a_symbol_used_but_not_asked_for_still_becomes_a_parameter() {
    let mut g: Graph<F64> = Graph::new();
    let outside = g.sym("t");
    let mut s = Scope::new(&mut g, "f");
    let x = s.param("x", ParamRole::Free);
    let e = s.add(x, outside);
    let f = s.close([(e, OutputRole::Plain)]);
    let names: Vec<&str> = g
        .func(f)
        .params()
        .iter()
        .map(|&p| g.symbol_name(p))
        .collect();
    assert_eq!(names, ["x", "t"]);
    let tape = Tape::compile(&g, &[e], &g.func(f).params());
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[2.0f64, 3.0], &mut w, &mut o);
    assert_eq!(o[0], 5.0);
}

#[test]
fn asking_twice_for_a_name_gives_one_parameter() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "f");
    let a = s.param("v", ParamRole::Free);
    let b = s.param("v", ParamRole::State { id: 3 });
    assert_eq!(a, b);
    let f = s.close([(a, OutputRole::Plain)]);
    assert_eq!(g.func(f).params().len(), 1);
    assert_eq!(g.func(f).param_roles()[0], ParamRole::State { id: 3 });
}

/// Defining interns: the same parameters, roles and outputs with their
/// roles give the same function whatever its name, a different output role
/// a different one, and derivative outputs appended to the first since do
/// not keep a later identical definition from finding it.
#[test]
fn identical_definitions_intern_to_one_function() {
    let mut g: Graph<F64> = Graph::new();
    let define = |g: &mut Graph<F64>, name: &str, role: OutputRole| {
        let mut s = Scope::new(g, name);
        let x = s.param("x", ParamRole::State { id: 0 });
        let k = s.param("k", ParamRole::Param);
        let kx = s.mul(k, x);
        let y = s.sin(kx);
        s.close([(y, role)])
    };
    let f = define(&mut g, "f", OutputRole::Plain);
    assert_eq!(define(&mut g, "g", OutputRole::Plain), f);
    assert_ne!(define(&mut g, "r", OutputRole::Residual { id: 0 }), f);

    // differentiating a call of f appends d y / d x to f
    let (a, b) = (g.sym("a"), g.sym("b"));
    let call = g.call(f, 0, &[a, b]);
    let sa = g.as_symbol(a).unwrap();
    let _ = differentiate(&mut g, call, sa);
    assert_eq!(g.func(f).outputs().len(), 2);
    assert_eq!(define(&mut g, "h", OutputRole::Plain), f);
}
