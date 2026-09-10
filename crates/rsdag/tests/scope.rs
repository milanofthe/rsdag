use rsdag::*;
use rsdag::{OutputRole, Tape, F64};

#[test]
fn parameters_keep_the_order_they_were_asked_for() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "f");
    // Asked for in the opposite order to their symbol ids.
    let z = s.param("z");
    let a = s.param("a");
    let e = s.sub(z, a);
    let f = s.close(vec![e]);
    let names: Vec<&str> = g.func(f).params.iter().map(|&p| g.symbol_name(p)).collect();
    assert_eq!(names, ["z", "a"]);
}

#[test]
fn roles_survive_and_select_jacobian_blocks() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "sys");
    let x = s.param_with_role("x", ParamRole::State { id: 0 });
    let k = s.param("k");
    let kx = s.mul(k, x);
    let f = s.close_with_roles(vec![(OutputRole::Residual { id: 0 }, kx)]);
    let blocks = g.jacobian_by_role(
        f,
        |o| matches!(o, OutputRole::Residual { .. }),
        |p| matches!(p, ParamRole::State { .. }),
    );
    assert_eq!(blocks.len(), 1);
}

#[test]
fn a_symbol_used_but_not_asked_for_still_becomes_a_parameter() {
    let mut g: Graph<F64> = Graph::new();
    let outside = g.sym("t");
    let mut s = Scope::new(&mut g, "f");
    let x = s.param("x");
    let e = s.add(x, outside);
    let f = s.close(vec![e]);
    let names: Vec<&str> = g.func(f).params.iter().map(|&p| g.symbol_name(p)).collect();
    assert_eq!(names, ["x", "t"]);
    let tape = Tape::compile(&g, &[e], &g.func(f).params.clone());
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[2.0f64, 3.0], &mut w, &mut o);
    assert_eq!(o[0], 5.0);
}

#[test]
fn asking_twice_for_a_name_gives_one_parameter() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "f");
    let a = s.param("v");
    let b = s.param_with_role("v", ParamRole::State { id: 3 });
    assert_eq!(a, b);
    assert_eq!(s.params().len(), 1);
    let f = s.close(vec![a]);
    assert_eq!(g.func(f).param_roles[0], ParamRole::State { id: 3 });
}
