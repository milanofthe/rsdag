//! With variants off (a process-wide switch, hence a test binary of its
//! own) a parameter-branching body is the full one.

use rsdag::{CmpOp, Graph, Node, ParamRole, Tape, F64};

#[test]
fn variants_off_runs_the_full_body() {
    rsdag::variant::set_enabled(false);
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    for n in ["v", "p", "x", "q"] {
        let e = g.sym(n);
        let Node::Symbol(s) = *g.node(e) else {
            unreachable!()
        };
        syms.push((e, s));
    }
    let [(v, vs), (p, ps), (x, xs), (q, qs)] = syms[..] else {
        unreachable!()
    };
    let zero = g.zero();
    let pos = g.cmp(CmpOp::Gt, p, zero);
    let (e, t) = (g.exp(v), g.tanh(v));
    let o = g.select(pos, e, t);
    let f = g.define_func("dev", vec![vs, ps], vec![o]);
    g.set_param_role(f, 0, ParamRole::State { id: 0 });
    g.set_param_role(f, 1, ParamRole::Param);
    let root = g.call(f, 0, &[x, q]);
    let tape = Tape::compile_split(&g, &[root], &[xs, qs], &[false, true]);
    let backend = rsdag::BodyBackend {
        compile: std::sync::Arc::new(|_: &Tape, _: &[bool], _: usize| None),
        submit: std::sync::Arc::new(|job: Box<dyn FnOnce() + Send>| job()),
    };
    assert!(tape.bundles()[0].with_backend(&backend).is_none());
}
