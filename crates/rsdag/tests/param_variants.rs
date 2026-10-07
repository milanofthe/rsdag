//! A body that branches on its parameters runs each binding's variant: the
//! selects the binding decides resolved to their arms, the outputs and the
//! derivatives bit for bit the full body's, through bindings that flip the
//! branches.

use rsdag::{CmpOp, ExprId, FuncId, Graph, Node, ParamRole, SymbolId, Tape, F64};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// A chain of `n` transcendental steps over `x`: an arm worth removing.
fn chain(g: &mut Graph<F64>, x: ExprId, n: usize, c: f64) -> ExprId {
    let mut y = x;
    for i in 0..n {
        let k = g.konst_f64(c + 0.01 * i as f64);
        let t = g.tanh(y);
        y = g.mul(k, t);
    }
    y
}

/// `f(v, p, q)`, `p` and `q` parameters: a polarity switch between two
/// heavy arms, a level switch on `q`, and a select whose condition reads
/// the state (no binding decides it).
pub fn device(g: &mut Graph<F64>) -> FuncId {
    let (v, vs) = sym(g, "dev.v");
    let (p, ps) = sym(g, "dev.p");
    let (q, qs) = sym(g, "dev.q");
    let zero = g.zero();
    let one = g.one();
    let pos = g.cmp(CmpOp::Gt, p, zero);
    let a = chain(g, v, 30, 0.9);
    let b = chain(g, v, 30, 1.1);
    let pa = g.mul(p, a);
    let arm = g.select(pos, pa, b);
    let lvl = g.cmp(CmpOp::Gt, q, one);
    let vp = g.mul(v, p);
    let lev = g.select(lvl, vp, v);
    let on = g.cmp(CmpOp::Gt, v, zero);
    let vv = g.mul(v, v);
    let st = g.select(on, vv, zero);
    let s0 = g.add(arm, lev);
    let o0 = g.add(s0, st);
    let tv = g.tanh(v);
    let pv = g.mul(vv, p);
    let o1 = g.select(pos, pv, tv);
    let f = g.define_func("dev", vec![vs, ps, qs], vec![o0, o1]);
    g.set_param_role(f, 0, ParamRole::State { id: 0 });
    g.set_param_role(f, 1, ParamRole::Param);
    g.set_param_role(f, 2, ParamRole::Param);
    f
}

pub struct Circuit {
    pub g: Graph<F64>,
    pub roots: Vec<ExprId>,
    pub syms: Vec<SymbolId>,
    /// Per input whether it is a parameter.
    pub pure: Vec<bool>,
}

/// `n` instances, each `f(x_i, p_i, q_i)`, both outputs summed.
pub fn circuit(n: usize) -> Circuit {
    let mut g: Graph<F64> = Graph::new();
    let f = device(&mut g);
    let (mut syms, mut roots, mut pure) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (x, xs) = sym(&mut g, &format!("x{i}"));
        let (p, ps) = sym(&mut g, &format!("p{i}"));
        let (q, qs) = sym(&mut g, &format!("q{i}"));
        syms.extend([xs, ps, qs]);
        pure.extend([false, true, true]);
        let a = g.call(f, 0, &[x, p, q]);
        let b = g.call(f, 1, &[x, p, q]);
        roots.push(g.add(a, b));
    }
    Circuit {
        g,
        roots,
        syms,
        pure,
    }
}

/// The inputs: states `t`-shifted, parameters by `binding` (one sign per
/// instance flips its polarity, the level alternates).
pub fn inputs(n: usize, t: f64, binding: u32) -> Vec<f64> {
    (0..n)
        .flat_map(|i| {
            let flip = (binding >> (i % 8)) & 1 == 1;
            let p = if flip { -0.7 } else { 0.4 + 0.01 * i as f64 };
            let q = if (i + binding as usize).is_multiple_of(3) {
                2.0
            } else {
                0.5
            };
            [t + 0.1 * i as f64 - 0.3, p, q]
        })
        .collect()
}

/// Whether `body` runs per-binding variants: only such a body has tapes
/// of its own for a backend to make.
pub fn runs_variants(body: &std::sync::Arc<dyn rsdag::ExternBundle>) -> bool {
    let backend = rsdag::BodyBackend {
        compile: std::sync::Arc::new(|_: &Tape, _: &[bool], _: usize| None),
        submit: std::sync::Arc::new(|job: Box<dyn FnOnce() + Send>| job()),
    };
    body.with_backend(&backend).is_some()
}

pub fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn a_parameter_branching_body_runs_its_variants() {
    for n in [1usize, 12] {
        let c = circuit(n);
        let split = Tape::compile_split(&c.g, &c.roots, &c.syms, &c.pure);
        let whole = Tape::compile(&c.g, &c.roots, &c.syms);
        assert!(runs_variants(&split.bundles()[0]), "the body runs variants");
        let mut w = vec![0.0; split.work_len()];
        let mut got = vec![0.0; split.out_len()];
        let (mut ww, mut want) = (Vec::new(), Vec::new());
        for binding in [0u32, 0b1010_0101, 0xff, 0] {
            split.eval_prolog_into(&inputs(n, 0.0, binding), &mut w);
            for t in [0.0, 0.5, -1.25] {
                let ins = inputs(n, t, binding);
                split.eval_main_into(&ins, &mut w, &mut got);
                whole.eval(&ins, &mut ww, &mut want);
                assert!(same(&got, &want), "{n} at {binding:#x}, t = {t}");
            }
        }
    }
}

#[test]
fn derivatives_through_the_variants_are_the_full_bodys() {
    let n = 12;
    let mut c = circuit(n);
    let jac = rsdag::sparse_jacobian(&mut c.g, &c.roots, &c.syms);
    let entries: Vec<ExprId> = jac.iter().flat_map(|r| r.iter().map(|&(_, e)| e)).collect();
    let split = Tape::compile_split(&c.g, &entries, &c.syms, &c.pure);
    let whole = Tape::compile(&c.g, &entries, &c.syms);
    let mut w = vec![0.0; split.work_len()];
    let mut got = vec![0.0; split.out_len()];
    let (mut ww, mut want) = (Vec::new(), Vec::new());
    for binding in [0u32, 0x5a, 0] {
        split.eval_prolog_into(&inputs(n, 0.0, binding), &mut w);
        let ins = inputs(n, 0.25, binding);
        split.eval_main_into(&ins, &mut w, &mut got);
        whole.eval(&ins, &mut ww, &mut want);
        assert!(same(&got, &want), "at {binding:#x}");
    }
}

/// A select of the prolog between parameters: decided, the main phase reads
/// the parameter it picks itself, where the full body reads the select's
/// value. The caller passes it all the same.
#[test]
fn a_decided_prolog_select_reads_its_arm() {
    let mut g: Graph<F64> = Graph::new();
    let (v, vs) = sym(&mut g, "dev.v");
    let (p, ps) = sym(&mut g, "dev.p");
    let (q, qs) = sym(&mut g, "dev.q");
    let zero = g.zero();
    let pos = g.cmp(CmpOp::Gt, p, zero);
    let s = g.select(pos, q, p);
    let heavy = chain(&mut g, v, 40, 0.9);
    let o = g.mul(heavy, s);
    let f = g.define_func("dev", vec![vs, ps, qs], vec![o]);
    g.set_param_role(f, 0, ParamRole::State { id: 0 });
    g.set_param_role(f, 1, ParamRole::Param);
    g.set_param_role(f, 2, ParamRole::Param);
    let (mut roots, mut syms, mut pure) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..4 {
        let (x, xs) = sym(&mut g, &format!("x{i}"));
        let (pp, pps) = sym(&mut g, &format!("p{i}"));
        let (qq, qqs) = sym(&mut g, &format!("q{i}"));
        syms.extend([xs, pps, qqs]);
        pure.extend([false, true, true]);
        roots.push(g.call(f, 0, &[x, pp, qq]));
    }
    let split = Tape::compile_split(&g, &roots, &syms, &pure);
    let whole = Tape::compile(&g, &roots, &syms);
    // The body asks for `q` (argument 2) per evaluation: its variants read it.
    let reads = split.bundles()[0].main_reads().expect("reads some");
    assert!(reads.contains(&2), "{reads:?}");
    let ins = |t: f64| -> Vec<f64> {
        (0..4)
            .flat_map(|i| {
                [
                    t + 0.1 * i as f64,
                    if i % 2 == 0 { 0.5 } else { -0.5 },
                    2.0 + i as f64,
                ]
            })
            .collect()
    };
    let mut w = vec![0.0; split.work_len()];
    let mut got = vec![0.0; split.out_len()];
    let (mut ww, mut want) = (Vec::new(), Vec::new());
    split.eval_prolog_into(&ins(0.0), &mut w);
    for t in [0.0, 0.3] {
        split.eval_main_into(&ins(t), &mut w, &mut got);
        whole.eval(&ins(t), &mut ww, &mut want);
        assert!(same(&got, &want), "t = {t}: {got:?} vs {want:?}");
    }
}
