//! Function bodies called by several instances run natively in lanes,
//! instances side by side in every register: the results are the scalar
//! code's and the interpreter's, bit for bit, for every kind of op, for
//! special values (NaN, infinities, signed zeros) and for a batch that does
//! not fill its last block of lanes; the prolog's states included.

use rsdag::node::{BinOp, CmpOp, ReduceOp, UnaryOp};
use rsdag::{ExprId, FuncId, Graph, Node, ParamRole, Scope, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

const UNARY: [UnaryOp; 26] = [
    UnaryOp::Exp,
    UnaryOp::Ln,
    UnaryOp::Sqrt,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Sinh,
    UnaryOp::Cosh,
    UnaryOp::Tanh,
    UnaryOp::Atan,
    UnaryOp::Floor,
    UnaryOp::Tan,
    UnaryOp::Log10,
    UnaryOp::Log2,
    UnaryOp::Log1p,
    UnaryOp::Expm1,
    UnaryOp::Cbrt,
    UnaryOp::Abs,
    UnaryOp::Sign,
    UnaryOp::Ceil,
    UnaryOp::Round,
    UnaryOp::Trunc,
    UnaryOp::Asin,
    UnaryOp::Acos,
    UnaryOp::Asinh,
    UnaryOp::Atanh,
    UnaryOp::Erf,
];
const BINARY: [BinOp; 4] = [BinOp::Powf, BinOp::Mod, BinOp::Atan2, BinOp::Hypot];
const CMP: [CmpOp; 6] = [
    CmpOp::Lt,
    CmpOp::Le,
    CmpOp::Gt,
    CmpOp::Ge,
    CmpOp::Eq,
    CmpOp::Ne,
];

/// A body over two states and two parameters with every kind of op, the
/// parameters' work (a prolog) feeding the states' (the main phase).
fn body(g: &mut Graph<F64>) -> FuncId {
    let mut s = Scope::new(g, "every");
    let a = s.param_with_role("a", ParamRole::State { id: 0 });
    let b = s.param_with_role("b", ParamRole::State { id: 1 });
    let p = s.param_with_role("p", ParamRole::Param);
    let q = s.param_with_role("q", ParamRole::Param);
    let mut outs = Vec::new();
    // parameter-only work: lands in the prolog, read through the state
    let pq = s.mul(p, q);
    let pe = s.exp(pq);
    let ab = s.mul(a, b);
    let x = s.add(ab, pe);
    outs.push(x);
    let d = s.sub(a, b);
    let dv = s.div(a, b);
    outs.extend([d, dv]);
    for n in [-3i64, -1, 2, 3, 5, 70] {
        outs.push(s.pow_i(a, n));
    }
    for op in UNARY {
        outs.push(s.unary(op, d));
    }
    for op in BINARY {
        outs.push(s.binary(op, a, b));
    }
    for op in CMP {
        outs.push(s.cmp(op, a, b));
    }
    let c = s.cmp(CmpOp::Lt, a, q);
    outs.push(s.select(c, b, p));
    outs.push(s.reduce(ReduceOp::Sum, vec![a, b, p, q, x]));
    outs.push(s.reduce(ReduceOp::Product, vec![a, b, q]));
    outs.push(s.reduce(ReduceOp::Min, vec![a, b, p]));
    outs.push(s.reduce(ReduceOp::Max, vec![a, b, q, d]));
    outs.push(s.dot(vec![a, b, p], vec![q, a, b]));
    let neg = s.neg(a);
    outs.push(neg);
    s.close(outs)
}

/// Values of every kind, so every lane meets them in turn.
const SPECIAL: [f64; 13] = [
    0.75,
    -1.5,
    0.0,
    -0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    1e-300,
    -2.5e8,
    0.5,
    3.0,
    -0.999,
    1.0,
];

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// `n` instances of the body at the top, every output of each a root.
fn program(n: usize) -> (Graph<F64>, Vec<ExprId>, Vec<SymbolId>, Vec<bool>) {
    let mut g: Graph<F64> = Graph::new();
    let f = body(&mut g);
    let n_out = g.func(f).outputs().len() as u32;
    let outs: Vec<u32> = (0..n_out).collect();
    let mut roots = Vec::new();
    let (mut syms, mut pure) = (Vec::new(), Vec::new());
    for k in 0..n {
        let (a, sa) = sym(&mut g, &format!("a{k}"));
        let (b, sb) = sym(&mut g, &format!("b{k}"));
        let (p, sp) = sym(&mut g, &format!("p{k}"));
        let (q, sq) = sym(&mut g, &format!("q{k}"));
        roots.extend(g.calls(f, &outs, &[a, b, p, q]));
        syms.extend([sa, sb, sp, sq]);
        pure.extend([false, false, true, true]);
    }
    (g, roots, syms, pure)
}

fn inputs(n: usize, shift: usize) -> Vec<f64> {
    (0..4 * n)
        .map(|k| SPECIAL[(k * 5 + shift) % SPECIAL.len()])
        .collect()
}

#[test]
fn lanes_are_the_scalar_code_for_every_op_and_value() {
    for n in [2usize, 3, 7] {
        let (g, roots, syms, pure) = program(n);
        let tape = Tape::compile_split(&g, &roots, &syms, &pure);
        let native = NativeTape::compile(&tape).expect("native code");
        for shift in 0..SPECIAL.len() {
            let x = inputs(n, shift);
            let (mut w, mut o) = (Vec::new(), Vec::new());
            tape.eval_prolog(&x, &mut w);
            tape.eval_main(&x, &mut w, &mut o);
            let (mut wn, mut on) = (Vec::new(), Vec::new());
            native.eval_prolog(&x, &mut wn);
            native.eval_main(&x, &mut wn, &mut on);
            for (k, (a, b)) in o.iter().zip(&on).enumerate() {
                assert!(
                    same(*a, *b),
                    "n={n} shift={shift} output {k}: {a:e} vs {b:e}"
                );
            }
            let s = tape.state_len();
            for (k, (a, b)) in w[..s].iter().zip(&wn[..s]).enumerate() {
                assert!(
                    same(*a, *b),
                    "n={n} shift={shift} state {k}: {a:e} vs {b:e}"
                );
            }
        }
    }
}

#[test]
fn lanes_without_state_are_the_scalar_code() {
    let n = 5;
    let (g, roots, syms, _) = program(n);
    let tape = Tape::compile(&g, &roots, &syms);
    let native = NativeTape::compile(&tape).expect("native code");
    for shift in 0..SPECIAL.len() {
        let x = inputs(n, shift);
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&x, &mut w, &mut o);
        let (mut wn, mut on) = (Vec::new(), Vec::new());
        native.eval(&x, &mut wn, &mut on);
        for (k, (a, b)) in o.iter().zip(&on).enumerate() {
            assert!(same(*a, *b), "shift={shift} output {k}: {a:e} vs {b:e}");
        }
    }
}
