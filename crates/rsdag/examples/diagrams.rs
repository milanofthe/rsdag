//! The README's diagrams, drawn by rsdag itself: DOT files for
//! `scripts/diagrams.sh`, which renders them to `docs/diagrams/*.svg`.
//!
//!   cargo run -p rsdag --example diagrams -- <out dir>

use std::collections::HashMap;
use std::fmt::Write;

use rsdag::dot::{reachable, Blocks, GraphView, Kind, TapeView, Theme};
use rsdag::symbolic::solve::{plan, LuProgram, Pattern};
use rsdag::{differentiate, CmpOp, ExprId, Graph, Node, ParamRole, SymbolId, Tape, F64};

fn sym(g: &Graph<F64>, e: ExprId) -> SymbolId {
    match g.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!("a symbol"),
    }
}

/// Graph, transforms, tape, the two backends, the solver driving them.
fn pipeline() -> String {
    Blocks::new(Theme::default(), "LR")
        .block(
            "dag",
            "Graph",
            &[
                "hash-consed expression DAG",
                "over exact rationals or f64",
                "constructors fold",
                "functions with calls and roles",
            ],
        )
        .block(
            "tf",
            "Transforms",
            &[
                "differentiate: forward, reverse",
                "sparse_jacobian, hessian",
                "substitute, inline, simplify",
                "sparse solve, Newton step",
            ],
        )
        .block(
            "tape",
            "Tape",
            &[
                "flat instruction IR",
                "scheduled for register pressure",
                "kernels: Gemv, Gemm, Solve",
                "batched calls of bodies",
                "prolog / main split",
            ],
        )
        .block(
            "interp",
            "Interpreter",
            &["one loop over any Scalar:", "f64, f32, Complex64"],
        )
        .block(
            "native",
            "Native code",
            &["AArch64, x86-64", "chunked functions", "bit-identical"],
        )
        .block(
            "solver",
            "Solver",
            &["Newton, ODE, sweeps", "through Program", "or Adaptive"],
        )
        .edge("dag", "tf", "")
        .edge("tf", "tape", "compile")
        .edge("tape", "interp", "")
        .edge("tape", "native", "")
        .edge("interp", "solver", "")
        .edge("native", "solver", "")
        .caption("semantics: one reference arithmetic every backend mirrors")
        .render()
}

/// How the graph and tape diagrams draw a node, and the state edge.
fn legend() -> String {
    let t = Theme::default();
    let mut s = t.header("LR");
    let kinds = [
        (Kind::Input, "input"),
        (Kind::Const, "constant"),
        (Kind::Op, "operation"),
        (Kind::Choice, "branch"),
        (Kind::Kernel, "kernel"),
        (Kind::Call, "call"),
        (Kind::Output, "result"),
    ];
    // One row: the kinds chained by invisible edges, then the state edge.
    for (k, (kind, label)) in kinds.iter().enumerate() {
        let _ = writeln!(s, "  k{k} [{}];", t.node(*kind, label, false));
        if k > 0 {
            let _ = writeln!(s, "  k{} -> k{k} [style=invis];", k - 1);
        }
    }
    let _ = writeln!(
        s,
        "  k{} -> s0 [style=invis];\n  s0 [label=\"prolog\", shape=plaintext];\n  \
         s1 [label=\"main\", shape=plaintext];\n  \
         s0 -> s1 [style=dashed, color=\"{c}\", fontcolor=\"{c}\", label=\"state\", minlen=2];",
        kinds.len() - 1,
        c = t.state
    );
    s.push_str("}\n");
    s
}

/// Planning a sparse solve, and the program it builds.
fn solve() -> String {
    Blocks::new(Theme::default(), "LR")
        .pattern(
            "pat",
            &["x..x..", ".x..xx", "x.x...", ".x.x..", "..x.x.", "...x.x"],
            "pattern fixed at build time",
        )
        .block("btf", "BTF", &["block triangular", "form"])
        .block("amd", "AMD", &["minimum degree", "ordering"])
        .block(
            "cost",
            "cost predictor",
            &["fill and flops from", "the elimination tree"],
        )
        .block(
            "lu",
            "static LU",
            &["Crout form,", "guarded pivots", "as graph ops"],
        )
        .block("prolog", "prolog", &["factorization", "over the entries"])
        .block(
            "main",
            "main",
            &["substitution", "over the right-hand side"],
        )
        .group("tape", &["prolog", "main"])
        .note(
            "guard",
            "pivot guard",
            &[
                "|pivot| >= 1e-3 max|column|",
                "fails: Plan::repivot on the values, rebuild",
            ],
        )
        .edge("pat", "btf", "")
        .edge("btf", "amd", "")
        .edge("amd", "cost", "")
        .edge("cost", "lu", "")
        .edge("lu", "prolog", "LuProgram")
        .accent_edge("prolog", "main", "state")
        .accent_edge("lu", "guard", "")
        .raw("{ rank=same; lu; guard; }")
        .render()
}

/// A function body over many instances.
fn bodies_concept() -> String {
    Blocks::new(Theme::default(), "LR")
        .block("i1", "instance 1", &["args x1, p1"])
        .block("i2", "instance 2", &["args x2, p2"])
        .text("dots", &["..."])
        .block("in", "instance N", &["args xN, pN"])
        .block(
            "batch",
            "batched call",
            &[
                "one op, all instances",
                "serially or on the",
                "current thread pool",
            ],
        )
        .block(
            "body",
            "body",
            &["one tape,", "compiled once,", "called per instance"],
        )
        .note(
            "pro",
            "prolog per instance",
            &[
                "over its parameter-pure",
                "arguments, in the",
                "caller's prolog",
            ],
        )
        .edge("i1", "batch", "")
        .edge("i2", "batch", "")
        .raw("dots -> batch [style=invis];")
        .edge("in", "batch", "")
        .edge("batch", "body", "runs the body")
        .accent_edge("body", "pro", "")
        .raw("{ rank=same; body; pro; }")
        .render()
}

/// Specializing a tape on the arms its selects took.
fn specialize_concept() -> String {
    Blocks::new(Theme::default(), "LR")
        .block(
            "tape",
            "tape",
            &[
                "v1 = mul x, p",
                "**v2 = select c1, v1, x**",
                "v3 = exp v2",
                "**v4 = select c2, v3, p**",
                "v5 = add v4, v1",
                "**v6 = select c3, v5, v3**",
            ],
        )
        .block(
            "trace",
            "trace",
            &["eval_with records", "the arm taken:", "c1=1, c2=0, c3=1"],
        )
        .block(
            "spec",
            "specialized tape",
            &["v1 = mul x, p", "v3 = exp v1", "v5 = add p, v1"],
        )
        .note(
            "guards",
            "guards",
            &["c1 == 1, c2 == 0, c3 == 1", "checked on every eval"],
        )
        .edge("tape", "trace", "eval")
        .edge("trace", "spec", "specialize")
        .accent_edge("spec", "guards", "")
        .accent_edge("guards", "trace", "fails: retrace")
        .raw("{ rank=same; spec; guards; }")
        .render()
}

/// `f = sin(x y) + x y` and its derivative in `x`: the derivative's nodes,
/// its own and the ones it shares with `f`, at full strength, the nodes
/// only `f` reads faded.
fn derivative() -> String {
    let mut g: Graph<F64> = Graph::new();
    let (x, y) = (g.sym("x"), g.sym("y"));
    let xy = g.mul(x, y);
    let s = g.sin(xy);
    let f = g.add(s, xy);
    let wrt = sym(&g, x);
    let df = differentiate(&mut g, f, wrt);
    GraphView::new(&g)
        .root(f, "f")
        .root(df, "df/dx")
        .focus(reachable(&g, &[df]))
        .render()
}

/// A device with parameters `is`, `n`, `vt` and state `v` as a split tape:
/// what depends on the parameters only is the prolog, and what the main
/// phase reads of it the state.
fn split() -> String {
    let mut g: Graph<F64> = Graph::new();
    let (v, is, n, vt) = (g.sym("v"), g.sym("is"), g.sym("n"), g.sym("vt"));
    let nvt = g.mul(n, vt);
    let u = g.div(v, nvt);
    let e = g.exp(u);
    let one = g.one();
    let em1 = g.sub(e, one);
    let i = g.mul(is, em1);
    let wrt = sym(&g, v);
    let di = differentiate(&mut g, i, wrt);
    let syms = [sym(&g, v), sym(&g, is), sym(&g, n), sym(&g, vt)];
    let pure = [false, true, true, true];
    let tape = Tape::compile_split(&g, &[i, di], &syms, &pure);
    TapeView::new(&tape)
        .inputs(&["v", "is", "n", "vt"])
        .params(&pure)
        .outputs(&["i", "di/dv"])
        .render()
}

/// Three instances of one body in a ring: the body's parameter-pure part
/// runs once per instance in the prolog, all instances in one batched call.
fn bodies_tape() -> String {
    let mut g: Graph<F64> = Graph::new();
    let (a, b, is, n) = (g.sym("a"), g.sym("b"), g.sym("is"), g.sym("n"));
    let vt = g.konst_f64(0.025);
    let nvt = g.mul(n, vt);
    let d = g.sub(a, b);
    let u = g.div(d, nvt);
    let e = g.exp(u);
    let one = g.one();
    let em1 = g.sub(e, one);
    let body = g.mul(is, em1);
    let params = vec![sym(&g, a), sym(&g, b), sym(&g, is), sym(&g, n)];
    let f = g.define_func("diode", params, vec![body]);
    g.set_param_role(f, 2, ParamRole::Param);
    g.set_param_role(f, 3, ParamRole::Param);
    let v: Vec<ExprId> = (0..3).map(|k| g.sym(&format!("v{k}"))).collect();
    let p: Vec<ExprId> = (0..3).map(|k| g.sym(&format!("is{k}"))).collect();
    let nn = g.sym("n");
    let calls: Vec<ExprId> = (0..3)
        .map(|k| g.call(f, 0, &[v[k], v[(k + 1) % 3], p[k], nn]))
        .collect();
    let kcl: Vec<ExprId> = (0..3)
        .map(|k| g.sub(calls[k], calls[(k + 2) % 3]))
        .collect();
    let mut syms: Vec<SymbolId> = v.iter().map(|&e| sym(&g, e)).collect();
    syms.extend(p.iter().map(|&e| sym(&g, e)));
    syms.push(sym(&g, nn));
    let pure = [false, false, false, true, true, true, true];
    let tape = Tape::compile_split(&g, &kcl, &syms, &pure);
    TapeView::new(&tape)
        .inputs(&["v0", "v1", "v2", "is0", "is1", "is2", "n"])
        .params(&pure)
        .outputs(&["kcl0", "kcl1", "kcl2"])
        .bundle(0, "diode")
        .render()
}

/// A piecewise model at one operating point: the arms a specialization
/// keeps at full strength, the ones it drops faded.
fn specialize_graph() -> String {
    let mut g: Graph<F64> = Graph::new();
    let (v, vth, vsat, k) = (g.sym("v"), g.sym("vth"), g.sym("vsat"), g.sym("k"));
    let vo = g.sub(v, vth);
    let lin = g.mul(k, vo);
    let sq = g.mul(vo, vo);
    let half = g.konst_f64(0.5);
    let ks = g.mul(k, half);
    let sat = g.mul(ks, sq);
    let on = g.cmp(CmpOp::Gt, v, vth);
    let hi = g.cmp(CmpOp::Gt, vo, vsat);
    let conducting = g.select(hi, lin, sat);
    let zero = g.zero();
    let i = g.select(on, conducting, zero);
    let point: HashMap<SymbolId, f64> = [(v, 1.0), (vth, 0.4), (vsat, 0.3), (k, 2.0)]
        .into_iter()
        .map(|(e, x)| (sym(&g, e), x))
        .collect();
    // The nodes the program keeps: a select's condition (its guard) and
    // the arm taken.
    let mut keep = Vec::new();
    let mut stack = vec![i];
    while let Some(e) = stack.pop() {
        if keep.contains(&e) {
            continue;
        }
        keep.push(e);
        match *g.node(e) {
            Node::Select(c, t, f) => {
                let taken = rsdag::eval::<f64, F64>(&g, &[c], &point)[0] != 0.0;
                stack.push(c);
                stack.push(if taken { t } else { f });
            }
            _ => stack.extend(g.operands(e).iter().copied()),
        }
    }
    GraphView::new(&g)
        .params(&[sym(&g, vth), sym(&g, vsat), sym(&g, k)])
        .root(i, "i")
        .focus(keep)
        .render()
}

/// The LU of a 3 by 3 arrow pattern as a program: the factorization over
/// the entries is the prolog, the substitution the main phase, the pivot
/// guard an output.
fn lu() -> String {
    let n = 3;
    let mut entries = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j || i == n - 1 || j == n - 1 {
                entries.push((i, j));
            }
        }
    }
    let mut pattern: Pattern = vec![Vec::new(); n];
    for &(i, j) in &entries {
        pattern[i].push(j);
    }
    let names: Vec<String> = entries
        .iter()
        .map(|(i, j)| format!("a{i}{j}"))
        .chain((0..n).map(|i| format!("b{i}")))
        .collect();
    let lu = LuProgram::build(n, entries, plan(&pattern).expect("a plan"), None);
    let names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let mut pure = vec![true; lu.entries().len()];
    pure.resize(lu.input_len(), false);
    TapeView::new(lu.tape())
        .inputs(&names)
        .params(&pure)
        .outputs(&["x0", "x1", "x2", "pivots ok"])
        .render()
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/diagrams".into());
    std::fs::create_dir_all(&dir).expect("create the output directory");
    for (name, dot) in [
        ("pipeline", pipeline()),
        ("legend", legend()),
        ("derivative", derivative()),
        ("split", split()),
        ("bodies", bodies_concept()),
        ("bodies_tape", bodies_tape()),
        ("specialize", specialize_concept()),
        ("specialize_graph", specialize_graph()),
        ("solve", solve()),
        ("lu", lu()),
    ] {
        let path = format!("{dir}/{name}.dot");
        std::fs::write(&path, dot).expect("write a diagram");
        println!("{path}");
    }
}
