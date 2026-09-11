//! Tape compile time of a circuit-like solve program, best of ten.
use rsdag::symbolic::solve::{pattern_of, plan, solve_planned};
use rsdag::{Graph, Node, SymbolId, Tape, F64};
use std::time::Instant;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(42);
    let mut g: Graph<F64> = Graph::new();
    let mut syms: Vec<SymbolId> = Vec::new();
    let mut rows: rsdag::symbolic::solve::SparseRows = vec![Vec::new(); n];
    let sym = |g: &mut Graph<F64>, name: String, syms: &mut Vec<SymbolId>| {
        let e = g.sym(&name);
        if let Node::Symbol(s) = g.node(e) {
            syms.push(*s);
        }
        e
    };
    for i in 0..n {
        let mut cols = vec![i, (i + 1) % n, (i * 7 + 3) % n];
        cols.sort_unstable();
        cols.dedup();
        for j in cols {
            let e = sym(&mut g, format!("a{i}_{j}"), &mut syms);
            rows[i].push((j, e));
        }
    }
    let nnz = syms.len();
    let b: Vec<_> = (0..n)
        .map(|i| sym(&mut g, format!("b{i}"), &mut syms))
        .collect();
    let plan = plan(&pattern_of(&rows)).expect("plan");
    let solved = solve_planned(&mut g, &rows, &plan, &b);
    let mut roots = solved.x;
    roots.push(solved.pivots_ok);
    let mut pure = vec![true; nnz];
    pure.resize(nnz + n, false);
    let mut best = f64::INFINITY;
    let mut ops = 0;
    for _ in 0..10 {
        let t = Instant::now();
        let tape = Tape::compile_split(&g, &roots, &syms, &pure);
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
        ops = tape.n_ops();
    }
    println!("n={n} ops={ops} compile_split best {best:.0} us");
}
