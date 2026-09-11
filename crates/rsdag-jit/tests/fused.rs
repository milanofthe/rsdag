//! The accumulating kernels natively: the same bits as the interpreter on
//! a block solve (subtractions folded into the products) and on the
//! complex product pattern (self folds).

use rsdag::symbolic::solve::{block_pattern, plan, solve_block_planned, Block, BlockRows, Cx};
use rsdag::{ExprId, Graph, Node, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

fn sym(g: &mut Graph<F64>, name: &str, syms: &mut Vec<SymbolId>) -> ExprId {
    let e = g.sym(name);
    if let Node::Symbol(s) = g.node(e) {
        syms.push(*s);
    }
    e
}

#[test]
fn a_complex_block_solve_runs_natively_with_folded_kernels() {
    let (nb, b) = (4usize, 6usize);
    let mut g: Graph<F64> = Graph::new();
    let mut syms = Vec::new();
    let mut vals = Vec::new();
    let mut rows: BlockRows<Cx> = vec![Vec::new(); nb];
    for i in 0..nb {
        for j in [i, (i + 1) % nb] {
            let mut blk = Vec::new();
            for r in 0..b {
                for c in 0..b {
                    let re = sym(&mut g, &format!("a{i}_{j}_{r}_{c}r"), &mut syms);
                    let im = sym(&mut g, &format!("a{i}_{j}_{r}_{c}i"), &mut syms);
                    vals.push(
                        ((i * 3 + j + r * 7 + c) % 5) as f64 * 0.2 - 0.4
                            + if i == j && r == c { 5.0 } else { 0.0 },
                    );
                    vals.push(
                        ((i + j * 2 + r + c * 3) % 7) as f64 * 0.1 - 0.3
                            + if i == j && r == c { 2.0 } else { 0.0 },
                    );
                    blk.push(Cx::new(re, im));
                }
            }
            rows[i].push((j, Block::Dense(blk)));
        }
    }
    let n_entries = vals.len();
    let rhs: Vec<Cx> = (0..nb * b)
        .map(|k| {
            let re = sym(&mut g, &format!("b{k}r"), &mut syms);
            let im = sym(&mut g, &format!("b{k}i"), &mut syms);
            vals.push(1.0 + 0.1 * k as f64);
            vals.push(-0.5);
            Cx::new(re, im)
        })
        .collect();
    let plan = plan(&block_pattern(&rows)).expect("plan");
    let solved = solve_block_planned(&mut g, &rows, b, &plan, &rhs);
    let mut roots: Vec<ExprId> = solved.x.iter().flat_map(|c| [c.re, c.im]).collect();
    roots.push(solved.pivots_ok);
    let mut pure = vec![true; n_entries];
    pure.resize(vals.len(), false);
    let tape = Tape::compile_split(&g, &roots, &syms, &pure);
    let d = tape.dump();
    assert!(d.contains(" acc "), "{d}");
    let native = NativeTape::compile(&tape).expect("compile");
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&vals, &mut w, &mut o);
    let (mut wn, mut on) = (Vec::new(), Vec::new());
    native.eval(&vals, &mut wn, &mut on);
    assert_eq!(o.len(), on.len());
    for (k, (a, b)) in o.iter().zip(&on).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {k}: {a} vs {b}");
    }
    // The prolog and main split evaluate the same.
    native.eval_prolog(&vals, &mut wn);
    native.eval_main(&vals, &mut wn, &mut on);
    for (a, b) in o.iter().zip(&on) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
