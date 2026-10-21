//! A program's ops without a call run in regions of independent pieces
//! (`rsdag::Region`) on an installed pool, bit for bit as serially, in the
//! interpreter and in native code; no piece writes a slot another piece
//! reads or writes.

use std::collections::HashSet;
use std::sync::Arc;

use rsdag::node::CmpOp;
use rsdag::parallel::{self, Lent};
use rsdag::tape::Op;
use rsdag::{ExprId, Graph, Node, ReduceOp, SymbolId, Tape, F64};
use rsdag_jit::NativeTape;

const N: usize = 12000;
const ENTRIES: usize = 3000;

/// An assembly: entries summing products, transcendentals and selects of
/// the inputs, some through a partial sum other entries share, some with
/// a min (a gather) among their terms.
fn assembly() -> (Graph<F64>, Vec<ExprId>, Vec<SymbolId>) {
    let mut g: Graph<F64> = Graph::new();
    let (xs, syms): (Vec<ExprId>, Vec<SymbolId>) = (0..N)
        .map(|k| {
            let e = g.sym(&format!("x{k}"));
            let Node::Symbol(s) = *g.node(e) else {
                unreachable!()
            };
            (e, s)
        })
        .unzip();
    let x = |k: usize| xs[k % N];
    let mut shared = Vec::new();
    for k in 0..40 {
        let p = g.mul(x(3 * k), x(3 * k + 1));
        shared.push(g.sin(p));
    }
    let mut roots = Vec::new();
    for e in 0..ENTRIES {
        let a = g.mul(x(2 * e), x(2 * e + 1));
        let b = g.cos(x(3 * e + 2));
        let c = g.add(x(4 * e + 1), x(4 * e + 3));
        let lt = g.cmp(CmpOp::Lt, x(e), x(e + 9));
        let s = g.select(lt, a, c);
        let mut terms = vec![a, b, s];
        if e % 7 == 0 {
            terms.push(shared[e % shared.len()]);
        }
        if e % 1000 == 999 {
            let m = g.reduce(ReduceOp::Min, vec![a, b, c]);
            terms.push(m);
        }
        let sum = g.reduce(ReduceOp::Sum, terms);
        roots.push(sum);
        if e % 3 == 0 {
            let chained = g.add(sum, c);
            roots.push(g.mul(chained, b));
        }
    }
    (g, roots, syms)
}

fn pool(threads: usize) -> Lent {
    Lent {
        pool: Arc::new(parallel::Workers::new(threads)),
        dispatch: rsdag::parallel::Dispatch::free(threads),
    }
}

fn inputs(shift: f64) -> Vec<f64> {
    (0..N)
        .map(|k| 0.3 + shift + 0.011 * k as f64 - (k % 7) as f64 * 0.09)
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn the_pieces_of_a_region_share_no_slot_one_of_them_writes() {
    let (g, roots, syms) = assembly();
    let tape = Tape::compile(&g, &roots, &syms);
    assert!(!tape.regions().is_empty(), "the assembly is a region");
    for r in tape.regions() {
        assert!(r.starts.len() >= 2);
        let mut writes: Vec<HashSet<u32>> = Vec::new();
        let mut reads: Vec<HashSet<u32>> = Vec::new();
        for k in 0..r.starts.len() {
            let (mut w, mut rd) = (HashSet::new(), HashSet::new());
            for i in r.piece(k) {
                assert!(
                    !matches!(
                        tape.ops()[i],
                        Op::Call { .. }
                            | Op::CallProlog { .. }
                            | Op::Gemv { .. }
                            | Op::Gemm { .. }
                            | Op::Solve { .. }
                            | Op::Reduce(ReduceOp::Min | ReduceOp::Max, ..)
                    ),
                    "op {i} is no op of a region"
                );
                tape.for_each_operand(i, |s| {
                    if rsdag::tape::input_index(s).is_none() {
                        rd.insert(s);
                    }
                });
                w.insert(tape.dst(i));
            }
            writes.push(w);
            reads.push(rd);
        }
        for p in 0..writes.len() {
            for q in 0..writes.len() {
                if p != q {
                    assert!(
                        writes[p].is_disjoint(&writes[q]),
                        "pieces {p}, {q} write one slot"
                    );
                    assert!(
                        writes[p].is_disjoint(&reads[q]),
                        "piece {q} reads what {p} writes"
                    );
                }
            }
        }
    }
}

#[test]
fn a_region_on_a_pool_is_the_serial_one() {
    let (g, roots, syms) = assembly();
    let tape = Tape::compile(&g, &roots, &syms);
    let native = NativeTape::compile(&tape).expect("native code");
    for shift in [0.0, 0.37, -1.2] {
        let x = inputs(shift);
        let (mut w, mut out, mut trace) = (Vec::new(), Vec::new(), Vec::<u8>::new());
        tape.eval_with(&x, &mut w, &mut out, &mut trace);
        let (mut wn, mut on) = (Vec::new(), Vec::new());
        native.eval(&x, &mut wn, &mut on);
        assert_eq!(bits(&out), bits(&on), "native serial");
        for threads in [2, 5, 12] {
            let (mut wp, mut op, mut wq, mut oq) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut tp = Vec::<u8>::new();
            parallel::install_lent(pool(threads), || {
                for _ in 0..3 {
                    tape.eval(&x, &mut wp, &mut op);
                    native.eval(&x, &mut wq, &mut oq);
                }
                tape.eval_with(&x, &mut wp, &mut op, &mut tp);
            });
            assert_eq!(bits(&out), bits(&op), "interpreter, {threads} threads");
            assert_eq!(bits(&out), bits(&oq), "native, {threads} threads");
            assert_eq!(trace, tp, "the select trace in op order, {threads} threads");
        }
    }
}
