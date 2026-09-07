//! Three-leg differential parity for the chunked JIT.
//!
//! For many random hash-consed DAGs, arena sweep (`eval_real`), tape
//! interpreter (`Tape::eval`) and chunked native code (`ChunkedTape::eval`)
//! must agree over random inputs, within floating-point codegen freedom
//! (see `close`). Chunk sizes down to 3 ops
//! force values across chunk boundaries constantly, so the work-array
//! store-through contract is exercised hard, not incidentally.

#[path = "../../rsgb/tests/common/mod.rs"]
mod common;
use std::collections::HashMap;

use rsgb::node::Node;
use rsgb::{eval_real, ExprId, Graph, ReduceOp, SymbolId, Tape};
use rsgb_jit::ChunkedTape;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn val(&mut self) -> f64 {
        (self.next_u64() % 5001) as f64 / 1000.0 - 2.5
    }
}

fn sym_id(ctx: &Graph, e: ExprId) -> SymbolId {
    match ctx.node(e) {
        Node::Symbol(s) => *s,
        _ => unreachable!(),
    }
}

/// Parity within floating-point codegen freedom. Bit-exactness is the wrong
/// metric across differently generated code: Cranelift's instruction selection
/// is host-ISA dependent (observed: 1-ULP drift vs the rustc-compiled scalar
/// interpreter on an AVX-512 host, none on the CI runners), and the random
/// DAGs amplify a mid-chain ULP through cancellation and exp chains (observed
/// worst on the fixed seed: 3.5e-11 relative). A genuine lowering, lane or
/// chunk-boundary bug reads a wrong slot and lands at O(1) relative error --
/// seven orders above this bound. NaN must match NaN; a finite/non-finite
/// mismatch always fails.
fn close(a: f64, b: f64) -> bool {
    if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a - b).abs() <= 1e-9 + 1e-9 * a.abs().max(b.abs())
}

/// Random DAG over `syms`, full op set (cmp/select/floor/min/max/dot included).
fn build(ctx: &mut Graph, rng: &mut Rng, syms: &[ExprId], steps: usize) -> Vec<ExprId> {
    let mut pool: Vec<ExprId> = syms.to_vec();
    for _ in 0..2 {
        let k = ctx.konst_f64(rng.val());
        pool.push(k);
    }
    let rand_list = |rng: &mut Rng, pool: &[ExprId]| -> Vec<ExprId> {
        let k = 2 + rng.below(3);
        (0..k).map(|_| pool[rng.below(pool.len())]).collect()
    };
    for _ in 0..steps {
        let a = pool[rng.below(pool.len())];
        let b = pool[rng.below(pool.len())];
        let c = pool[rng.below(pool.len())];
        let (idx, ext) = common::draw_op(&mut |n| rng.below(n), 21, 16, false, true);
        let e = if ext {
            common::ext_op(ctx, idx, a, b)
        } else {
            match idx {
                0 => ctx.add(a, b),
                1 => ctx.sub(a, b),
                2 => ctx.mul(a, b),
                3 => ctx.neg(a),
                4 => {
                    let mut k = rng.below(6) as i64 - 2;
                    if k < 0 && ctx.is_zero(a) {
                        k = 2;
                    }
                    ctx.pow_i(a, k)
                }
                5 => ctx.exp(a),
                6 => ctx.ln(if ctx.is_zero(a) { syms[0] } else { a }),
                7 => ctx.sqrt(if ctx.is_zero(a) { syms[0] } else { a }),
                8 => ctx.sin(a),
                9 => ctx.cos(a),
                10 => ctx.sinh(a),
                11 => ctx.cosh(a),
                12 => ctx.tanh(a),
                13 => ctx.reduce(ReduceOp::Sum, rand_list(rng, &pool)),
                14 => ctx.reduce(ReduceOp::Product, rand_list(rng, &pool)),
                15 => {
                    let la = rand_list(rng, &pool);
                    let lb: Vec<ExprId> =
                        (0..la.len()).map(|_| pool[rng.below(pool.len())]).collect();
                    ctx.dot(la, lb)
                }
                16 => {
                    let op = [rsgb::CmpOp::Gt, rsgb::CmpOp::Le, rsgb::CmpOp::Lt][rng.below(3)];
                    ctx.cmp(op, a, b)
                }
                17 => ctx.select(a, b, c),
                18 => ctx.floor(a),
                19 => ctx.reduce(ReduceOp::Min, rand_list(rng, &pool)),
                _ => ctx.reduce(ReduceOp::Max, rand_list(rng, &pool)),
            }
        };
        pool.push(e);
    }
    // Several roots so the output-copy path is exercised too.
    let n_roots = 1 + rng.below(4).min(pool.len() - 1);
    (0..n_roots)
        .map(|_| pool[pool.len() - 1 - rng.below(n_roots)])
        .collect()
}

/// arena == tape == chunked JIT across random DAGs, random inputs and
/// adversarial chunk sizes.
#[test]
fn chunked_jit_matches_arena_and_tape() {
    #[allow(clippy::unusual_byte_groupings)] // mnemonic seed
    let mut rng = Rng(0x0dd_b1a5_ed_c0ffee);
    for case in 0..250 {
        let mut ctx: Graph = Graph::new();
        let syms: Vec<ExprId> = ["x", "y", "z"].iter().map(|n| ctx.sym(n)).collect();
        let steps = 6 + rng.below(30);
        let roots = build(&mut ctx, &mut rng, &syms, steps);

        let sym_ids: Vec<SymbolId> = syms.iter().map(|&s| sym_id(&ctx, s)).collect();
        let tape = Tape::compile(&ctx, &roots, &sym_ids);
        // Tiny chunks (3 ops) force cross-chunk traffic; the default size is
        // covered by one compile per case as well.
        let chunk_ops = [3, 7, rsgb_jit::CHUNK_OPS][case % 3];
        let jit = ChunkedTape::compile_with(&tape, chunk_ops).expect("compile chunks");

        let (mut w1, mut o1) = (Vec::new(), Vec::new());
        let (mut w2, mut o2) = (Vec::new(), Vec::new());
        for _ in 0..8 {
            let args: Vec<f64> = (0..syms.len()).map(|_| rng.val()).collect();
            let want = {
                let env: HashMap<SymbolId, f64> =
                    sym_ids.iter().copied().zip(args.iter().copied()).collect();
                eval_real(&ctx, &env, &roots)
            };
            tape.eval(&args, &mut w1, &mut o1);
            jit.eval(&args, &mut w2, &mut o2);
            for j in 0..roots.len() {
                assert!(
                    close(want[j], o1[j]),
                    "case {case}: arena vs tape, out {j}: {:?} vs {:?}",
                    want[j],
                    o1[j]
                );
                assert!(
                    close(o1[j], o2[j]),
                    "case {case} (chunk {chunk_ops}): tape vs jit, out {j}: {:?} vs {:?}",
                    o1[j],
                    o2[j]
                );
            }
        }
    }
}

/// Specialize-then-compile: a choice-specialized (shortened) tape compiled
/// with the chunked backend must reproduce the interpreted specialization --
/// real outputs and guard outputs alike -- across random DAGs with `Select`s,
/// random inputs, and adversarial chunk sizes.
#[test]
fn compiled_specialized_tape_matches_interpreter() {
    #[allow(clippy::unusual_byte_groupings)] // mnemonic seed
    let mut rng = Rng(0x5bec_1a11_ced_c0de);
    for case in 0..150 {
        let mut ctx: Graph = Graph::new();
        let syms: Vec<ExprId> = ["x", "y", "z"].iter().map(|n| ctx.sym(n)).collect();
        let steps = 10 + rng.below(30);
        let roots = build(&mut ctx, &mut rng, &syms, steps);
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&s| sym_id(&ctx, s)).collect();
        let tape = Tape::compile(&ctx, &roots, &sym_ids);

        let args: Vec<f64> = (0..syms.len()).map(|_| rng.val()).collect();
        let (mut w, mut o, mut choices) = (Vec::new(), Vec::new(), Vec::new());
        tape.eval_traced(&args, &mut w, &mut o, &mut choices);
        let spec = tape.specialize(&choices);
        let jit = ChunkedTape::compile_with(spec.tape(), [3, rsgb_jit::CHUNK_OPS][case % 2])
            .expect("compile spec tape");

        // Raw shortened-tape outputs (real ++ guards) must agree.
        let (mut wi, mut oi) = (Vec::new(), Vec::new());
        let (mut wn, mut on) = (Vec::new(), Vec::new());
        for probe in 0..6 {
            let pargs: Vec<f64> = if probe == 0 {
                args.clone()
            } else {
                (0..syms.len()).map(|_| rng.val()).collect()
            };
            spec.tape().eval(&pargs, &mut wi, &mut oi);
            jit.eval(&pargs, &mut wn, &mut on);
            assert_eq!(oi.len(), on.len());
            for (a, b) in oi.iter().zip(&on) {
                assert!(
                    close(*a, *b),
                    "case {case} probe {probe}: interp {a:?} vs native {b:?}"
                );
            }
        }
    }
}

/// A call into a symbolic function evaluates through its interpreted body on
/// both backends (bit-identical), and a short input array pads to NaN instead
/// of reading out of bounds.
#[test]
fn function_call_and_short_input_parity() {
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let y = ctx.sym("y");
    // f(p) = p*p + 1, applied to x.
    let p = ctx.sym("p");
    let ps = sym_id(&ctx, p);
    let pp = ctx.mul(p, p);
    let one = ctx.one();
    let body = ctx.add(pp, one);
    let f = ctx.define_func("sq1", vec![ps], vec![body]);
    let o = ctx.call(f, 0, &[x]);
    let s = ctx.add(o, y);
    let e = ctx.exp(s);

    let (xi, yi) = (sym_id(&ctx, x), sym_id(&ctx, y));
    let tape = Tape::compile(&ctx, &[e, s], &[xi, yi]);
    let jit = ChunkedTape::compile_with(&tape, 2).expect("compile");

    let (mut w1, mut o1) = (Vec::new(), Vec::new());
    let (mut w2, mut o2) = (Vec::new(), Vec::new());
    tape.eval(&[0.5, 2.0], &mut w1, &mut o1);
    jit.eval(&[0.5, 2.0], &mut w2, &mut o2);
    assert_eq!(o1[1], 0.25 + 1.0 + 2.0);
    assert_eq!(o1[0].to_bits(), o2[0].to_bits());
    assert_eq!(o1[1].to_bits(), o2[1].to_bits());

    // Short input array: interpreter yields NaN for the missing input; the
    // JIT's padding must reproduce that instead of reading out of bounds.
    tape.eval(&[0.5], &mut w1, &mut o1);
    jit.eval(&[0.5], &mut w2, &mut o2);
    assert!(o1[0].is_nan() && o2[0].is_nan());
    assert!(o1[1].is_nan() && o2[1].is_nan());
}
