//! Lane-batched (SIMD) parity: for random DAGs, evaluating TWO independent
//! input sets through `LaneTape` must match two scalar `Tape::eval` passes
//! per lane, within floating-point codegen freedom (see `close`).

use rsgb::node::Node;
use rsgb::{ExprId, Graph, ReduceOp, SymbolId, Tape};
use rsgb_jit::{LaneTape, LANES};

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
        let e = match rng.below(21) {
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
                let lb: Vec<ExprId> = (0..la.len()).map(|_| pool[rng.below(pool.len())]).collect();
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
        };
        pool.push(e);
    }
    let n_roots = 1 + rng.below(4).min(pool.len() - 1);
    (0..n_roots)
        .map(|_| pool[pool.len() - 1 - rng.below(n_roots)])
        .collect()
}

#[test]
fn lane_tape_matches_scalar_tape_per_lane() {
    #[allow(clippy::unusual_byte_groupings)] // mnemonic seed
    let mut rng = Rng(0x1a4e_5eed_0dd_b1a5);
    for case in 0..250 {
        let mut ctx: Graph = Graph::new();
        let syms: Vec<ExprId> = ["x", "y", "z"].iter().map(|n| ctx.sym(n)).collect();
        let steps = 6 + rng.below(30);
        let roots = build(&mut ctx, &mut rng, &syms, steps);

        let sym_ids: Vec<SymbolId> = syms.iter().map(|&s| sym_id(&ctx, s)).collect();
        let tape = Tape::compile(&ctx, &roots, &sym_ids);
        let chunk_ops = [3, 7, rsgb_jit::CHUNK_OPS][case % 3];
        let lanes: Vec<LaneTape> =
            vec![LaneTape::compile_with(&tape, chunk_ops).expect("pair compile")];

        let (mut ws, mut os) = (Vec::new(), Vec::new());
        let (mut wl, mut ol) = (Vec::new(), Vec::new());
        for _ in 0..6 {
            for lane in &lanes {
                let w = lane.width();
                let sets: Vec<Vec<f64>> = (0..w)
                    .map(|_| (0..syms.len()).map(|_| rng.val()).collect())
                    .collect();
                let mut inter = Vec::with_capacity(syms.len() * w);
                for k in 0..syms.len() {
                    for set in &sets {
                        inter.push(set[k]);
                    }
                }
                lane.eval(&inter, &mut wl, &mut ol);
                for (l, args) in sets.iter().enumerate() {
                    tape.eval(args, &mut ws, &mut os);
                    for j in 0..roots.len() {
                        assert!(
                            close(os[j], ol[j * w + l]),
                            "case {case}, width {w}, lane {l}, out {j}: scalar {:?} vs lane {:?}",
                            os[j],
                            ol[j * w + l]
                        );
                    }
                }
            }
        }
    }
}

/// The prolog split carries over: pure inputs prepared once, main re-run per
/// impure binding, per lane.
#[test]
fn lane_tape_split_prolog_matches() {
    let mut ctx: Graph = Graph::new();
    let (p, x) = (ctx.sym("p"), ctx.sym("x"));
    let ep = ctx.exp(p);
    let mx = ctx.mul(ep, x);
    let root = ctx.sin(mx);
    let syms = [sym_id(&ctx, p), sym_id(&ctx, x)];
    let tape = Tape::compile_split(&ctx, &[root], &syms, &[true, false]);
    let lane = LaneTape::compile(&tape).expect("lane compile");

    let (mut ws, mut os) = (Vec::new(), Vec::new());
    let (mut wl, mut ol) = (Vec::new(), Vec::new());
    let inter0 = [0.3, 0.7, 1.1, -0.4]; // p lanes, x lanes
    lane.eval_prolog(&inter0, &mut wl);
    for step in 0..5 {
        let xs = [1.1 + step as f64, -0.4 * step as f64];
        let inter = [0.3, 0.7, xs[0], xs[1]];
        lane.eval_main(&inter, &mut wl, &mut ol);
        for l in 0..LANES {
            let args = [inter[l], xs[l]];
            tape.eval(&args, &mut ws, &mut os);
            assert!(
                close(os[0], ol[l]),
                "step {step} lane {l}: {:?} vs {:?}",
                os[0],
                ol[l]
            );
        }
    }
}
