//! The C backend against the interpreter: bit-exact on the ring and the
//! helper routines, within a tolerance where the platform `libm` is called.

#[path = "../../rsgb/tests/common/mod.rs"]
mod common;

use rsgb::{ExprId, Graph, Node, ReduceOp, SymbolId, Tape};
use rsgb_c::verify::{find_compiler, run_c};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn val(&mut self) -> f64 {
        (self.next() % 20000) as f64 / 1000.0 - 10.0
    }
}

fn sym_id(g: &Graph, e: ExprId) -> SymbolId {
    match *g.node(e) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    }
}

/// Random graphs over the classic ring ops plus reductions (bit-exact set)
/// or the full vocabulary.
fn build(g: &mut Graph, rng: &mut Rng, syms: &[ExprId], steps: usize, ext: bool) -> ExprId {
    let mut pool: Vec<ExprId> = syms.to_vec();
    for _ in 0..2 {
        let k = g.konst_f64(rng.val());
        pool.push(k);
    }
    for _ in 0..steps {
        let a = pool[rng.below(pool.len())];
        let b = pool[rng.below(pool.len())];
        let c = pool[rng.below(pool.len())];
        let list = |rng: &mut Rng, pool: &[ExprId], n: usize| -> Vec<ExprId> {
            (0..n).map(|_| pool[rng.below(pool.len())]).collect()
        };
        let (idx, is_ext) = common::draw_op(&mut |n| rng.below(n), 12, 12, false, ext);
        let e = if is_ext {
            common::ext_op(g, idx, a, b)
        } else {
            match idx {
                0 => g.add(a, b),
                1 => g.sub(a, b),
                2 => g.mul(a, b),
                3 => g.neg(a),
                4 => g.pow_i(a, rng.below(4) as i64 + 1),
                5 => g.select(a, b, c),
                6 => g.cmp(rsgb::CmpOp::Lt, a, b),
                7 => g.reduce(ReduceOp::Sum, {
                    let n = 2 + rng.below(3);
                    list(rng, &pool, n)
                }),
                8 => g.reduce(ReduceOp::Sum, {
                    let n = 17 + rng.below(4);
                    list(rng, &pool, n)
                }),
                9 => g.reduce(ReduceOp::Product, {
                    let n = 2 + rng.below(2);
                    list(rng, &pool, n)
                }),
                10 => {
                    let n = 2 + rng.below(3);
                    let la = list(rng, &pool, n);
                    let lb = list(rng, &pool, n);
                    g.dot(la, lb)
                }
                _ => g.reduce(ReduceOp::Max, {
                    let n = 2 + rng.below(3);
                    list(rng, &pool, n)
                }),
            }
        };
        pool.push(e);
    }
    *pool.last().unwrap()
}

fn cases(ext: bool, seeds: std::ops::Range<u64>) -> (usize, usize) {
    let Some(cc) = find_compiler() else {
        eprintln!("no C compiler on the path, skipping");
        return (0, 0);
    };
    let mut exact = 0;
    let mut close = 0;
    for seed in seeds {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut g: Graph = Graph::new();
        let nsym = 1 + rng.below(3);
        let syms: Vec<ExprId> = (0..nsym).map(|i| g.sym(&format!("x{i}"))).collect();
        let ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&g, e)).collect();
        let steps = 4 + rng.below(16);
        let root = build(&mut g, &mut rng, &syms, steps, ext);
        let tape = Tape::compile(&g, &[root], &ids);
        let inputs: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..nsym).map(|_| rng.val()).collect())
            .collect();
        let got = run_c(&cc, &tape, &inputs).expect("C build");
        let (mut w, mut o) = (Vec::new(), Vec::new());
        for (row, cr) in inputs.iter().zip(&got) {
            tape.eval(row, &mut w, &mut o);
            let (a, b) = (o[0], cr[0]);
            let mut env = std::collections::HashMap::new();
            for (k, &sid) in ids.iter().enumerate() {
                env.insert(sid, row[k]);
            }
            let arena = rsgb::eval_real(&g, &env, &[root])[0];
            assert!(
                arena.to_bits() == a.to_bits() || (arena.is_nan() && a.is_nan()),
                "seed {seed}: arena {arena:?} vs interpreter {a:?}"
            );
            if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
                exact += 1;
            } else if (a - b).abs() <= 1e-12 * (1.0 + a.abs()) {
                close += 1;
            } else {
                panic!(
                    "seed {seed}: inputs {row:?} interpreter {a:?} vs C {b:?}\n{}\n{}",
                    tape.dump(),
                    rsgb_c::emit(&tape, "f").unwrap()
                );
            }
        }
    }
    (exact, close)
}

#[test]
fn ring_and_helpers_are_bit_exact() {
    let (exact, close) = cases(false, 1..40);
    assert_eq!(close, 0, "ring ops must be bit-exact ({exact} exact)");
}

#[test]
fn full_vocabulary_is_within_libm_tolerance() {
    let (exact, close) = cases(true, 100..140);
    assert!(
        exact + close > 0 || find_compiler().is_none(),
        "no cases ran"
    );
    eprintln!("exact {exact}, within tolerance {close}");
}
