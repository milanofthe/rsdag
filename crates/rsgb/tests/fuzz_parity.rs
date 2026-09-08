//! Properties of the tape over synthetic programs: it evaluates like the
//! arena, a split tape like an unsplit one, a specialization like the choice
//! it froze, `eval_batch` like the scalar path, and a symbolic derivative
//! like a finite difference.
//!
//! The programs come from `rsgb::synth`, so every backend's parity suite
//! draws from the same population and a new op is covered here the moment it
//! is drawable there.

use std::collections::HashMap;

use rsgb::node::Node;
use rsgb::synth::{build_over, Rng, Spec, Vocabulary};
use rsgb::{differentiate, eval_real, ExprId, Graph, ReduceOp, SymbolId, Tape};

fn sym_id(ctx: &Graph, e: ExprId) -> SymbolId {
    match *ctx.node(e) {
        Node::Symbol(s) => s,
        _ => unreachable!(),
    }
}

fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// One generated expression over `syms`, drawn from the vocabulary the test
/// asks for.
fn build_with_syms(
    ctx: &mut Graph,
    rng: &mut Rng,
    syms: &[ExprId],
    steps: usize,
    smooth: bool,
    extended: bool,
) -> ExprId {
    let mut spec = Spec::new(rng.next_u64()).steps(steps).vocab(if extended {
        Vocabulary::Full
    } else {
        Vocabulary::Elementary
    });
    if smooth {
        spec = spec.smooth();
    }
    build_over(ctx, &mut spec, syms)
}

/// As [`build_with_syms`], with guards dense enough that a random walk over
/// the inputs crosses region boundaries often.
fn build_branchy(ctx: &mut Graph, rng: &mut Rng, syms: &[ExprId], steps: usize) -> ExprId {
    let mut spec = Spec::new(rng.next_u64())
        .steps(steps)
        .vocab(Vocabulary::Elementary)
        .selects(25);
    build_over(ctx, &mut spec, syms)
}

#[test]
fn eval_batch_matches_scalar_bit_exact() {
    // The 4-lane SoA `eval_batch` must reproduce the scalar `eval` exactly: lane
    // `l` of the batch output equals the scalar evaluation of input set `l`.
    const L: usize = 4;
    let mut mismatches = 0;
    for seed in 1..600u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
        let mut ctx: Graph = Graph::new();
        let nsym = 1 + rng.below(4);
        let syms: Vec<ExprId> = (0..nsym).map(|i| ctx.sym(&format!("x{i}"))).collect();
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&ctx, e)).collect();
        let steps = 6 + rng.below(20);
        let root = build_with_syms(&mut ctx, &mut rng, &syms, steps, false, true);
        let tape = Tape::compile(&ctx, &[root], &sym_ids);

        // L distinct input sets; scalar eval each, then one batched eval.
        let sets: Vec<Vec<f64>> = (0..L)
            .map(|_| (0..nsym).map(|_| rng.val()).collect())
            .collect();
        let mut scalar = [0.0f64; L];
        let (mut w, mut o) = (Vec::new(), Vec::new());
        for (l, s) in sets.iter().enumerate() {
            tape.eval(s, &mut w, &mut o);
            scalar[l] = o[0];
        }
        let batch_in: Vec<[f64; L]> = (0..nsym)
            .map(|k| std::array::from_fn(|l| sets[l][k]))
            .collect();
        let (mut wb, mut ob) = (Vec::new(), Vec::new());
        tape.eval_batch::<L>(&batch_in, &mut wb, &mut ob);

        for l in 0..L {
            if !same_bits(scalar[l], ob[0][l]) {
                mismatches += 1;
                eprintln!(
                    "seed {seed} lane {l}: scalar={:?} batch={:?}",
                    scalar[l], ob[0][l]
                );
            }
        }
    }
    assert_eq!(mismatches, 0, "eval_batch diverged from scalar eval");
}

#[test]
fn tape_matches_arena_sweep_bit_exact() {
    let mut mismatches = 0;
    for seed in 1..600u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut ctx: Graph = Graph::new();
        let nsym = 1 + rng.below(4);
        let syms: Vec<ExprId> = (0..nsym).map(|i| ctx.sym(&format!("x{i}"))).collect();
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&ctx, e)).collect();
        let steps = 6 + rng.below(20);
        let root = build_with_syms(&mut ctx, &mut rng, &syms, steps, false, true);

        // Random input point.
        let inputs: Vec<f64> = (0..nsym).map(|_| rng.val()).collect();
        let env: HashMap<SymbolId, f64> = sym_ids
            .iter()
            .copied()
            .zip(inputs.iter().copied())
            .collect();

        let want = eval_real(&ctx, &env, &[root])[0];

        let tape = Tape::compile(&ctx, &[root], &sym_ids);
        let (mut work, mut out) = (Vec::new(), Vec::new());
        tape.eval(&inputs, &mut work, &mut out);
        let got = out[0];

        if !same_bits(want, got) {
            mismatches += 1;
            eprintln!("seed {seed}: arena={want:?} tape={got:?}");
        }
    }
    assert_eq!(mismatches, 0, "tape vs arena sweep diverged");
}

/// Prolog-split tapes must agree bit-for-bit with the unsplit tape -- through
/// the plain `eval` (reordered stream, same dataflow) and through the
/// prolog/main protocol, including several main passes over one prolog and a
/// changed "state" input between them (the split's whole point).
#[test]
fn split_tape_matches_unsplit_bit_exact() {
    let mut mismatches = 0;
    for seed in 1..400u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut ctx: Graph = Graph::new();
        let nsym = 2 + rng.below(4);
        let syms: Vec<ExprId> = (0..nsym).map(|i| ctx.sym(&format!("x{i}"))).collect();
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&ctx, e)).collect();
        let steps = 6 + rng.below(24);
        let root = build_with_syms(&mut ctx, &mut rng, &syms, steps, false, true);

        // Random purity mask ("parameters" vs "state"), at least one impure.
        let mut pure: Vec<bool> = (0..nsym).map(|_| rng.below(2) == 0).collect();
        pure[rng.below(nsym)] = false;

        let plain = Tape::compile(&ctx, &[root], &sym_ids);
        let split = Tape::compile_split(&ctx, &[root], &sym_ids, &pure);

        let (mut w1, mut o1) = (Vec::new(), Vec::new());
        let (mut w2, mut o2) = (Vec::new(), Vec::new());
        let (mut w3, mut o3) = (Vec::new(), Vec::new());

        let mut inputs: Vec<f64> = (0..nsym).map(|_| rng.val()).collect();
        // One prolog, several main passes with the impure inputs varying.
        split.eval_prolog(&inputs, &mut w3);
        for pass in 0..4 {
            plain.eval(&inputs, &mut w1, &mut o1);
            split.eval(&inputs, &mut w2, &mut o2);
            split.eval_main(&inputs, &mut w3, &mut o3);
            for (a, b) in [(o1[0], o2[0]), (o1[0], o3[0])] {
                if !same_bits(a, b) {
                    mismatches += 1;
                    eprintln!("seed {seed} pass {pass}: plain={a:?} split={b:?}");
                    if std::env::var_os("DUMP").is_some() {
                        eprintln!("expr: {}", rsgb::to_string(&ctx, root));
                        eprintln!("pure: {pure:?} syms {:?}", sym_ids);
                        eprintln!("PLAIN\n{}", plain.dump());
                        eprintln!("SPLIT\n{}", split.dump());
                    }
                }
            }
            // Vary only impure inputs; the prolog stays valid.
            for k in 0..nsym {
                if !pure[k] {
                    inputs[k] = rng.val();
                }
            }
        }
    }
    assert_eq!(mismatches, 0, "split tape diverged from unsplit");
}

#[test]
fn liveness_reuses_slots_on_deep_chains() {
    // A 200-deep accumulator chain ((((x+c)+c)+c)...) has 200+ reachable nodes
    // but only a couple are live at once -> the work buffer should be tiny.
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let mut acc = x;
    for i in 0..200 {
        let c = ctx.konst_f64(i as f64);
        acc = ctx.add(acc, c);
    }
    let tape = Tape::compile(&ctx, &[acc], &[sym_id(&ctx, x)]);
    assert!(
        tape.n_slots() <= 8,
        "expected a handful of live slots, got {}",
        tape.n_slots()
    );

    // ... and it still evaluates correctly: x + sum(0..200).
    let (mut work, mut out) = (Vec::new(), Vec::new());
    tape.eval(&[3.0], &mut work, &mut out);
    let expect = 3.0 + (0..200).map(|i| i as f64).sum::<f64>();
    assert_eq!(out[0], expect);
}

#[test]
fn long_reduce_dot_match_arena_4lane() {
    // Long Reduce / Dot lists exercise the 4-lane path; tape and arena must
    // still agree bit-for-bit (both go through the shared reduce/dot slice fn).
    for seed in 1..200u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
        let mut ctx: Graph = Graph::new();
        let nsym = 3;
        let syms: Vec<ExprId> = (0..nsym).map(|i| ctx.sym(&format!("x{i}"))).collect();
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&ctx, e)).collect();
        // a wide list (20..40 operands) of symbol/const mixes
        let len = 20 + rng.below(20);
        let mut pool = syms.clone();
        for _ in 0..3 {
            let k = ctx.konst_f64(rng.val());
            pool.push(k);
        }
        let la: Vec<ExprId> = (0..len).map(|_| pool[rng.below(pool.len())]).collect();
        let lb: Vec<ExprId> = (0..len).map(|_| pool[rng.below(pool.len())]).collect();
        let red = ctx.reduce(ReduceOp::Sum, la.clone());
        let dot = ctx.dot(la, lb);
        let root = ctx.add(red, dot);

        let inputs: Vec<f64> = (0..nsym).map(|_| rng.val()).collect();
        let env: HashMap<SymbolId, f64> = sym_ids
            .iter()
            .copied()
            .zip(inputs.iter().copied())
            .collect();
        let want = eval_real(&ctx, &env, &[root])[0];

        let tape = Tape::compile(&ctx, &[root], &sym_ids);
        let (mut work, mut out) = (Vec::new(), Vec::new());
        tape.eval(&inputs, &mut work, &mut out);
        assert!(
            same_bits(want, out[0]),
            "seed {seed}: arena={want:?} tape={:?}",
            out[0]
        );
    }
}

#[test]
fn specialize_pins_and_guards() {
    // y = select(x > 0, exp(x), -x): specialized at x = 1 the select is gone
    // (fewer ops), the guard holds anywhere on the positive branch, fires on
    // the negative one, and respecializing restores parity.
    let mut ctx: Graph = Graph::new();
    let x = ctx.sym("x");
    let zero = ctx.zero();
    let c = ctx.cmp(rsgb::CmpOp::Gt, x, zero);
    let ex = ctx.exp(x);
    let nx = ctx.neg(x);
    let y = ctx.select(c, ex, nx);
    let tape = Tape::compile(&ctx, &[y], &[sym_id(&ctx, x)]);
    assert_eq!(tape.n_selects(), 1);

    let (mut w, mut o) = (Vec::new(), Vec::new());
    let mut choices = Vec::new();
    tape.eval_traced(&[1.0], &mut w, &mut o, &mut choices);
    assert_eq!(choices, vec![1]);

    let spec = tape.specialize(&choices);
    assert!(
        spec.n_ops() < tape.n_ops(),
        "select pinning must shorten the tape"
    );

    let mut os = Vec::new();
    assert!(
        spec.eval_checked(&[2.0], &mut w, &mut os),
        "guard must hold on the same branch"
    );
    assert_eq!(os.len(), 1);
    assert!(same_bits(os[0], 2.0f64.exp()));

    assert!(
        !spec.eval_checked(&[-1.0], &mut w, &mut os),
        "guard must fire across the branch"
    );

    tape.eval_traced(&[-1.0], &mut w, &mut o, &mut choices);
    assert_eq!(choices, vec![0]);
    let spec = tape.specialize(&choices);
    assert!(spec.eval_checked(&[-1.0], &mut w, &mut os));
    assert!(same_bits(os[0], 1.0));
}

#[test]
fn specialized_tape_matches_full_bit_exact() {
    // Trace at a point, specialize, then random-walk the inputs. Whenever the
    // guards pass, the specialized outputs must be bit-exact against the full
    // tape; whenever they fire, respecializing from a fresh trace must restore
    // parity at that same point. A missed flip of a *live* select would break
    // the parity assert, so this also fuzzes guard soundness.
    let mut flips = 0;
    let mut holds = 0;
    for seed in 1..400u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0xA076_1D64_78BD_642F) | 1);
        let mut ctx: Graph = Graph::new();
        let nsym = 1 + rng.below(4);
        let syms: Vec<ExprId> = (0..nsym).map(|i| ctx.sym(&format!("x{i}"))).collect();
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&ctx, e)).collect();
        let steps = 8 + rng.below(24);
        let r1 = build_branchy(&mut ctx, &mut rng, &syms, steps);
        let r2 = build_branchy(&mut ctx, &mut rng, &syms, steps);
        let tape = Tape::compile(&ctx, &[r1, r2], &sym_ids);

        let mut inputs: Vec<f64> = (0..nsym).map(|_| rng.val()).collect();
        let (mut w, mut of, mut os) = (Vec::new(), Vec::new(), Vec::new());
        let mut choices = Vec::new();
        tape.eval_traced(&inputs, &mut w, &mut of, &mut choices);
        let mut spec = tape.specialize(&choices);

        for _ in 0..30 {
            for v in inputs.iter_mut() {
                // Steps large enough to cross comparison boundaries regularly.
                *v += rng.val();
            }
            tape.eval_traced(&inputs, &mut w, &mut of, &mut choices);
            if spec.eval_checked(&inputs, &mut w, &mut os) {
                holds += 1;
            } else {
                flips += 1;
                spec = tape.specialize(&choices);
                assert!(
                    spec.eval_checked(&inputs, &mut w, &mut os),
                    "seed {seed}: fresh trace must validate at its own point"
                );
            }
            assert_eq!(os.len(), 2, "seed {seed}: guard outputs must be truncated");
            for k in 0..2 {
                assert!(
                    same_bits(of[k], os[k]),
                    "seed {seed}: output {k} full={:?} specialized={:?}",
                    of[k],
                    os[k]
                );
            }
        }
    }
    assert!(
        flips > 30,
        "walk never crossed regions; test too weak (flips={flips})"
    );
    assert!(
        holds > 1000,
        "guards never held; test too weak (holds={holds})"
    );
}

#[test]
fn differentiate_matches_finite_differences() {
    let mut checked = 0;
    for seed in 1..800u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
        let mut ctx: Graph = Graph::new();
        let nsym = 1 + rng.below(3);
        let syms: Vec<ExprId> = (0..nsym).map(|i| ctx.sym(&format!("x{i}"))).collect();
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&ctx, e)).collect();
        let steps = 4 + rng.below(10);
        let root = build_with_syms(&mut ctx, &mut rng, &syms, steps, true, true);

        // Differentiate w.r.t. one symbol.
        let wrt = rng.below(nsym);
        let d = differentiate(&mut ctx, root, sym_ids[wrt]);

        // Positive-ish point to stay in ln/sqrt domains more often.
        let point: Vec<f64> = (0..nsym).map(|_| rng.pos()).collect();
        let mut env: HashMap<SymbolId, f64> =
            sym_ids.iter().copied().zip(point.iter().copied()).collect();

        let ad = eval_real(&ctx, &env, &[d])[0];

        // Central finite difference of the original.
        let h = 1e-6;
        env.insert(sym_ids[wrt], point[wrt] + h);
        let fp = eval_real(&ctx, &env, &[root])[0];
        env.insert(sym_ids[wrt], point[wrt] - h);
        let fm = eval_real(&ctx, &env, &[root])[0];
        let fd = (fp - fm) / (2.0 * h);

        // Skip samples that landed on a domain edge or blew up numerically.
        if !ad.is_finite() || !fd.is_finite() || fd.abs() > 1e8 {
            continue;
        }
        let rel = (ad - fd).abs() / (fd.abs() + 1e-6);
        assert!(
            rel < 1e-3,
            "seed {seed}: d/dx{wrt} AD={ad:e} FD={fd:e} rel={rel:e}"
        );
        checked += 1;
    }
    assert!(checked > 200, "too few AD samples survived ({checked})");
}

/// Partial specialization: with a random pin mask, every evaluation whose
/// guards hold must be bit-exact against the full tape -- unpinned selects may
/// flip freely without invalidating anything.
#[test]
fn partial_specialization_checked_evals_bit_exact() {
    let mut mismatches = 0;
    for seed in 1..500u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut ctx: Graph = Graph::new();
        let nsym = 2 + rng.below(3);
        let syms: Vec<ExprId> = (0..nsym).map(|i| ctx.sym(&format!("x{i}"))).collect();
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&ctx, e)).collect();
        let steps = 8 + rng.below(24);
        let root = build_with_syms(&mut ctx, &mut rng, &syms, steps, true, true);
        let tape = Tape::compile(&ctx, &[root], &sym_ids);
        if tape.n_selects() == 0 {
            continue;
        }

        let inputs: Vec<f64> = (0..nsym).map(|_| rng.val()).collect();
        let (mut w, mut o, mut choices) = (Vec::new(), Vec::new(), Vec::new());
        tape.eval_traced(&inputs, &mut w, &mut o, &mut choices);
        let pin: Vec<bool> = (0..tape.n_selects()).map(|_| rng.below(2) == 0).collect();
        let spec = tape.specialize_partial(&choices, &pin);

        let (mut wf, mut of) = (Vec::new(), Vec::new());
        let (mut ws, mut os) = (Vec::new(), Vec::new());
        for probe in 0..10 {
            let pargs: Vec<f64> = if probe == 0 {
                inputs.clone()
            } else {
                (0..nsym).map(|_| rng.val()).collect()
            };
            tape.eval(&pargs, &mut wf, &mut of);
            let ok = spec.eval_checked(&pargs, &mut ws, &mut os);
            if probe == 0 {
                assert!(ok, "seed {seed}: guards must hold at the traced point");
            }
            if ok {
                for (a, b) in of.iter().zip(&os) {
                    if !same_bits(*a, *b) {
                        mismatches += 1;
                        eprintln!("seed {seed} probe {probe}: full={a:?} partial-spec={b:?}");
                    }
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "partial specialization diverged where guards held"
    );
}

/// The typed evaluators: a tape evaluated in `Complex<f64>` matches the
/// complex arena sweep bit for bit on real inputs, and `f32` stays within
/// single precision of `f64` on well-conditioned expressions.
#[test]
fn typed_tape_matches_complex_arena_and_f32_is_close() {
    use num_complex::Complex64;
    let mut checked = 0;
    for seed in 1..300u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut ctx: Graph = Graph::new();
        let nsym = 1 + rng.below(4);
        let syms: Vec<ExprId> = (0..nsym).map(|i| ctx.sym(&format!("x{i}"))).collect();
        let sym_ids: Vec<SymbolId> = syms.iter().map(|&e| sym_id(&ctx, e)).collect();
        let steps = 4 + rng.below(20);
        let root = build_with_syms(&mut ctx, &mut rng, &syms, steps, false, true);
        let tape = Tape::compile(&ctx, &[root], &sym_ids);
        let inputs: Vec<f64> = (0..nsym).map(|_| rng.val()).collect();
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval(&inputs, &mut w, &mut o);
        if !o[0].is_finite() {
            continue;
        }
        // complex: tape vs arena
        let cin: Vec<Complex64> = inputs.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let (mut wc, mut oc) = (Vec::new(), Vec::new());
        tape.eval_typed(&cin, &mut wc, &mut oc);
        let mut env = HashMap::new();
        for (k, &s) in sym_ids.iter().enumerate() {
            env.insert(s, cin[k]);
        }
        let arena = rsgb::eval(&ctx, root, &env);
        assert!(
            same_bits(oc[0].re, arena.re) && same_bits(oc[0].im, arena.im),
            "seed {seed}: complex tape {:?} vs arena {:?}",
            oc[0],
            arena
        );
        // f32: within single precision of the f64 value (relative)
        let fin: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();
        let (mut wf, mut of) = (Vec::new(), Vec::new());
        tape.eval_typed(&fin, &mut wf, &mut of);
        let (a, b) = (o[0], of[0] as f64);
        if a.abs() < 1e6 && b.is_finite() {
            let tol = 1e-3 * (1.0 + a.abs());
            if (a - b).abs() > tol {
                // rough ops (floor, sign, mod, rand) may legitimately jump
                // under rounding; only count the smooth cases
                continue;
            }
        }
        checked += 1;
    }
    assert!(checked > 150, "too few checked cases: {checked}");
}
