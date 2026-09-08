//! Lane parity: every lane of a `LaneTape` reproduces the scalar tape on
//! that lane's parameter set, over synthetic programs.

use rsdag::synth::{cases, Spec, Vocabulary};
use rsdag::{Graph, Tape};
use rsdag_jit::{suggest_lanes, LaneTape, LANES, LANE_WIDTHS};

/// The symbol behind a symbol node, for the hand-written case below.
fn symbol_of(g: &Graph, e: rsdag::ExprId) -> rsdag::SymbolId {
    match *g.node(e) {
        rsdag::Node::Symbol(s) => s,
        _ => unreachable!(),
    }
}

/// Lane `k` computes the same bits at every width: a wider tape is more
/// parameter sets per pass, never a different arithmetic. This is what makes
/// the width a tuning knob rather than a semantic choice.
#[test]
fn every_width_agrees_with_every_other() {
    for case in cases(0..40, |seed| {
        let spec = Spec::new(seed).steps(30 + (seed as usize % 40)).params(3);
        match seed % 3 {
            0 => spec.vocab(Vocabulary::Ring).max_list(20),
            1 => spec.vocab(Vocabulary::Elementary),
            _ => spec.vocab(Vocabulary::Full),
        }
    }) {
        let n_in = case.syms.len();
        let want = case.reference();
        for width in LANE_WIDTHS {
            let Ok(lane) = LaneTape::compile_lanes(&case.tape, width, rsdag_jit::CHUNK_OPS) else {
                continue;
            };
            assert_eq!(lane.width(), width);
            // Row `l % rows` into lane `l`, so every lane carries a
            // different parameter set at every width.
            let rows = case.rows.len();
            let mut interleaved = Vec::with_capacity(n_in * width);
            for k in 0..n_in {
                for l in 0..width {
                    interleaved.push(case.rows[l % rows][k]);
                }
            }
            let (mut w, mut o) = (Vec::new(), Vec::new());
            lane.eval(&interleaved, &mut w, &mut o);
            for l in 0..width {
                let expect = &want[l % rows];
                for (j, &e) in expect.iter().enumerate() {
                    let got = o[j * width + l];
                    assert!(
                        got.to_bits() == e.to_bits() || (got.is_nan() && e.is_nan()),
                        "seed {}, width {width}, lane {l}, output {j}: {e:?} vs {got:?}",
                        case.seed
                    );
                }
            }
        }
        // The suggestion is one of the widths, or "use the scalar backend".
        if let Some(w) = suggest_lanes(&case.tape) {
            assert!(LANE_WIDTHS.contains(&w), "suggested width {w}");
        }
    }
}

#[test]
fn every_lane_matches_the_scalar_tape() {
    for (i, case) in cases(0..250, |seed| {
        let spec = Spec::new(seed)
            .steps(6 + (seed as usize % 30))
            .params(3)
            .outputs(1 + seed as usize % 3);
        match seed % 3 {
            0 => spec.vocab(Vocabulary::Ring).max_list(20),
            1 => spec.vocab(Vocabulary::Elementary),
            _ => spec.vocab(Vocabulary::Full),
        }
    })
    .enumerate()
    {
        let chunk_ops = [3, 7, rsdag_jit::CHUNK_OPS][i % 3];
        let lane = LaneTape::compile_with(&case.tape, chunk_ops).expect("lane compile");
        let w = lane.width();
        let n_in = case.syms.len();

        // The case's rows fill the lanes; the lane tape wants them
        // input-major (input k of every lane consecutive).
        let sets: Vec<&Vec<f64>> = case.rows.iter().take(w).collect();
        assert!(sets.len() == w, "the corpus must supply one row per lane");
        let mut interleaved = Vec::with_capacity(n_in * w);
        for k in 0..n_in {
            for set in &sets {
                interleaved.push(set[k]);
            }
        }
        let (mut wl, mut ol) = (Vec::new(), Vec::new());
        lane.eval(&interleaved, &mut wl, &mut ol);

        // Lane `l` must equal the reference for row `l`.
        let want = case.reference();
        for (l, row_want) in want.iter().take(w).enumerate() {
            for (j, &expect) in row_want.iter().enumerate() {
                let got = ol[j * w + l];
                assert!(
                    got.to_bits() == expect.to_bits() || (got.is_nan() && expect.is_nan()),
                    "seed {}, width {w}, lane {l}, output {j}: reference {expect:?} vs lane {got:?}",
                    case.seed
                );
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
    let syms = [symbol_of(&ctx, p), symbol_of(&ctx, x)];
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
                os[0].to_bits() == ol[l].to_bits(),
                "step {step} lane {l}: {:?} vs {:?}",
                os[0],
                ol[l]
            );
        }
    }
}
