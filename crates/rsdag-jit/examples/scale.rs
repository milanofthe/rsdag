//! A circuit-shaped hierarchy at scale, for measuring how the stages a
//! solver runs grow with it. A device function reads its node voltages
//! (states), two instance parameters and the parameters of its model card
//! (`--card`, 400 by default, as a compact model has); a gate is four
//! devices over two cards, a NAND with an internal node and one terminal on
//! ground; a block is `--gates` gates over a pool of nodes. The block is
//! placed once at the top, as a netlist's top-level subcircuit is
//! (`placed=once`), or its gates are placed at the top directly
//! (`placed=flat`). Every function takes everything it reads as a
//! parameter, the cards' parameters passed down the hierarchy; with
//! `--bind` a device is called through a binding of its card instead (see
//! `Graph::bind`), and the cards are globals above it. The tapes compile the
//! hierarchy as it stands (its composite functions as templates); with
//! `--inline` the composite functions are inlined first. `--trace` prints
//! the stage timings of the compiles.
//!
//! Per configuration one CSV row: graph nodes, then the time of each stage
//! (building, the sparse Jacobian in the states, the calls specialized to
//! the ground constant, the composite functions inlined (`--inline`), the residual and
//! Jacobian tapes, their native code), the time per evaluation of each
//! (interpreted and native), and a hash of the native outputs, the same
//! across versions that compute the same bits.
//!
//!     cargo run --release -p rsdag-jit --example scale -- [--gates <n>] [--card <n>] [--placed once|flat|both]
//!         [--bind] [--inline] [--trace]

use std::time::Instant;

use rsdag::{
    sparse_jacobian, Bound, ExprId, FuncId, Graph, Node, ParamRole, ReduceOp, Scope, SymbolId,
    Tape, F64,
};
use rsdag_jit::NativeTape;

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    match *g.node(e) {
        Node::Symbol(s) => (e, s),
        _ => unreachable!(),
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Seconds per call: the best of five batches of at least 50 ms.
fn per_call(mut f: impl FnMut()) -> f64 {
    let mut reps = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        if t.elapsed().as_secs_f64() >= 0.05 {
            break;
        }
        reps *= 2;
    }
    (0..5)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_secs_f64() / reps as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// The device: `(d, g, s)` states, `w`, `l`, then the card. The card's
/// parameters enter a threshold and a gain (work on the card alone), the
/// instance's a ratio; the current from `d` to `s` and its negation out.
fn device(g: &mut Graph<F64>, card: usize) -> FuncId {
    let mut s = Scope::new(g, "device");
    let vd = s.param_with_role("d", ParamRole::State { id: 0 });
    let vg = s.param_with_role("g", ParamRole::State { id: 1 });
    let vs = s.param_with_role("s", ParamRole::State { id: 2 });
    let w = s.param_with_role("w", ParamRole::Param);
    let l = s.param_with_role("l", ParamRole::Param);
    let c: Vec<ExprId> = (0..card)
        .map(|k| s.param_with_role(&format!("c{k}"), ParamRole::Param))
        .collect();
    let weights: Vec<ExprId> = (0..card)
        .map(|k| s.konst_f64(1e-3 / (1.0 + k as f64)))
        .collect();
    let vth = s.dot(c.clone(), weights);
    let kp = s.exp(c[1 % card]);
    let wl = s.div(w, l);
    let beta = s.mul(kp, wl);
    let vgs = s.sub(vg, vs);
    let vds = s.sub(vd, vs);
    let ov = s.sub(vgs, vth);
    let e = s.exp(ov);
    let one = s.konst_f64(1.0);
    let sp = s.add(one, e);
    let sp = s.ln(sp);
    let sq = s.mul(sp, sp);
    let clm = s.mul(c[2 % card], vds);
    let clm = s.add(one, clm);
    let th = s.tanh(vds);
    let id = s.mul(beta, sq);
    let id = s.mul(id, clm);
    let id = s.mul(id, th);
    let is = s.neg(id);
    s.close(vec![id, is])
}

/// The gate over `(a, b, y, vdd, m)` and both cards: two devices of card
/// `p` from `vdd` to `y`, two of card `n` from `y` through `m` to ground.
/// Out: the currents into `y`, `m` and `vdd`.
fn gate(g: &mut Graph<F64>, dev: FuncId, card: usize, bound: Option<[Bound; 2]>) -> FuncId {
    let mut s = Scope::new(g, "gate");
    let node: Vec<ExprId> = ["a", "b", "y", "vdd", "m"]
        .iter()
        .enumerate()
        .map(|(k, n)| s.param_with_role(n, ParamRole::State { id: k as u32 }))
        .collect();
    let (a, b, y, vdd, m) = (node[0], node[1], node[2], node[3], node[4]);
    // the cards: parameters passed down, or bound into the devices and
    // globals of the gate
    let (cn, cp): (Vec<ExprId>, Vec<ExprId>) = match bound {
        Some(_) => {
            for k in 0..2 * card {
                s.global(&format!("card{k}"));
            }
            (Vec::new(), Vec::new())
        }
        None => (
            (0..card)
                .map(|k| s.param_with_role(&format!("n{k}"), ParamRole::Param))
                .collect(),
            (0..card)
                .map(|k| s.param_with_role(&format!("p{k}"), ParamRole::Param))
                .collect(),
        ),
    };
    let (w, l) = (s.konst_f64(2e-6), s.konst_f64(1e-7));
    let zero = s.konst_f64(0.0);
    // a device: its card passed with the arguments, or bound
    let place = |s: &mut Scope<F64>, d: ExprId, gt: ExprId, src: ExprId, c: &[ExprId], k| {
        let args: Vec<ExprId> = [d, gt, src, w, l]
            .into_iter()
            .chain(c.iter().copied())
            .collect();
        match bound {
            Some(b) => s.calls_bound(b[k], &[0, 1], &args),
            None => s.calls(dev, &[0, 1], &args),
        }
    };
    let p1 = place(&mut s, y, a, vdd, &cp, 1);
    let p2 = place(&mut s, y, b, vdd, &cp, 1);
    let n1 = place(&mut s, y, a, m, &cn, 0);
    let n2 = place(&mut s, m, b, zero, &cn, 0);
    let iy = s.reduce(ReduceOp::Sum, vec![p1[0], p2[0], n1[0]]);
    let im = s.reduce(ReduceOp::Sum, vec![n1[1], n2[0]]);
    let ivdd = s.reduce(ReduceOp::Sum, vec![p1[1], p2[1]]);
    s.close(vec![iy, im, ivdd])
}

/// The gates' wiring over `n` nodes: gate `k` reads two nodes, drives one
/// and has its own internal node `n + k`.
fn wiring(gates: usize, n: usize) -> Vec<[usize; 3]> {
    (0..gates)
        .map(|k| [(7 * k + 3) % n, (13 * k + 5) % n, k % n])
        .collect()
}

/// The residuals over node expressions `v` (the pool, then one internal
/// node per gate) and `vdd`: per node the sum of the currents into it.
fn residuals(
    g: &mut Graph<F64>,
    gt: FuncId,
    v: &[ExprId],
    vdd: ExprId,
    cards: &[ExprId],
    gates: usize,
    n: usize,
) -> Vec<ExprId> {
    let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); n + gates + 1];
    for (k, [a, b, y]) in wiring(gates, n).into_iter().enumerate() {
        let args: Vec<ExprId> = [v[a], v[b], v[y], vdd, v[n + k]]
            .into_iter()
            .chain(cards.iter().copied())
            .collect();
        let i = g.calls(gt, &[0, 1, 2], &args);
        terms[y].push(i[0]);
        terms[n + k].push(i[1]);
        terms[n + gates].push(i[2]);
    }
    terms
        .into_iter()
        .map(|t| g.reduce(ReduceOp::Sum, t))
        .collect()
}

fn hash(v: &[f64]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ x.to_bits()).wrapping_mul(0x0100_0000_01b3)
    })
}

fn run(gates: usize, card: usize, once: bool, bind: bool, inline_first: bool) {
    let n = gates / 2;
    let t = Instant::now();
    let mut g: Graph<F64> = Graph::new();
    let dev = device(&mut g, card);
    let cards: Vec<(ExprId, SymbolId)> = (0..2 * card)
        .map(|k| sym(&mut g, &format!("card{k}")))
        .collect();
    let all_cards: Vec<ExprId> = cards.iter().map(|&(e, _)| e).collect();
    // bound: a device function per card, the cards its globals
    let bound = bind.then(|| {
        let at = |c: usize| -> Vec<(u32, ExprId)> {
            (0..card)
                .map(|k| ((5 + k) as u32, all_cards[c * card + k]))
                .collect()
        };
        let (bn, bp) = (at(0), at(1));
        [g.bind(dev, &bn), g.bind(dev, &bp)]
    });
    let gt = gate(&mut g, dev, card, bound);
    let (vdd, vdd_s) = sym(&mut g, "vdd");
    let top: Vec<(ExprId, SymbolId)> = (0..n + gates)
        .map(|k| sym(&mut g, &format!("v{k}")))
        .collect();
    let card_e: Vec<ExprId> = if bind { Vec::new() } else { all_cards.clone() };
    let roots = if once {
        // the block over formal nodes, placed once on the top-level ones
        let mut s = Scope::new(&mut g, "block");
        let fv: Vec<ExprId> = (0..n + gates)
            .map(|k| s.param_with_role(&format!("blk.v{k}"), ParamRole::State { id: k as u32 }))
            .collect();
        let fvdd = s.param_with_role(
            "blk.vdd",
            ParamRole::State {
                id: (n + gates) as u32,
            },
        );
        let fc: Vec<ExprId> = (0..card_e.len())
            .map(|k| s.param_with_role(&format!("blk.card{k}"), ParamRole::Param))
            .collect();
        let outs = residuals(&mut s, gt, &fv, fvdd, &fc, gates, n);
        let n_out = outs.len() as u32;
        let block = s.close(outs);
        let args: Vec<ExprId> = top
            .iter()
            .map(|&(e, _)| e)
            .chain([vdd])
            .chain(card_e.iter().copied())
            .collect();
        g.calls(block, &(0..n_out).collect::<Vec<_>>(), &args)
    } else {
        let v: Vec<ExprId> = top.iter().map(|&(e, _)| e).collect();
        residuals(&mut g, gt, &v, vdd, &card_e, gates, n)
    };
    let build = ms(t);
    let states: Vec<SymbolId> = top.iter().map(|&(_, s)| s).chain([vdd_s]).collect();

    let t = Instant::now();
    let rows = sparse_jacobian(&mut g, &roots, &states);
    let jacobian = ms(t);
    let entries: Vec<ExprId> = rows.iter().flatten().map(|&(_, e)| e).collect();

    let t = Instant::now();
    let both: Vec<ExprId> = roots.iter().chain(&entries).copied().collect();
    let spec = g.specialize_calls(&both);
    let specialize = ms(t);

    // the hierarchy compiled as it stands, or (`--inline`) inlined first
    let t = Instant::now();
    let flat = if inline_first {
        g.inline_composite(&spec)
    } else {
        spec
    };
    let inline = ms(t);
    let (res, jac) = flat.split_at(roots.len());
    let nodes = g.len();

    // inputs: the states vary, the cards are bound (the prolog's share)
    let syms: Vec<SymbolId> = states
        .iter()
        .copied()
        .chain(cards.iter().map(|&(_, s)| s))
        .collect();
    let pure: Vec<bool> = (0..syms.len()).map(|k| k >= states.len()).collect();
    let t = Instant::now();
    let tape_f = Tape::compile_split(&g, res, &syms, &pure);
    let tape_j = Tape::compile_split(&g, jac, &syms, &pure);
    let tapes = ms(t);
    let t = Instant::now();
    let nat_f = NativeTape::compile(&tape_f).expect("native code");
    let nat_j = NativeTape::compile(&tape_j).expect("native code");
    let native = ms(t);

    let x: Vec<f64> = (0..syms.len())
        .map(|k| {
            if k < states.len() {
                0.3 + 0.4 * ((k * 37 % 101) as f64 / 101.0)
            } else {
                0.01 * ((k * 17 % 23) as f64 + 1.0)
            }
        })
        .collect();
    let eval = |tape: &Tape, nat: &NativeTape| {
        let (mut w, mut o) = (Vec::new(), Vec::new());
        tape.eval_prolog(&x, &mut w);
        let interp = per_call(|| tape.eval_main(&x, &mut w, &mut o));
        let (mut wn, mut on) = (Vec::new(), Vec::new());
        nat.eval_prolog(&x, &mut wn);
        let nat_t = per_call(|| nat.eval_main(&x, &mut wn, &mut on));
        assert_eq!(hash(&o), hash(&on), "native and interpreter differ");
        (interp * 1e6, nat_t * 1e6, hash(&on))
    };
    let (fi, fnat, fh) = eval(&tape_f, &nat_f);
    let (ji, jnat, jh) = eval(&tape_j, &nat_j);
    println!(
        "{}{}{},{gates},{card},{},{nodes},{build:.1},{jacobian:.1},{specialize:.1},{inline:.1},{tapes:.1},{native:.1},{fi:.1},{fnat:.1},{ji:.1},{jnat:.1},{:016x}",
        if once { "once" } else { "flat" },
        if bind { "+bind" } else { "" },
        if inline_first { "+inline" } else { "" },
        states.len(),
        fh ^ jh.rotate_left(1),
    );
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |flag: &str| {
        args.iter().position(|a| a == flag).map(|i| {
            let v = args.remove(i + 1);
            args.remove(i);
            v
        })
    };
    let gates: usize = option("--gates").map_or(2400, |v| v.parse().expect("--gates <n>"));
    let card: usize = option("--card").map_or(400, |v| v.parse().expect("--card <n>"));
    let placed = option("--placed").unwrap_or_else(|| "both".into());
    let bind = args.iter().any(|a| a == "--bind");
    let inline_first = args.iter().any(|a| a == "--inline");
    if args.iter().any(|a| a == "--trace") {
        // the stage timings of the compiles, on stderr
        struct Stderr;
        impl rsdag::hooks::Log for Stderr {
            fn log(&self, _: rsdag::hooks::Level, message: &str) {
                eprintln!("{message}");
            }
        }
        rsdag::hooks::set_log(&Stderr);
    }
    println!(
        "placed,gates,card,states,nodes,build_ms,jacobian_ms,specialize_ms,inline_ms,tapes_ms,native_ms,\
         f_interp_us,f_native_us,j_interp_us,j_native_us,hash"
    );
    for once in [true, false] {
        if placed == "both" || (placed == "once") == once {
            run(gates, card, once, bind, inline_first);
        }
    }
}
