//! rsdag on DAE modules: a module JSON as SANE's `export_module` writes it
//! (the rsdag module, the index of the system function, the parameter
//! values by name and a point `x0`). The residual is the DC one, as the
//! other tools get it: the derivatives and time bound to the constant zero
//! and every call specialized to that (`Graph::specialize_calls`). The
//! residual and its Jacobian are derived on the module's functions and
//! compiled as programs over them: a subcircuit's function a template, the
//! device bodies called.
//! The states are the inputs; the parameters are bound once, so their work
//! runs in the prolog. Per module: the setup and the time per call of the
//! residual and of its sparse Jacobian in the states, interpreted and
//! native; then natively with the parameters folded (the calls inlined and
//! every parameter a constant, each instance's parameter branches decided
//! at build time, as a tool that folds constants does). One CSV row each.
//! `--values <dir>` also writes the residual and the Jacobian at
//! `x0 + 0.01` there (off the solution, where the residual is not zero),
//! for comparing other tools against. `--composed <dir>` writes each module
//! with its subcircuit functions inlined and the device bodies functions,
//! the form the other tools are given.
//!
//! `--no-variants` runs every device body whole, its parameter branches
//! both computed (see `rsdag::variant`), for comparing against.
//! `--no-folded` leaves the folded build out (its columns empty), which
//! takes minutes on a large module. `--interleave <MB>` has the calling
//! thread write through a buffer of that many megabytes before every timed
//! call, as a solver's factorization does between its evaluations: the
//! calls then find their data where a solver's find it, not in the caches
//! the previous call left them in.
//!
//! `--threads <n>` runs the residual and the Jacobian with a pool of `n`
//! threads installed, their independent calls (the device instances) in
//! parallel (`rsdag::parallel`, rsdag's `Workers`). A list (`--threads
//! 1,2,4,12`) measures the native code at each count in one process, the
//! counts taken in turn over `--rounds <n>` rounds (3 by default) so the
//! machine's drift falls on all alike; one `threads` row per count, the
//! median of its rounds, after the module's row (which is the first
//! count's). `--native` leaves the interpreted calls unmeasured. `--lanes
//! auto|always|never` sets the device bodies' lane
//! code (`rsdag_jit::Lanes`, `auto` by default).
//!
//!     cargo run --release -p rsdag-jit --example modules -- [--values <dir>] [--composed <dir>] [--threads <n>[,<n>...]]
//!         [--rounds <n>] [--native] [--interleave <MB>] [--no-folded] [--lanes <mode>] [--no-variants] <module.json>...

use std::time::Instant;

use rsdag::{
    sparse_jacobian, substitute, ExprId, Graph, Module, Output, OutputRole, ParamRole, Tape, F64,
};
use rsdag_jit::{Lanes, NativeTape, Options};

static INTERLEAVE: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

/// Write through the `--interleave` buffer (a no-op without one).
fn interleave() {
    let mut buf = INTERLEAVE.lock().unwrap();
    for v in buf.iter_mut().step_by(8) {
        *v = v.wrapping_add(1);
    }
}

/// Seconds per call: the best of five batches of at least 50 ms; with
/// `--interleave`, the calls alone, the buffer written before each.
fn per_call(mut f: impl FnMut()) -> f64 {
    if !INTERLEAVE.lock().unwrap().is_empty() {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let (mut sum, mut n) = (0.0, 0usize);
            while sum < 0.05 || n < 3 {
                interleave();
                let t = Instant::now();
                f();
                sum += t.elapsed().as_secs_f64();
                n += 1;
            }
            best = best.min(sum / n as f64);
        }
        return best;
    }
    let mut reps = 1usize;
    let mut dt;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        dt = t.elapsed().as_secs_f64();
        if dt > 0.05 {
            break;
        }
        reps *= 4;
    }
    let mut best = dt / reps as f64;
    for _ in 0..4 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    best
}

fn timed<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64())
}

/// The pools `--threads` asks for, by count (none for one thread): the
/// programs' independent calls run on them (`rsdag::parallel`).
static POOLS: std::sync::OnceLock<Vec<(usize, Option<rsdag::parallel::Parallel>)>> =
    std::sync::OnceLock::new();

/// Rounds over the thread counts (`--rounds`), and whether the interpreted
/// calls go unmeasured (`--native`).
static ROUNDS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static NATIVE_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// The lane code `--lanes` asks for.
static LANES: std::sync::OnceLock<Lanes> = std::sync::OnceLock::new();

/// Whether device bodies run their per-binding variants (`--no-variants`).
static VARIANTS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// `f` with `pool` installed, if any.
fn on_pool<R: Send>(pool: &Option<rsdag::parallel::Parallel>, f: impl FnOnce() -> R + Send) -> R {
    match pool.clone() {
        Some(p) => rsdag::parallel::install(p, f),
        None => f(),
    }
}

/// `--stats`: per called body, its instances and the ops of its prolog and
/// main phase, the main phase's by kind.
fn print_stats(tape: &Tape) {
    use rsdag::tape::Op;
    for (b, bundle) in tape.bundles().iter().enumerate() {
        let Some(body) = bundle.body() else {
            continue;
        };
        let instances: u32 = (tape.ops().iter())
            .filter_map(|op| match *op {
                Op::Call {
                    bundle, n_groups, ..
                } if bundle as usize == b => Some(n_groups),
                _ => None,
            })
            .sum();
        let (all, prolog) = (body.n_ops(), body.prolog_len());
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for op in &body.ops()[prolog..] {
            let name = match op {
                Op::Unary(u, _) => format!("Unary::{u:?}"),
                Op::Binary(b, _, _) => format!("Binary::{b:?}"),
                Op::Reduce(r, _, _) => format!("Reduce::{r:?}"),
                other => format!("{other:?}")
                    .split(['(', ' ', '{'])
                    .next()
                    .unwrap_or("")
                    .to_string(),
            };
            *kinds.entry(name).or_default() += 1;
        }
        println!(
            "stats,body {b},instances {instances},ops {all},prolog {prolog},main {},state {},outputs {}",
            all - prolog,
            body.state_len(),
            bundle.n_outputs()
        );
        let mut kinds: Vec<_> = kinds.into_iter().collect();
        kinds.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (k, n) in kinds {
            println!("stats,  {k},{n}");
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Setup (seconds) and per-call time (seconds) of `roots` over the inputs,
/// interpreted then native, on the first `--threads` count; the values at
/// `check`; the native per-call time at every count (see `--rounds`).
fn measure(
    g: &Graph<F64>,
    roots: &[ExprId],
    syms: &[rsdag::SymbolId],
    pure: &[bool],
    vals: &[f64],
    check: &[f64],
) -> ([f64; 4], Vec<f64>, Vec<f64>) {
    let (mut tape, s_tape) = timed(|| Tape::compile_split(g, roots, syms, pure));
    if STATS.get().copied().unwrap_or(false) {
        print_stats(&tape);
    }
    tape.with_backend(&rsdag::BodyBackend {
        variants: Some(rsdag::variant::VariantPolicy {
            enabled: VARIANTS.get().copied().unwrap_or(true),
            ..Default::default()
        }),
        ..Default::default()
    });
    // Variants built by the first prolog, so the main phases measured run
    // on them.
    let opts = Options {
        lanes: LANES.get().copied().unwrap_or_default(),
        background: false,
        ..Options::default()
    };
    let (native, s_native) =
        timed(|| NativeTape::compile_opts(&tape, &opts, &[]).expect("native code"));
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let (mut wn, mut on) = (Vec::new(), Vec::new());
    let pools = POOLS.get().expect("pools");
    let first = &pools[0].1;
    let interpreted = !NATIVE_ONLY.get().copied().unwrap_or(false);
    let c_tape = on_pool(first, || {
        tape.eval_prolog(vals, &mut w);
        if interpreted {
            per_call(|| tape.eval_main(vals, &mut w, &mut o))
        } else {
            tape.eval_main(vals, &mut w, &mut o);
            f64::NAN
        }
    });
    on_pool(first, || native.eval_prolog(vals, &mut wn));
    let mut samples = vec![Vec::new(); pools.len()];
    for _ in 0..ROUNDS.get().copied().unwrap_or(1) {
        for ((_, pool), at) in pools.iter().zip(samples.iter_mut()) {
            at.push(on_pool(pool, || {
                per_call(|| native.eval_main(vals, &mut wn, &mut on))
            }));
        }
    }
    let per_count: Vec<f64> = samples.into_iter().map(median).collect();
    let c_native = per_count[0];
    if let Some((k, (a, b))) = o
        .iter()
        .zip(&on)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
    {
        let n = o
            .iter()
            .zip(&on)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        panic!(
            "native differs at output {k} of {}: {a:e} vs {b:e} ({n} outputs differ)",
            o.len()
        );
    }
    native.eval_main(check, &mut wn, &mut on);
    ([s_tape, s_tape + s_native, c_tape, c_native], on, per_count)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut option = |flag: &str| {
        args.iter().position(|a| a == flag).map(|i| {
            let d = args.remove(i + 1);
            args.remove(i);
            d
        })
    };
    let values_dir = option("--values");
    let composed_dir = option("--composed");
    let lanes = match option("--lanes").as_deref() {
        None | Some("auto") => Lanes::Auto,
        Some("always") => Lanes::Always,
        Some("never") => Lanes::Never,
        Some(m) => panic!("--lanes auto|always|never, not {m}"),
    };
    LANES.set(lanes).ok();
    let counts: Vec<usize> = option("--threads").map_or(vec![1], |t| {
        t.split(',')
            .map(|n| n.trim().parse().expect("--threads <n>[,<n>...]"))
            .collect()
    });
    let rounds: usize = option("--rounds").map_or(3, |r| r.parse().expect("--rounds <n>"));
    if let Some(mb) = option("--interleave") {
        let mb: usize = mb.parse().expect("--interleave <MB>");
        *INTERLEAVE.lock().unwrap() = vec![0u64; mb << 17];
    }
    if let Some(i) = args.iter().position(|a| a == "--no-variants") {
        args.remove(i);
        VARIANTS.set(false).ok();
    }
    let folding = match args.iter().position(|a| a == "--no-folded") {
        Some(i) => {
            args.remove(i);
            false
        }
        None => true,
    };
    if let Some(i) = args.iter().position(|a| a == "--stats") {
        args.remove(i);
        STATS.set(true).ok();
    }
    if let Some(i) = args.iter().position(|a| a == "--native") {
        args.remove(i);
        NATIVE_ONLY.set(true).ok();
    }
    let pools = counts
        .iter()
        .map(|&n| {
            let pool = (n > 1).then(|| {
                rsdag::parallel::Parallel::new(std::sync::Arc::new(rsdag::parallel::Workers::new(
                    n,
                )))
            });
            (n, pool)
        })
        .collect();
    POOLS.set(pools).ok();
    ROUNDS.set(if counts.len() > 1 { rounds } else { 1 }).ok();
    println!(
        "{}",
        [
            "module,states",
            "setup_f_interp_s,setup_f_native_s,call_f_interp_us,call_f_native_us",
            "setup_j_interp_s,setup_j_native_s,call_j_interp_us,call_j_native_us,j_nnz",
            "setup_f_folded_s,call_f_folded_us,setup_j_folded_s,call_j_folded_us",
        ]
        .join(",")
    );
    for path in &args {
        let text = std::fs::read_to_string(path).expect("read the module");
        let v: serde_json::Value = serde_json::from_str(&text).expect("parse the module");
        let mut module: Module<F64> =
            serde_json::from_value(v["module"].clone()).expect("a module");
        // The derivatives the exporter derived for its own Jacobian lose
        // their role, so the Jacobian here derives afresh, as the other
        // tools do.
        for r in module
            .funcs
            .iter_mut()
            .flat_map(|f| f.output_roles.iter_mut())
        {
            if matches!(r, OutputRole::Derivative { .. }) {
                *r = OutputRole::Plain;
            }
        }
        let (mut g, map) = Graph::from_module(&module).expect("a valid module");
        let f = map.funcs[v["circuit"].as_u64().expect("circuit") as usize];
        let x0: Vec<f64> = serde_json::from_value(v["x0"].clone()).expect("x0");
        if let Some(dir) = &composed_dir {
            write_composed(&mut g, f, &v, dir, path);
        }
        let zero = g.zero();
        let func = g.func(f);
        // Inputs: the function's parameters; the states vary, the rest is bound.
        let syms = func.params().to_vec();
        let roles = func.param_roles().to_vec();
        let mut states = Vec::new();
        let (mut vals, mut pure) = (Vec::new(), Vec::new());
        for (&s, role) in syms.iter().zip(&roles) {
            let (value, is_state) = match *role {
                ParamRole::State { id } => (x0[id as usize], true),
                ParamRole::Param => (
                    v["params"][g.symbol_name(s)].as_f64().expect("a value"),
                    false,
                ),
                _ => (0.0, false), // derivatives and time: the DC residual
            };
            if is_state {
                states.push(s);
            }
            vals.push(value);
            pure.push(!is_state);
        }
        let residuals: Vec<ExprId> = func
            .outputs()
            .iter()
            .zip(func.output_roles())
            .filter(|(_, r)| matches!(r, OutputRole::Residual { .. }))
            .map(|(o, _)| match *o {
                Output::Expr(e) => e,
                _ => zero,
            })
            .collect();
        // x' and t as the constant zero, every call specialized to it.
        let at_rest = syms
            .iter()
            .zip(&roles)
            .filter(|(_, r)| matches!(r, ParamRole::StateDot { .. } | ParamRole::Time))
            .map(|(&s, _)| (s, zero))
            .collect();
        let rest = substitute(&mut g, &residuals, &at_rest);
        let residuals = g.specialize_calls(&rest);
        let check: Vec<f64> = vals
            .iter()
            .zip(&pure)
            .map(|(&v, &p)| if p { v } else { v + 0.01 })
            .collect();
        let (fm, fvals, f_counts) = measure(&g, &residuals, &syms, &pure, &vals, &check);
        let (rows, s_diff) = timed(|| sparse_jacobian(&mut g, &residuals, &states));
        let (mut ri, mut ci, mut entries) = (Vec::new(), Vec::new(), Vec::new());
        for (i, row) in rows.iter().enumerate() {
            for &(j, e) in row {
                ri.push(i);
                ci.push(j);
                entries.push(e);
            }
        }
        let (jm, jvals, j_counts) = measure(&g, &entries, &syms, &pure, &vals, &check);
        let by_count = |name: &str| {
            if counts.len() > 1 {
                for ((n, f), j) in counts.iter().zip(&f_counts).zip(&j_counts) {
                    println!("threads,{name},{n},{:.4},{:.4}", f * 1e6, j * 1e6);
                }
            }
        };
        let name = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        if !folding {
            println!(
                "{name},{},{:.6},{:.6},{:.4},{:.4},{:.6},{:.6},{:.4},{:.4},{},,,,",
                states.len(),
                fm[0],
                fm[1],
                fm[2] * 1e6,
                fm[3] * 1e6,
                s_diff + jm[0],
                s_diff + jm[1],
                jm[2] * 1e6,
                jm[3] * 1e6,
                entries.len()
            );
            by_count(&name);
            continue;
        }
        let (folded, s_fold) = timed(|| {
            let inlined = g.inline_all(&residuals);
            let map = syms
                .iter()
                .zip(&pure)
                .zip(&vals)
                .filter(|((_, &p), _)| p)
                .map(|((&s, _), &v)| (s, g.konst_f64(v)))
                .collect();
            substitute(&mut g, &inlined, &map)
        });
        let xs: Vec<f64> = vals
            .iter()
            .zip(&pure)
            .filter(|(_, &p)| !p)
            .map(|(&v, _)| v)
            .collect();
        let xc: Vec<f64> = xs.iter().map(|v| v + 0.01).collect();
        let no_split = vec![false; states.len()];
        let (ff, ffv, _) = measure(&g, &folded, &states, &no_split, &xs, &xc);
        let (frows, s_fdiff) = timed(|| sparse_jacobian(&mut g, &folded, &states));
        let fentries: Vec<ExprId> = frows.iter().flatten().map(|&(_, e)| e).collect();
        let (fj, fjv, _) = measure(&g, &fentries, &states, &no_split, &xs, &xc);
        let close = |a: &[f64], b: &[f64]| {
            let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
            a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-9 * scale)
        };
        assert!(close(&ffv, &fvals), "folded residual differs");
        let dense = |rows: &[usize], cols: &[usize], v: &[f64]| {
            let n = states.len();
            let mut d = vec![0.0; n * n];
            for ((&r, &c), &x) in rows.iter().zip(cols).zip(v) {
                d[r * n + c] = x;
            }
            d
        };
        let (fr, fc): (Vec<usize>, Vec<usize>) = frows
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().map(move |&(j, _)| (i, j)))
            .unzip();
        assert!(
            close(&dense(&fr, &fc, &fjv), &dense(&ri, &ci, &jvals)),
            "folded Jacobian differs"
        );
        println!(
            "{name},{},{:.6},{:.6},{:.4},{:.4},{:.6},{:.6},{:.4},{:.4},{},{:.6},{:.4},{:.6},{:.4}",
            states.len(),
            fm[0],
            fm[1],
            fm[2] * 1e6,
            fm[3] * 1e6,
            s_diff + jm[0],
            s_diff + jm[1],
            jm[2] * 1e6,
            jm[3] * 1e6,
            entries.len(),
            s_fold + ff[1],
            ff[3] * 1e6,
            s_fold + s_fdiff + fj[1],
            fj[3] * 1e6
        );
        by_count(&name);
        if let Some(dir) = &values_dir {
            let out = serde_json::json!({ "f": fvals, "j_rows": ri, "j_cols": ci, "j": jvals });
            std::fs::write(format!("{dir}/{name}.values.json"), out.to_string())
                .expect("write values");
        }
    }
}

/// The module at `path` (its graph `g`, its system `f`, its JSON `v`) as
/// rsdag compiles it: the system's outputs with the composite functions
/// inlined, a new system function over them with the same roles, written
/// to `dir` under the module's file name.
fn write_composed(
    g: &mut Graph<F64>,
    f: rsdag::FuncId,
    v: &serde_json::Value,
    dir: &str,
    path: &str,
) {
    let zero = g.zero();
    let func = g.func(f);
    let params = func.params().to_vec();
    let roles = func.param_roles().to_vec();
    let out_roles = func.output_roles().to_vec();
    let outs: Vec<ExprId> = func
        .outputs()
        .iter()
        .map(|o| match *o {
            Output::Expr(e) => e,
            _ => zero,
        })
        .collect();
    let program = g.inline_composite(&outs);
    let c = g.define_func("circuit", params, program);
    for (k, &r) in roles.iter().enumerate() {
        g.set_param_role(c, k as u32, r);
    }
    for (k, &r) in out_roles.iter().enumerate() {
        g.set_output_role(c, k as u32, r);
    }
    let mut out = v.clone();
    out["module"] = serde_json::to_value(g.to_module()).expect("a module as JSON");
    out["circuit"] = serde_json::json!(c.0);
    let name = std::path::Path::new(path).file_name().unwrap();
    std::fs::write(std::path::Path::new(dir).join(name), out.to_string()).expect("write");
}
