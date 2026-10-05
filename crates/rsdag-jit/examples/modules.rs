//! rsdag on DAE modules: a module JSON as SANE's `export_module` writes it
//! (the rsdag module, the index of the system function, the parameter
//! values by name and a point `x0`). The residual is the DC one, as the
//! other tools get it: the derivatives and time bound to the constant zero
//! and every call specialized to that (`Graph::specialize_calls`). The
//! residual and its Jacobian are derived on the module's functions and
//! compiled as programs over them (`Graph::inline_composite`, the setup
//! includes it): a subcircuit's function inlined, the device bodies called.
//! The states are the inputs; the parameters are bound once, so their work
//! runs in the prolog. Per module: the setup and the time per call of the
//! residual and of its sparse Jacobian in the states, interpreted and
//! native; then natively with the parameters folded (the calls inlined and
//! every parameter a constant, each instance's parameter branches decided
//! at build time, as a tool that folds constants does). One CSV row each.
//! `--values <dir>` also writes the residual and the Jacobian at
//! `x0 + 0.01` there (off the solution, where the residual is not zero),
//! for comparing other tools against. `--composed <dir>` writes each module
//! as rsdag compiles it, the subcircuit functions inlined and the device
//! bodies functions, the form the other tools are given.
//!
//! `--threads <n>` runs the residual and the Jacobian with a pool of `n`
//! threads installed, their independent calls (the device instances) in
//! parallel (`rsdag::parallel`): rsdag's `Workers`, or with `--rayon` a
//! rayon pool.
//!
//!     cargo run --release -p rsdag-jit --example modules -- [--values <dir>] [--composed <dir>] [--threads <n> [--rayon]] <module.json>...

use std::time::Instant;

use rsdag::{
    sparse_jacobian, substitute, ExprId, Graph, Module, Output, OutputRole, ParamRole, Tape, F64,
};
use rsdag_jit::NativeTape;

/// Seconds per call: the best of five batches of at least 50 ms.
fn per_call(mut f: impl FnMut()) -> f64 {
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

/// The pool `--threads` asks for: the programs' independent calls run on
/// it (`rsdag::parallel`).
static POOL: std::sync::OnceLock<Option<rsdag::parallel::Parallel>> = std::sync::OnceLock::new();

/// `f` with the `--threads` pool installed, if any.
fn on_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match POOL.get().cloned().flatten() {
        Some(p) => rsdag::parallel::install(p, f),
        None => f(),
    }
}

/// Setup (seconds) and per-call time (seconds) of `roots` over the inputs,
/// interpreted then native; the values at `check`.
fn measure(
    g: &Graph<F64>,
    roots: &[ExprId],
    syms: &[rsdag::SymbolId],
    pure: &[bool],
    vals: &[f64],
    check: &[f64],
) -> ([f64; 4], Vec<f64>) {
    let (tape, s_tape) = timed(|| Tape::compile_split(g, roots, syms, pure));
    let (native, s_native) = timed(|| NativeTape::compile(&tape).expect("native code"));
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let (mut wn, mut on) = (Vec::new(), Vec::new());
    let (c_tape, c_native) = on_pool(|| {
        tape.eval_prolog(vals, &mut w);
        let c_tape = per_call(|| tape.eval_main(vals, &mut w, &mut o));
        native.eval_prolog(vals, &mut wn);
        let c_native = per_call(|| native.eval_main(vals, &mut wn, &mut on));
        (c_tape, c_native)
    });
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
    ([s_tape, s_tape + s_native, c_tape, c_native], on)
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
    let threads: usize = option("--threads").map_or(1, |t| t.parse().expect("--threads <n>"));
    let rayon_pool = args
        .iter()
        .position(|a| a == "--rayon")
        .map(|i| args.remove(i))
        .is_some();
    let pool = (threads > 1).then(|| {
        let pool: std::sync::Arc<dyn rsdag::parallel::Pool> = if rayon_pool {
            std::sync::Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("a thread pool"),
            )
        } else {
            std::sync::Arc::new(rsdag::parallel::Workers::new(threads))
        };
        rsdag::parallel::Parallel::new(pool)
    });
    POOL.set(pool).ok();
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
        let (program, s_inline) = timed(|| g.inline_composite(&residuals));
        let (fm, fvals) = measure(&g, &program, &syms, &pure, &vals, &check);
        let (rows, s_diff) = timed(|| sparse_jacobian(&mut g, &residuals, &states));
        let (mut ri, mut ci, mut entries) = (Vec::new(), Vec::new(), Vec::new());
        for (i, row) in rows.iter().enumerate() {
            for &(j, e) in row {
                ri.push(i);
                ci.push(j);
                entries.push(e);
            }
        }
        let (entries_program, s_jinline) = timed(|| g.inline_composite(&entries));
        let (jm, jvals) = measure(&g, &entries_program, &syms, &pure, &vals, &check);
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
        let (ff, ffv) = measure(&g, &folded, &states, &no_split, &xs, &xc);
        let (frows, s_fdiff) = timed(|| sparse_jacobian(&mut g, &folded, &states));
        let fentries: Vec<ExprId> = frows.iter().flatten().map(|&(_, e)| e).collect();
        let (fj, fjv) = measure(&g, &fentries, &states, &no_split, &xs, &xc);
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
        let name = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy();
        println!(
            "{name},{},{:.6},{:.6},{:.4},{:.4},{:.6},{:.6},{:.4},{:.4},{},{:.6},{:.4},{:.6},{:.4}",
            states.len(),
            s_inline + fm[0],
            s_inline + fm[1],
            fm[2] * 1e6,
            fm[3] * 1e6,
            s_diff + s_jinline + jm[0],
            s_diff + s_jinline + jm[1],
            jm[2] * 1e6,
            jm[3] * 1e6,
            entries.len(),
            s_fold + ff[1],
            ff[3] * 1e6,
            s_fold + s_fdiff + fj[1],
            fj[3] * 1e6
        );
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
