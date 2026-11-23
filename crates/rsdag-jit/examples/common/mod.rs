//! What the examples share.

use std::time::Instant;

/// The shortest batch of calls a timing takes, in seconds.
const BATCH_S: f64 = 0.05;

/// Seconds per call of `f`: the best of five batches of at least
/// [`BATCH_S`]. Every call adds to the measure `evaluations` at the current
/// rsprobe node.
pub fn per_call(mut f: impl FnMut()) -> f64 {
    let mut reps = 1usize;
    let mut dt;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        dt = t.elapsed().as_secs_f64();
        rsprobe::add!("evaluations", reps);
        if dt > BATCH_S {
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
        rsprobe::add!("evaluations", reps);
    }
    best
}

/// An rsprobe context entered on this thread for as long as it lives, its
/// parameters set through it (`Deref`). On drop its record goes to the path
/// in `RSPROBE_RECORD`: that file, or in that directory a file of its own
/// (`<name>.<process id>.json`). Without the feature `rsprobe/on` the record
/// holds the run's parameters alone.
pub struct Run {
    name: String,
    ctx: rsprobe::Context,
    on: Option<rsprobe::Guard>,
}

impl Run {
    /// A run named `name`, a parameter of its record.
    pub fn new(name: &str) -> Run {
        let ctx = rsprobe::Context::new();
        ctx.param("name", name);
        let on = Some(ctx.enter());
        Run {
            name: name.to_string(),
            ctx,
            on,
        }
    }
}

impl std::ops::Deref for Run {
    type Target = rsprobe::Context;
    fn deref(&self) -> &rsprobe::Context {
        &self.ctx
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        // the root's span is in the record once its guard has dropped
        self.on.take();
        let Some(path) = std::env::var_os("RSPROBE_RECORD") else {
            return;
        };
        let mut path = std::path::PathBuf::from(path);
        if path.is_dir() {
            path.push(format!("{}.{}.json", self.name, std::process::id()));
        }
        std::fs::write(&path, self.ctx.record().to_string()).expect("write the record");
    }
}
