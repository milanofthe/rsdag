//! A hot program on the fastest backend that serves it, switched per call.
//!
//! Three backends evaluate one tape, and every one is bit-exact against the
//! others, so switching mid-solve is sound:
//!
//! 1. **Native code**, when a [`Compiler`] is given (`rsdag-jit` has one):
//!    after a few evaluations the tape is compiled in the background; once
//!    it lands it wins. A large tape, one whose interpreted pass is long,
//!    is compiled at once, and its passes wait for the compile rather than
//!    interpret alongside it.
//! 2. **Choice specialization** (interpreter): models branch by operating
//!    region (`Select`) and the full tape evaluates both arms. After a
//!    traced evaluation the tape is shortened against the current choices
//!    ([`Tape::specialize`]) and runs guarded, under the policy of
//!    [`crate::choices`]: a region flip retraces and respecializes, a
//!    select seen flipping is unpinned for good, a tape whose choices
//!    thrash opts out, and a specialization that holds is compiled too.
//! 3. **The interpreter**, always correct, always available.
//!
//! The prolog/main split runs in episodes: [`Adaptive::eval_prolog`] once
//! per parameter binding, [`Adaptive::eval_main`] per iteration over the
//! same buffer. The interpreter and the native code share the state layout
//! ([`Program::state_len`]), so an episode moves from one to the other at any
//! time; only a specialized episode is tied to its specialization, and
//! falls back to the full tape when a region flips.
//!
//! Each pass adds one to `evals interpreted`, `evals native` or `evals
//! specialized` at the caller's rsprobe node, by the form that ran it (a
//! specialized pass whose guards fail counts on the full form too).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use rspool::Instant;

use crate::choices::{ChoicePolicy, Specializer};
use crate::{Program, SpecializedTape, Tape};

/// Native code for [`Adaptive`], from outside the core: `rsdag-jit`
/// implements it; without one the interpreter and its specialization serve.
pub trait Compiler: Send + Sync {
    /// `tape` as native code, keeping the slots `live` in the work array
    /// after a run (the prolog guards of a specialization are read there);
    /// `None` when it cannot be compiled.
    fn compile(&self, tape: &Tape, live: &[u32]) -> Option<Box<dyn Program>>;
    /// Run `job` in the background.
    fn submit(&self, job: Box<dyn FnOnce() + Send>);
}

type Native = Arc<dyn Program>;

/// When to compile and when to specialize. The defaults are what a circuit
/// simulator's Newton loop measured best with.
#[derive(Clone, Debug)]
pub struct Policy {
    /// Compile natively at all.
    pub jit: bool,
    /// Evaluations before the background compile is kicked off: a couple of
    /// interpreted passes filter out one-shot tapes.
    pub kick_after: u32,
    /// Whether a pass waits for the native code rather than interpret
    /// where that is expected to finish it sooner: while the compile is in
    /// flight, a pass waits when the compile's expected remainder is
    /// shorter than an interpreted pass (the slowest so far, before any
    /// the tape's ops times [`INTERPRETED_NS_PER_OP`](crate::constants::INTERPRETED_NS_PER_OP)),
    /// at most as long as that pass, before it interprets; and a tape whose
    /// one interpreted pass is expected to cost more than its whole compile
    /// ([`COMPILE_NS_PER_OP`](crate::constants::COMPILE_NS_PER_OP) per op of
    /// it and its bodies) has the compile kicked at its first pass. A pass
    /// so loses at most what it would have spent interpreting.
    pub wait: bool,
    /// How the program specializes over its region choices (its bodies'
    /// are [`VariantPolicy::choices`](crate::variant::VariantPolicy::choices)).
    pub choices: ChoicePolicy,
    /// How the program's bodies specialize per parameter binding. Where
    /// the program is compiled, the compiled form builds the variants (on
    /// the compiler's background queue) and the interpreted one runs its
    /// bodies whole meanwhile; without, the interpreter specializes.
    pub variants: crate::variant::VariantPolicy,
}

impl Default for Policy {
    fn default() -> Self {
        Policy::DEFAULT
    }
}

struct Jit {
    compiled: OnceLock<Option<Native>>,
    kicked: AtomicBool,
    evals: AtomicU32,
    slowest_interpreted_pass_ns: AtomicU64,
    pass_prior: Duration,
    compile_expected: Duration,
    kicked_at: OnceLock<Instant>,
    landed: (Mutex<bool>, Condvar),
}

struct Spec {
    learned: Specializer,
    spec: Option<(u64, Arc<SpecializedTape>, Option<Native>)>,
}

impl Spec {
    fn current(&self) -> Option<(u64, Arc<SpecializedTape>, Option<Native>)> {
        let v = self.learned.current()?;
        self.spec.clone().filter(|(at, ..)| *at == v)
    }
}

/// What an [`Adaptive`] has done so far (see [`Adaptive::stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// The full tape's native code has landed.
    pub native: bool,
    /// Specializations built, and region flips seen.
    pub specializations: u64,
    pub flips: u64,
    /// A specialization runs as native code now.
    pub spec_native: bool,
    /// Specialization is off for this tape (thrash, or too little shrink).
    pub spec_off: bool,
}

/// Which specialization, if any, prepared an episode's buffer (see
/// [`Adaptive::eval_prolog`]), and the forms it saw (see
/// [`Adaptive::eval_main`]). The full tape and its native code share their
/// layout, so they are one kind of episode.
#[derive(Clone)]
pub struct Episode {
    kind: Kind,
    forms_epoch: u64,
}

#[derive(Clone)]
enum Kind {
    Full,
    SpecInterp(Arc<SpecializedTape>, u64),
    SpecNative(Native, Arc<SpecializedTape>, u64),
}

/// A tape and its accelerated forms, arbitrated per call.
pub struct Adaptive {
    tape: Arc<Tape>,
    policy: Policy,
    compiler: Option<Arc<dyn Compiler>>,
    spec: Option<Arc<Mutex<Spec>>>,
    jit: Arc<Jit>,
}

impl Adaptive {
    /// `compiler` supplies the native code; without it the interpreter and
    /// its specialization serve.
    pub fn new(mut tape: Tape, policy: Policy, compiler: Option<Arc<dyn Compiler>>) -> Adaptive {
        let submit = compiler.clone().map(|c| -> crate::Submit {
            Arc::new(move |job: Box<dyn FnOnce() + Send>| c.submit(job))
        });
        let compiled = policy.jit && compiler.is_some();
        tape.with_backend(&crate::BodyBackend {
            submit,
            variants: Some(crate::variant::VariantPolicy {
                interpreted: policy.variants.interpreted && !compiled,
                ..policy.variants
            }),
            ..crate::BodyBackend::default()
        });
        let learned = Specializer::new(tape.n_selects());
        let spec = learned.eligible(&policy.choices).then(|| {
            Arc::new(Mutex::new(Spec {
                learned,
                spec: None,
            }))
        });
        let (pass_prior, compile_expected) = expected(&tape);
        Adaptive {
            tape: Arc::new(tape),
            policy,
            compiler,
            spec,
            jit: Arc::new(Jit {
                compiled: OnceLock::new(),
                kicked: AtomicBool::new(false),
                evals: AtomicU32::new(0),
                slowest_interpreted_pass_ns: AtomicU64::new(0),
                pass_prior,
                compile_expected,
                kicked_at: OnceLock::new(),
                landed: (Mutex::new(false), Condvar::new()),
            }),
        }
    }

    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// What the program has done so far (diagnostics; the specialization
    /// counters are read only when no evaluation holds them).
    pub fn stats(&self) -> Stats {
        let mut s = Stats {
            native: self.native().is_some(),
            ..Stats::default()
        };
        if let Some(Ok(st)) = self.spec.as_ref().map(|c| c.try_lock()) {
            s.specializations = st.learned.built;
            s.flips = st.learned.flips;
            s.spec_native = st.current().is_some_and(|(.., native)| native.is_some());
            s.spec_off = st.learned.off;
        }
        s
    }

    /// The native code of the full tape, once compiled.
    pub fn native(&self) -> Option<&dyn Program> {
        self.jit.compiled.get().and_then(Option::as_deref)
    }

    fn forms_epoch(&self) -> u64 {
        let native = self.jit.compiled.get().is_some();
        self.tape.forms_epoch().wrapping_add(u64::from(native))
    }

    /// The whole program on the best rung: native specialization, native
    /// full tape, interpreted specialization, interpreter.
    pub fn eval(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        self.maybe_kick();
        self.await_native();
        let clock = self.clock();
        self.eval_rung(inputs, work, out);
        self.clocked(clock);
    }

    // https://github.com/milanofthe/rsdag/issues/242
    fn eval_rung(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        if let Some(cache) = &self.spec {
            if let Ok(mut st) = cache.try_lock() {
                if self.eval_spec(&mut st, cache, inputs, work, out) {
                    return;
                }
            }
        }
        self.eval_full(inputs, work, out);
    }

    fn eval_full(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        match self.native() {
            Some(native) => {
                rsprobe::add!("evals native", 1);
                native.eval(inputs, work, out)
            }
            None => {
                rsprobe::add!("evals interpreted", 1);
                self.tape.eval(inputs, work, out)
            }
        }
    }

    fn eval_spec(
        &self,
        st: &mut Spec,
        cache: &Arc<Mutex<Spec>>,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) -> bool {
        st.learned.ran(1);
        let Some((ver, sp, native)) = st.current() else {
            if !st.learned.due(&self.policy.choices) {
                return false;
            }
            self.retrace(st, inputs, work, out);
            return true;
        };
        let held = match native {
            Some(native) => {
                rsprobe::add!("evals specialized", 1);
                native.eval(inputs, work, out);
                sp.check_outputs(out)
            }
            None if !self.interpreted_serves(&sp) => {
                self.maybe_compile_spec(st, cache);
                return false;
            }
            None => {
                rsprobe::add!("evals specialized", 1);
                sp.eval_checked(inputs, work, out)
            }
        };
        if held {
            st.learned.held(ver, 1);
            self.maybe_compile_spec(st, cache);
        } else {
            self.on_flip(st, ver, inputs, work, out);
        }
        true
    }

    fn interpreted_serves(&self, sp: &SpecializedTape) -> bool {
        let (ops, full) = (sp.n_ops(), self.tape.n_ops());
        Specializer::interpreted_serves(&self.policy.choices, ops, full, self.native().is_some())
    }

    /// Phase 1 of an episode: prepare `work` for the main passes and name
    /// the backend that prepared it, which [`eval_main`](Self::eval_main)
    /// stays compatible with.
    pub fn eval_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) -> Episode {
        self.maybe_kick();
        self.await_native();
        let clock = self.clock();
        let ep = self.prolog_rung(inputs, work);
        self.clocked(clock);
        ep
    }

    fn prolog_rung(&self, inputs: &[f64], work: &mut Vec<f64>) -> Episode {
        let forms_epoch_before_prolog = self.forms_epoch();
        if let Some(cache) = &self.spec {
            if let Ok(mut st) = cache.try_lock() {
                if let Some(kind) = self.spec_prolog(&mut st, inputs, work) {
                    return Episode {
                        kind,
                        forms_epoch: forms_epoch_before_prolog,
                    };
                }
            }
        }
        self.full_prolog(inputs, work);
        Episode {
            kind: Kind::Full,
            forms_epoch: forms_epoch_before_prolog,
        }
    }

    fn spec_prolog(&self, st: &mut Spec, inputs: &[f64], work: &mut Vec<f64>) -> Option<Kind> {
        st.learned.ran(1);
        if st.current().is_none() {
            if !st.learned.due(&self.policy.choices) {
                return None;
            }
            self.retrace(st, inputs, work, &mut Vec::new());
        }
        let (ver, sp, native) = st.current()?;
        if let Some(native) = native {
            native.eval_prolog(inputs, work);
            if sp.check_prolog_guards(inputs, work) {
                return Some(Kind::SpecNative(native, sp, ver));
            }
        } else if !self.interpreted_serves(&sp) {
            return None;
        } else if sp.eval_prolog_checked(inputs, work) {
            return Some(Kind::SpecInterp(sp, ver));
        }
        self.on_flip(st, ver, inputs, work, &mut Vec::new());
        Some(Kind::Full)
    }

    fn full_prolog(&self, inputs: &[f64], work: &mut Vec<f64>) {
        match self.native() {
            Some(native) => native.eval_prolog(inputs, work),
            None => self.tape.eval_prolog(inputs, work),
        }
    }

    /// Phase 2: a main pass over the buffer the episode's prolog prepared.
    /// A region flip under a specialization moves the episode to the full
    /// tape (the retrace that handles the flip leaves the full tape's state
    /// in `work`), for its remainder. `inputs` hold the binding the prolog
    /// ran at, so the episode may run it again when a faster form lands:
    /// the native code, or a function body's variant (see
    /// [`Tape::forms_epoch`]).
    pub fn eval_main(
        &self,
        ep: &mut Episode,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        self.maybe_kick();
        self.await_native();
        let clock = self.clock();
        self.main_rung(ep, inputs, work, out);
        self.clocked(clock);
    }

    // https://github.com/milanofthe/rsdag/issues/242
    fn main_rung(&self, ep: &mut Episode, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        if self.forms_epoch() != ep.forms_epoch {
            *ep = self.eval_prolog(inputs, work);
        }
        match &ep.kind {
            Kind::SpecNative(native, sp, ver) => {
                rsprobe::add!("evals specialized", 1);
                native.eval_main(inputs, work, out);
                if sp.check_outputs(out) {
                    return;
                }
                self.flip_retrace(*ver, inputs, work, out);
                ep.kind = Kind::Full;
            }
            Kind::SpecInterp(sp, ver) => {
                rsprobe::add!("evals specialized", 1);
                if sp.eval_main_checked(inputs, work, out) {
                    if let Some(cache) = &self.spec {
                        if let Ok(mut st) = cache.try_lock() {
                            st.learned.ran(1);
                            st.learned.held(*ver, 1);
                            self.maybe_compile_spec(&mut st, cache);
                        }
                    }
                    return;
                }
                self.flip_retrace(*ver, inputs, work, out);
                ep.kind = Kind::Full;
            }
            Kind::Full => match self.native() {
                Some(native) => {
                    rsprobe::add!("evals native", 1);
                    native.eval_main(inputs, work, out)
                }
                None => {
                    rsprobe::add!("evals interpreted", 1);
                    self.tape.eval_main(inputs, work, out)
                }
            },
        }
    }

    fn retrace(&self, st: &mut Spec, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        let mut fresh = Vec::new();
        rsprobe::add!("evals interpreted", 1);
        self.tape.eval_with(inputs, work, out, &mut fresh);
        let tape = &self.tape;
        let built = st
            .learned
            .learn(&self.policy.choices, [&fresh[..]], tape.n_ops(), |t, p| {
                let sp = tape.specialize(t, p);
                let ops = sp.n_ops();
                (sp, ops)
            });
        if let Some((ver, sp)) = built {
            st.spec = Some((ver, Arc::new(sp), None));
        }
    }

    fn flip_retrace(&self, ver: u64, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) {
        match self.spec.as_ref().map(|c| c.lock()) {
            Some(Ok(mut st)) => self.on_flip(&mut st, ver, inputs, work, out),
            _ => {
                rsprobe::add!("evals interpreted", 1);
                self.tape.eval(inputs, work, out)
            }
        }
    }

    fn on_flip(
        &self,
        st: &mut Spec,
        ver: u64,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        if st.learned.failed(&self.policy.choices, ver, 1) {
            self.retrace(st, inputs, work, out);
        } else {
            self.eval_full(inputs, work, out);
        }
    }

    fn maybe_compile_spec(&self, st: &mut Spec, cache: &Arc<Mutex<Spec>>) {
        let Some(compiler) = self.compiler.clone().filter(|_| self.policy.jit) else {
            return;
        };
        let Some((at, sp, native)) = st.current() else {
            return;
        };
        let (have, serves) = (native.map(|_| at), self.interpreted_serves(&sp));
        let Some(ver) = st.learned.compile_due(&self.policy.choices, have, serves) else {
            return;
        };
        let (cache, c) = (cache.clone(), compiler.clone());
        compiler.submit(Box::new(move || {
            let materialized_guard_slots: Vec<u32> =
                sp.prolog_guards().iter().map(|&(s, _)| s).collect();
            let compiled = c.compile(sp.tape(), &materialized_guard_slots);
            let mut st = cache.lock().unwrap_or_else(|e| e.into_inner());
            if let (true, Some(compiled)) = (st.learned.landed(ver), compiled) {
                rsprobe::debug!("specialized tape compiled, native backend active");
                if let Some((_, _, native)) = &mut st.spec {
                    *native = Some(Arc::from(compiled));
                }
            }
        }));
    }

    fn pass(&self) -> Duration {
        match self.jit.slowest_interpreted_pass_ns.load(Ordering::Relaxed) {
            0 => self.jit.pass_prior,
            ns => Duration::from_nanos(ns),
        }
    }

    fn clock(&self) -> Option<Instant> {
        let timed = self.policy.jit
            && self.policy.wait
            && self.compiler.is_some()
            && self.jit.compiled.get().is_none();
        timed.then(Instant::now)
    }

    fn clocked(&self, clock: Option<Instant>) {
        if let Some(t) = clock {
            let ns = u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.jit
                .slowest_interpreted_pass_ns
                .fetch_max(ns, Ordering::Relaxed);
        }
    }

    fn await_native(&self) {
        let Some(kicked) = self.jit.kicked_at.get() else {
            return;
        };
        let pass = self.pass();
        let remaining = self.jit.compile_expected.saturating_sub(kicked.elapsed());
        if !self.policy.wait || self.jit.compiled.get().is_some() || remaining >= pass {
            return;
        }
        let (lock, cv) = &self.jit.landed;
        let landed = lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = cv.wait_timeout_while(landed, pass, |landed| !*landed);
    }

    fn maybe_kick(&self) {
        let Some(compiler) = self.compiler.clone() else {
            return;
        };
        let evals = self.jit.evals.fetch_add(1, Ordering::Relaxed) + 1;
        let costly = self.policy.wait && self.pass() >= self.jit.compile_expected;
        if !self.policy.jit
            || (evals < self.policy.kick_after && !costly)
            || self.jit.kicked.swap(true, Ordering::Relaxed)
        {
            return;
        }
        let _ = self.jit.kicked_at.set(Instant::now());
        let (tape, jit) = (self.tape.clone(), self.jit.clone());
        let c = compiler.clone();
        compiler.submit(Box::new(move || {
            let compiled = c.compile(&tape, &[]);
            match compiled {
                Some(_) => rsprobe::debug!("hot tape compiled, native backend active"),
                None => rsprobe::warn!("tape compile failed, staying interpreted"),
            }
            let _ = jit.compiled.set(compiled.map(Arc::from));
            let (lock, cv) = &jit.landed;
            *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
            cv.notify_all();
        }));
    }
}

/// A program's evaluations at one binding: its inputs, its work buffer and
/// the episode its prolog prepared, owned together. Binding runs the
/// parameter-pure prolog once; an evaluation runs the main phase at the
/// inputs as they are, the ones that vary set through
/// [`inputs_mut`](Self::inputs_mut). It belongs to one program, which it
/// does not hold, so a holder may keep the program next to it; clone it per
/// thread that evaluates the program.
#[derive(Clone)]
pub struct Session {
    episode: Episode,
    inputs: Vec<f64>,
    work: Vec<f64>,
}

impl Session {
    /// `program` bound at `inputs`, its prolog run there.
    pub fn new(program: &Adaptive, inputs: Vec<f64>) -> Session {
        let mut work = Vec::new();
        let episode = program.eval_prolog(&inputs, &mut work);
        Session {
            episode,
            inputs,
            work,
        }
    }

    /// The inputs the next evaluation runs at.
    pub fn inputs(&self) -> &[f64] {
        &self.inputs
    }

    /// The inputs, to set the ones that vary between evaluations (a pure
    /// one keeps the value it was bound at).
    pub fn inputs_mut(&mut self) -> &mut [f64] {
        &mut self.inputs
    }

    /// `program` at the inputs into `out`.
    pub fn eval(&mut self, program: &Adaptive, out: &mut Vec<f64>) {
        program.eval_main(&mut self.episode, &self.inputs, &mut self.work, out);
    }

    /// `program` at `n` points into `out`, output-major: output `o` of point
    /// `k` at `out[o * n + k]`. `set(k, inputs)` sets point `k`'s inputs;
    /// `row` is the caller's scratch.
    pub fn eval_points(
        &mut self,
        program: &Adaptive,
        n: usize,
        mut set: impl FnMut(usize, &mut [f64]),
        row: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        let width = program.tape().n_outputs();
        out.clear();
        out.resize(width * n, 0.0);
        for k in 0..n {
            set(k, &mut self.inputs);
            self.eval(program, row);
            for (o, &v) in row.iter().take(width).enumerate() {
                out[o * n + k] = v;
            }
        }
    }
}

fn expected(tape: &Tape) -> (Duration, Duration) {
    let body = |b: u32| tape.bundles()[b as usize].body();
    let pass: usize = (tape.ops()[tape.prolog_len()..].iter())
        .map(|op| match *op {
            crate::tape::Op::Call {
                bundle, n_groups, ..
            } => body(bundle).map_or(1, |b| b.n_ops() - b.prolog_len()) * n_groups as usize,
            _ => 1,
        })
        .sum();
    let bodies: usize = tape
        .bundles()
        .iter()
        .filter_map(|b| b.body())
        .map(Tape::n_ops)
        .sum();
    let ns = |n: usize, per: f64| Duration::from_nanos((n as f64 * per) as u64);
    (
        ns(pass, f64::from(crate::constants::INTERPRETED_NS_PER_OP)),
        ns(tape.n_ops() + bodies, crate::constants::COMPILE_NS_PER_OP),
    )
}
