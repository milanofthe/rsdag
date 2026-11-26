//! Function bodies specialized per parameter binding.
//!
//! A device model branches on its parameters: a polarity, a model level, a
//! switch that shorts a resistance. Lowered exactly, each branch is a
//! `Select`, and a tape computes both of its arms at every evaluation. Its
//! condition, though, is fixed once the parameters are bound: a prolog
//! value. A [`VariantBody`] reads those conditions in each instance's
//! prolog, the pattern they take, and runs the instance on the body that
//! pattern decides ([`Tape::decide`]), its prolog and its main phase: the
//! untaken arms gone, the outputs bit for bit the full body's. A variant is
//! built when its pattern first turns up and is shared by every instance
//! that takes it; a new binding that flips a condition moves its instances
//! to another variant at their next prolog, so nothing is ever
//! invalidated. A derivative body is a body of the function like any
//! other, its selects decided the same way, so derivatives with respect to
//! the parameters stay exact.
//!
//! An instance's state is a block of the full body's state length and one
//! value more, its variant's index: the variant's own state at the start
//! of the block, laid out as the variant lays it out. A batch runs the
//! instances of each variant together, listed in place ([`Instances`]), so
//! the instances of one variant fill its lanes however they interleave (a
//! polarity alternating along a ring) and nothing is copied.
//!
//! How a body specializes is its program's to say ([`VariantPolicy`],
//! through [`BodyBackend`]): a program's owner sets the policy, a backend
//! that compiles the program keeps it. Without a place for background work
//! ([`BodyBackend::submit`]) the selects are found and a variant is built in
//! the prolog that first needs it, so what runs is a function of the calls
//! made alone. With one, that work goes there, and an instance runs the
//! full body, which computes the same values, until its variant is ready:
//! a prolog never waits on a compilation.
//!
//! The selects left in a variant are decided at run time, by the state
//! (an operating region), and each variant specializes over them too
//! ([`crate::choices`]): a trace of some of its instances pins the arms
//! they take, the main phase shortened against it over the variant's own
//! state ([`Tape::specialize_main`], one output counting the guards that
//! fail). A batch runs on it, and an instance whose guards fail runs again
//! on the variant's main phase, its result the same either way; a trace of
//! those instances unpins the selects that flipped, and the main phase is
//! shortened again without them. When to trace, rebuild, compile and give
//! up is the one policy of [`crate::choices`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::choices::{ChoicePolicy, Specializer};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::extern_fn::{BackendCache, BodyBackend, BodyCompiler, ExternBundle, Instances};
use crate::func::InterpretedBody;
use crate::tape::{Guards, ParamSelects, Program as _, Tape};

use crate::constants::{PERCENT, TRACED};

const FULL_BODY: u32 = 0;

/// How bodies specialize per binding (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariantPolicy {
    /// Specialize at all; off, every body runs whole.
    pub enabled: bool,
    /// Whether a form that interprets its bodies specializes them too. Off
    /// where a compiler takes the program over: the interpreted phase then
    /// spends nothing on variants the compiled form builds anyway.
    pub interpreted: bool,
    /// The bodies a function keeps, the full one included: a binding whose
    /// pattern turns up past them runs the full body.
    pub max_variants: usize,
    /// The share of the body, in percent, that deciding the selects must be
    /// able to remove (all of them one way, or all the other) for the body
    /// to run variants at all.
    pub min_shrink_pct: usize,
    /// How each variant specializes over the choices its instances take at
    /// run time (see [`crate::choices`]).
    pub choices: ChoicePolicy,
}

impl Default for VariantPolicy {
    fn default() -> Self {
        VariantPolicy::DEFAULT
    }
}

struct Found {
    selects: ParamSelects,
    shrink_pct: usize,
}

struct Shared {
    found: OnceLock<Option<Found>>,
    full_ops: usize,
    full_state_len: usize,
    bodies: RwLock<Vec<Arc<InterpretedBody>>>,
    choices: RwLock<Vec<Arc<Choice>>>,
    variant_of_pattern: Mutex<HashMap<Box<[bool]>, u32>>,
    build_lock: Mutex<()>,
    exhaustion_reported: AtomicBool,
    epoch: AtomicU64,
}

struct Choice {
    learned: Mutex<Specializer>,
    shortened_main: RwLock<Option<(u64, Arc<InterpretedBody>)>>,
}

impl Choice {
    fn new(body: &InterpretedBody) -> Arc<Choice> {
        let t = &body.tape;
        Arc::new(Choice {
            learned: Mutex::new(Specializer::new(t.n_main_selects())),
            shortened_main: RwLock::new(None),
        })
    }
}

fn main_ops(t: &Tape) -> usize {
    t.n_ops() - t.prolog_len()
}

fn form(body: &InterpretedBody, key: u64, compile: &BodyCompiler) -> Option<Arc<dyn ExternBundle>> {
    let made = body.backends.get_or_try_insert(key, || {
        let t = &body.tape;
        compile(t, body.pure_args(), body.n_outputs()).ok_or(())
    });
    made.ok()
}

impl Shared {
    fn body(&self, v: u32) -> Arc<InterpretedBody> {
        self.bodies.read().unwrap()[v as usize].clone()
    }

    fn choice(&self, v: usize) -> Arc<Choice> {
        self.choices.read().unwrap()[v].clone()
    }

    fn keep(&self, body: InterpretedBody) -> u32 {
        let mut bodies = self.bodies.write().unwrap();
        self.choices.write().unwrap().push(Choice::new(&body));
        bodies.push(Arc::new(body));
        bodies.len() as u32 - 1
    }

    fn find(&self) -> Option<Found> {
        let full = self.body(FULL_BODY);
        let t = &full.tape;
        let selects = t.param_selects(full.pure_args())?;
        let n = selects.n_conds();
        let (all, decided) = t.ops_decided(&selects, &[vec![true; n], vec![false; n]]);
        let least = decided.into_iter().min().unwrap_or(all).min(all);
        let shrink_pct = (all - least) * PERCENT / all.max(1);
        Some(Found {
            selects,
            shrink_pct,
        })
    }

    fn variant_of(&self, ps: &ParamSelects, pattern: &[bool], max: usize) -> u32 {
        let built = |s: &Shared| s.variant_of_pattern.lock().unwrap().get(pattern).copied();
        if let Some(v) = built(self) {
            return v;
        }
        let _building = self.build_lock.lock().unwrap();
        if let Some(v) = built(self) {
            return v;
        }
        let v = self.build(ps, pattern, max);
        self.variant_of_pattern
            .lock()
            .unwrap()
            .insert(pattern.into(), v);
        v
    }

    fn build(&self, ps: &ParamSelects, pattern: &[bool], max: usize) -> u32 {
        if self.bodies.read().unwrap().len() >= max {
            if !self.exhaustion_reported.swap(true, Ordering::Relaxed) {
                rsprobe::info!("a body keeps {max} variants; later patterns run it whole");
            }
            return FULL_BODY;
        }
        let full = self.body(FULL_BODY);
        let t = &full.tape;
        let decided = t.decide(ps, pattern);
        if decided.n_ops() >= self.full_ops || decided.state_len() > self.full_state_len {
            return FULL_BODY;
        }
        rsprobe::add!("variants", 1);
        let (pure, n_out) = (full.pure_args().to_vec(), full.n_outputs());
        self.keep(InterpretedBody::new(decided, n_out, pure))
    }

    fn learn(&self, v: usize, traces: &[Vec<u8>], policy: &ChoicePolicy) {
        let choice = self.choice(v);
        let mut learned = choice.learned.lock().unwrap();
        let body = self.body(v as u32);
        let t = &body.tape;
        let traces = traces.iter().map(Vec::as_slice);
        let built = learned.learn(policy, traces, main_ops(t), |trace, pin| {
            let spec = t.specialize_main(trace, pin, Guards::Count);
            let ops = main_ops(spec.tape());
            (spec, ops)
        });
        let mut spec = choice.shortened_main.write().unwrap();
        if let Some((ver, sp)) = built {
            let (n_out, pure) = (body.n_outputs() + 1, body.pure_args().to_vec());
            *spec = Some((
                ver,
                Arc::new(InterpretedBody::new(sp.into_tape(), n_out, pure)),
            ));
        } else if learned.current().is_none() {
            *spec = None;
        }
    }

    fn pattern(
        &self,
        ps: &ParamSelects,
        mask: &[bool],
        pure: &[f64],
        conds: Option<&Arc<dyn ExternBundle>>,
    ) -> Vec<bool> {
        crate::scratch::with(|args: &mut Vec<f64>| {
            args.clear();
            args.resize(mask.len(), f64::NAN);
            crate::extern_fn::spread_pure(mask, pure, args);
            crate::scratch::with(|o: &mut Vec<f64>| match conds {
                Some(b) => {
                    o.resize(ps.n_conds(), 0.0);
                    crate::scratch::with_len(b.work_len(), 0.0, |w| b.call_into(args, w, o));
                    o.iter().map(|&c| c != 0.0).collect()
                }
                None => crate::scratch::with(|w: &mut Vec<f64>| ps.pattern(args, w, o)),
            })
        })
    }
}

struct Form {
    shared: Arc<Shared>,
    full: Arc<InterpretedBody>,
    compile: Option<Arc<BodyCompiler>>,
    key: u64,
    policy: VariantPolicy,
    made: Box<[OnceLock<Arc<dyn ExternBundle>>]>,
    making: Box<[AtomicBool]>,
    conds: OnceLock<Arc<dyn ExternBundle>>,
    pending: Mutex<Pending>,
}

#[derive(Default)]
struct Pending {
    rows: Vec<Box<[f64]>>,
    seen: HashSet<Box<[u64]>>,
    queued: bool,
}

impl Form {
    fn ready(&self, pure: &[f64]) -> Option<u32> {
        let Some(found) = self.shared.found.get()? else {
            return Some(FULL_BODY);
        };
        if found.shrink_pct < self.policy.min_shrink_pct {
            return Some(FULL_BODY);
        }
        let mask = self.full.pure_args();
        let pattern = self
            .shared
            .pattern(&found.selects, mask, pure, self.conds.get());
        let v = *self
            .shared
            .variant_of_pattern
            .lock()
            .unwrap()
            .get(pattern.as_slice())? as usize;
        if v >= self.made.len() {
            return Some(FULL_BODY);
        }
        (self.compile.is_none() || self.made[v].get().is_some()).then_some(v as u32)
    }

    fn advance(&self, rows: &[Box<[f64]>]) {
        let _s = rsprobe::scope!("variants");
        rsprobe::add!("bindings", rows.len());
        let shared = &self.shared;
        let Some(found) = shared.found.get_or_init(|| shared.find()) else {
            return;
        };
        if found.shrink_pct < self.policy.min_shrink_pct {
            return;
        }
        for pure in rows {
            let mask = self.full.pure_args();
            let pattern = shared.pattern(&found.selects, mask, pure, self.conds.get());
            let v = shared.variant_of(&found.selects, &pattern, self.policy.max_variants);
            if (v as usize) < self.made.len() {
                self.make(v as usize);
            }
        }
        self.make_conds();
    }

    fn defer(self: &Arc<Self>, pure: &[f64], submit: &crate::Submit) {
        let mut p = self.pending.lock().unwrap();
        if p.seen.insert(pure.iter().map(|x| x.to_bits()).collect()) {
            p.rows.push(pure.into());
        }
        if std::mem::replace(&mut p.queued, true) {
            return;
        }
        drop(p);
        let form = self.clone();
        submit(Box::new(move || loop {
            let rows = {
                let mut p = form.pending.lock().unwrap();
                if p.rows.is_empty() {
                    p.queued = false;
                    p.seen.clear();
                    return;
                }
                std::mem::take(&mut p.rows)
            };
            form.advance(&rows);
            form.shared
                .choice(FULL_BODY as usize)
                .learned
                .lock()
                .unwrap()
                .cool(&form.policy.choices);
            form.shared.epoch.fetch_add(1, Ordering::Release);
        }));
    }

    fn advancing(&self) -> bool {
        self.pending.lock().unwrap().queued
    }

    fn serves(&self, v: usize, body: &InterpretedBody) -> bool {
        let ops = main_ops(&body.tape);
        let full = self.shared.body(v as u32);
        let full = main_ops(&full.tape);
        let compiled = self.compile.is_some();
        Specializer::interpreted_serves(&self.policy.choices, ops, full, compiled)
    }

    fn compiled(&self, body: &InterpretedBody) -> Option<Arc<dyn ExternBundle>> {
        self.compile.as_ref()?;
        body.backends.get(self.key)
    }

    fn compile_due(
        &self,
        v: usize,
        learned: &mut Specializer,
    ) -> Option<impl FnOnce() + Send + 'static> {
        learned.current()?;
        let compile = self.compile.clone()?;
        let choice = self.shared.choice(v);
        let (at, body) = choice.shortened_main.read().unwrap().clone()?;
        let (have, serves) = (self.compiled(&body).map(|_| at), self.serves(v, &body));
        let ver = learned.compile_due(&self.policy.choices, have, serves)?;
        let key = self.key;
        Some(move || {
            form(&body, key, &*compile);
            choice.learned.lock().unwrap().landed(ver);
        })
    }

    fn make_conds(&self) {
        let (Some(compile), Some(Some(f))) = (&self.compile, self.shared.found.get()) else {
            return;
        };
        if self.conds.get().is_none() {
            if let Some(b) = compile(f.selects.tape(), &[], f.selects.n_conds()) {
                let _ = self.conds.set(b);
            }
        }
    }

    fn make(&self, v: usize) {
        if let (Some(compile), None) = (&self.compile, self.made[v].get()) {
            if let Some(f) = form(&self.shared.body(v as u32), self.key, &**compile) {
                let _ = self.made[v].set(f);
            }
        }
    }
}

/// A function body that runs each instance on its variant (see the module
/// docs). One per form of the body: its own (interpreted, the default
/// policy) and what [`ExternBundle::with_backend`] makes of it for a
/// program.
pub struct VariantBody {
    shared: Arc<Shared>,
    full: Arc<InterpretedBody>,
    interp: Box<[OnceLock<Arc<InterpretedBody>>]>,
    form: Arc<Form>,
    reads: Arc<[u32]>,
    backend: BodyBackend,
    policy: VariantPolicy,
    backends: BackendCache,
    rate: crate::parallel::Rate,
}

impl VariantBody {
    pub(crate) fn new(full: InterpretedBody) -> VariantBody {
        let t = &full.tape;
        let choice = Choice::new(&full);
        let shared = Arc::new(Shared {
            found: OnceLock::new(),
            full_ops: t.n_ops(),
            full_state_len: t.state_len(),
            bodies: RwLock::new(vec![Arc::new(full)]),
            choices: RwLock::new(vec![choice]),
            variant_of_pattern: Mutex::new(HashMap::default()),
            build_lock: Mutex::new(()),
            exhaustion_reported: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
        });
        VariantBody::over(shared, BodyBackend::default(), VariantPolicy::default())
    }

    fn over(shared: Arc<Shared>, backend: BodyBackend, policy: VariantPolicy) -> VariantBody {
        let full = shared.body(FULL_BODY);
        let t = &full.tape;
        let reads = t.main_reads();
        let n = policy.max_variants.max(1);
        let form = Arc::new(Form {
            shared: shared.clone(),
            full: full.clone(),
            compile: backend.compile.clone(),
            key: backend.key,
            policy,
            made: (0..n).map(|_| OnceLock::new()).collect(),
            making: (0..n).map(|_| AtomicBool::new(false)).collect(),
            conds: OnceLock::new(),
            pending: Mutex::default(),
        });
        let v = VariantBody {
            reads: reads.into(),
            interp: (0..n).map(|_| OnceLock::new()).collect(),
            form,
            full,
            shared,
            backend,
            policy,
            backends: BackendCache::default(),
            rate: crate::parallel::Rate::new(crate::constants::INTERPRETED_NS_PER_OP),
        };
        let _ = v.interp[FULL_BODY as usize].set(v.full.clone());
        let built = v.shared.bodies.read().unwrap().len().min(n);
        for k in 0..built {
            v.form.make(k);
        }
        v.form.make_conds();
        v
    }

    fn variant_slot(&self) -> usize {
        self.shared.full_state_len
    }

    fn variant(&self, pure: &[f64]) -> u32 {
        let compiles = self.backend.compile.is_some();
        if !self.policy.enabled || !(compiles || self.policy.interpreted) {
            return FULL_BODY;
        }
        if let Some(v) = self.form.ready(pure) {
            return v;
        }
        match &self.backend.submit {
            None => {
                self.form.advance(&[pure.into()]);
                self.form.ready(pure).unwrap_or(FULL_BODY)
            }
            Some(submit) => {
                self.form.defer(pure, submit);
                FULL_BODY
            }
        }
    }

    fn run_then_move_epoch(&self, now: bool, job: impl FnOnce() + Send + 'static) {
        match (&self.backend.submit, now) {
            (Some(_), false) => {
                let shared = self.shared.clone();
                self.later_keeping_epoch(move || {
                    job();
                    shared.epoch.fetch_add(1, Ordering::Release);
                })
            }
            _ => job(),
        }
    }

    fn later_keeping_epoch(&self, job: impl FnOnce() + Send + 'static) {
        match &self.backend.submit {
            Some(submit) => submit(Box::new(job)),
            None => job(),
        }
    }

    fn make(&self, v: usize, now: bool) {
        if self.backend.compile.is_none() || self.form.making[v].swap(true, Ordering::AcqRel) {
            return;
        }
        let form = self.form.clone();
        self.run_then_move_epoch(now, move || form.make(v));
    }

    fn bundle(&self, v: usize) -> &dyn ExternBundle {
        assert!(
            v < self.interp.len(),
            "a state laid out under another variant policy"
        );
        if let Some(b) = self.form.made[v].get() {
            return b.as_ref();
        }
        let b = self.interp[v].get_or_init(|| self.shared.body(v as u32));
        self.make(v, false);
        b.as_ref()
    }

    fn specializes(&self) -> bool {
        self.policy.choices.enabled && (self.backend.compile.is_some() || self.policy.interpreted)
    }

    fn spec_form(&self, v: usize) -> Option<(u64, Arc<dyn ExternBundle>, bool)> {
        if !self.specializes() {
            return None;
        }
        let (ver, body) = self
            .shared
            .choice(v)
            .shortened_main
            .read()
            .unwrap()
            .clone()?;
        if let Some(form) = self.form.compiled(&body) {
            return Some((ver, form, true));
        }
        self.form.serves(v, &body).then_some((ver, body, false))
    }

    fn main_variant(
        &self,
        v: usize,
        args: &[f64],
        states: &[f64],
        out: &mut [f64],
        at: &Instances,
    ) {
        let full = self.bundle(v);
        let Some((ver, spec, compiled)) = self.spec_form(v) else {
            full.main_batch(args, states, out, at);
            if self.specializes() {
                self.observe(v, args, states, at, None, false);
            }
            return;
        };
        let no = full.n_outputs();
        let ns = no + 1;
        let n = at.iter().max().map_or(0, |g| g + 1);
        crate::scratch::with_len(n * ns, 0.0, |so| {
            spec.main_batch(args, states, so, at);
            crate::scratch::with(|failed: &mut Vec<u32>| {
                failed.clear();
                for g in at.iter() {
                    let o = &so[g * ns..(g + 1) * ns];
                    if o[no] == 0.0 {
                        out[g * no..(g + 1) * no].copy_from_slice(&o[..no]);
                    } else {
                        failed.push(g as u32);
                    }
                }
                if !failed.is_empty() {
                    full.main_batch(args, states, out, &at.listed(failed));
                }
                self.observe(v, args, states, at, Some((ver, failed)), compiled);
            });
        });
    }

    fn observe(
        &self,
        v: usize,
        args: &[f64],
        states: &[f64],
        at: &Instances,
        ran: Option<(u64, &[u32])>,
        compiled: bool,
    ) {
        let policy = self.policy.choices;
        let choice = self.shared.choice(v);
        let Ok(mut learned) = choice.learned.try_lock() else {
            return;
        };
        learned.ran(at.len() as u64);
        let traced: Vec<u32> = match ran {
            Some((ver, failed)) if !failed.is_empty() => {
                let trace = learned.failed(&policy, ver, failed.len() as u64);
                if learned.current().is_none() {
                    *choice.shortened_main.write().unwrap() = None;
                }
                if !trace {
                    return;
                }
                spread(failed.len(), |k| failed[k])
            }
            ran => {
                if let Some((ver, _)) = ran {
                    learned.held(ver, at.len() as u64);
                }
                let full_body_instances_move_soon =
                    v == FULL_BODY as usize && learned.current().is_none() && self.form.advancing();
                if !full_body_instances_move_soon && learned.due(&policy) {
                    spread(at.len(), |k| at.at(k) as u32)
                } else {
                    let compile = (!compiled).then(|| self.form.compile_due(v, &mut learned));
                    drop(learned);
                    if let Some(job) = compile.flatten() {
                        self.later_keeping_epoch(job);
                    }
                    return;
                }
            }
        };
        drop(learned);
        let body = self.shared.body(v as u32);
        let (na, st, sl) = (at.n_args, at.stride, body.state_len());
        let copied_rows: Vec<(Vec<f64>, Vec<f64>)> = traced
            .iter()
            .map(|&g| {
                let (g, r) = (g as usize, at.row(g as usize));
                (
                    args[g * na..(g + 1) * na].to_vec(),
                    states[r * st..r * st + sl].to_vec(),
                )
            })
            .collect();
        let form = self.form.clone();
        self.later_keeping_epoch(move || {
            let t = &body.tape;
            let mut w = vec![0.0; t.work_len()];
            let mut o = vec![0.0; t.out_len()];
            let traces: Vec<Vec<u8>> = copied_rows
                .iter()
                .map(|(a, s)| {
                    w[..s.len()].copy_from_slice(s);
                    let mut trace = Vec::new();
                    t.eval_main_with(a, &mut w, &mut o, &mut trace);
                    trace
                })
                .collect();
            form.shared.learn(v, &traces, &policy);
            let job = form.compile_due(v, &mut form.shared.choice(v).learned.lock().unwrap());
            if let Some(job) = job {
                job();
            }
        });
    }

    fn by_variant(&self, at: &Instances, vs: &[u32], mut f: impl FnMut(usize, &Instances)) {
        let first = vs[0];
        if vs.iter().all(|&v| v == first) {
            f(first as usize, at);
            return;
        }
        let nv = self.interp.len();
        crate::scratch::with(|start: &mut Vec<usize>| {
            crate::scratch::with(|order: &mut Vec<u32>| {
                start.clear();
                start.resize(nv + 1, 0);
                for &v in vs {
                    start[v as usize + 1] += 1;
                }
                for v in 0..nv {
                    start[v + 1] += start[v];
                }
                order.clear();
                order.resize(vs.len(), 0);
                crate::scratch::with(|next: &mut Vec<usize>| {
                    next.clear();
                    next.extend_from_slice(&start[..nv]);
                    for (k, &v) in vs.iter().enumerate() {
                        order[next[v as usize]] = at.at(k) as u32;
                        next[v as usize] += 1;
                    }
                });
                for v in 0..nv {
                    if start[v] < start[v + 1] {
                        f(v, &at.listed(&order[start[v]..start[v + 1]]));
                    }
                }
            })
        });
    }
}

impl ExternBundle for VariantBody {
    fn n_outputs(&self) -> usize {
        self.full.n_outputs()
    }
    fn rate(&self) -> &crate::parallel::Rate {
        &self.rate
    }
    fn work_len(&self) -> usize {
        self.full.work_len()
    }
    fn batch_lanes(&self) -> usize {
        self.full.batch_lanes()
    }
    fn call_into(&self, args: &[f64], work: &mut [f64], out: &mut [f64]) {
        self.full.call_into(args, work, out);
    }
    fn state_len(&self) -> usize {
        self.variant_slot() + 1
    }
    fn pure_args(&self) -> &[bool] {
        self.full.pure_args()
    }
    fn prolog_into(&self, pure: &[f64], _work: &mut [f64], state: &mut [f64]) {
        let v = self.variant(pure) as usize;
        let b = self.bundle(v);
        let sl = b.state_len();
        crate::scratch::with_len(b.work_len(), 0.0, |w| {
            b.prolog_into(pure, w, &mut state[..sl])
        });
        state[sl..self.variant_slot()].fill(0.0);
        state[self.variant_slot()] = v as f64;
    }
    fn main_into(&self, args: &[f64], state: &[f64], _work: &mut [f64], out: &mut [f64]) {
        let at = Instances::first(1, args.len(), state.len());
        self.main_batch(args, state, out, &at);
    }
    fn prolog_batch(&self, pure: &[f64], states: &mut [f64], at: &Instances) {
        if at.is_empty() {
            return;
        }
        let (na, st, slot) = (at.n_args, at.stride, self.variant_slot());
        crate::scratch::with(|vs: &mut Vec<u32>| {
            vs.clear();
            vs.extend(at.iter().map(|g| self.variant(&pure[g * na..(g + 1) * na])));
            self.by_variant(at, vs, |v, ins| {
                let b = self.bundle(v);
                b.prolog_batch(pure, states, ins);
                let sl = b.state_len();
                for g in ins.iter() {
                    let r = ins.row(g);
                    states[r * st + sl..r * st + slot].fill(0.0);
                    states[r * st + slot] = v as f64;
                }
            });
        });
    }
    fn main_batch(&self, args: &[f64], states: &[f64], out: &mut [f64], at: &Instances) {
        if at.is_empty() {
            return;
        }
        let (st, slot) = (at.stride, self.variant_slot());
        crate::scratch::with(|vs: &mut Vec<u32>| {
            vs.clear();
            vs.extend(at.iter().map(|g| states[at.row(g) * st + slot] as u32));
            self.by_variant(at, vs, |v, ins| {
                self.main_variant(v, args, states, out, ins);
            });
        });
    }
    fn body(&self) -> Option<&Tape> {
        self.full.body()
    }
    fn main_reads(&self) -> Option<Vec<u32>> {
        Some(self.reads.to_vec())
    }
    fn backend_cache(&self) -> Option<&BackendCache> {
        Some(&self.backends)
    }
    fn with_backend(&self, backend: &BodyBackend) -> Option<Arc<dyn ExternBundle>> {
        let policy = backend.variants.unwrap_or(self.policy);
        let v = VariantBody::over(self.shared.clone(), backend.clone(), policy);
        Some(Arc::new(v))
    }
    fn forms_epoch(&self) -> u64 {
        self.shared.epoch.load(Ordering::Acquire)
    }
}

fn spread(n: usize, item: impl Fn(usize) -> u32) -> Vec<u32> {
    let m = n.min(TRACED);
    (0..m).map(|j| item(j * n / m.max(1))).collect()
}
