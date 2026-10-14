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
//! shortened again without them. Choices that flip too often turn the
//! specialization off for the variant.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use rustc_hash::FxHashMap as HashMap;

use crate::choices::Choices;
use crate::extern_fn::{BackendCache, BodyBackend, ExternBundle, Instances};
use crate::func::InterpretedBody;
use crate::hooks::{log, Level};
use crate::tape::{Guards, ParamSelects, Tape};

use crate::constants::{PERCENT, TRACED};

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
    pub choices: crate::ChoicePolicy,
}

impl Default for VariantPolicy {
    fn default() -> Self {
        VariantPolicy::DEFAULT
    }
}

/// The selects a binding decides, and what deciding them can save.
struct Found {
    selects: ParamSelects,
    /// The share of the body's ops, in percent, that the better of
    /// deciding every select one way or every one the other removes.
    shrink_pct: usize,
}

/// Where a pattern's variant stands.
#[derive(Clone, Copy)]
enum Slot {
    /// Being built in the background.
    Building,
    /// Built: its index (`0`, the full body, where deciding shortens
    /// nothing or no room is left).
    Ready(u32),
}

/// What every form of one body shares, whatever runs it: the selects a
/// binding decides and the variants, by index.
struct Shared {
    /// The selects, once looked for; `None` inside when the body has none
    /// a binding decides.
    found: OnceLock<Option<Found>>,
    /// Whether looking for them is on its way.
    finding: AtomicBool,
    /// The full body's ops, what a variant must undercut.
    ops: usize,
    /// The full body's state, the longest a variant may keep.
    state_len: usize,
    /// The full body (`0`) and every variant built.
    bodies: RwLock<Vec<Arc<InterpretedBody>>>,
    /// Per body, its specialization over the run-time choices.
    choices: RwLock<Vec<Arc<Choice>>>,
    /// The variant of each pattern seen.
    index: Mutex<HashMap<Box<[bool]>, Slot>>,
    /// Whether a pattern found no room left (reported once).
    exhausted: AtomicBool,
    /// Moves when background work lands (see
    /// [`ExternBundle::forms_epoch`]).
    epoch: AtomicU64,
}

/// A variant's specialization over the choices its instances take at run
/// time: what they taught so far, and its main phase shortened against it.
#[derive(Default)]
struct Choice {
    learned: Mutex<Choices>,
    /// The shortened main phase as a body, its outputs the variant's and
    /// the count of guards that fail, and the version it was built at.
    spec: RwLock<Option<(u64, Arc<InterpretedBody>)>>,
    /// A trace of instances is on its way.
    tracing: AtomicBool,
}

impl Shared {
    fn body(&self, v: u32) -> Arc<InterpretedBody> {
        self.bodies.read().unwrap()[v as usize].clone()
    }

    fn choice(&self, v: usize) -> Arc<Choice> {
        self.choices.read().unwrap()[v].clone()
    }

    /// Body `body` kept as a variant: its index.
    fn keep(&self, body: InterpretedBody) -> u32 {
        let mut bodies = self.bodies.write().unwrap();
        bodies.push(Arc::new(body));
        self.choices.write().unwrap().push(Arc::default());
        bodies.len() as u32 - 1
    }

    fn find(&self) -> Option<Found> {
        let full = self.body(0);
        let t = full.body().expect("an interpreted body is a tape");
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

    /// The variant of `pattern`, built: the full body when deciding
    /// shortens nothing, or when `max` bodies are kept already.
    fn build(&self, ps: &ParamSelects, pattern: &[bool], max: usize) -> u32 {
        if self.bodies.read().unwrap().len() >= max {
            if !self.exhausted.swap(true, Ordering::Relaxed) {
                log(
                    Level::Info,
                    &format!("rsdag: a body keeps {max} variants; later patterns run it whole"),
                );
            }
            return 0;
        }
        let full = self.body(0);
        let t = full.body().expect("an interpreted body is a tape");
        let decided = t.decide(ps, pattern);
        if decided.n_ops() >= self.ops || decided.state_len() > self.state_len {
            return 0;
        }
        let (pure, n_out) = (full.pure_args().to_vec(), full.n_outputs());
        self.keep(InterpretedBody::new(decided, n_out, pure))
    }

    /// What the traces `traces` of instances of variant `v` teach: the
    /// selects they disagree in unpinned, and the main phase shortened
    /// again where that changed the pins (or where there is none yet).
    fn learn(&self, v: usize, traces: &[Vec<u8>], policy: &crate::ChoicePolicy) {
        let choice = self.choice(v);
        let mut learned = choice.learned.lock().unwrap();
        if learned.off {
            return;
        }
        let first = learned.built == 0;
        let fresh: usize = traces.iter().map(|t| learned.observe(t)).sum();
        if !first && fresh == 0 {
            return;
        }
        let body = self.body(v as u32);
        let t = body.body().expect("an interpreted body is a tape");
        let spec = t.specialize_main(learned.trace(), &learned.pin(), Guards::Count);
        let main = |t: &Tape| t.n_ops() - t.prolog_len();
        if !learned.built(policy, main(spec.tape()), main(t)) {
            *choice.spec.write().unwrap() = None;
            log(
                Level::Debug,
                &format!(
                    "rsdag: a body's run-time choices shorten it too little ({} of {} ops)",
                    main(spec.tape()),
                    main(t)
                ),
            );
            return;
        }
        learned.version += 1;
        log(
            Level::Debug,
            &format!(
                "rsdag: a body specialized over its run-time choices, {} of {} main ops",
                main(spec.tape()),
                main(t)
            ),
        );
        let n_out = body.n_outputs() + 1;
        let shortened = InterpretedBody::new(spec.into_tape(), n_out, body.pure_args().to_vec());
        *choice.spec.write().unwrap() = Some((learned.version, Arc::new(shortened)));
    }

    /// The pattern the selects `ps` take at the pure arguments `pure` (of
    /// the arguments `mask` flags), by `conds` (a backend's form of their
    /// tape) where given, else interpreted.
    fn pattern(
        &self,
        ps: &ParamSelects,
        mask: &[bool],
        pure: &[f64],
        conds: Option<&Arc<dyn ExternBundle>>,
    ) -> Vec<bool> {
        // The conditions read the pure arguments only; the others are NaN.
        crate::scratch::with(|args: &mut Vec<f64>| {
            args.clear();
            let mut p = pure.iter();
            args.extend(mask.iter().map(|&is_pure| match is_pure {
                true => *p.next().expect("one value per pure argument"),
                false => f64::NAN,
            }));
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

/// A function body that runs each instance on its variant (see the module
/// docs). One per form of the body: its own (interpreted, the default
/// policy) and what [`ExternBundle::with_backend`] makes of it for a
/// program.
pub struct VariantBody {
    shared: Arc<Shared>,
    /// The full body (variant `0`), for whole calls and as the body's tape.
    full: Arc<InterpretedBody>,
    /// Per variant this form runs, its interpreted body (set once the
    /// variant is built) and the backend's form of it (set once made).
    interp: Box<[OnceLock<Arc<InterpretedBody>>]>,
    made: Arc<[OnceLock<Arc<dyn ExternBundle>>]>,
    /// Per variant, whether the backend's form of it is on its way.
    making: Box<[AtomicBool]>,
    /// Per variant, the backend's form of its shortened main phase and the
    /// version it was made of, and the version being made.
    spec_made: Arc<[Mutex<Option<(u64, Arc<dyn ExternBundle>)>>]>,
    spec_making: Box<[AtomicU64]>,
    /// The backend's form of the conditions' tape, once made.
    conds: Arc<OnceLock<Arc<dyn ExternBundle>>>,
    conds_making: AtomicBool,
    /// What any variant's main phase reads: the full body's reads (see
    /// [`Tape::decide`]).
    reads: Arc<[u32]>,
    backend: BodyBackend,
    policy: VariantPolicy,
    backends: BackendCache,
    /// What an op of the body costs as its instances run (whatever variant
    /// and form), learned from the runs.
    rate: crate::parallel::Rate,
}

impl VariantBody {
    /// `full`, a body with a select a binding decides (see
    /// [`Tape::has_param_selects`]): interpreted under the default policy,
    /// everything done in the evaluations that ask.
    pub(crate) fn new(full: InterpretedBody) -> VariantBody {
        let t = full.body().expect("an interpreted body is a tape");
        let shared = Arc::new(Shared {
            found: OnceLock::new(),
            finding: AtomicBool::new(false),
            ops: t.n_ops(),
            state_len: t.state_len(),
            bodies: RwLock::new(vec![Arc::new(full)]),
            choices: RwLock::new(vec![Arc::default()]),
            index: Mutex::new(HashMap::default()),
            exhausted: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
        });
        VariantBody::over(shared, BodyBackend::default(), VariantPolicy::default())
    }

    fn over(shared: Arc<Shared>, backend: BodyBackend, policy: VariantPolicy) -> VariantBody {
        let full = shared.body(0);
        let t = full.body().expect("an interpreted body is a tape");
        let reads = t.main_reads();
        let n = policy.max_variants.max(1);
        let v = VariantBody {
            reads: reads.into(),
            interp: (0..n).map(|_| OnceLock::new()).collect(),
            made: (0..n).map(|_| OnceLock::new()).collect(),
            making: (0..n).map(|_| AtomicBool::new(false)).collect(),
            spec_made: (0..n).map(|_| Mutex::new(None)).collect(),
            spec_making: (0..n).map(|_| AtomicU64::new(0)).collect(),
            conds: Arc::default(),
            conds_making: AtomicBool::new(false),
            full,
            shared,
            backend,
            policy,
            backends: BackendCache::default(),
            rate: crate::parallel::Rate::new(crate::constants::INTERPRETED_NS_PER_OP),
        };
        // An instance runs the full body until its variant is ready: this
        // form of it from the start.
        let _ = v.interp[0].set(v.full.clone());
        // Making a form of the body makes what it runs: the full body, the
        // variants built so far and the conditions' tape, here; a variant
        // built later is made where the backend puts background work.
        let built = v.shared.bodies.read().unwrap().len().min(n);
        for k in 0..built {
            v.make(k, true);
        }
        if let Some(Some(_)) = v.shared.found.get() {
            v.make_conds(true);
        }
        v
    }

    /// Where in its block an instance keeps its variant.
    fn tag(&self) -> usize {
        self.shared.state_len
    }

    /// The variant to run for the pure arguments `pure`: its pattern's,
    /// once built and made, else the full body.
    fn variant(&self, pure: &[f64]) -> u32 {
        let compiles = self.backend.compile.is_some();
        if !self.policy.enabled || !(compiles || self.policy.interpreted) {
            return 0;
        }
        let shared = &self.shared;
        let found = match &self.backend.submit {
            None => shared.found.get_or_init(|| shared.find()),
            Some(submit) => match shared.found.get() {
                Some(found) => found,
                None => {
                    if !shared.finding.swap(true, Ordering::AcqRel) {
                        let shared = shared.clone();
                        submit(Box::new(move || {
                            let _ = shared.found.set(shared.find());
                            shared.epoch.fetch_add(1, Ordering::Release);
                        }));
                    }
                    return 0;
                }
            },
        };
        let Some(found) = found.as_ref() else {
            return 0;
        };
        if found.shrink_pct < self.policy.min_shrink_pct {
            return 0;
        }
        let ps = &found.selects;
        let pattern = shared.pattern(ps, self.full.pure_args(), pure, self.conds());
        let mut index = shared.index.lock().unwrap();
        let v = match (index.get(pattern.as_slice()).copied(), &self.backend.submit) {
            (Some(Slot::Ready(v)), _) => v,
            (Some(Slot::Building), _) => return 0,
            (None, None) => {
                let v = shared.build(ps, &pattern, self.policy.max_variants);
                index.insert(pattern.into(), Slot::Ready(v));
                v
            }
            (None, Some(submit)) => {
                let pattern: Box<[bool]> = pattern.into();
                index.insert(pattern.clone(), Slot::Building);
                drop(index);
                let (shared, max) = (shared.clone(), self.policy.max_variants);
                submit(Box::new(move || {
                    let found = shared.found.get().and_then(Option::as_ref);
                    let v = found.map_or(0, |f| shared.build(&f.selects, &pattern, max));
                    shared.index.lock().unwrap().insert(pattern, Slot::Ready(v));
                    shared.epoch.fetch_add(1, Ordering::Release);
                }));
                return 0;
            }
        };
        drop(index);
        let v = v as usize;
        if v >= self.interp.len() {
            return 0; // past what this form keeps
        }
        // A variant runs once this form has it at its best: where the
        // backend compiles, its compiled form, else the full body's
        // compiled one runs meanwhile (faster than the variant
        // interpreted).
        if compiles && self.made[v].get().is_none() {
            self.interp[v].get_or_init(|| shared.body(v as u32));
            self.make(v, false);
            if self.made[v].get().is_none() {
                return 0;
            }
        }
        v as u32
    }

    /// The backend's form of the conditions' tape, where it compiles one,
    /// once made (asked for here the first time).
    fn conds(&self) -> Option<&Arc<dyn ExternBundle>> {
        self.backend.compile.as_ref()?;
        if self.conds.get().is_none() {
            self.make_conds(false);
        }
        self.conds.get()
    }

    /// Run `job`: `now`, or where there is no place for background work;
    /// else there.
    fn run(&self, now: bool, job: impl FnOnce() + Send + 'static) {
        match (&self.backend.submit, now) {
            (Some(submit), false) => {
                let shared = self.shared.clone();
                submit(Box::new(move || {
                    job();
                    shared.epoch.fetch_add(1, Ordering::Release);
                }))
            }
            _ => job(),
        }
    }

    /// Have the backend's form of the conditions' tape made (see
    /// [`run`](Self::run)), once the selects are found.
    fn make_conds(&self, now: bool) {
        let Some(compile) = self.backend.compile.clone() else {
            return;
        };
        if self.conds_making.swap(true, Ordering::AcqRel) {
            return;
        }
        let (shared, conds) = (self.shared.clone(), self.conds.clone());
        self.run(now, move || {
            let Some(Some(f)) = shared.found.get() else {
                return;
            };
            if let Some(b) = compile(f.selects.tape(), &[], f.selects.n_conds()) {
                let _ = conds.set(b);
            }
        });
    }

    /// Have the backend's form of variant `v` made (see [`run`](Self::run)).
    /// It is kept with the variant's body under the backend's key, so every
    /// program of the backend shares it.
    fn make(&self, v: usize, now: bool) {
        let Some(compile) = self.backend.compile.clone() else {
            return;
        };
        if self.making[v].swap(true, Ordering::AcqRel) {
            return;
        }
        let (shared, made, key) = (self.shared.clone(), self.made.clone(), self.backend.key);
        let job = move || {
            let body = shared.body(v as u32);
            let cache = body
                .backend_cache()
                .expect("an interpreted body keeps its forms");
            let form = cache.get_or_try_insert(key, || {
                let t = body.body().expect("an interpreted body is a tape");
                compile(t, body.pure_args(), body.n_outputs()).ok_or(())
            });
            if let Ok(form) = form {
                let _ = made[v].set(form);
            }
        };
        self.run(now, job);
    }

    /// What runs variant `v` in this form: the backend's form once made,
    /// else the interpreted body (the same state layout). A variant another
    /// form of the program built (its prolog ran elsewhere) is taken up
    /// here on first sight.
    fn bundle(&self, v: usize) -> &dyn ExternBundle {
        assert!(
            v < self.interp.len(),
            "a state laid out under another variant policy"
        );
        if let Some(b) = self.made[v].get() {
            return b.as_ref();
        }
        let b = self.interp[v].get_or_init(|| self.shared.body(v as u32));
        self.make(v, false);
        b.as_ref()
    }

    /// Whether this form specializes its variants over their run-time
    /// choices.
    fn specializes(&self) -> bool {
        self.policy.choices.enabled && (self.backend.compile.is_some() || self.policy.interpreted)
    }

    /// What runs variant `v`'s shortened main phase in this form, and its
    /// version: the backend's form once made (asked for here), else the
    /// interpreted body where the form interprets.
    fn spec_form(&self, v: usize) -> Option<(u64, Arc<dyn ExternBundle>)> {
        if !self.specializes() {
            return None;
        }
        let (ver, body) = self.shared.choice(v).spec.read().unwrap().clone()?;
        let Some(compile) = self.backend.compile.clone() else {
            return Some((ver, body));
        };
        if let Some((made, form)) = &*self.spec_made[v].lock().unwrap() {
            if *made == ver {
                return Some((ver, form.clone()));
            }
        }
        if self.spec_making[v].swap(ver, Ordering::AcqRel) != ver {
            let (made, key) = (self.spec_made.clone(), self.backend.key);
            let job = move || {
                let cache = body
                    .backend_cache()
                    .expect("an interpreted body keeps its forms");
                let form = cache.get_or_try_insert(key, || {
                    let t = body.body().expect("an interpreted body is a tape");
                    compile(t, body.pure_args(), body.n_outputs()).ok_or(())
                });
                if let Ok(form) = form {
                    let mut m = made[v].lock().unwrap();
                    if m.as_ref().is_none_or(|(at, _)| *at < ver) {
                        *m = Some((ver, form));
                    }
                }
            };
            // Nothing a prolog runs on changes: no epoch to move.
            match &self.backend.submit {
                Some(submit) => submit(Box::new(job)),
                None => job(),
            }
            if let Some((made, form)) = &*self.spec_made[v].lock().unwrap() {
                if *made == ver {
                    return Some((ver, form.clone()));
                }
            }
        }
        None
    }

    /// The instances `at` of variant `v`: on its shortened main phase where
    /// this form has one, an instance whose guards fail again on the
    /// variant's own; what they show of the choices learned on the way.
    fn main_variant(
        &self,
        v: usize,
        args: &[f64],
        states: &[f64],
        out: &mut [f64],
        at: &Instances,
    ) {
        let full = self.bundle(v);
        let Some((ver, spec)) = self.spec_form(v) else {
            full.main_batch(args, states, out, at);
            if self.specializes() {
                self.observe(v, args, states, at, None);
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
                self.observe(v, args, states, at, Some((ver, failed)));
            });
        });
    }

    /// The instances `at` of variant `v` ran: with no shortened main phase
    /// yet, a trace of some of them builds one; with one of version `ver`,
    /// the instances whose guards failed (`failed`) count as flips, and a
    /// trace of some of them unpins what they flipped. One trace at a time,
    /// where the backend puts background work.
    fn observe(
        &self,
        v: usize,
        args: &[f64],
        states: &[f64],
        at: &Instances,
        ran: Option<(u64, &[u32])>,
    ) {
        let policy = self.policy.choices;
        let choice = self.shared.choice(v);
        // Contended: another batch of the variant keeps the count.
        let Ok(mut learned) = choice.learned.try_lock() else {
            return;
        };
        if learned.off {
            return;
        }
        learned.ran(at.len() as u64);
        let traced: Vec<u32> = match ran {
            None => {
                if learned.built > 0 || choice.spec.read().unwrap().is_some() {
                    return; // built, this form's form of it on its way
                }
                let body = self.shared.body(v as u32);
                let t = body.body().expect("an interpreted body is a tape");
                if t.n_main_selects() < policy.min_selects {
                    learned.off = true;
                    return;
                }
                spread(at.len(), |k| at.at(k) as u32)
            }
            Some((ver, failed)) => {
                if failed.is_empty() || ver != learned.version {
                    return;
                }
                if learned.flipped(&policy, failed.len() as u64) {
                    *choice.spec.write().unwrap() = None;
                    log(
                        Level::Debug,
                        &format!(
                            "rsdag: a body's run-time choices thrash ({} flips in {} instances), specialization off",
                            learned.flips, learned.evals
                        ),
                    );
                    return;
                }
                spread(failed.len(), |k| failed[k])
            }
        };
        drop(learned);
        if choice.tracing.swap(true, Ordering::AcqRel) {
            return;
        }
        // What the trace reads, copied: the batch's buffers are the caller's.
        let body = self.shared.body(v as u32);
        let (na, st, sl) = (at.n_args, at.stride, body.state_len());
        let rows: Vec<(Vec<f64>, Vec<f64>)> = traced
            .iter()
            .map(|&g| {
                let (g, r) = (g as usize, at.row(g as usize));
                (
                    args[g * na..(g + 1) * na].to_vec(),
                    states[r * st..r * st + sl].to_vec(),
                )
            })
            .collect();
        let shared = self.shared.clone();
        let job = move || {
            let t = body.body().expect("an interpreted body is a tape");
            let mut w = vec![0.0; t.work_len()];
            let mut o = vec![0.0; t.out_len()];
            let traces: Vec<Vec<u8>> = rows
                .iter()
                .map(|(a, s)| {
                    w[..s.len()].copy_from_slice(s);
                    let mut trace = Vec::new();
                    t.eval_main_with(a, &mut w, &mut o, &mut trace);
                    trace
                })
                .collect();
            shared.learn(v, &traces, &policy);
            shared.choice(v).tracing.store(false, Ordering::Release);
        };
        // Nothing a prolog runs on changes: no epoch to move.
        match &self.backend.submit {
            Some(submit) => submit(Box::new(job)),
            None => job(),
        }
    }

    /// `f(v, instances)` for each variant among the instances `at`, given
    /// the variant `vs[k]` of the `k`th: the instances of a variant listed
    /// in their order, the variants ascending.
    fn by_variant(&self, at: &Instances, vs: &[u32], mut f: impl FnMut(usize, &Instances)) {
        let first = vs[0];
        if vs.iter().all(|&v| v == first) {
            f(first as usize, at);
            return;
        }
        // A counting sort of the instances by variant.
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
    fn call_into(&self, args: &[f64], work: &mut [f64], out: &mut [f64]) {
        self.full.call_into(args, work, out);
    }
    fn state_len(&self) -> usize {
        self.tag() + 1
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
        // What the variant leaves of the block is no one's: zero, so a
        // state is a function of its binding alone.
        state[sl..self.tag()].fill(0.0);
        state[self.tag()] = v as f64;
    }
    fn main_into(&self, args: &[f64], state: &[f64], _work: &mut [f64], out: &mut [f64]) {
        let at = Instances::first(1, args.len(), state.len());
        self.main_batch(args, state, out, &at);
    }
    fn prolog_batch(&self, pure: &[f64], states: &mut [f64], at: &Instances) {
        if at.is_empty() {
            return;
        }
        let (na, st, tag) = (at.n_args, at.stride, self.tag());
        crate::scratch::with(|vs: &mut Vec<u32>| {
            vs.clear();
            vs.extend(at.iter().map(|g| self.variant(&pure[g * na..(g + 1) * na])));
            self.by_variant(at, vs, |v, ins| {
                let b = self.bundle(v);
                b.prolog_batch(pure, states, ins);
                let sl = b.state_len();
                for g in ins.iter() {
                    let r = ins.row(g);
                    states[r * st + sl..r * st + tag].fill(0.0);
                    states[r * st + tag] = v as f64;
                }
            });
        });
    }
    fn main_batch(&self, args: &[f64], states: &[f64], out: &mut [f64], at: &Instances) {
        if at.is_empty() {
            return;
        }
        let (st, tag) = (at.stride, self.tag());
        crate::scratch::with(|vs: &mut Vec<u32>| {
            vs.clear();
            vs.extend(at.iter().map(|g| states[at.row(g) * st + tag] as u32));
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

/// Up to [`TRACED`] of `n` items, `item(k)` the `k`th, spread evenly.
fn spread(n: usize, item: impl Fn(usize) -> u32) -> Vec<u32> {
    let m = n.min(TRACED);
    (0..m).map(|j| item(j * n / m.max(1))).collect()
}
