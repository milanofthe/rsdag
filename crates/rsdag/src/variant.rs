//! Function bodies specialized per parameter binding.
//!
//! A device model branches on its parameters: a polarity, a model level, a
//! switch that shorts a resistance. Lowered exactly, each branch is a
//! `Select`, and a tape computes both of its arms at every evaluation. Its
//! condition, though, is fixed once the parameters are bound: a prolog
//! value. A [`VariantBody`] reads those conditions in each instance's
//! prolog, the pattern they take, and runs the instance's main phase on the
//! body that pattern decides ([`Tape::decide`]): the untaken arms gone, the
//! outputs bit for bit the full body's. A variant is built when its pattern
//! first turns up and is shared by every instance that takes it; a new
//! binding that flips a condition moves its instances to another variant
//! at their next prolog, so nothing is ever invalidated. A derivative body
//! is a body of the function like any other, its selects decided the same
//! way, so derivatives with respect to the parameters stay exact.
//!
//! Every variant lays its state out in one block of the same length, the
//! last value of which is the variant's index: a batch of instances on one
//! variant runs straight over the caller's states, and a state left by one
//! backend's prolog reads alike in another's main phase.
//!
//! A backend that compiles ([`BodyBackend`]) gets the full body compiled up
//! front and builds each variant in the background: an instance whose
//! variant is not ready yet runs the full body, which computes the same
//! values, and takes its variant at a later prolog. A prolog never waits on
//! a compilation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use rustc_hash::FxHashMap as HashMap;

use crate::extern_fn::{BackendCache, BodyBackend, ExternBundle};
use crate::func::InterpretedBody;
use crate::tape::{ParamSelects, Tape};

/// Variants a body builds at most; a pattern beyond them runs the full body.
const MAX_VARIANTS: usize = 64;

/// The share of the main phase the decided selects must be able to remove
/// (deciding them all one way or all the other) for a body to run variants,
/// in percent.
const MIN_SHRINK_PCT: usize = 10;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Whether the bodies built from now on specialize per binding (they do by
/// default): off, every body is the full one, for comparing against.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// See [`set_enabled`].
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Ops of a tape's main phase.
fn main_ops(t: &Tape) -> usize {
    t.n_ops() - t.prolog_len()
}

/// Where a pattern's variant stands.
#[derive(Clone, Copy)]
enum Slot {
    /// Being built in the background.
    Building,
    /// Built: its index (`0`, the full body, where deciding shortens nothing).
    Ready(u32),
}

/// What every backend's form of one body shares: the variants, by index.
struct Shared {
    selects: ParamSelects,
    /// The full body's tape as built, deciding works on.
    full: Arc<InterpretedBody>,
    /// The full body's main ops, what a variant must undercut.
    main_ops: usize,
    /// The state block of every variant, its index the last value.
    state_len: usize,
    /// The full body (`0`) and every variant built, the bodies interpreted
    /// over the common state layout.
    bodies: RwLock<Vec<Arc<InterpretedBody>>>,
    /// The variant of each pattern seen.
    index: Mutex<HashMap<Box<[bool]>, Slot>>,
}

impl Shared {
    /// Build the variant of `pattern`: the full body when deciding shortens
    /// nothing, or past [`MAX_VARIANTS`].
    fn build(&self, pattern: &[bool]) -> u32 {
        let t = self.full.body().expect("an interpreted body is a tape");
        if self.bodies.read().unwrap().len() >= MAX_VARIANTS {
            return 0;
        }
        let decided = t.decide(&self.selects, Some(pattern), self.state_len);
        if main_ops(&decided) >= self.main_ops || decided.state_len() != self.state_len {
            return 0;
        }
        let (pure, n_out) = (self.full.pure_args().to_vec(), self.full.n_outputs());
        let mut bodies = self.bodies.write().unwrap();
        bodies.push(Arc::new(InterpretedBody::new(decided, n_out, pure)));
        bodies.len() as u32 - 1
    }

    /// The variant of `pattern`, built here on its first turn.
    fn variant_now(&self, pattern: &[bool]) -> u32 {
        let mut index = self.index.lock().unwrap();
        match index.get(pattern) {
            Some(Slot::Ready(v)) => *v,
            Some(Slot::Building) => 0,
            None => {
                let v = self.build(pattern);
                index.insert(pattern.into(), Slot::Ready(v));
                v
            }
        }
    }
}

/// A function body that runs each instance on its variant (see the module
/// docs).
pub struct VariantBody {
    shared: Arc<Shared>,
    /// This backend's form of each variant, made on first use.
    run: Arc<[OnceLock<Arc<dyn ExternBundle>>]>,
    /// Per variant, whether this backend's form of it is on its way.
    pending: Arc<[AtomicBool]>,
    /// The backend the variants are made by; `None` interprets them.
    backend: Option<BodyBackend>,
    backends: BackendCache,
}

impl VariantBody {
    /// Whether deciding the selects `selects` of `full` can remove at
    /// least [`MIN_SHRINK_PCT`] of its main phase, deciding them all one way
    /// or all the other.
    pub(crate) fn pays(full: &InterpretedBody, selects: &ParamSelects) -> bool {
        let t = full.body().expect("an interpreted body is a tape");
        let n = selects.n_conds();
        let least = [true, false]
            .map(|b| main_ops(&t.decide(selects, Some(&vec![b; n]), 0)))
            .into_iter()
            .min()
            .unwrap_or(0);
        let all = main_ops(t);
        (all - least.min(all)) * 100 >= MIN_SHRINK_PCT * all
    }

    /// `full` running each binding's variant of the selects `selects`
    /// decides (see [`Tape::param_selects`]).
    pub(crate) fn new(full: InterpretedBody, selects: ParamSelects) -> VariantBody {
        let t = full.body().expect("an interpreted body is a tape");
        let state_len = 1 + t.state_len();
        let main_ops = main_ops(t);
        let padded = t.decide(&selects, None, state_len);
        let (pure, n_out) = (full.pure_args().to_vec(), full.n_outputs());
        let shared = Arc::new(Shared {
            selects,
            full: Arc::new(full),
            main_ops,
            state_len,
            bodies: RwLock::new(vec![Arc::new(InterpretedBody::new(padded, n_out, pure))]),
            index: Mutex::new(HashMap::default()),
        });
        VariantBody::over(shared, None)
    }

    fn over(shared: Arc<Shared>, backend: Option<BodyBackend>) -> VariantBody {
        let v = VariantBody {
            shared,
            run: (0..MAX_VARIANTS).map(|_| OnceLock::new()).collect(),
            pending: (0..MAX_VARIANTS).map(|_| AtomicBool::new(false)).collect(),
            backend,
            backends: BackendCache::default(),
        };
        // The full body is what an instance runs until its variant is
        // ready: this backend's form of it from the start.
        if let Some(b) = &v.backend {
            let full = v.shared.bodies.read().unwrap()[0].clone();
            let made = made_by(&b.compile, &full);
            let _ = v.run[0].set(made);
        }
        v
    }

    /// The variant of `pattern` this backend runs now: a ready one, else
    /// the full body while it is built in the background.
    fn variant_soon(&self, pattern: &[bool], backend: &BodyBackend) -> u32 {
        let slot = self.shared.index.lock().unwrap().get(pattern).copied();
        match slot {
            Some(Slot::Ready(v)) if self.run[v as usize].get().is_some() => v,
            Some(Slot::Ready(v)) => {
                self.make_soon(v as usize, backend);
                0
            }
            Some(Slot::Building) => 0,
            None => {
                let mut index = self.shared.index.lock().unwrap();
                if index.contains_key(pattern) {
                    return 0;
                }
                index.insert(pattern.into(), Slot::Building);
                drop(index);
                let (shared, run, pending) =
                    (self.shared.clone(), self.run.clone(), self.pending.clone());
                let (compile, pattern): (_, Box<[bool]>) =
                    (backend.compile.clone(), pattern.into());
                (backend.submit)(Box::new(move || {
                    let v = shared.build(&pattern);
                    shared.index.lock().unwrap().insert(pattern, Slot::Ready(v));
                    if !pending[v as usize].swap(true, Ordering::AcqRel) {
                        let body = shared.bodies.read().unwrap()[v as usize].clone();
                        let _ = run[v as usize].set(made_by(&compile, &body));
                    }
                }));
                0
            }
        }
    }

    /// Have this backend's form of variant `v` made in the background.
    fn make_soon(&self, v: usize, backend: &BodyBackend) {
        if self.pending[v].swap(true, Ordering::AcqRel) {
            return;
        }
        let (shared, run, compile) = (
            self.shared.clone(),
            self.run.clone(),
            backend.compile.clone(),
        );
        (backend.submit)(Box::new(move || {
            let body = shared.bodies.read().unwrap()[v].clone();
            let _ = run[v].set(made_by(&compile, &body));
        }));
    }

    /// This backend's form of variant `v`, else its interpreted form (the
    /// same state layout) while this backend's is on its way.
    fn bundle(&self, v: usize) -> Arc<dyn ExternBundle> {
        if let Some(b) = self.run[v].get() {
            return b.clone();
        }
        let body: Arc<dyn ExternBundle> = self.shared.bodies.read().unwrap()[v].clone();
        match &self.backend {
            Some(b) => {
                self.make_soon(v, b);
                body
            }
            None => self.run[v].get_or_init(|| body).clone(),
        }
    }

    /// The variant of the instance whose state starts `state`.
    fn of(&self, state: &[f64]) -> usize {
        state[self.shared.state_len - 1] as usize
    }
}

/// What `compile` makes of `body`, else `body` itself.
fn made_by(
    compile: &Arc<crate::extern_fn::BodyCompiler>,
    body: &Arc<InterpretedBody>,
) -> Arc<dyn ExternBundle> {
    let t = body.body().expect("an interpreted body is a tape");
    compile(t, body.pure_args(), body.n_outputs()).unwrap_or_else(|| body.clone())
}

impl ExternBundle for VariantBody {
    fn n_outputs(&self) -> usize {
        self.shared.full.n_outputs()
    }
    fn work_len(&self) -> usize {
        self.shared.full.work_len()
    }
    fn call_into(&self, args: &[f64], work: &mut [f64], out: &mut [f64]) {
        self.shared.full.call_into(args, work, out);
    }
    fn state_len(&self) -> usize {
        self.shared.state_len
    }
    fn pure_args(&self) -> &[bool] {
        self.shared.full.pure_args()
    }
    fn prolog_into(&self, pure: &[f64], _work: &mut [f64], state: &mut [f64]) {
        let mask = self.pure_args();
        // The conditions read the pure arguments only; the others are NaN.
        let pattern = crate::scratch::with(|args: &mut Vec<f64>| {
            args.clear();
            let mut p = pure.iter();
            args.extend(mask.iter().map(|&is_pure| match is_pure {
                true => *p.next().expect("one value per pure argument"),
                false => f64::NAN,
            }));
            crate::scratch::with(|w: &mut Vec<f64>| {
                crate::scratch::with(|o: &mut Vec<f64>| self.shared.selects.pattern(args, w, o))
            })
        });
        let v = match &self.backend {
            Some(b) => self.variant_soon(&pattern, b),
            None => self.shared.variant_now(&pattern),
        };
        let b = self.bundle(v as usize);
        crate::scratch::with_len(b.work_len(), 0.0, |w| b.prolog_into(pure, w, state));
        state[self.shared.state_len - 1] = v as f64;
    }
    fn main_into(&self, args: &[f64], state: &[f64], _work: &mut [f64], out: &mut [f64]) {
        let b = self.bundle(self.of(state));
        crate::scratch::with_len(b.work_len(), 0.0, |w| b.main_into(args, state, w, out));
    }
    fn main_batch(
        &self,
        args: &[f64],
        states: &[f64],
        n_groups: usize,
        n_args: usize,
        out: &mut [f64],
    ) {
        // Runs of instances on one variant, each a batch of its own.
        let (sl, no) = (self.shared.state_len, self.n_outputs());
        let mut g = 0;
        while g < n_groups {
            let v = self.of(&states[g * sl..]);
            let h = (g + 1..n_groups)
                .find(|&k| self.of(&states[k * sl..]) != v)
                .unwrap_or(n_groups);
            self.bundle(v).main_batch(
                &args[g * n_args..h * n_args],
                &states[g * sl..h * sl],
                h - g,
                n_args,
                &mut out[g * no..h * no],
            );
            g = h;
        }
    }
    fn body(&self) -> Option<&Tape> {
        self.shared.full.body()
    }
    fn backend_cache(&self) -> Option<&BackendCache> {
        Some(&self.backends)
    }
    fn with_backend(&self, backend: &BodyBackend) -> Option<Arc<dyn ExternBundle>> {
        let v = VariantBody::over(self.shared.clone(), Some(backend.clone()));
        Some(Arc::new(v))
    }
}
