//! Specialization over region choices, the one policy of everything that
//! specializes against them: [`Adaptive`](crate::Adaptive) for a program
//! evaluated point by point, the function bodies ([`crate::variant`]) for
//! their instances.
//!
//! A model branches by operating region (`Select`), and evaluating it
//! computes both arms. A trace of the choices taken shortens it
//! ([`Tape::specialize`](crate::Tape::specialize),
//! [`Tape::specialize_main`](crate::Tape::specialize_main)), guarded: where
//! a guard fails, the evaluation runs again on the full form and the result
//! is the same, bit for bit. What changes is only what pays, and a
//! `Specializer` decides it: when to trace and to build, that a select
//! seen flipping is unpinned for good, when to compile, which form serves,
//! and that choices flipping too often turn specialization off. Its owner
//! evaluates, traces, builds and keeps the forms, by the version it hands
//! out.

use crate::constants::PERCENT;
use crate::hooks::{log, Level};

/// When specializing against the choices pays. The defaults are what a
/// circuit simulator's Newton loop measured best with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChoicePolicy {
    /// Specialize over region choices at all.
    pub enabled: bool,
    /// Below this many `Select`s the shortening cannot pay for its guards.
    pub min_selects: usize,
    /// A specialization must remove this percentage of the instructions.
    pub min_shrink_pct: usize,
    /// Evaluations that must run flip-free before a specialization is
    /// rebuilt (rebuilding is O(tape); a cold Newton cascade hops regions).
    pub flip_cooldown: u64,
    /// Choices thrash, and specialization goes off, once at least
    /// `thrash_flips` flips came at one per fewer than `thrash_evals`
    /// evaluations.
    pub thrash_flips: u64,
    pub thrash_evals: u64,
    /// Guard-holding evaluations before a specialization is compiled.
    pub compile_after: u64,
    /// Compiles a specialization may spend across region flips.
    pub compile_budget: u32,
    /// Next to compiled code of the full form, the interpreted
    /// specialization serves only when this many times shorter (an
    /// interpreted op costs about this many compiled ones).
    pub interp_cost: usize,
}

impl Default for ChoicePolicy {
    fn default() -> Self {
        ChoicePolicy::DEFAULT
    }
}

/// One program's or one body's specialization over its choices (see the
/// module docs). An evaluation is one point (a program) or one instance (a
/// body).
#[derive(Default)]
pub(crate) struct Specializer {
    /// The selects of the form it specializes, the length of a trace.
    selects: usize,
    /// The trace held, the one the next specialization pins.
    trace: Vec<u8>,
    /// Selects seen flipping: unpinned for good, they stay real `Select`s.
    unpinned: Vec<bool>,
    evals: u64,
    pub(crate) flips: u64,
    /// No specialization is rebuilt before this evaluation count.
    cooldown_until: u64,
    /// Choices thrash, or the shrink is too small: the full form serves.
    pub(crate) off: bool,
    /// The version of the current specialization; it moves whenever one is
    /// built or dropped, so work for an outdated one is dropped.
    version: u64,
    /// Whether a specialization is current.
    current: bool,
    /// Specializations built.
    pub(crate) built: u64,
    /// Guard-holding evaluations since the current one was built.
    stable: u64,
    compiles: u32,
    compiling: bool,
    /// A trace is on its way.
    tracing: bool,
}

impl Specializer {
    /// For a form of `selects` selects.
    pub(crate) fn new(selects: usize) -> Specializer {
        Specializer {
            selects,
            unpinned: vec![false; selects],
            ..Specializer::default()
        }
    }

    /// Whether specializing a form of this many selects can pay at all.
    pub(crate) fn eligible(&self, policy: &ChoicePolicy) -> bool {
        policy.enabled && self.selects >= policy.min_selects
    }

    /// The version of the specialization to run, if one is current.
    pub(crate) fn current(&self) -> Option<u64> {
        (self.current && !self.off).then_some(self.version)
    }

    /// `n` evaluations ran, on whatever form.
    pub(crate) fn ran(&mut self, n: u64) {
        self.evals += n;
    }

    /// `n` evaluations on the specialization `version` held their guards.
    pub(crate) fn held(&mut self, version: u64, n: u64) {
        if self.current() == Some(version) {
            self.stable += n;
        }
    }

    /// `n` evaluations on the specialization `version` failed a guard: the
    /// region changed, and the specialization is dropped. Whether to trace
    /// them (on the full form), so the selects they flipped are unpinned;
    /// `false` also where the choices thrash and specialization goes off.
    pub(crate) fn failed(&mut self, policy: &ChoicePolicy, version: u64, n: u64) -> bool {
        if self.current() != Some(version) {
            return false;
        }
        self.current = false;
        self.version += 1;
        self.stable = 0;
        self.cool(policy);
        self.flips += n;
        if self.flips >= policy.thrash_flips && self.evals < self.flips * policy.thrash_evals {
            self.turn_off("choices thrash");
            return false;
        }
        self.trace()
    }

    /// Whether to trace evaluations of the full form now, to build a
    /// specialization from: none is current, no cooldown runs and no trace
    /// is on its way.
    pub(crate) fn due(&mut self, policy: &ChoicePolicy) -> bool {
        self.eligible(policy) && self.current().is_none() && self.ready() && self.trace()
    }

    /// No specialization is built for the policy's cooldown from here.
    pub(crate) fn cool(&mut self, policy: &ChoicePolicy) {
        self.cooldown_until = self.evals + policy.flip_cooldown;
    }

    fn ready(&self) -> bool {
        !self.off && self.evals >= self.cooldown_until
    }

    fn trace(&mut self) -> bool {
        !std::mem::replace(&mut self.tracing, true)
    }

    /// What the traces `traces` teach: the selects they disagree in (with
    /// each other and the trace held) unpinned, and, where due, a new
    /// specialization `build(trace, pin)` makes, with its op count, kept
    /// when it removes enough of the `full` ops (else specialization goes
    /// off): it and its version.
    pub(crate) fn learn<'t, S>(
        &mut self,
        policy: &ChoicePolicy,
        traces: impl IntoIterator<Item = &'t [u8]>,
        full: usize,
        build: impl FnOnce(&[u8], &[bool]) -> (S, usize),
    ) -> Option<(u64, S)> {
        self.tracing = false;
        for t in traces {
            self.observe(t);
        }
        if self.current().is_some() || !self.ready() {
            return None;
        }
        let pin: Vec<bool> = self.unpinned.iter().map(|&u| !u).collect();
        let (spec, ops) = build(&self.trace, &pin);
        if ops * PERCENT > full * (PERCENT - policy.min_shrink_pct.min(PERCENT)) {
            self.turn_off(&format!("shrink too small, {ops} of {full} ops"));
            return None;
        }
        log(
            Level::Debug,
            &format!("rsdag: specialized over run-time choices, {ops} of {full} ops"),
        );
        self.built += 1;
        self.version += 1;
        self.current = true;
        self.stable = 0;
        Some((self.version, spec))
    }

    /// A trace seen: the selects it differs in from the one held are
    /// unpinned for good, and it becomes the one held.
    fn observe(&mut self, trace: &[u8]) {
        if self.trace.len() == trace.len() {
            for (u, (a, b)) in self.unpinned.iter_mut().zip(self.trace.iter().zip(trace)) {
                *u |= a != b;
            }
        }
        self.trace.clear();
        self.trace.extend_from_slice(trace);
    }

    /// Whether the interpreted specialization of `ops` ops serves next to
    /// the full form of `full` ops, `compiled` where that is compiled code.
    pub(crate) fn interpreted_serves(
        policy: &ChoicePolicy,
        ops: usize,
        full: usize,
        compiled: bool,
    ) -> bool {
        !compiled || ops * policy.interp_cost <= full
    }

    /// The version to compile now, `have` the one compiled already: the
    /// current specialization, within the budget, one compile at a time,
    /// once it held through its probation where the interpreted one serves
    /// (`serves`); where it does not, nothing runs it to hold, and it is
    /// compiled at once.
    pub(crate) fn compile_due(
        &mut self,
        policy: &ChoicePolicy,
        have: Option<u64>,
        serves: bool,
    ) -> Option<u64> {
        let v = self.current()?;
        if have == Some(v)
            || self.compiling
            || (serves && self.stable < policy.compile_after)
            || self.compiles >= policy.compile_budget
        {
            return None;
        }
        self.compiling = true;
        self.compiles += 1;
        Some(v)
    }

    /// The compile of `version` landed: whether that is still current.
    pub(crate) fn landed(&mut self, version: u64) -> bool {
        self.compiling = false;
        self.current() == Some(version)
    }

    fn turn_off(&mut self, why: &str) {
        self.off = true;
        self.current = false;
        log(
            Level::Debug,
            &format!(
                "rsdag: specialization over run-time choices off ({why}), {} flips in {} evaluations",
                self.flips, self.evals
            ),
        );
    }
}
