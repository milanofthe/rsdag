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
    /// `thrash_flips` specializations were dropped and their flips came at
    /// one per fewer than `thrash_evals` evaluations.
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

#[derive(Default)]
pub(crate) struct Specializer {
    n_selects: usize,
    held_trace: Vec<u8>,
    unpinned: Vec<bool>,
    evals: u64,
    pub(crate) flips: u64,
    region_changes: u64,
    cooldown_until: u64,
    pub(crate) off: bool,
    version: u64,
    has_current: bool,
    pub(crate) built: u64,
    held_since_built: u64,
    compiles: u32,
    compiling: bool,
    trace_on_its_way: bool,
}

impl Specializer {
    pub(crate) fn new(selects: usize) -> Specializer {
        Specializer {
            n_selects: selects,
            unpinned: vec![false; selects],
            ..Specializer::default()
        }
    }

    pub(crate) fn eligible(&self, policy: &ChoicePolicy) -> bool {
        policy.enabled && self.n_selects >= policy.min_selects
    }

    pub(crate) fn current(&self) -> Option<u64> {
        (self.has_current && !self.off).then_some(self.version)
    }

    pub(crate) fn ran(&mut self, n: u64) {
        self.evals += n;
    }

    pub(crate) fn held(&mut self, version: u64, n: u64) {
        if self.current() == Some(version) {
            self.held_since_built += n;
        }
    }

    pub(crate) fn failed(&mut self, policy: &ChoicePolicy, version: u64, n: u64) -> bool {
        if self.current() != Some(version) {
            return false;
        }
        self.has_current = false;
        self.version += 1;
        self.held_since_built = 0;
        self.cool(policy);
        self.flips += n;
        self.region_changes += 1;
        rsprobe::add!("specializations dropped", 1);
        if self.region_changes >= policy.thrash_flips
            && self.evals < self.flips * policy.thrash_evals
        {
            self.turn_off("choices thrash");
            return false;
        }
        self.claim_trace()
    }

    pub(crate) fn due(&mut self, policy: &ChoicePolicy) -> bool {
        self.eligible(policy) && self.current().is_none() && self.ready() && self.claim_trace()
    }

    pub(crate) fn cool(&mut self, policy: &ChoicePolicy) {
        self.cooldown_until = self.evals + policy.flip_cooldown;
    }

    fn ready(&self) -> bool {
        !self.off && self.evals >= self.cooldown_until
    }

    fn claim_trace(&mut self) -> bool {
        !std::mem::replace(&mut self.trace_on_its_way, true)
    }

    pub(crate) fn learn<'t, S>(
        &mut self,
        policy: &ChoicePolicy,
        traces: impl IntoIterator<Item = &'t [u8]>,
        full: usize,
        build: impl FnOnce(&[u8], &[bool]) -> (S, usize),
    ) -> Option<(u64, S)> {
        self.trace_on_its_way = false;
        for t in traces {
            self.observe(t);
        }
        if self.current().is_some() || !self.ready() {
            return None;
        }
        let pin: Vec<bool> = self.unpinned.iter().map(|&u| !u).collect();
        let (spec, ops) = build(&self.held_trace, &pin);
        if ops * PERCENT > full * (PERCENT - policy.min_shrink_pct.min(PERCENT)) {
            self.turn_off(&format!("shrink too small, {ops} of {full} ops"));
            return None;
        }
        rsprobe::add!("specializations built", 1);
        rsprobe::debug!("specialized over run-time choices, {ops} of {full} ops");
        self.built += 1;
        self.version += 1;
        self.has_current = true;
        self.held_since_built = 0;
        Some((self.version, spec))
    }

    fn observe(&mut self, trace: &[u8]) {
        if self.held_trace.len() == trace.len() {
            for (u, (a, b)) in self
                .unpinned
                .iter_mut()
                .zip(self.held_trace.iter().zip(trace))
            {
                *u |= a != b;
            }
        }
        self.held_trace.clear();
        self.held_trace.extend_from_slice(trace);
    }

    pub(crate) fn interpreted_serves(
        policy: &ChoicePolicy,
        ops: usize,
        full: usize,
        compiled: bool,
    ) -> bool {
        !compiled || ops * policy.interp_cost <= full
    }

    pub(crate) fn compile_due(
        &mut self,
        policy: &ChoicePolicy,
        have: Option<u64>,
        serves: bool,
    ) -> Option<u64> {
        let v = self.current()?;
        if have == Some(v)
            || self.compiling
            || (serves && self.held_since_built < policy.compile_after)
            || self.compiles >= policy.compile_budget
        {
            return None;
        }
        self.compiling = true;
        self.compiles += 1;
        Some(v)
    }

    pub(crate) fn landed(&mut self, version: u64) -> bool {
        self.compiling = false;
        self.current() == Some(version)
    }

    fn turn_off(&mut self, why: &str) {
        self.off = true;
        self.has_current = false;
        rsprobe::add!("specializations off", 1);
        rsprobe::debug!(
            "specialization over run-time choices off ({why}), {} flips in {} evaluations",
            self.flips,
            self.evals
        );
    }
}
