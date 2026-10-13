//! What a program learns of its region choices, shared by everything that
//! specializes against them: [`Adaptive`](crate::Adaptive) for a program
//! evaluated point by point, the function bodies ([`crate::variant`]) for
//! their instances.
//!
//! A model branches by operating region (`Select`), and evaluating it
//! computes both arms. A trace of the choices taken shortens it
//! ([`Tape::specialize`](crate::Tape::specialize),
//! [`Tape::specialize_main`](crate::Tape::specialize_main)), guarded: where
//! a guard fails, the full program serves and the result is the same, bit
//! for bit. What changes is only what pays: a select seen taking the other
//! arm is unpinned for good, so the next specialization keeps it a real
//! `Select`; choices that flip too often turn specialization off.

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
}

impl Default for ChoicePolicy {
    fn default() -> Self {
        ChoicePolicy::DEFAULT
    }
}

/// The choices learned so far: the trace a specialization pins, the
/// selects seen flipping and the counts the policy decides on. An
/// evaluation is one point (a program) or one instance (a body).
#[derive(Default)]
pub(crate) struct Choices {
    /// The trace the current specialization was built from.
    trace: Vec<u8>,
    /// Selects seen flipping: unpinned for good, they stay real `Select`s.
    unpinned: Vec<bool>,
    pub(crate) evals: u64,
    pub(crate) flips: u64,
    /// No specialization is rebuilt before this evaluation count.
    cooldown_until: u64,
    /// Choices thrash, or the shrink is too small: the full program serves.
    pub(crate) off: bool,
    /// Bumped whenever the specialization is replaced, so work for an
    /// outdated one is dropped when it lands.
    pub(crate) version: u64,
    /// Specializations built.
    pub(crate) built: u64,
}

impl Choices {
    /// A trace seen: the selects it differs in from the one held are
    /// unpinned for good, and it becomes the one held. Returns how many
    /// selects it unpinned.
    pub(crate) fn observe(&mut self, trace: &[u8]) -> usize {
        if self.unpinned.len() != trace.len() {
            self.unpinned.resize(trace.len(), false);
        }
        let mut fresh = 0;
        if self.trace.len() == trace.len() {
            for (u, (a, b)) in self.unpinned.iter_mut().zip(self.trace.iter().zip(trace)) {
                if a != b && !*u {
                    *u = true;
                    fresh += 1;
                }
            }
        }
        self.trace.clear();
        self.trace.extend_from_slice(trace);
        fresh
    }

    /// The trace held, the one to specialize against.
    pub(crate) fn trace(&self) -> &[u8] {
        &self.trace
    }

    /// Which selects of the trace a specialization pins.
    pub(crate) fn pin(&self) -> Vec<bool> {
        self.unpinned.iter().map(|&u| !u).collect()
    }

    /// Whether a specialization may be built now (no cooldown runs).
    pub(crate) fn ready(&self) -> bool {
        !self.off && self.evals >= self.cooldown_until
    }

    /// `n` evaluations ran.
    pub(crate) fn ran(&mut self, n: u64) {
        self.evals += n;
    }

    /// `n` evaluations flipped a pinned region: counted, and `true` when
    /// choices thrash, specialization off from here.
    pub(crate) fn flipped(&mut self, policy: &ChoicePolicy, n: u64) -> bool {
        self.flips += n;
        if self.flips >= policy.thrash_flips && self.evals < self.flips * policy.thrash_evals {
            self.off = true;
        }
        self.off
    }

    /// No specialization is rebuilt for the policy's cooldown from here.
    pub(crate) fn cool(&mut self, policy: &ChoicePolicy) {
        self.cooldown_until = self.evals + policy.flip_cooldown;
    }

    /// A specialization of `ops` instructions against a program of `full`
    /// is built: kept when it shrinks the program enough, else
    /// specialization goes off (`false`).
    pub(crate) fn built(&mut self, policy: &ChoicePolicy, ops: usize, full: usize) -> bool {
        if ops * 100 > full * (100 - policy.min_shrink_pct.min(100)) {
            self.off = true;
            return false;
        }
        self.built += 1;
        true
    }
}
