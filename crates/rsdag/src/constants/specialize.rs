//! Specialization over run-time choices ([`crate::choices`],
//! [`crate::variant`], [`crate::adaptive`]): the policies' defaults, what
//! a circuit simulator's Newton loop measured best with. Time only: a
//! specialized program computes the full one's bits wherever its guards
//! hold, and the full program serves where one fails.

use crate::adaptive::Policy;
use crate::choices::ChoicePolicy;
use crate::variant::VariantPolicy;

/// The instances of a batch that a trace takes, spread over the batch.
/// They are enough to see the choices that most instances take, and few
/// enough to cost little next to the batch.
pub(crate) const TRACED: usize = 16;

/// The whole against which a share in percent (`min_shrink_pct`) is taken.
pub(crate) const PERCENT: usize = 100;

impl ChoicePolicy {
    /// The defaults ([`Default`] returns them). Its `min_shrink_pct` is the
    /// share that a built specialization must remove. Compare
    /// `VariantPolicy::min_shrink_pct`, a bound on a body before any variant
    /// is built.
    pub const DEFAULT: ChoicePolicy = ChoicePolicy {
        enabled: true,
        min_selects: 16,
        min_shrink_pct: 15,
        flip_cooldown: 32,
        thrash_flips: 8,
        thrash_evals: 8,
        compile_after: 16,
        compile_budget: 4,
        interp_cost: 4,
    };
}

impl VariantPolicy {
    /// The defaults ([`Default`] returns them). Its `min_shrink_pct` bounds
    /// what deciding the selects of a body could remove at best, before any
    /// variant is built. It is lower than `ChoicePolicy::min_shrink_pct`,
    /// the share that a built specialization must remove.
    pub const DEFAULT: VariantPolicy = VariantPolicy {
        enabled: true,
        interpreted: true,
        max_variants: 64,
        min_shrink_pct: 10,
        choices: ChoicePolicy::DEFAULT,
    };
}

const _: () = assert!(VariantPolicy::DEFAULT.min_shrink_pct < ChoicePolicy::DEFAULT.min_shrink_pct);

impl Policy {
    /// The defaults ([`Default`] returns them).
    pub const DEFAULT: Policy = Policy {
        jit: true,
        kick_after: 3,
        wait: true,
        choices: ChoicePolicy::DEFAULT,
        variants: VariantPolicy::DEFAULT,
    };
}
