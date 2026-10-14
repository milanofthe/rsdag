//! Specialization over run-time choices ([`crate::choices`],
//! [`crate::variant`], [`crate::adaptive`]): the policies' defaults, what
//! a circuit simulator's Newton loop measured best with. Time only: a
//! specialized program computes the full one's bits wherever its guards
//! hold, and the full program serves where one fails.

use std::time::Duration;

use crate::adaptive::Policy;
use crate::choices::ChoicePolicy;
use crate::variant::VariantPolicy;

/// Instances a trace takes of a batch, spread over it: enough to see the
/// choices most of them take, few enough to cost little next to the batch.
pub(crate) const TRACED: usize = 16;

/// A share in percent (`min_shrink_pct`) against this whole.
pub(crate) const PERCENT: usize = 100;

impl ChoicePolicy {
    /// The defaults ([`Default`] returns them).
    pub const DEFAULT: ChoicePolicy = ChoicePolicy {
        enabled: true,
        min_selects: 16,
        // What a built specialization must remove; compare
        // `VariantPolicy::min_shrink_pct`, a bound on a body before any
        // variant is built.
        min_shrink_pct: 15,
        flip_cooldown: 32,
        thrash_flips: 8,
        thrash_evals: 8,
    };
}

impl VariantPolicy {
    /// The defaults ([`Default`] returns them).
    pub const DEFAULT: VariantPolicy = VariantPolicy {
        enabled: true,
        interpreted: true,
        max_variants: 64,
        // What deciding a body's selects could remove at best, before any
        // variant is built: lower than `ChoicePolicy::min_shrink_pct`, the
        // share a built specialization must remove.
        min_shrink_pct: 10,
        choices: ChoicePolicy::DEFAULT,
    };
}

impl Policy {
    /// The defaults ([`Default`] returns them).
    pub const DEFAULT: Policy = Policy {
        jit: true,
        kick_after: 3,
        eager_pass: Some(Duration::from_millis(10)),
        choices: ChoicePolicy::DEFAULT,
        spec_compile_after: 16,
        spec_compile_budget: 4,
        spec_interp_cost: 4,
        variants: VariantPolicy::DEFAULT,
    };
}
