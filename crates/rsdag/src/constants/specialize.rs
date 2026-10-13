//! Specialization over run-time choices ([`crate::choices`],
//! [`crate::variant`], [`crate::adaptive`]). Time only: a specialized
//! program computes the full one's bits wherever its guards hold.

/// Instances a trace takes of a batch, spread over it: enough to see the
/// choices most of them take, few enough to cost little next to the batch.
pub(crate) const TRACED: usize = 16;
