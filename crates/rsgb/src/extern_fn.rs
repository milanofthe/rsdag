//! Compiled multi-output bodies backing [`Node::Opaque`](crate::node::Node::Opaque).
//!
//! An [`ExternId`](crate::node::ExternId) names an opaque operator in a
//! [`Graph`](crate::graph::Graph). On its own that operator has no body and
//! evaluates to `NaN` (its partial-derivative markers still give a Jacobian its
//! structure); binding it to an output slot of an [`ExternBundle`] gives it a
//! numeric implementation, so the eval paths (arena sweep and compiled
//! [`Tape`](crate::tape::Tape)) call the bundle instead. This is the seam a
//! device template body or an externally-compiled model plugs into.
//!
//! The trait is object-safe and shared as `Arc<dyn ExternBundle>`, so a
//! compiled body survives `Graph` mutation and crosses thread boundaries with
//! the per-thread tapes the solver clones.

/// A multi-output compiled body shared by several opaque operators.
///
/// A compiled multi-output body typically produces many correlated outputs at
/// once: a device's terminal currents *and* the entries of its Jacobian.
/// Computing them in one call (the shared interior runs once) is the whole point of
/// compilation, so several `Opaque` operators are bound to slots of a single
/// `ExternBundle`. The [`Graph`](crate::graph::Graph) records, per
/// [`ExternId`](crate::node::ExternId), which bundle and which output slot it
/// reads; the compiled tape then calls the bundle once and scatters its outputs
/// to all sibling operators that share the same arguments.
pub trait ExternBundle: Send + Sync {
    /// Number of outputs this bundle writes.
    fn n_outputs(&self) -> usize;

    /// Evaluate all outputs from the arguments. `out` has length
    /// [`n_outputs`](Self::n_outputs); `args` holds one value per boundary
    /// input, in input order.
    fn call(&self, args: &[f64], out: &mut [f64]);

    /// Evaluate `n_groups` independent argument groups at once (instance
    /// batching): `args` is group-major (`n_groups * n_args`), `out` likewise
    /// (`n_groups * n_outputs`). The default loops over [`call`](Self::call);
    /// implementations may evaluate the groups as SIMD lanes -- results must
    /// stay bit-identical to the sequential loop.
    fn call_batch(&self, args: &[f64], n_groups: usize, n_args: usize, out: &mut [f64]) {
        let n_out = self.n_outputs();
        for g in 0..n_groups {
            self.call(
                &args[g * n_args..(g + 1) * n_args],
                &mut out[g * n_out..(g + 1) * n_out],
            );
        }
    }
}
