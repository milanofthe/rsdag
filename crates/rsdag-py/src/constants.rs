//! Every number the bindings' behavior depends on, in one place. Time and
//! text only: what a traced program computes is rsdag's.

/// Node budget of a tracer's `repr`: an expression written out as a tree
/// past this prints as its id and the bound instead.
pub(crate) const REPR_NODES: usize = 200;

/// Programs from this many ops on release the GIL while they run; below,
/// releasing and retaking it costs more than the evaluation.
pub(crate) const DETACH_OPS: usize = 256;
