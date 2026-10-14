//! Every number the bindings' behavior depends on, in one place. Time and
//! text only: what a traced program computes is rsdag's.

/// Node budget of a tracer's `repr`: an expression written out as a tree
/// past this prints as its id and the bound instead.
pub(crate) const REPR_NODES: usize = 200;

/// Programs from this many ops on release the GIL while they run; below,
/// releasing and retaking it costs more than the evaluation.
pub(crate) const DETACH_OPS: usize = 256;

/// The `min_ops` of the parallel batches of a natively compiled program
/// (`rsdag_jit::Batch::Parallel`): ops a batch of function-body calls
/// carries from which it runs on the thread pool. The same unit and the
/// same decision as rsdag's [`MIN_OPS`](rsdag::parallel::MIN_OPS) for a
/// stage, at another value (`20_000` there); both stay as they are until a
/// gate measured in time replaces them.
pub(crate) const PY_JIT_BATCH_MIN_OPS: usize = 1 << 16;
