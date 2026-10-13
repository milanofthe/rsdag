//! How expressions print ([`crate::display`], [`crate::dot`]). Text only.

/// Expressions whose infix form stays under this many bytes print as one
/// infix string; a larger one prints as a listing, one line per node (a DAG
/// printed as a tree grows exponentially with its sharing).
pub(crate) const INFIX_MAX: usize = 4096;
