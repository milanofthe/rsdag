//! Tape compilation, in five passes over the reachable forest:
//!
//! 1. [`Forest::analyze`] -- reachability, parameter-purity for the prolog
//!    split, use counts, and superinstruction fusion.
//! 2. [`Forest::schedule`] -- register-pressure list scheduling into an
//!    instruction order, and the position tables over it.
//! 3. [`Forest::liveness`] -- last-use positions and the pins that keep a
//!    root or a prolog value alive.
//! 4. [`batch_groups`] -- the instance-batching pre-scan.
//! 5. [`Forest::emit`] -- slot allocation and the instruction stream.
//!
//! Each pass reads only what the ones before it produced, which is what
//! makes them separable, and each names its output in a struct rather than
//! in a dozen parallel vectors of a single long function. See the module
//! docs of [`super`] for the IR itself.

use crate::field::Field;
use std::collections::BTreeSet;
use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use super::{BatchTable, Op, Tape};
use crate::extern_fn::ExternBundle;
use crate::func::{Body, FuncId, Output};
use crate::graph::Graph;
use crate::node::{ExprId, Node, SymbolId};

impl Tape {
    /// Compile a tape computing `roots`, where `inputs[k]` (passed to
    /// [`eval`](Self::eval)) is the value of symbol `input_syms[k]`. Symbols not
    /// listed evaluate to `NaN`.
    pub fn compile<K: Field>(ctx: &Graph<K>, roots: &[ExprId], input_syms: &[SymbolId]) -> Tape {
        Self::compile_inner(ctx, roots, input_syms, None)
    }

    /// [`compile`](Self::compile) with a prolog split: `pure_inputs[k]` marks
    /// input `k` as solve-constant (a parameter), and every op depending only
    /// on such inputs is scheduled into a *prolog prefix* of the instruction
    /// stream. A Newton loop evaluates the prolog once per parameter binding
    /// ([`eval_prolog`](Self::eval_prolog)) and then only the remainder per
    /// iteration ([`eval_main`](Self::eval_main)); values crossing the boundary
    /// are pinned so iteration-loop slot reuse cannot clobber them. Plain
    /// [`eval`](Self::eval) still runs the whole stream, so the split is
    /// invisible to callers that ignore it.
    pub fn compile_split<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: &[bool],
    ) -> Tape {
        Self::compile_inner(ctx, roots, input_syms, Some(pure_inputs))
    }

    fn compile_inner<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> Tape {
        // Five passes over the reachable forest, each reading only what the
        // ones before it produced: analysis marks and counts, scheduling
        // picks an order, liveness turns that order into lifetimes, the
        // batch scan groups repeated calls, and emission allocates slots and
        // writes the instruction stream.
        // Each pass reports its time through `hooks` at debug level, which
        // is what a consumer's pipeline profile reads.
        use crate::hooks::timed;
        let forest = timed("tape analyze", || {
            Forest::analyze(ctx, roots, input_syms, pure_inputs)
        });
        let schedule = timed("tape schedule", || forest.schedule(ctx));
        let liveness = timed("tape liveness", || {
            forest.liveness(ctx, roots, &schedule, pure_inputs.is_some())
        });
        let batch_group_args = timed("tape batch scan", || batch_groups(ctx, &schedule.order));
        timed("tape emit", || {
            forest.emit(ctx, roots, &schedule, &liveness, &batch_group_args)
        })
    }
}

/// The reachable forest of one compilation, in dependency order, with the
/// tables the later passes index by *base position* (the index into
/// [`Forest::base`], not the arena id).
struct Forest {
    /// Reachable nodes in ascending arena id, which is a dependency order.
    base: Vec<ExprId>,
    /// Base position by arena id; `u32::MAX` for an unreachable node.
    bpos: Vec<u32>,
    /// Input index of each mapped symbol.
    input_of: HashMap<SymbolId, u32>,
    /// Parameter-purity for the prolog split; all false without one.
    pure: Vec<bool>,
    /// `fused_into_b[p]` is the base position of the `Add` that absorbed the
    /// node at `p` as a superinstruction operand.
    fused_into_b: Vec<Option<usize>>,
}

/// The instruction order, and the tables indexed by *schedule position*.
struct Schedule {
    order: Vec<ExprId>,
    /// First position of the main (impure) phase.
    boundary: usize,
    /// Schedule position by arena id.
    pos: Vec<u32>,
    /// The consuming op's schedule position, for a fused operand.
    fused_into: Vec<Option<usize>>,
}

/// Lifetimes over the schedule.
struct Liveness {
    /// Highest position that reads each position's value; `usize::MAX` for a
    /// root, or for a prolog value the main phase reads.
    last: Vec<usize>,
    pinned: Vec<bool>,
    /// Which positions belong to the prolog; empty without a split.
    is_pure: Vec<bool>,
}

impl Forest {
    /// Base position of a reachable node.
    #[inline]
    fn pos(&self, e: ExprId) -> usize {
        self.bpos[e.0 as usize] as usize
    }

    /// Pass 1: reachability, purity, use counts and superinstruction fusion.
    fn analyze<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> Forest {
        let mut input_of: HashMap<SymbolId, u32> =
            HashMap::with_capacity_and_hasher(input_syms.len(), Default::default());
        for (k, &s) in input_syms.iter().enumerate() {
            input_of.insert(s, k as u32);
        }

        // Reachable nodes in ascending id (= dependencies first). `mark` is a
        // dense table over the arena, so the sweep and every later position
        // lookup is an array index, not a hash.
        let n_arena = ctx.len();
        let mut mark = vec![false; n_arena];
        let mut stack = roots.to_vec();
        while let Some(id) = stack.pop() {
            let k = id.0 as usize;
            if mark[k] {
                continue;
            }
            mark[k] = true;
            stack.extend_from_slice(&ctx.operands(id));
        }
        let base: Vec<ExprId> = (0..n_arena)
            .filter(|&k| mark[k])
            .map(|k| ExprId(k as u32))
            .collect();
        let m = base.len();
        // Dense position table (index into `base`) by ExprId.
        let mut bpos = vec![u32::MAX; n_arena];
        for (i, id) in base.iter().enumerate() {
            bpos[id.0 as usize] = i as u32;
        }
        let bp = |e: ExprId| bpos[e.0 as usize] as usize;

        // Purity (prolog split): a node is parameter-pure when every operand is.
        let mut pure = vec![false; m];
        if let Some(mask) = pure_inputs {
            for (i, id) in base.iter().enumerate() {
                pure[i] = match ctx.node(*id) {
                    Node::Const(_) => true,
                    Node::Symbol(s) => match input_of.get(s) {
                        // unmapped symbols evaluate to a NaN constant: pure
                        None => true,
                        Some(&k) => mask.get(k as usize).copied().unwrap_or(false),
                    },
                    _ => ctx.operands(*id).iter().all(|a| pure[bp(*a)]),
                };
            }
        }

        let mut is_root = vec![false; m];
        for r in roots {
            is_root[bp(*r)] = true;
        }

        // Use counts (operand occurrences over the reachable forest).
        let mut uses = vec![0u32; m];
        for id in &base {
            for a in ctx.operands(*id).iter() {
                uses[bp(*a)] += 1;
            }
        }

        // Superinstruction fusion. An `Add` one of whose operands is a
        // single-use `Mul` becomes one `MulAdd` dispatch (a single-use `Neg`
        // becomes `Sub`); the fused operand is never materialized. This is a
        // dispatch fusion only -- the arithmetic stays the identical sequence
        // of IEEE operations, so parity with the arena sweep is bit-exact.
        // `fused_into_b[p]` is the base index of the consuming `Add`.
        let mut fused_into_b: Vec<Option<usize>> = vec![None; m];
        for (i, id) in base.iter().enumerate() {
            if let Node::Add(a, b) = ctx.node(*id) {
                let fusable = |x: &ExprId, fused: &[Option<usize>]| {
                    let px = bp(*x);
                    !is_root[px]
                        && uses[px] == 1
                        && fused[px].is_none()
                        && matches!(ctx.node(*x), Node::Mul(..) | Node::Neg(..))
                };
                if fusable(a, &fused_into_b) {
                    fused_into_b[bp(*a)] = Some(i);
                } else if fusable(b, &fused_into_b) {
                    fused_into_b[bp(*b)] = Some(i);
                }
            }
        }

        Forest {
            base,
            bpos,
            input_of,
            pure,
            fused_into_b,
        }
    }

    /// Pass 2: the instruction order, and the position tables over it.
    fn schedule<K: Field>(&self, ctx: &Graph<K>) -> Schedule {
        let base = &self.base;
        let pure = &self.pure;
        let fused_into_b = &self.fused_into_b;
        let m = base.len();
        let n_arena = self.bpos.len();
        let bp = |e: ExprId| self.pos(e);
        // ---- schedule: register-pressure-driven list scheduling -------------
        //
        // Ascending id (creation order) is a valid schedule but a poor one:
        // every residual is created before any Jacobian entry, so the primal
        // intermediates of every device stay live until their derivatives are
        // evaluated at the very end -- the live set is the whole primal forest.
        // Instead, a greedy list scheduler: among the ready ops prefer the one
        // that kills the most live values (its operands' last use), and among
        // equals the most recently enabled one (LIFO: finish the computation
        // in flight before starting another). A device's derivatives then
        // follow its residual directly, and the live set is one device's
        // worth, not the circuit's. The prolog split is a strict phase: every
        // parameter-pure op precedes every impure one (the pure subgraph is
        // closed under operands, so it is always fully schedulable first).
        //
        // A fused operand is not scheduled on its own: its consumer's
        // dependencies are the union of both operand lists, and it is emitted
        // (as a no-op placeholder) right before that consumer.
        let deps_of = |i: usize| -> Vec<usize> {
            // Distinct dependencies of base node `i` as base indices, seeing
            // through a fused operand to its operands.
            let mut out: Vec<usize> = Vec::with_capacity(4);
            for &a in ctx.operands(base[i]).iter() {
                let pa = bp(a);
                if fused_into_b[pa] == Some(i) {
                    for &x in ctx.operands(a).iter() {
                        out.push(bp(x));
                    }
                } else {
                    out.push(pa);
                }
            }
            out.sort_unstable();
            out.dedup();
            out
        };
        // Leaves (inputs / constants) are never scheduled on their own: they
        // are materialised on demand right before their first consumer, so a
        // multiply-used parameter is live from its first use, not from the
        // start of the tape.
        let is_leaf: Vec<bool> = base
            .iter()
            .map(|id| matches!(ctx.node(*id), Node::Const(_) | Node::Symbol(_)))
            .collect();
        // Reverse edges (CSR) among the scheduled (non-leaf, non-fused) ops.
        let mut user_count = vec![0u32; m];
        let mut pending = vec![0u32; m];
        let mut dep_lists: Vec<Vec<usize>> = Vec::with_capacity(m);
        for i in 0..m {
            if fused_into_b[i].is_some() || is_leaf[i] {
                dep_lists.push(Vec::new());
                continue;
            }
            let d = deps_of(i);
            pending[i] = d.iter().filter(|&&x| !is_leaf[x]).count() as u32;
            for &x in &d {
                user_count[x] += 1;
            }
            dep_lists.push(d);
        }
        let mut user_start = vec![0u32; m + 1];
        for i in 0..m {
            user_start[i + 1] = user_start[i] + user_count[i];
        }
        let mut users = vec![0u32; user_start[m] as usize];
        let mut fill = user_start.clone();
        for (i, d) in dep_lists.iter().enumerate() {
            for &x in d {
                users[fill[x] as usize] = i as u32;
                fill[x] += 1;
            }
        }
        // Remaining (distinct-user) count per node: a scheduled op kills an
        // operand when it is that operand's last remaining user.
        let mut remaining: Vec<u32> = user_count.clone();
        let kills_of = |i: usize, remaining: &[u32]| -> u32 {
            dep_lists[i].iter().filter(|&&d| remaining[d] == 1).count() as u32
        };
        // Max-heap on (pure phase first, most kills, most recently enabled).
        // The initial candidates (leaf-only operands) are seeded in creation
        // order -- lowest id pops first -- so the schedule starts where the
        // graph was built; everything enabled later is LIFO.
        let mut heap: std::collections::BinaryHeap<(bool, u32, u64, u32)> =
            std::collections::BinaryHeap::new();
        let mut seq: u64 = 0;
        for i in (0..m).rev() {
            if fused_into_b[i].is_none() && !is_leaf[i] && pending[i] == 0 {
                heap.push((pure[i], kills_of(i, &remaining), seq, i as u32));
                seq += 1;
            }
        }
        let mut order: Vec<ExprId> = Vec::with_capacity(m);
        let mut emitted = vec![false; m];
        // First position of the main (impure) phase, once known.
        let mut boundary: Option<usize> = None;
        // The op emitted last (base index), for the interleave below.
        let mut last_op: Option<usize> = None;
        while let Some(top) = heap.pop() {
            let (is_pure_op, kills, _, iu) = top;
            let mut i = iu as usize;
            if emitted[i] {
                continue;
            }
            // ILP interleave: a pure depth-first order chains every op onto
            // the one just emitted (its value round-trips through the slot
            // memory, so the interpreter serialises on store-to-load
            // latency). When an equally good candidate that does NOT read
            // the last op is available, take it first, so two chains
            // alternate and the CPU overlaps them. Register pressure is
            // unaffected to first order (same kills).
            if let (Some(lo), Some(&(p2, k2, _, iu2))) = (last_op, heap.peek()) {
                let i2 = iu2 as usize;
                if p2 == is_pure_op
                    && k2 == kills
                    && !emitted[i2]
                    && dep_lists[i].contains(&lo)
                    && !dep_lists[i2].contains(&lo)
                {
                    heap.pop();
                    heap.push(top);
                    i = i2;
                }
            }
            if !is_pure_op && boundary.is_none() {
                boundary = Some(order.len());
            }
            last_op = Some(i);
            // Materialise this op's not-yet-emitted leaf operands (its own and
            // its fused operand's), then a fused operand as a placeholder,
            // then the op itself.
            for &d in &dep_lists[i] {
                if is_leaf[d] && !emitted[d] {
                    emitted[d] = true;
                    order.push(base[d]);
                }
            }
            for &a in ctx.operands(base[i]).iter() {
                let pa = bp(a);
                if fused_into_b[pa] == Some(i) {
                    emitted[pa] = true;
                    order.push(a);
                }
            }
            emitted[i] = true;
            order.push(base[i]);
            for &d in &dep_lists[i] {
                remaining[d] -= 1;
            }
            for k in user_start[i]..user_start[i + 1] {
                let u = users[k as usize] as usize;
                pending[u] -= 1;
                if pending[u] == 0 {
                    heap.push((pure[u], kills_of(u, &remaining), seq, u as u32));
                    seq += 1;
                }
            }
        }
        // Leaves that are roots themselves (never consumed by an op): pure
        // ones may still join the prolog, impure ones must start the main
        // phase if nothing else did (a root that is a bare state input).
        for phase_pure in [true, false] {
            for i in 0..m {
                if !emitted[i] && pure[i] == phase_pure {
                    debug_assert!(is_leaf[i], "list scheduler left an op unscheduled");
                    if !phase_pure && boundary.is_none() {
                        boundary = Some(order.len());
                    }
                    emitted[i] = true;
                    order.push(base[i]);
                }
            }
        }
        debug_assert_eq!(order.len(), m, "list scheduler must emit every node");
        let boundary = boundary.unwrap_or(order.len());
        // Dense position table over the schedule, and the per-position views.
        // The prolog is exactly the emitted prefix before `boundary`: a pure
        // leaf first demanded by a main-phase op is (re)materialised there,
        // which is correct and keeps the phase boundary a plain prefix.
        let mut pos_t = vec![u32::MAX; n_arena];
        for (i, id) in order.iter().enumerate() {
            pos_t[id.0 as usize] = i as u32;
        }
        let pos = pos_t;
        let p = |e: ExprId| pos[e.0 as usize] as usize;
        let fused_into: Vec<Option<usize>> = order
            .iter()
            .map(|id| fused_into_b[bp(*id)].map(|c| p(base[c])))
            .collect();

        Schedule {
            order,
            boundary,
            pos,
            fused_into,
        }
    }

    /// Pass 3: lifetimes and pins.
    fn liveness<K: Field>(
        &self,
        ctx: &Graph<K>,
        roots: &[ExprId],
        schedule: &Schedule,
        split: bool,
    ) -> Liveness {
        let m = self.base.len();
        let order = &schedule.order;
        let boundary = schedule.boundary;
        let fused_into = &schedule.fused_into;
        let p = |e: ExprId| schedule.pos[e.0 as usize] as usize;
        // The prolog is exactly the emitted prefix before `boundary`.
        let is_pure: Vec<bool> = if split {
            (0..m).map(|i| i < boundary).collect()
        } else {
            Vec::new()
        };
        let n_pure_nodes = boundary;

        let mut pinned = vec![false; m];
        for r in roots {
            pinned[p(*r)] = true;
        }

        // Last-use position of each node (highest op index that reads it). A
        // fused node executes inside its consumer, so its operands are read at
        // the consumer's position; roots are pinned so their slot survives.
        let mut last = vec![0usize; m];
        for (i, id) in order.iter().enumerate() {
            let read_at = fused_into[i].unwrap_or(i);
            for a in ctx.operands(*id).iter() {
                let q = p(*a);
                last[q] = last[q].max(read_at);
            }
        }
        for r in roots {
            last[p(*r)] = usize::MAX;
        }
        // Prolog split: a pure value read by the main phase must survive every
        // later `eval_main` pass, so its slot is pinned like a root -- the main
        // phase's slot reuse must never clobber a prolog result it will read
        // again on the next Newton iteration. (Fused nodes materialize nothing
        // and need no pin; their pure operands are pinned through their
        // consumer's read position.)
        if !is_pure.is_empty() {
            for i in 0..m {
                if is_pure[i] && fused_into[i].is_none() && last[i] >= n_pure_nodes {
                    last[i] = usize::MAX;
                }
            }
        }

        Liveness {
            last,
            pinned,
            is_pure,
        }
    }

    /// Pass 5: slot allocation and the instruction stream.
    fn emit<K: Field>(
        &self,
        ctx: &Graph<K>,
        roots: &[ExprId],
        schedule: &Schedule,
        liveness: &Liveness,
        batch_group_args: &HashMap<u32, Vec<Vec<ExprId>>>,
    ) -> Tape {
        let base = &self.base;
        let input_of = &self.input_of;
        let m = base.len();
        let order = &schedule.order;
        let fused_into = &schedule.fused_into;
        let p = |e: ExprId| schedule.pos[e.0 as usize] as usize;
        let last = &liveness.last;
        let pinned = &liveness.pinned;
        let is_pure = &liveness.is_pure;
        let n_pure_nodes = schedule.boundary;
        // Per function: the evaluating bundle and its output-to-slot map --
        // the solver's registered body, the extern body, or an interpreted
        // body built here, so a tape is total without any registration.
        let mut bodies: HashMap<u32, Body> = HashMap::default();
        let mut evaluator = |ctx: &Graph<K>, f: FuncId, out: u32| {
            let body = bodies.entry(f.0).or_insert_with(|| ctx.func(f).body(ctx));
            (
                body.bundle.clone(),
                body.slot_of.get(out as usize).copied().flatten(),
            )
        };
        let mut emitted = vec![false; m];

        // Allocate slots with a LIFO free list; free an op's dying operands
        // before picking its dst, so the op can reuse a dying operand's slot.
        let mut slot_by_pos = vec![0u32; m];
        let mut free: Vec<u32> = Vec::new();
        let mut next: u32 = 0;
        let mut ops = Vec::with_capacity(m);
        let mut dst = Vec::with_capacity(m);
        let mut arg_pool: Vec<u32> = Vec::new();
        let mut max_args = 0usize;
        let mut bundles: Vec<Arc<dyn ExternBundle>> = Vec::new();
        let mut bundle_tape_idx: HashMap<u32, u32> = HashMap::default();
        let mut group_base: HashMap<(u32, Vec<u32>), u32> = HashMap::default();
        let mut bundle_scratch_len: u32 = 0;
        let mut sink: Option<u32> = None;
        let mut split_at: Option<usize> = None;
        let mut batches: Vec<BatchTable> = Vec::new();
        let mut batch_args_len = 0usize;
        for i in 0..m {
            if !is_pure.is_empty() && i == n_pure_nodes && split_at.is_none() {
                split_at = Some(ops.len()); // first main-phase op starts here
            }
            if fused_into[i].is_some() {
                continue; // materialized inside its consuming Add
            }
            if emitted[i] {
                continue; // leaf materialised early by a BundleBatch below
            }
            let node = ctx.node(order[i]);
            // Instance batching: the FIRST bundled opaque of a batchable bundle
            // materialises every group's leaf arguments (inputs/constants have
            // no dependencies, so early emission is always legal) and emits ONE
            // `BundleBatch`; every group is pre-registered in `group_base`, so
            // this and all later opaques of the bundle lower to cheap picks.
            if let Node::Call(o, _) = node {
                let (cf, cout) = ctx.output(*o);
                let cbidx = cf.0;
                if !matches!(ctx.func(cf).outputs[cout as usize], Output::Zero) {
                    if let Some(groups) = batch_group_args.get(&cbidx) {
                        // not an entry(): ops are emitted between check and insert
                        #[allow(clippy::map_entry)]
                        if !bundle_tape_idx.contains_key(&cbidx) {
                            for a in groups.iter().flatten() {
                                let ap = p(*a);
                                if ap > i && !emitted[ap] {
                                    let op = match ctx.node(*a) {
                                        Node::Const(c) => Op::Const(ctx.const_val(*c).to_f64()),
                                        Node::Symbol(sym) => Op::Input(
                                            input_of.get(sym).copied().unwrap_or(u32::MAX),
                                        ),
                                        _ => unreachable!("batch pre-scan admits only leaves"),
                                    };
                                    let d = free.pop().unwrap_or_else(|| {
                                        let d = next;
                                        next += 1;
                                        d
                                    });
                                    slot_by_pos[ap] = d;
                                    ops.push(op);
                                    dst.push(d);
                                    emitted[ap] = true;
                                }
                            }
                            let (cb, _) = evaluator(ctx, cf, cout);
                            let n_out = cb.n_outputs() as u32;
                            let tape_bidx = {
                                let k = bundles.len() as u32;
                                bundles.push(cb);
                                bundle_tape_idx.insert(cbidx, k);
                                k
                            };
                            let n_args = groups[0].len() as u32;
                            let n_groups = groups.len() as u32;
                            let base0 = bundle_scratch_len;
                            bundle_scratch_len += n_groups * n_out;
                            let start = arg_pool.len() as u32;
                            for (gi, g) in groups.iter().enumerate() {
                                let arg_slots: Vec<u32> =
                                    g.iter().map(|a| slot_by_pos[p(*a)]).collect();
                                arg_pool.extend(arg_slots.iter().copied());
                                group_base.insert((cbidx, arg_slots), base0 + gi as u32 * n_out);
                            }
                            max_args = max_args.max(n_args as usize);
                            batch_args_len = batch_args_len.max((n_groups * n_args) as usize);
                            let tidx = batches.len() as u32;
                            batches.push(BatchTable {
                                start,
                                n_args,
                                n_groups,
                                base0,
                                n_out,
                            });
                            let snk = *sink.get_or_insert_with(|| {
                                let d = next;
                                next += 1;
                                d
                            });
                            ops.push(Op::BundleBatch(tape_bidx, tidx));
                            dst.push(snk);
                        }
                    }
                }
            }
            let s = |a: &ExprId| slot_by_pos[p(*a)];
            let op = match node {
                Node::Const(c) => Op::Const(ctx.const_val(*c).to_f64()),
                Node::Symbol(sym) => Op::Input(input_of.get(sym).copied().unwrap_or(u32::MAX)),
                Node::Add(a, b) => {
                    let fused_operand = if fused_into[p(*a)] == Some(i) {
                        Some(a)
                    } else if fused_into[p(*b)] == Some(i) {
                        Some(b)
                    } else {
                        None
                    };
                    match fused_operand {
                        Some(f) => {
                            let other = if f == a { b } else { a };
                            match ctx.node(*f) {
                                Node::Mul(x, y) => Op::MulAdd(s(x), s(y), s(other)),
                                Node::Neg(x) => Op::Sub(s(other), s(x)),
                                _ => unreachable!("only Mul/Neg operands are fused"),
                            }
                        }
                        None => Op::Add(s(a), s(b)),
                    }
                }
                Node::Mul(a, b) => Op::Mul(s(a), s(b)),
                Node::Neg(a) => Op::Neg(s(a)),
                Node::Pow(a, n) => Op::Powi(s(a), *n as i32),
                Node::Unary(op, a) => Op::Unary(*op, s(a)),
                Node::Cmp(op, a, b) => Op::Cmp(*op, s(a), s(b)),
                Node::Binary(op, a, b) => Op::Binary(*op, s(a), s(b)),
                Node::Select(c, t, e) => Op::Select(s(c), s(t), s(e)),
                Node::Reduce(op, l) => {
                    let args = ctx.args(*l);
                    let start = arg_pool.len() as u32;
                    arg_pool.extend(args.iter().map(&s));
                    max_args = max_args.max(args.len());
                    Op::Reduce(*op, start, args.len() as u32)
                }
                Node::Dot(l) => {
                    let (a, b) = ctx.dot_args(*l);
                    let start = arg_pool.len() as u32;
                    arg_pool.extend(a.iter().map(&s));
                    arg_pool.extend(b.iter().map(&s));
                    max_args = max_args.max(2 * a.len());
                    Op::Dot(start, a.len() as u32)
                }
                Node::Call(o, l) => {
                    let args = ctx.args(*l);
                    let (cf, cout) = ctx.output(*o);
                    let cbidx = cf.0;
                    let (cb, slot) = evaluator(ctx, cf, cout);
                    if let Some(slot) = slot {
                        // Emit one BundleCall per distinct (function, args)
                        // group, then a cheap pick for this output's slot.
                        let arg_slots: Vec<u32> = args.iter().map(&s).collect();
                        let key = (cbidx, arg_slots.clone());
                        let n_out_all = cb.n_outputs() as u32;
                        let base = *group_base.entry(key).or_insert_with(|| {
                            let tape_bidx = *bundle_tape_idx.entry(cbidx).or_insert_with(|| {
                                let k = bundles.len() as u32;
                                bundles.push(cb.clone());
                                k
                            });
                            let n_out = n_out_all;
                            let base = bundle_scratch_len;
                            bundle_scratch_len += n_out;
                            let start = arg_pool.len() as u32;
                            arg_pool.extend(arg_slots.iter().copied());
                            max_args = max_args.max(arg_slots.len());
                            let snk = *sink.get_or_insert_with(|| {
                                let s = next;
                                next += 1;
                                s
                            });
                            ops.push(Op::BundleCall(
                                tape_bidx,
                                start,
                                arg_slots.len() as u32,
                                base,
                            ));
                            dst.push(snk);
                            base
                        });
                        Op::BundlePick(base + slot)
                    } else {
                        // A zero output (a derivative the body does not carry).
                        Op::Const(0.0)
                    }
                }
            };
            // Free distinct operands that die at this step. A fused operand was
            // never materialized (no slot); what dies here instead are *its*
            // operands, whose last use was attributed to this position.
            let mut dying: Vec<usize> = Vec::new();
            for &a in ctx.operands(order[i]).iter() {
                let ap = p(a);
                if fused_into[ap] == Some(i) {
                    for x in ctx.operands(a).iter() {
                        dying.push(p(*x));
                    }
                } else {
                    dying.push(ap);
                }
            }
            dying.retain(|&ap| last[ap] == i && !pinned[ap] && fused_into[ap].is_none());
            dying.sort_unstable();
            dying.dedup();
            for ap in dying {
                free.push(slot_by_pos[ap]);
            }
            let d = free.pop().unwrap_or_else(|| {
                let s = next;
                next += 1;
                s
            });
            slot_by_pos[i] = d;
            ops.push(op);
            dst.push(d);
        }
        let outputs = roots.iter().map(|r| slot_by_pos[p(*r)]).collect();
        let prolog_ops = if is_pure.is_empty() {
            0
        } else {
            split_at.unwrap_or(ops.len())
        };
        let n_selects = ops.iter().filter(|o| matches!(o, Op::Select(..))).count();
        Tape {
            ops,
            dst,
            n_selects,
            arg_pool,
            outputs,
            n_work: next as usize,
            max_args,
            bundles,
            bundle_scratch_len: bundle_scratch_len as usize,
            batches,
            batch_args_len,
            prolog_ops,
        }
    }
}

/// Pass 4: instance batching pre-scan. For every function referenced by at
/// least two distinct argument groups whose arguments are all plain inputs
/// or constants, the per-group calls can coalesce into one batched call at
/// the first call site.
fn batch_groups<K: Field>(ctx: &Graph<K>, order: &[ExprId]) -> HashMap<u32, Vec<Vec<ExprId>>> {
    // Groups are keyed and ordered by first encounter, so the batched call
    // lands at the first call site, where the leaf arguments can always be
    // materialised.
    let mut batch_group_args: HashMap<u32, Vec<Vec<ExprId>>> = HashMap::default();
    let mut seen: BTreeSet<(u32, Vec<ExprId>)> = BTreeSet::new();
    for id in order {
        if let Node::Call(o, l) = *ctx.node(*id) {
            let (f, _) = ctx.output(o);
            let args = ctx.args(l);
            if seen.insert((f.0, args.to_vec())) {
                batch_group_args.entry(f.0).or_default().push(args.to_vec());
            }
        }
    }
    batch_group_args.retain(|_, groups| {
        groups.len() >= 2
            && groups.iter().all(|g| g.len() == groups[0].len())
            && groups
                .iter()
                .flatten()
                .all(|a| matches!(ctx.node(*a), Node::Const(_) | Node::Symbol(_)))
    });
    batch_group_args
}
