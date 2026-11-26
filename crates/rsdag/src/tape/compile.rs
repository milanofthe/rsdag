//! The tape compiler: the reachable forest lowered to instructions, then
//! scheduled, then given slots.
//!
//! Five passes, each reading only what the ones before it produced:
//!
//! 1. [`Forest::analyze`]: reachability, purity for the prolog split, use
//!    counts and the dispatch fusions (`MulAdd`, `Sub`).
//! 2. [`Forest::lower`]: one [`Inst`] per node, or one per *kernel*: the
//!    calls of a function with distinct argument lists at one call depth
//!    are one batched call, the row dots of one vector one matrix-vector
//!    product, the components of one dense system one solve. A kernel is
//!    an instruction with several outputs; the nodes it stands for are its
//!    output values. Leaves are operands, never instructions: an input is
//!    read where it is used, a constant is an instruction of its own.
//! 3. [`Program::fuse_accumulators`]: a product kernel's output whose one
//!    consumer adds, subtracts or negates it is folded by the kernel.
//! 4. [`Program::schedule`]: register-pressure list scheduling over the
//!    instructions, the prolog as a strict first phase.
//! 5. [`Program::emit`]: lifetimes, slots (a kernel's outputs a block of
//!    consecutive slots), and the instruction stream.
//!
//! Passes 1 and 2 serve every tape over one set of roots: a [`Lowered`]
//! program is lowered once, and a tape of some of its roots is a view of
//! it, the instructions those roots reach, its calls computing the outputs
//! read ([`Program::narrow_calls`]), then passes 3 to 5 alone.
//!
//! A compiled tape can be lifted back into its program ([`Tape::lift`]),
//! which is how a transform of a tape (a specialization) is compiled.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use super::topo::Topo;
use super::{input_index, Accum, Fold, Op, Region, Src, Tape, INPUT};
use crate::extern_fn::ExternBundle;
use crate::field::Field;
use crate::func::{Body, FuncId, Output};
use crate::graph::Graph;
use crate::graph::NO_CONTEXT;
use crate::node::{ArgList, ExprId, Node, SymbolId};

type Site = (u32, ArgList);
use crate::constants::{
    GEMV_MIN_ROWS, PIECE_OPS, PLACE_MIN_RUN, REGION_MIN_OPS, SOLVE_BATCH_MAX_K, SOLVE_BATCH_MAX_N,
};

fn split_state(o: &[u32], stateful: bool) -> (&[u32], u32) {
    match (stateful, o.split_last()) {
        (true, Some((&state, args))) => (args, state),
        _ => (o, super::NO_STATE),
    }
}

fn through_folded_consumers(alias: &HashMap<u32, Ref>, mut r: Ref) -> Ref {
    while let Ref::Value(i, _) = r {
        match alias.get(&i) {
            Some(&to) => r = to,
            None => break,
        }
    }
    r
}

impl Tape {
    /// Compile a tape computing `roots`, where `inputs[k]` (passed to
    /// [`eval`](super::Program::eval)) is the value of symbol `input_syms[k]`. Symbols not
    /// listed evaluate to `NaN`.
    ///
    /// A hierarchy is compiled as it stands: a composite function (one
    /// whose body calls others) is lowered once into a template, its
    /// instructions over its operands, and a call of it appends the
    /// template with its operands in their place; no graph is rewritten
    /// per instance. The calls of a leaf function (a device model's body)
    /// from every instance then run as one batch, as in a flat graph.
    /// Differentiate and specialize on the hierarchy first: that work stays
    /// on its functions.
    pub fn compile<K: Field>(ctx: &Graph<K>, roots: &[ExprId], input_syms: &[SymbolId]) -> Tape {
        Self::compile_inner(ctx, roots, input_syms, None)
    }

    /// [`compile`](Self::compile) with a prolog split: `pure_inputs[k]` marks
    /// input `k` as solve-constant (a parameter), and every op depending only
    /// on such inputs is scheduled into a *prolog prefix* of the instruction
    /// stream. A Newton loop evaluates the prolog once per parameter binding
    /// ([`eval_prolog`](super::Program::eval_prolog)) and then only the remainder per
    /// iteration ([`eval_main`](super::Program::eval_main)); values crossing the boundary
    /// are pinned so iteration-loop slot reuse cannot clobber them. Plain
    /// [`eval`](super::Program::eval) still runs the whole stream, so the split is
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
        let _s = rsprobe::scope!("compile");
        let program = lower(ctx, roots, input_syms, pure_inputs);
        finish(program, pure_inputs, input_syms.len())
    }
}

fn lower<K: Field>(
    ctx: &Graph<K>,
    roots: &[ExprId],
    input_syms: &[SymbolId],
    pure_inputs: Option<&[bool]>,
) -> Program {
    let forest = {
        let _s = rsprobe::scope!("analyze");
        Forest::analyze(ctx, roots, input_syms, pure_inputs)
    };
    rsprobe::add!("nodes", forest.base.len());
    let mut templates = Templates::default();
    let mut program = {
        let _s = rsprobe::scope!("lower");
        forest.lower(ctx, roots, &mut templates)
    };
    if templates.expanded_leaf_calls_apart {
        let _s = rsprobe::scope!("merge");
        program.merge_calls();
    }
    program
}

fn finish(mut program: Program, pure_inputs: Option<&[bool]>, n_inputs: usize) -> Tape {
    {
        let _s = rsprobe::scope!("fuse");
        program.fuse_accumulators(pure_inputs);
    }
    let order = {
        let _s = rsprobe::scope!("schedule");
        program.schedule()
    };
    let split = pure_inputs.is_some();
    let mut tape = {
        let _s = rsprobe::scope!("emit");
        program.emit(&order, split)
    };
    tape.n_inputs = n_inputs;
    rsprobe::add!("ops", tape.ops.len());
    rsprobe::add!("slots", tape.n_work);
    rsprobe::add!("stages", tape.stages.len());
    rsprobe::add!("regions", tape.regions.len());
    let calls = tape.ops.iter().filter_map(|op| match *op {
        Op::Call { n_groups, .. } => Some(n_groups as usize),
        _ => None,
    });
    rsprobe::add!("instances", calls.sum::<usize>());
    rsprobe::add!(
        "bodies",
        tape.bundles.iter().filter(|b| b.body().is_some()).count()
    );
    tape
}

/// The program of a set of roots over one signature, lowered once (the
/// reachable forest analyzed, its kernels grouped, its templates expanded
/// and their calls merged): the tapes of a system that compute some of the
/// same roots (its residuals; with its Jacobian; with its charges) are views
/// of one program rather than compilations of their own. A view takes the
/// instructions its roots reach, its calls computing only the outputs it
/// reads, and is scheduled and given slots alone; it computes what
/// [`Tape::compile`] of its roots computes, value for value.
pub struct Lowered {
    program: Program,
    roots: Vec<ExprId>,
    first_index_of_root: HashMap<ExprId, u32>,
    inputs: Vec<SymbolId>,
    pure_inputs: Option<Vec<bool>>,
}

impl Lowered {
    /// `roots` lowered over the inputs `input_syms`; with `pure_inputs`,
    /// every view split into a prolog as by [`Tape::compile_split`].
    pub fn new<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> Lowered {
        let _s = rsprobe::scope!("lowered");
        let program = lower(ctx, roots, input_syms, pure_inputs);
        let mut first_index_of_root: HashMap<ExprId, u32> = HashMap::default();
        for (k, &r) in roots.iter().enumerate() {
            first_index_of_root.entry(r).or_insert(k as u32);
        }
        Lowered {
            program,
            roots: roots.to_vec(),
            first_index_of_root,
            inputs: input_syms.to_vec(),
            pure_inputs: pure_inputs.map(<[bool]>::to_vec),
        }
    }

    /// The roots it was lowered for.
    pub fn roots(&self) -> &[ExprId] {
        &self.roots
    }

    /// Whether it was lowered over the inputs `input_syms`, split by
    /// `pure_inputs`.
    pub fn signature(&self, input_syms: &[SymbolId], pure_inputs: Option<&[bool]>) -> bool {
        self.inputs == input_syms && self.pure_inputs.as_deref() == pure_inputs
    }

    /// Whether a tape of `roots` over that signature is a view of it.
    pub fn serves(
        &self,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> bool {
        self.signature(input_syms, pure_inputs)
            && roots
                .iter()
                .all(|r| self.first_index_of_root.contains_key(r))
    }

    /// The tape computing `roots`, each one of the lowered roots.
    pub fn tape<K: Field>(&self, ctx: &Graph<K>, roots: &[ExprId]) -> Tape {
        let _s = rsprobe::scope!("view");
        let mut p = self.program.clone();
        p.roots = (roots.iter())
            .map(|r| {
                self.program.roots
                    [*self.first_index_of_root.get(r).expect("a lowered root") as usize]
            })
            .collect();
        let pure = self.pure_inputs.as_deref();
        {
            let _s = rsprobe::scope!("narrow");
            p.retain_reachable();
            p.narrow_calls(ctx, pure);
        }
        finish(p, pure, self.inputs.len())
    }
}

// https://github.com/milanofthe/rsdag/issues/241
#[derive(Default)]
struct Scratch {
    generation: u32,
    seen: Vec<u32>,
    pos: Vec<u32>,
    input_seen: Vec<u32>,
    input: Vec<u32>,
}

std::thread_local! {
    static SCRATCH: std::cell::Cell<Option<Scratch>> = const { std::cell::Cell::new(None) };
}

impl Scratch {
    fn take(n_nodes: usize, n_symbols: usize) -> Scratch {
        let mut t = SCRATCH.with(|c| c.take()).unwrap_or_default();
        t.generation = t.generation.wrapping_add(1);
        let generation_wrapped = t.generation == 0;
        if generation_wrapped {
            t.seen.fill(0);
            t.input_seen.fill(0);
            t.generation = 1;
        }
        if t.seen.len() < n_nodes {
            t.seen.resize(n_nodes, 0);
            t.pos.resize(n_nodes, u32::MAX);
        }
        if t.input_seen.len() < n_symbols {
            t.input_seen.resize(n_symbols, 0);
            t.input.resize(n_symbols, u32::MAX);
        }
        t
    }

    #[inline]
    fn mark(&mut self, e: ExprId) -> bool {
        let k = e.0 as usize;
        let fresh = self.seen[k] != self.generation;
        self.seen[k] = self.generation;
        fresh
    }

    fn set_input(&mut self, s: SymbolId, k: u32) {
        if let Some(slot) = self.input.get_mut(s.0 as usize) {
            *slot = k;
            self.input_seen[s.0 as usize] = self.generation;
        }
    }

    #[inline]
    fn input(&self, s: SymbolId) -> Option<u32> {
        let k = s.0 as usize;
        (self.input_seen.get(k) == Some(&self.generation)).then(|| self.input[k])
    }
}

impl Drop for Forest {
    fn drop(&mut self) {
        let t = std::mem::take(&mut self.tables);
        SCRATCH.with(|c| c.set(Some(t)));
    }
}

struct Forest {
    base: Vec<ExprId>,
    tables: Scratch,
    pure: Vec<bool>,
    fused_into: Vec<Option<usize>>,
    split: bool,
    extended_call_operands: HashMap<(u32, Site), Vec<ExprId>>,
}

#[derive(Clone)]
pub(super) struct Inst {
    pub(super) kind: Kind,
    // https://github.com/milanofthe/rsdag/issues/241
    pub(super) ins: (u32, u32),
    pub(super) n_out: u32,
    pub(super) pure: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Ref {
    Value(u32, u32),
    Input(u32),
}

#[derive(Clone, Debug)]
pub(super) enum Kind {
    Const(f64),
    Add,
    Mul,
    MulAdd,
    Sub,
    Neg,
    Powi(i32),
    Unary(crate::node::UnaryOp),
    Binary(crate::node::BinOp),
    Cmp(crate::node::CmpOp),
    Select,
    Reduce(crate::node::ReduceOp),
    Dot(u32),
    Call {
        bundle: u32,
        func: u32,
        n_groups: u32,
        n_args: u32,
        reads: Option<Arc<[u32]>>,
        stateful: bool,
    },
    CallProlog {
        bundle: u32,
        n_groups: u32,
        n_pure: u32,
    },
    Gemv {
        m: u32,
        n: u32,
        acc: Option<Vec<u32>>,
    },
    Gemm {
        m: u32,
        k: u32,
        n: u32,
        acc: Option<Vec<u32>>,
    },
    Solve {
        n: u32,
        k: u32,
        count: u32,
    },
}

pub(super) const NO_FUNC: u32 = u32::MAX;

#[derive(Clone)]
pub(crate) struct Program {
    pub(super) insts: Vec<Inst>,
    pub(super) pool: Vec<Ref>,
    pub(super) bundles: Vec<Arc<dyn ExternBundle>>,
    pub(super) roots: Vec<Ref>,
    pub(super) placeholder: Option<u32>,
    pub(super) state: Option<(Vec<u32>, u32)>,
}

impl Program {
    pub(super) fn ins(&self, i: usize) -> &[Ref] {
        let (s, l) = self.insts[i].ins;
        &self.pool[s as usize..(s + l) as usize]
    }

    fn fold_hints(&self, uses: &HashMap<(u32, u32), (u32, u32)>) -> Vec<(u32, u32)> {
        let mut hints: Vec<(u32, u32)> = Vec::new();
        for (&(k, c), &(count, j)) in uses {
            if count != 1 || j == u32::MAX {
                continue;
            }
            let this = Ref::Value(k, c);
            let ins = self.ins(j as usize);
            let other = match self.insts[j as usize].kind {
                Kind::Sub if ins[1] == this => ins[0],
                Kind::Add if ins[0] == this => ins[1],
                Kind::Add => ins[0],
                _ => continue,
            };
            if let Ref::Value(x, _) = other {
                let reads_kernel = |r: &Ref| matches!(*r, Ref::Value(y, _) if y == k);
                let is_or_reads_kernel_output =
                    x == k || self.ins(x as usize).iter().any(reads_kernel);
                if !is_or_reads_kernel_output {
                    hints.push((x, k));
                }
            }
        }
        hints.sort_unstable();
        hints.dedup();
        hints
    }

    fn fuse_accumulators(&mut self, pure_inputs: Option<&[bool]>) {
        let m = self.insts.len();
        let kernels: Vec<usize> = (0..m)
            .filter(|&i| {
                matches!(
                    self.insts[i].kind,
                    Kind::Gemv { acc: None, .. } | Kind::Gemm { acc: None, .. }
                )
            })
            .collect();
        if kernels.is_empty() {
            return;
        }
        let mut uses: HashMap<(u32, u32), (u32, u32)> = HashMap::default();
        for j in 0..m {
            for r in self.ins(j) {
                if let Ref::Value(i, c) = *r {
                    if matches!(
                        self.insts[i as usize].kind,
                        Kind::Gemv { .. } | Kind::Gemm { .. }
                    ) {
                        let e = uses.entry((i, c)).or_insert((0, j as u32));
                        e.0 += 1;
                        e.1 = j as u32;
                    }
                }
            }
        }
        for r in &self.roots {
            if let Ref::Value(i, c) = *r {
                uses.entry((i, c)).or_insert((0, u32::MAX)).0 += 2;
            }
        }
        let input_pure =
            |k: u32| pure_inputs.is_some_and(|p| p.get(k as usize).copied().unwrap_or(true));
        let mut alias: HashMap<u32, Ref> = HashMap::default();
        let mut dead = vec![false; m];
        let mut topo: Option<Topo> = None;
        let mut unread_placeholder: Option<Ref> = None;
        for &k in &kernels {
            let n_out = self.insts[k].n_out;
            let kernel_pure = self.insts[k].pure;
            let mut codes: Vec<u32> = vec![Fold::PLAIN.0; n_out as usize];
            let placeholder = *unread_placeholder.get_or_insert_with(|| {
                self.insts.push(Inst {
                    kind: Kind::Const(f64::NAN),
                    ins: (self.pool.len() as u32, 0),
                    n_out: 1,
                    pure: true,
                });
                dead.push(false);
                self.placeholder = Some(self.insts.len() as u32 - 1);
                Ref::Value(self.insts.len() as u32 - 1, 0)
            });
            let mut accs: Vec<Ref> = vec![placeholder; n_out as usize];
            let mut fused_output_consumer: Vec<(u32, u32)> = Vec::new();
            let mut taken_accumulators: Vec<u32> = Vec::new();
            let mut read_as_self_accumulator = vec![false; n_out as usize];
            for c in 0..n_out {
                let Some(&(count, j)) = uses.get(&(k as u32, c)) else {
                    continue;
                };
                let single_live_consumer = count == 1 && !dead[j as usize];
                if !single_live_consumer || read_as_self_accumulator[c as usize] {
                    continue;
                }
                let this = Ref::Value(k as u32, c);
                let ins = self.ins(j as usize);
                let (code, acc) = match self.insts[j as usize].kind {
                    Kind::Sub if ins[1] == this => (Fold::SUB.0, ins[0]),
                    Kind::Neg => (Fold::NEG.0, placeholder),
                    Kind::Add => (Fold::ADD.0, if ins[0] == this { ins[1] } else { ins[0] }),
                    _ => continue,
                };
                let acc = through_folded_consumers(&alias, acc);
                let (code, acc) = match acc {
                    Ref::Value(i, c2) if i as usize == k && code != Fold::NEG.0 => {
                        let other_output_stays_plain =
                            c2 != c && codes[c2 as usize] == Fold::PLAIN.0;
                        if !other_output_stays_plain {
                            continue;
                        }
                        read_as_self_accumulator[c2 as usize] = true;
                        (code | Fold::SELF | (c2 << Fold::SELF_SHIFT), placeholder)
                    }
                    Ref::Value(i, _) if code != Fold::NEG.0 => {
                        if kernel_pure && !self.insts[i as usize].pure {
                            continue;
                        }
                        let lowering_order_puts_acc_first = (i as usize) < k;
                        let acc_independent_of_kernel = match topo.as_mut() {
                            Some(t) => t.read(self, k as u32, i),
                            None if lowering_order_puts_acc_first => true,
                            None => {
                                let hints = self.fold_hints(&uses);
                                topo.insert(Topo::new(self, &hints, k as u32, &taken_accumulators))
                                    .read(self, k as u32, i)
                            }
                        };
                        if !acc_independent_of_kernel {
                            continue;
                        }
                        taken_accumulators.push(i);
                        (code, acc)
                    }
                    Ref::Input(i) if code != Fold::NEG.0 => {
                        if kernel_pure && !input_pure(i) {
                            continue;
                        }
                        (code, acc)
                    }
                    _ => (code, placeholder),
                };
                codes[c as usize] = code;
                accs[c as usize] = acc;
                fused_output_consumer.push((c, j));
            }
            if fused_output_consumer.is_empty() {
                continue;
            }
            let start = self.pool.len() as u32;
            let old: Vec<Ref> = self.ins(k).to_vec();
            let len = old.len() as u32 + n_out;
            self.pool.extend(old);
            self.pool.extend(accs);
            let inst = &mut self.insts[k];
            inst.ins = (start, len);
            match &mut inst.kind {
                Kind::Gemv { acc, .. } | Kind::Gemm { acc, .. } => *acc = Some(codes),
                _ => unreachable!(),
            }
            for (c, j) in fused_output_consumer {
                dead[j as usize] = true;
                alias.insert(j, Ref::Value(k as u32, c));
            }
        }
        if alias.is_empty() {
            return;
        }
        self.compact(&dead, |r| through_folded_consumers(&alias, r));
    }

    pub(super) fn compact(&mut self, dead: &[bool], resolve: impl Fn(Ref) -> Ref) {
        let mut renumber = vec![u32::MAX; self.insts.len()];
        let mut next = 0u32;
        for (i, &d) in dead.iter().enumerate() {
            if !d {
                renumber[i] = next;
                next += 1;
            }
        }
        let map = |r: Ref| -> Ref {
            match resolve(r) {
                Ref::Value(i, c) => Ref::Value(renumber[i as usize], c),
                r => r,
            }
        };
        let old_pool = std::mem::take(&mut self.pool);
        let old = std::mem::take(&mut self.insts);
        self.pool.reserve(old_pool.len());
        for (mut inst, _) in old.into_iter().zip(dead).filter(|(_, &d)| !d) {
            let (s, l) = inst.ins;
            let start = self.pool.len() as u32;
            self.pool.extend(
                old_pool[s as usize..(s + l) as usize]
                    .iter()
                    .map(|&r| map(r)),
            );
            inst.ins = (start, l);
            self.insts.push(inst);
        }
        for r in self.roots.iter_mut() {
            *r = map(*r);
        }
        self.placeholder = self
            .placeholder
            .filter(|&i| !dead[i as usize])
            .map(|i| renumber[i as usize]);
        if let Some((at, _)) = &mut self.state {
            *at = at
                .iter()
                .zip(dead)
                .filter(|(_, &d)| !d)
                .map(|(&a, _)| a)
                .collect();
        }
    }

    // https://github.com/milanofthe/rsdag/issues/241
    pub(super) fn merge_calls(&mut self) {
        #[derive(PartialEq, Eq, Hash)]
        struct Key {
            bundle: u32,
            n_args: u32,
            reads: Option<Vec<u32>>,
            stateful: bool,
            pure: bool,
            depth: u32,
        }
        let m = self.insts.len();
        let mut depth = vec![0u32; m];
        for i in 0..m {
            let d = self
                .ins(i)
                .iter()
                .filter_map(|r| match *r {
                    Ref::Value(j, _) => {
                        debug_assert!((j as usize) < i, "operands come before their reader");
                        Some(depth[j as usize])
                    }
                    Ref::Input(_) => None,
                })
                .max()
                .unwrap_or(0);
            depth[i] = d + u32::from(matches!(self.insts[i].kind, Kind::Call { .. }));
        }
        let mut index: HashMap<Key, usize> = HashMap::default();
        let mut groups: Vec<Vec<u32>> = Vec::new();
        for i in 0..m {
            if let Kind::Call {
                bundle,
                n_args,
                ref reads,
                stateful,
                ..
            } = self.insts[i].kind
            {
                let key = Key {
                    bundle,
                    n_args,
                    reads: reads.as_ref().map(|r| r.to_vec()),
                    stateful,
                    pure: self.insts[i].pure,
                    depth: depth[i],
                };
                let g = *index.entry(key).or_insert_with(|| {
                    groups.push(Vec::new());
                    groups.len() - 1
                });
                groups[g].push(i as u32);
            }
        }
        let mut dead = vec![false; m];
        let mut moved: HashMap<u32, (u32, Vec<u32>, u32)> = HashMap::default();
        for members in groups.iter().filter(|g| g.len() >= 2) {
            let first = &self.insts[members[0] as usize];
            let Kind::Call {
                bundle,
                func,
                n_args,
                ref reads,
                stateful,
                ..
            } = first.kind
            else {
                unreachable!()
            };
            let (reads, pure) = (reads.clone(), first.pure);
            let n_groups_of = |c: u32| match self.insts[c as usize].kind {
                Kind::Call { n_groups, .. } | Kind::CallProlog { n_groups, .. } => {
                    n_groups as usize
                }
                _ => unreachable!(),
            };
            let prolog_of = |c: u32| match *self.ins(c as usize).last().expect("a state") {
                Ref::Value(p, 0) => p,
                _ => unreachable!("a stateful call's last operand is its state"),
            };
            let inst: Vec<(u32, usize)> = (members.iter())
                .flat_map(|&c| (0..n_groups_of(c)).map(move |g| (c, g)))
                .collect();
            let key = |k: usize| -> (&[Ref], &[Ref]) {
                let (c, g) = inst[k];
                let ng = n_groups_of(c);
                let own = self.ins(c as usize);
                let ni = (own.len() - usize::from(stateful)) / ng;
                let pro: &[Ref] = match stateful {
                    true => {
                        let p = self.ins(prolog_of(c) as usize);
                        let np = p.len() / ng;
                        &p[g * np..(g + 1) * np]
                    }
                    false => &[],
                };
                (&own[g * ni..(g + 1) * ni], pro)
            };
            let hash = |k: usize| {
                let mut h = FxHasher::default();
                key(k).hash(&mut h);
                Some(h.finish())
            };
            let mut row = vec![0usize; inst.len()];
            super::calls::find_rows(inst.len(), hash, |a, b| key(a) == key(b), |k, r| row[k] = r);
            let (mut ins, mut pure_ins): (Vec<Ref>, Vec<Ref>) = (Vec::new(), Vec::new());
            let mut at = vec![0u32; inst.len()];
            let mut n_groups = 0u32;
            for k in 0..inst.len() {
                if row[k] != k {
                    at[k] = at[row[k]];
                    continue;
                }
                let (own, pro) = key(k);
                ins.extend_from_slice(own);
                pure_ins.extend_from_slice(pro);
                at[k] = n_groups;
                n_groups += 1;
            }
            let state_g = match stateful {
                true => self.bundles[bundle as usize].state_len() as u32,
                false => 0,
            };
            let mut out_g = 0u32;
            let mut from = 0;
            let mut maps = Vec::with_capacity(members.len());
            for &c in members {
                let ng = n_groups_of(c);
                out_g = self.insts[c as usize].n_out / ng as u32;
                if stateful {
                    dead[prolog_of(c) as usize] = true;
                }
                dead[c as usize] = true;
                maps.push((c, at[from..from + ng].to_vec()));
                from += ng;
            }
            if stateful {
                let Kind::CallProlog { n_pure, .. } =
                    self.insts[prolog_of(members[0]) as usize].kind
                else {
                    unreachable!()
                };
                let kind = Kind::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                };
                let p = self.push(
                    kind,
                    pure_ins,
                    super::calls::state_block(n_groups, state_g),
                    true,
                );
                ins.push(Ref::Value(p, 0));
            }
            let kind = Kind::Call {
                bundle,
                func,
                n_groups,
                n_args,
                reads,
                stateful,
            };
            let at = self.push(kind, ins, n_groups * out_g, pure);
            for (c, map) in maps {
                moved.insert(c, (at, map, out_g));
            }
        }
        dead.resize(self.insts.len(), false);
        self.compact(&dead, |r| match r {
            Ref::Value(i, o) => match moved.get(&i) {
                Some(&(at, ref map, w)) => Ref::Value(at, map[(o / w) as usize] * w + o % w),
                None => r,
            },
            r => r,
        });
    }

    fn push(&mut self, kind: Kind, ins: Vec<Ref>, n_out: u32, pure: bool) -> u32 {
        let start = self.pool.len() as u32;
        self.pool.extend(ins);
        self.insts.push(Inst {
            kind,
            ins: (start, self.pool.len() as u32 - start),
            n_out,
            pure,
        });
        self.insts.len() as u32 - 1
    }

    pub(super) fn retain_reachable(&mut self) {
        let m = self.insts.len();
        let mut dead = vec![true; m];
        let mut stack: Vec<u32> = self
            .roots
            .iter()
            .filter_map(|r| match *r {
                Ref::Value(i, _) => Some(i),
                Ref::Input(_) => None,
            })
            .collect();
        if let Some((kept_state_start, _)) = &self.state {
            let placed_in_kept_state = |i: &u32| kept_state_start[*i as usize] != u32::MAX;
            stack.extend((0..m as u32).filter(placed_in_kept_state));
        }
        while let Some(i) = stack.pop() {
            if !std::mem::replace(&mut dead[i as usize], false) {
                continue;
            }
            stack.extend(self.ins(i as usize).iter().filter_map(|r| match *r {
                Ref::Value(j, _) if dead[j as usize] => Some(j),
                _ => None,
            }));
        }
        self.compact(&dead, |r| r);
    }

    pub(super) fn narrow_calls<K: Field>(&mut self, ctx: &Graph<K>, pure_inputs: Option<&[bool]>) {
        let mut group_outputs_read: HashMap<u32, Vec<bool>> = HashMap::default();
        for (i, inst) in self.insts.iter().enumerate() {
            if let Kind::Call { func, n_groups, .. } = inst.kind {
                if func != NO_FUNC {
                    group_outputs_read
                        .insert(i as u32, vec![false; (inst.n_out / n_groups) as usize]);
                }
            }
        }
        if group_outputs_read.is_empty() {
            return;
        }
        let m = self.insts.len();
        for i in 0..m {
            for r in self.ins(i) {
                if let Ref::Value(j, o) = *r {
                    if let Some(u) = group_outputs_read.get_mut(&j) {
                        let w = u.len();
                        u[o as usize % w] = true;
                    }
                }
            }
        }
        for r in &self.roots {
            if let Ref::Value(j, o) = *r {
                if let Some(u) = group_outputs_read.get_mut(&j) {
                    let w = u.len();
                    u[o as usize % w] = true;
                }
            }
        }
        let is_pure = |p: &Program, r: Ref| match r {
            Ref::Value(j, _) => p.insts[j as usize].pure,
            Ref::Input(k) => pure_inputs.is_some_and(|m| m.get(k as usize) == Some(&true)),
        };
        let mut calls: Vec<(u32, Vec<bool>)> = group_outputs_read.into_iter().collect();
        calls.sort_unstable_by_key(|c| c.0);
        let mut dead = vec![false; m];
        let mut moved: HashMap<u32, (u32, u32, Vec<u32>)> = HashMap::default();
        for (i, used) in calls {
            if used.iter().all(|&u| u) {
                continue;
            }
            let Kind::Call {
                bundle,
                func,
                n_groups,
                n_args,
                ref reads,
                stateful,
            } = self.insts[i as usize].kind
            else {
                unreachable!()
            };
            let reads = reads.clone();
            let function = ctx.func(FuncId(func));
            let old = self.bundles[bundle as usize].clone();
            let Some(outs) = function.slot_outputs(&old) else {
                continue;
            };
            let needed: Vec<u32> = (used.iter().zip(&outs))
                .filter(|(&u, _)| u)
                .map(|(_, &k)| k)
                .collect();
            let body = function.body_exact(ctx, &needed);
            if Arc::ptr_eq(&body.bundle, &old) {
                continue;
            }
            let own = self.ins(i as usize).to_vec();
            let own = &own[..own.len() - usize::from(stateful)];
            let mask = old.pure_args().to_vec();
            let prolog = stateful.then(|| match own_last(self, i) {
                Ref::Value(p, 0) => p,
                _ => unreachable!("a stateful call's last operand is its state"),
            });
            let n_in = own.len() / n_groups as usize;
            let n_pure = mask.iter().filter(|&&p| p).count();
            let mut from_list_or_prolog: Vec<Result<usize, usize>> =
                Vec::with_capacity(n_args as usize);
            let mut rank = 0;
            let mut at_read: Vec<Option<usize>> = vec![None; n_args as usize];
            if let Some(r) = &reads {
                for (k, &q) in r.iter().enumerate() {
                    at_read[q as usize] = Some(k);
                }
            }
            for p in 0..n_args as usize {
                from_list_or_prolog.push(match (&reads, at_read[p]) {
                    (None, _) => Ok(p),
                    (Some(_), Some(k)) => Ok(k),
                    (Some(_), None) => Err(rank),
                });
                rank += usize::from(mask.get(p) == Some(&true));
            }
            let pro_ins: Option<Vec<Ref>> = prolog.map(|p| self.ins(p as usize).to_vec());
            let mut whole_args: Vec<Ref> = Vec::with_capacity(n_groups as usize * n_args as usize);
            for g in 0..n_groups as usize {
                let list = &own[g * n_in..(g + 1) * n_in];
                for f in &from_list_or_prolog {
                    whole_args.push(match *f {
                        Ok(k) => list[k],
                        Err(k) => {
                            let pro = pro_ins.as_ref().expect("an argument not read is pure");
                            pro[g * n_pure + k]
                        }
                    });
                }
            }
            if let Some(p) = prolog {
                dead[p as usize] = true;
            }
            dead[i as usize] = true;
            let pure = self.insts[i as usize].pure;
            let b = body.bundle.clone();
            let at_bundle = match self.bundles.iter().position(|x| Arc::ptr_eq(x, &b)) {
                Some(k) => k as u32,
                None => {
                    self.bundles.push(b.clone());
                    self.bundles.len() as u32 - 1
                }
            };
            let lists: Vec<&[Ref]> = whole_args.chunks(n_args as usize).collect();
            let mask = b.pure_args();
            let stateful = !pure
                && pure_inputs.is_some()
                && b.state_len() > 0
                && mask.iter().any(|&p| p)
                && lists.iter().all(|a| {
                    a.len() == mask.len()
                        && a.iter().zip(mask).all(|(&r, &p)| !p || is_pure(self, r))
                });
            let reads: Option<Arc<[u32]>> = stateful
                .then(|| b.main_reads())
                .flatten()
                .filter(|r| r.len() < n_args as usize)
                .map(Into::into);
            let mut ins: Vec<Ref> = match &reads {
                None => whole_args.clone(),
                Some(r) => (lists.iter())
                    .flat_map(|a| r.iter().map(move |&p| a[p as usize]))
                    .collect(),
            };
            if stateful {
                let pure_args: Vec<Ref> = (lists.iter())
                    .flat_map(|a| a.iter().zip(mask).filter(|&(_, &p)| p).map(|(&r, _)| r))
                    .collect();
                let kind = Kind::CallProlog {
                    bundle: at_bundle,
                    n_groups,
                    n_pure: mask.iter().filter(|&&p| p).count() as u32,
                };
                let width = super::calls::state_block(n_groups, b.state_len() as u32);
                let p = self.push(kind, pure_args, width, true);
                ins.push(Ref::Value(p, 0));
            }
            let width = b.n_outputs() as u32;
            let kind = Kind::Call {
                bundle: at_bundle,
                func,
                n_groups,
                n_args,
                reads,
                stateful,
            };
            let at = self.push(kind, ins, n_groups * width, pure);
            let slots: Vec<u32> = (outs.iter())
                .map(|&k| {
                    body.slot_of
                        .get(k as usize)
                        .copied()
                        .flatten()
                        .unwrap_or(u32::MAX)
                })
                .collect();
            moved.insert(i, (at, outs.len() as u32, slots));
        }
        if moved.is_empty() {
            return;
        }
        dead.resize(self.insts.len(), false);
        let width = |at: u32| -> u32 {
            match self.insts[at as usize].kind {
                Kind::Call { n_groups, .. } => self.insts[at as usize].n_out / n_groups,
                _ => unreachable!(),
            }
        };
        let widths: HashMap<u32, u32> = moved.values().map(|&(at, _, _)| (at, width(at))).collect();
        self.compact(&dead, |r| match r {
            Ref::Value(i, o) => match moved.get(&i) {
                Some((at, w_old, slots)) => {
                    let (g, s) = (o / w_old, o % w_old);
                    let slot = slots[s as usize];
                    debug_assert!(slot != u32::MAX, "a read output the body carries");
                    Ref::Value(*at, g * widths[at] + slot)
                }
                None => r,
            },
            r => r,
        });
    }
}

fn own_last(p: &Program, i: u32) -> Ref {
    *p.ins(i as usize).last().expect("an operand")
}

impl Tape {
    // https://github.com/milanofthe/rsdag/issues/241
    pub(super) fn lift(&self) -> Program {
        let m = self.ops.len();
        let mut op_holding_slot = vec![u32::MAX; self.n_work];
        let value = |k: u32, op_holding_slot: &[u32]| -> Ref {
            match input_index(k) {
                Some(j) => Ref::Input(j),
                None => {
                    let j = op_holding_slot[k as usize];
                    Ref::Value(j, k - self.dst[j as usize])
                }
            }
        };
        let placeholder_after_stream = Ref::Value(m as u32, 0);
        let mut uses_placeholder = false;
        let mut insts: Vec<Inst> = Vec::with_capacity(m + 1);
        let mut pool: Vec<Ref> = Vec::with_capacity(self.arg_pool.len() + 2 * m);
        for i in 0..m {
            let start = pool.len();
            self.for_each_operand(i, |k| pool.push(value(k, &op_holding_slot)));
            let width = self.width(i);
            let codes = |acc: Option<Accum>| acc.map(|a| self.pool(a.codes, width).to_vec());
            let kind = match self.ops[i] {
                Op::Const(v) => Kind::Const(v),
                Op::Add(..) => Kind::Add,
                Op::Mul(..) => Kind::Mul,
                Op::MulAdd(..) => Kind::MulAdd,
                Op::Sub(..) => Kind::Sub,
                Op::Neg(_) => Kind::Neg,
                Op::Powi(_, n) => Kind::Powi(n),
                Op::Unary(op, _) => Kind::Unary(op),
                Op::Binary(op, ..) => Kind::Binary(op),
                Op::Cmp(op, ..) => Kind::Cmp(op),
                Op::Select(..) => Kind::Select,
                Op::Reduce(op, ..) => Kind::Reduce(op),
                Op::Dot(_, n) => Kind::Dot(n),
                Op::Call {
                    bundle,
                    n_groups,
                    n_args,
                    n_in,
                    reads,
                    state,
                    ..
                } => Kind::Call {
                    bundle,
                    func: NO_FUNC,
                    n_groups,
                    n_args,
                    reads: (reads != super::ALL_ARGS).then(|| self.pool(reads, n_in).into()),
                    stateful: state != super::NO_STATE,
                },
                Op::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                    ..
                } => Kind::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                },
                Op::Gemv {
                    m: rows, n, acc, ..
                } => Kind::Gemv {
                    m: rows,
                    n,
                    acc: codes(acc),
                },
                Op::Gemm {
                    m: rows, k, n, acc, ..
                } => Kind::Gemm {
                    m: rows,
                    k,
                    n,
                    acc: codes(acc),
                },
                Op::Solve { n, k, count, .. } => Kind::Solve { n, k, count },
            };
            if let Op::Gemv { acc: Some(a), .. } | Op::Gemm { acc: Some(a), .. } = self.ops[i] {
                if a.c.is_none() {
                    pool.extend(std::iter::repeat_n(
                        placeholder_after_stream,
                        width as usize,
                    ));
                }
                let entries = pool.len() - width as usize;
                for (j, &code) in self.pool(a.codes, width).iter().enumerate() {
                    if !Fold(code).reads_operand() {
                        pool[entries + j] = placeholder_after_stream;
                        uses_placeholder = true;
                    }
                }
            }
            insts.push(Inst {
                kind,
                ins: (start as u32, (pool.len() - start) as u32),
                n_out: width,
                pure: i < self.prolog_ops,
            });
            let d = self.dst[i];
            op_holding_slot[d as usize..(d + width) as usize].fill(i as u32);
        }
        if uses_placeholder {
            insts.push(Inst {
                kind: Kind::Const(f64::NAN),
                ins: (pool.len() as u32, 0),
                n_out: 1,
                pure: true,
            });
        }
        Program {
            insts,
            pool,
            bundles: self.bundles.clone(),
            roots: self
                .outputs
                .iter()
                .map(|&k| value(k, &op_holding_slot))
                .collect(),
            placeholder: uses_placeholder.then_some(m as u32),
            state: None,
        }
    }
}

impl Forest {
    #[inline]
    fn pos(&self, e: ExprId) -> usize {
        self.tables.pos[e.0 as usize] as usize
    }

    fn stateful(&self, b: &dyn ExternBundle, lists: &[&[ExprId]]) -> bool {
        let mask = b.pure_args();
        self.split
            && b.state_len() > 0
            && mask.iter().any(|&p| p)
            && lists.iter().all(|args| {
                args.len() == mask.len()
                    && args
                        .iter()
                        .zip(mask)
                        .all(|(&a, &p)| !p || self.pure[self.pos(a)])
            })
    }

    #[inline]
    fn input(&self, s: SymbolId) -> Option<u32> {
        self.tables.input(s)
    }

    fn call_ops<'a, K: Field>(&'a self, ctx: &'a Graph<K>, f: u32, site: Site) -> &'a [ExprId] {
        match self.extended_call_operands.get(&(f, site)) {
            Some(ops) => ops,
            None => ctx.args(site.1),
        }
    }

    fn analyze<K: Field>(
        ctx: &Graph<K>,
        roots: &[ExprId],
        input_syms: &[SymbolId],
        pure_inputs: Option<&[bool]>,
    ) -> Forest {
        let mut t = Scratch::take(ctx.len(), ctx.n_symbols());
        for (k, &s) in input_syms.iter().enumerate() {
            t.set_input(s, k as u32);
        }
        let mut base: Vec<ExprId> = Vec::new();
        let mut stack = roots.to_vec();
        let mut walked: HashSet<Site> = HashSet::default();
        let mut ext: HashMap<(u32, Site), Vec<ExprId>> = HashMap::default();
        let mut funcs: HashSet<u32> = HashSet::default();
        while let Some(id) = stack.pop() {
            if !t.mark(id) {
                continue;
            }
            base.push(id);
            if let Node::Call(o, l) = *ctx.node(id) {
                let f = ctx.output(o).0;
                let site = (ctx.context_of(o), l);
                let globals = ctx.globals(f);
                if !globals.is_empty() || site.0 != NO_CONTEXT {
                    ext.entry((f.0, site)).or_insert_with(|| {
                        let mut ops = ctx.full_args(o, l).into_owned();
                        ops.extend_from_slice(&globals);
                        ops
                    });
                    if funcs.insert(f.0) {
                        stack.extend_from_slice(&globals);
                    }
                }
                if !walked.insert(site) {
                    continue;
                }
            }
            stack.extend_from_slice(&ctx.operands(id));
        }
        let call_ops = |f: u32, site: Site| -> &[ExprId] {
            match ext.get(&(f, site)) {
                Some(ops) => ops,
                None => ctx.args(site.1),
            }
        };
        base.sort_unstable_by_key(|e| e.0);
        debug_assert!(
            base.iter()
                .all(|&e| ctx.operands(e).iter().all(|a| a.0 < e.0)),
            "ascending arena ids are a dependency order"
        );
        let m = base.len();
        for (i, id) in base.iter().enumerate() {
            t.pos[id.0 as usize] = i as u32;
        }
        let bpos = &t.pos;
        let bp = |e: ExprId| bpos[e.0 as usize] as usize;

        // https://github.com/milanofthe/rsdag/issues/241
        let mut pure = vec![false; m];
        if let Some(mask) = pure_inputs {
            let mut pure_list: HashMap<(u32, Site), bool> = HashMap::default();
            for (i, id) in base.iter().enumerate() {
                pure[i] = match *ctx.node(*id) {
                    Node::Const(_) => true,
                    Node::Symbol(s) => match t.input(s) {
                        None => true,
                        Some(k) => mask.get(k as usize).copied().unwrap_or(false),
                    },
                    Node::Call(o, l) => {
                        let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                        *pure_list
                            .entry((f, site))
                            .or_insert_with(|| call_ops(f, site).iter().all(|a| pure[bp(*a)]))
                    }
                    _ => ctx.operands(*id).iter().all(|a| pure[bp(*a)]),
                };
            }
        }
        let mut is_root = vec![false; m];
        for r in roots {
            is_root[bp(*r)] = true;
        }
        let mut uses = vec![0u32; m];
        walked.clear();
        for id in &base {
            if let Node::Call(o, l) = *ctx.node(*id) {
                if !walked.insert((ctx.context_of(o), l)) {
                    continue;
                }
            }
            for a in ctx.operands(*id).iter() {
                uses[bp(*a)] += 1;
            }
        }
        // https://github.com/milanofthe/rsdag/issues/241
        let mut fused_into: Vec<Option<usize>> = vec![None; m];
        for (i, id) in base.iter().enumerate() {
            if let Node::Add(a, b) = ctx.node(*id) {
                let single_use_mul_or_neg = |x: &ExprId, fused: &[Option<usize>]| {
                    let px = bp(*x);
                    !is_root[px]
                        && uses[px] == 1
                        && fused[px].is_none()
                        && matches!(ctx.node(*x), Node::Mul(..) | Node::Neg(..))
                };
                if single_use_mul_or_neg(a, &fused_into) {
                    fused_into[bp(*a)] = Some(i);
                } else if single_use_mul_or_neg(b, &fused_into) {
                    fused_into[bp(*b)] = Some(i);
                }
            }
        }
        Forest {
            base,
            tables: t,
            pure,
            fused_into,
            split: pure_inputs.is_some(),
            extended_call_operands: ext,
        }
    }

    fn lower<K: Field>(
        &self,
        ctx: &Graph<K>,
        roots: &[ExprId],
        templates: &mut Templates,
    ) -> Program {
        let calls = self.call_sets(ctx);
        let mut composite: HashSet<u32> = HashSet::default();
        for &id in &self.base {
            if let Node::Call(o, _) = *ctx.node(id) {
                let f = ctx.output(o).0;
                if ctx.func(f).composite_in(ctx) {
                    composite.insert(f.0);
                }
            }
        }
        let groups = self.groups(ctx, &calls, &composite);
        let order = self.unit_order(ctx, roots, &groups);
        let m = self.base.len();
        let mut lw = Lowering {
            p: Program {
                insts: Vec::with_capacity(m),
                pool: Vec::with_capacity(2 * m),
                bundles: Vec::new(),
                roots: Vec::new(),
                placeholder: None,
                state: None,
            },
            value: vec![None; m],
            template_consts_by_bits: HashMap::default(),
        };
        let mut bodies = Bodies::default();
        let mut lowered = vec![false; groups.groups.len()];
        for &i in &order {
            let materialized_in_consuming_add = self.fused_into[i].is_some();
            let lowered_with_its_call_or_kernel = lw.value[i].is_some();
            if materialized_in_consuming_add || lowered_with_its_call_or_kernel {
                continue;
            }
            match (groups.kernel_of[i], *ctx.node(self.base[i])) {
                (Some(g), _) => {
                    if !std::mem::replace(&mut lowered[g], true) {
                        let (key, members) = &groups.groups[g];
                        self.lower_group(ctx, &mut lw, &mut bodies, &calls, key, members);
                    }
                }
                (None, Node::Call(o, l)) => {
                    let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                    let members = &calls.members[&(f, site)];
                    let set = calls.set_of[&(f, site)];
                    if composite.contains(&f) {
                        let outs = &calls.sets[set as usize];
                        let ops = self.call_ops(ctx, f, site);
                        let args: Vec<Ref> = ops.iter().map(|&a| self.val(&lw, a)).collect();
                        let mask: Option<Vec<bool>> = self
                            .split
                            .then(|| ops.iter().map(|&a| self.pure[self.pos(a)]).collect());
                        let t = templates.get(ctx, f, outs, mask);
                        let values = lw.expand(&mut bodies, &t, &args);
                        for &(mi, at) in members {
                            lw.value[mi] = Some(values[at as usize]);
                        }
                        continue;
                    }
                    let members: Vec<usize> = members.iter().map(|&(mi, _)| mi).collect();
                    self.lower_calls(ctx, &mut lw, &mut bodies, &calls, f, set, &members);
                }
                (None, _) => self.lower_node(ctx, &mut lw, i),
            }
        }
        lw.p.roots = roots.iter().map(|&r| self.val(&lw, r)).collect();
        lw.p
    }

    fn val(&self, lw: &Lowering, e: ExprId) -> Ref {
        lw.value[self.pos(e)].expect("an operand precedes its consumer")
    }

    fn call_sets<K: Field>(&self, ctx: &Graph<K>) -> CallSets {
        let mut per_call: HashMap<(u32, Site), Vec<(usize, u32)>> = HashMap::default();
        for (i, &id) in self.base.iter().enumerate() {
            if let Node::Call(o, l) = *ctx.node(id) {
                let (f, out) = ctx.output(o);
                per_call
                    .entry((f.0, (ctx.context_of(o), l)))
                    .or_default()
                    .push((i, out));
            }
        }
        let mut ids: HashMap<(u32, Vec<u32>), u32> = HashMap::default();
        let mut calls = CallSets::default();
        for ((f, site), mut calls_of_site) in per_call {
            calls_of_site.sort_unstable_by_key(|&(i, out)| (out, i));
            let mut outs: Vec<u32> = Vec::new();
            let mut members = Vec::with_capacity(calls_of_site.len());
            for (i, out) in calls_of_site {
                if outs.last() != Some(&out) {
                    outs.push(out);
                }
                members.push((i, outs.len() as u32 - 1));
            }
            members.sort_unstable();
            let id = *ids.entry((f, outs.clone())).or_insert_with(|| {
                calls.sets.push(outs);
                calls.sets.len() as u32 - 1
            });
            calls.set_of.insert((f, site), id);
            calls.members.insert((f, site), members);
        }
        calls
    }

    fn groups<K: Field>(
        &self,
        ctx: &Graph<K>,
        calls: &CallSets,
        composite: &HashSet<u32>,
    ) -> Groups {
        let base = &self.base;
        let m = base.len();
        let mut lists: HashMap<u32, Vec<Site>> = HashMap::default();
        let mut seen: HashSet<(u32, Site)> = HashSet::default();
        for &id in base {
            if let Node::Call(o, l) = *ctx.node(id) {
                let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                if seen.insert((f, site)) {
                    lists.entry(f).or_default().push(site);
                }
            }
        }
        lists.retain(|&f, g| {
            let width = |&s: &Site| self.call_ops(ctx, f, s).len();
            !composite.contains(&f) && g.len() >= 2 && g.iter().all(|a| width(a) == width(&g[0]))
        });
        let mut rows_of: HashMap<Vec<ExprId>, Vec<usize>> = HashMap::default();
        for (i, &id) in base.iter().enumerate() {
            if let Node::Dot(l) = *ctx.node(id) {
                let (_, x) = ctx.dot_args(l);
                if !x.is_empty() {
                    rows_of.entry(x.to_vec()).or_default().push(i);
                }
            }
        }
        rows_of.retain(|_, rows| rows.len() >= GEMV_MIN_ROWS);
        let mut is_member = vec![false; m];
        for rows in rows_of.values() {
            for &r in rows {
                is_member[r] = true;
            }
        }
        for (i, &id) in base.iter().enumerate() {
            is_member[i] |= match *ctx.node(id) {
                Node::Call(o, _) => lists.contains_key(&ctx.output(o).0 .0),
                Node::Solve(..) => true,
                _ => false,
            };
        }
        let mut depth = vec![0u32; m];
        let mut list_depth: HashMap<(u32, Site), u32> = HashMap::default();
        for (i, &id) in base.iter().enumerate() {
            let deepest = |depth: &[u32], ops: &[ExprId]| {
                ops.iter().map(|&a| depth[self.pos(a)]).max().unwrap_or(0)
            };
            let over = match *ctx.node(id) {
                Node::Call(o, l) => {
                    let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                    *list_depth
                        .entry((f, site))
                        .or_insert_with(|| deepest(&depth, self.call_ops(ctx, f, site)))
                }
                _ => deepest(&depth, &ctx.operands(id)),
            };
            depth[i] = over + u32::from(is_member[i]);
        }
        let mut index: HashMap<GroupKey, usize> = HashMap::default();
        let mut groups: Vec<(GroupKey, Vec<usize>)> = Vec::new();
        for (i, &id) in base.iter().enumerate() {
            if !is_member[i] {
                continue;
            }
            let key = match *ctx.node(id) {
                Node::Call(o, l) => {
                    let (f, site) = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                    GroupKey::Call {
                        func: f,
                        depth: depth[i],
                        pure: self.pure[i],
                        output_set: calls.set_of[&(f, site)],
                    }
                }
                Node::Dot(l) => GroupKey::Gemv {
                    vector: ctx.dot_args(l).1.to_vec(),
                    depth: depth[i],
                    pure: self.pure[i],
                },
                Node::Solve(l, _) => GroupKey::Solve(l, self.pure[i]),
                _ => unreachable!("members are calls, rows or components"),
            };
            let g = *index.entry(key.clone()).or_insert_with(|| {
                groups.push((key, Vec::new()));
                groups.len() - 1
            });
            groups[g].1.push(i);
        }
        self.merge_gemm(ctx, &mut groups);
        self.merge_solves(ctx, &mut groups, &depth);
        self.merge_solve_batches(ctx, &mut groups, &depth);
        let mut kernel_of: Vec<Option<usize>> = vec![None; m];
        for (g, (key, members)) in groups.iter().enumerate() {
            let is_kernel = match key {
                GroupKey::Gemv { .. } => members.len() >= GEMV_MIN_ROWS,
                GroupKey::Call { .. } => {
                    let mut distinct: HashSet<Site> = HashSet::default();
                    for &i in members {
                        if let Node::Call(o, l) = *ctx.node(base[i]) {
                            distinct.insert((ctx.context_of(o), l));
                        }
                    }
                    distinct.len() >= 2
                }
                GroupKey::Gemm { .. }
                | GroupKey::Solve(..)
                | GroupKey::SolveMany { .. }
                | GroupKey::SolveBatch(..) => true,
            };
            if is_kernel {
                for &i in members {
                    kernel_of[i] = Some(g);
                }
            }
        }
        Groups { groups, kernel_of }
    }

    // https://github.com/milanofthe/rsdag/issues/241
    fn merge_gemm<K: Field>(&self, ctx: &Graph<K>, groups: &mut [(GroupKey, Vec<usize>)]) {
        let row_of = |mi: usize| -> &[ExprId] {
            let Node::Dot(l) = *ctx.node(self.base[mi]) else {
                unreachable!()
            };
            ctx.dot_args(l).0
        };
        let row_hash = |mi: usize| -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = rustc_hash::FxHasher::default();
            row_of(mi).hash(&mut h);
            h.finish()
        };
        let mut by_row_fingerprint: HashMap<(Vec<u64>, u32, bool), Vec<usize>> = HashMap::default();
        for (g, (key, members)) in groups.iter().enumerate() {
            if let GroupKey::Gemv { depth: d, pure, .. } = key {
                if members.len() < GEMV_MIN_ROWS {
                    continue;
                }
                let mut hashes: Vec<u64> = members.iter().map(|&mi| row_hash(mi)).collect();
                hashes.sort_unstable();
                if hashes.windows(2).any(|w| w[0] == w[1]) {
                    continue;
                }
                by_row_fingerprint
                    .entry((hashes, *d, *pure))
                    .or_default()
                    .push(g);
            }
        }
        let mut merged: Vec<Vec<usize>> = by_row_fingerprint
            .into_values()
            .filter(|gs| gs.len() >= 2)
            .collect();
        merged.sort();
        let rows = |g: usize, groups: &[(GroupKey, Vec<usize>)]| -> Vec<Vec<ExprId>> {
            let mut r: Vec<Vec<ExprId>> =
                groups[g].1.iter().map(|&mi| row_of(mi).to_vec()).collect();
            r.sort();
            r
        };
        for gs in merged {
            let first = rows(gs[0], groups);
            let fingerprints_verified = gs[1..].iter().all(|&g| rows(g, groups) == first);
            if !fingerprints_verified {
                continue;
            }
            let xs: Vec<Vec<ExprId>> = gs
                .iter()
                .map(|&g| match &groups[g].0 {
                    GroupKey::Gemv { vector, .. } => vector.clone(),
                    _ => unreachable!(),
                })
                .collect();
            let members: Vec<usize> = gs
                .iter()
                .flat_map(|&g| std::mem::take(&mut groups[g].1))
                .collect();
            groups[gs[0]] = (
                GroupKey::Gemm {
                    sorted_rows: first,
                    vectors: xs,
                },
                members,
            );
        }
    }

    // https://github.com/milanofthe/rsdag/issues/241
    fn merge_solves<K: Field>(
        &self,
        ctx: &Graph<K>,
        groups: &mut [(GroupKey, Vec<usize>)],
        depth: &[u32],
    ) {
        let mut by_matrix: HashMap<(Vec<ExprId>, u32, bool), Vec<usize>> = HashMap::default();
        for (g, (key, members)) in groups.iter().enumerate() {
            if let GroupKey::Solve(l, pure) = key {
                let (n, all) = (Graph::<K>::solve_n(l.len()), ctx.args(*l));
                by_matrix
                    .entry((all[..n * n].to_vec(), depth[members[0]], *pure))
                    .or_default()
                    .push(g);
            }
        }
        let mut merged: Vec<(Vec<usize>, Vec<ExprId>)> = by_matrix
            .into_iter()
            .filter(|(_, gs)| gs.len() >= 2)
            .map(|((a, _, _), gs)| (gs, a))
            .collect();
        merged.sort();
        for (gs, a) in merged {
            debug_assert!(gs.windows(2).all(|w| w[0] < w[1]));
            let lists: Vec<ArgList> = gs
                .iter()
                .map(|&g| match &groups[g].0 {
                    GroupKey::Solve(l, _) => *l,
                    _ => unreachable!(),
                })
                .collect();
            let members: Vec<usize> = gs
                .iter()
                .flat_map(|&g| std::mem::take(&mut groups[g].1))
                .collect();
            groups[gs[0]] = (
                GroupKey::SolveMany {
                    matrix: a,
                    rhs_lists: lists,
                },
                members,
            );
        }
    }

    fn merge_solve_batches<K: Field>(
        &self,
        ctx: &Graph<K>,
        groups: &mut [(GroupKey, Vec<usize>)],
        depth: &[u32],
    ) {
        let system = |key: &GroupKey| -> Option<System> {
            match key {
                GroupKey::Solve(l, _) => {
                    let n = Graph::<K>::solve_n(l.len());
                    Some((ctx.args(*l)[..n * n].to_vec(), vec![*l]))
                }
                GroupKey::SolveMany {
                    matrix,
                    rhs_lists: lists,
                } => Some((matrix.clone(), lists.clone())),
                _ => None,
            }
        };
        let mut by_shape: HashMap<(usize, usize, u32, bool), Vec<usize>> = HashMap::default();
        for (g, (key, members)) in groups.iter().enumerate() {
            let (Some((a, lists)), Some(&first)) = (system(key), members.first()) else {
                continue;
            };
            let n_matrix_entries = a.len();
            if n_matrix_entries > SOLVE_BATCH_MAX_N * SOLVE_BATCH_MAX_N
                || lists.len() > SOLVE_BATCH_MAX_K
            {
                continue;
            }
            by_shape
                .entry((a.len(), lists.len(), depth[first], self.pure[first]))
                .or_default()
                .push(g);
        }
        let mut batches: Vec<Vec<usize>> =
            by_shape.into_values().filter(|gs| gs.len() >= 2).collect();
        batches.sort();
        for gs in batches {
            let systems: Vec<System> = gs.iter().filter_map(|&g| system(&groups[g].0)).collect();
            let members: Vec<usize> = gs
                .iter()
                .flat_map(|&g| std::mem::take(&mut groups[g].1))
                .collect();
            groups[gs[0]] = (GroupKey::SolveBatch(systems), members);
        }
    }

    fn unit_order<K: Field>(
        &self,
        ctx: &Graph<K>,
        roots: &[ExprId],
        groups: &Groups,
    ) -> Vec<usize> {
        let m = self.base.len();
        let (groups, kernel_of) = (&groups.groups, &groups.kernel_of);
        let first_member: Vec<usize> = groups
            .iter()
            .map(|(_, ms)| ms.first().copied().unwrap_or(0))
            .collect();
        let mut site_first: HashMap<(u32, Site), usize> = HashMap::default();
        let mut site_of: Vec<usize> = (0..m).collect();
        for (i, &id) in self.base.iter().enumerate() {
            if let Node::Call(o, l) = *ctx.node(id) {
                let key = (ctx.output(o).0 .0, (ctx.context_of(o), l));
                site_of[i] = *site_first.entry(key).or_insert(i);
            }
        }
        let unit_of = |i: usize| -> usize {
            match kernel_of[i] {
                Some(g) => first_member[g],
                None => site_of[i],
            }
        };
        let deps_of_node = |i: usize, out: &mut Vec<usize>| {
            if let Node::Call(o, l) = *ctx.node(self.base[i]) {
                let ops = self.call_ops(ctx, ctx.output(o).0 .0, (ctx.context_of(o), l));
                out.extend(ops.iter().map(|&a| unit_of(self.pos(a))));
                return;
            }
            for &a in ctx.operands(self.base[i]).iter() {
                let pa = self.pos(a);
                if self.fused_into[pa] == Some(i) {
                    for &x in ctx.operands(a).iter() {
                        out.push(unit_of(self.pos(x)));
                    }
                } else {
                    out.push(unit_of(pa));
                }
            }
        };
        // https://github.com/milanofthe/rsdag/issues/241
        let mut last_unit_listing: Vec<usize> = vec![usize::MAX; m];
        let mut instances_listed: HashSet<Site> = HashSet::default();
        let mut raw: Vec<usize> = Vec::new();
        let mut deps: Vec<usize> = Vec::new();
        let mut deps_of_unit = |u: usize, deps: &mut Vec<usize>| {
            raw.clear();
            match kernel_of[u] {
                Some(g) => {
                    instances_listed.clear();
                    for &mi in &groups[g].1 {
                        if let Node::Call(o, l) = *ctx.node(self.base[mi]) {
                            if !instances_listed.insert((ctx.context_of(o), l)) {
                                continue;
                            }
                        }
                        deps_of_node(mi, &mut raw);
                    }
                }
                None => deps_of_node(u, &mut raw),
            }
            deps.clear();
            for &d in &raw {
                if last_unit_listing[d] != u {
                    last_unit_listing[d] = u;
                    deps.push(d);
                }
            }
        };
        let mut done = vec![false; m];
        let mut order: Vec<usize> = Vec::with_capacity(m);
        let mut stack: Vec<(usize, bool)> = roots
            .iter()
            .map(|r| (unit_of(self.pos(*r)), false))
            .collect();
        while let Some((u, expanded)) = stack.pop() {
            if done[u] {
                continue;
            }
            if expanded {
                done[u] = true;
                order.push(u);
                continue;
            }
            stack.push((u, true));
            deps_of_unit(u, &mut deps);
            for &d in &deps {
                if !done[d] {
                    stack.push((d, false));
                }
            }
        }
        order
    }

    fn lower_group<K: Field>(
        &self,
        ctx: &Graph<K>,
        lw: &mut Lowering,
        bodies: &mut Bodies,
        calls: &CallSets,
        key: &GroupKey,
        members: &[usize],
    ) {
        let base = &self.base;
        let pure = members.iter().all(|&mi| self.pure[mi]);
        debug_assert!(
            members.iter().all(|&mi| self.pure[mi] == pure),
            "a kernel of pure and impure members would pull pure work out of the prolog"
        );
        let val = |lw: &Lowering, e: ExprId| self.val(lw, e);
        match key {
            GroupKey::Call {
                func, output_set, ..
            } => self.lower_calls(ctx, lw, bodies, calls, *func, *output_set, members),
            GroupKey::Gemv { vector: x, .. } => {
                let mut ins: Vec<Ref> = Vec::new();
                for &mi in members {
                    let Node::Dot(l) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    ins.extend(ctx.dot_args(l).0.iter().map(|&a| val(lw, a)));
                }
                ins.extend(x.iter().map(|&a| val(lw, a)));
                let kind = Kind::Gemv {
                    m: members.len() as u32,
                    n: x.len() as u32,
                    acc: None,
                };
                let inst = lw.push(kind, ins, members.len() as u32, pure);
                for (r, &mi) in members.iter().enumerate() {
                    lw.value[mi] = Some(Ref::Value(inst, r as u32));
                }
            }
            GroupKey::Gemm {
                sorted_rows: rows,
                vectors: xs,
            } => {
                let (rm, k, cn) = (rows.len(), xs[0].len(), xs.len());
                let ins: Vec<Ref> = rows
                    .iter()
                    .chain(xs)
                    .flat_map(|r| r.iter().map(|&a| val(lw, a)))
                    .collect();
                fn index(set: &[Vec<ExprId>]) -> HashMap<&[ExprId], usize> {
                    set.iter()
                        .enumerate()
                        .map(|(i, r)| (r.as_slice(), i))
                        .collect()
                }
                let (row_index, col_index) = (index(rows), index(xs));
                let kind = Kind::Gemm {
                    m: rm as u32,
                    k: k as u32,
                    n: cn as u32,
                    acc: None,
                };
                let inst = lw.push(kind, ins, (rm * cn) as u32, pure);
                for &mi in members {
                    let Node::Dot(l) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    let (a, x) = ctx.dot_args(l);
                    let (r, c) = (row_index[a], col_index[x]);
                    lw.value[mi] = Some(Ref::Value(inst, (r * cn + c) as u32));
                }
            }
            GroupKey::SolveMany {
                matrix: a,
                rhs_lists: lists,
            } => {
                let n = (a.len() as f64).sqrt() as u32;
                let k = lists.len() as u32;
                let mut ins: Vec<Ref> = a.iter().map(|&e| val(lw, e)).collect();
                for &l in lists {
                    ins.extend(ctx.args(l)[(n * n) as usize..].iter().map(|&e| val(lw, e)));
                }
                let col_of: HashMap<ArgList, u32> = lists
                    .iter()
                    .enumerate()
                    .map(|(c, &l)| (l, c as u32))
                    .collect();
                let inst = lw.push(Kind::Solve { n, k, count: 1 }, ins, n * k, pure);
                for &mi in members {
                    let Node::Solve(l, c) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    lw.value[mi] = Some(Ref::Value(inst, col_of[&l] * n + c));
                }
            }
            GroupKey::SolveBatch(systems) => {
                let n = Graph::<K>::solve_n(ctx.args(systems[0].1[0]).len());
                let k = systems[0].1.len();
                let mut ins: Vec<Ref> = systems
                    .iter()
                    .flat_map(|(a, _)| a.iter().map(|&e| val(lw, e)))
                    .collect();
                let mut col_of: HashMap<ArgList, usize> = HashMap::default();
                for (l, &list) in systems.iter().flat_map(|(_, lists)| lists).enumerate() {
                    ins.extend(ctx.args(list)[n * n..].iter().map(|&e| val(lw, e)));
                    col_of.insert(list, l);
                }
                let count = systems.len();
                let kind = Kind::Solve {
                    n: n as u32,
                    k: k as u32,
                    count: count as u32,
                };
                let inst = lw.push(kind, ins, (count * n * k) as u32, pure);
                for &mi in members {
                    let Node::Solve(l, c) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    lw.value[mi] = Some(Ref::Value(inst, (col_of[&l] * n) as u32 + c));
                }
            }
            GroupKey::Solve(l, _) => {
                let all = ctx.args(*l);
                let n = Graph::<K>::solve_n(all.len()) as u32;
                let ins: Vec<Ref> = all.iter().map(|&a| val(lw, a)).collect();
                let inst = lw.push(Kind::Solve { n, k: 1, count: 1 }, ins, n, pure);
                for &mi in members {
                    let Node::Solve(_, c) = *ctx.node(base[mi]) else {
                        unreachable!()
                    };
                    lw.value[mi] = Some(Ref::Value(inst, c));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_calls<K: Field>(
        &self,
        ctx: &Graph<K>,
        lw: &mut Lowering,
        bodies: &mut Bodies,
        calls: &CallSets,
        f: u32,
        set: u32,
        members: &[usize],
    ) {
        let (bundle, body) = bodies.get(ctx, &mut lw.p.bundles, f, set, &calls.sets);
        let b = body.bundle.clone();
        let n_out = b.n_outputs() as u32;
        let mut lists: Vec<Site> = Vec::new();
        let mut group_of: HashMap<Site, u32> = HashMap::default();
        for &mi in members {
            let Node::Call(o, l) = *ctx.node(self.base[mi]) else {
                unreachable!()
            };
            let site = (ctx.context_of(o), l);
            group_of.entry(site).or_insert_with(|| {
                lists.push(site);
                lists.len() as u32 - 1
            });
        }
        let args: Vec<&[ExprId]> = lists.iter().map(|&l| self.call_ops(ctx, f, l)).collect();
        let (n_groups, n_args) = (lists.len() as u32, args[0].len() as u32);
        let pure = members.iter().all(|&mi| self.pure[mi]);
        let stateful = !pure && self.stateful(&*b, &args);
        // https://github.com/milanofthe/rsdag/issues/241
        let main_phase_reads: Option<Arc<[u32]>> = stateful
            .then(|| b.main_reads())
            .flatten()
            .filter(|r| r.len() < n_args as usize)
            .map(Into::into);
        let mut ins: Vec<Ref> = match &main_phase_reads {
            None => args
                .iter()
                .flat_map(|a| a.iter())
                .map(|&a| self.val(lw, a))
                .collect(),
            Some(r) => args
                .iter()
                .flat_map(|a| r.iter().map(move |&p| a[p as usize]))
                .map(|a| self.val(lw, a))
                .collect(),
        };
        if stateful {
            let mask = b.pure_args();
            let pure_args: Vec<Ref> = args
                .iter()
                .flat_map(|a| a.iter().zip(mask).filter(|&(_, &p)| p))
                .map(|(&a, _)| self.val(lw, a))
                .collect();
            let n_pure = mask.iter().filter(|&&p| p).count() as u32;
            let kind = Kind::CallProlog {
                bundle,
                n_groups,
                n_pure,
            };
            let width = super::calls::state_block(n_groups, b.state_len() as u32);
            let prolog = lw.push(kind, pure_args, width, true);
            ins.push(Ref::Value(prolog, 0));
        }
        let kind = Kind::Call {
            bundle,
            func: f,
            n_groups,
            n_args,
            reads: main_phase_reads,
            stateful,
        };
        let inst = lw.push(kind, ins, n_groups * n_out, pure);
        for &mi in members {
            let Node::Call(o, l) = *ctx.node(self.base[mi]) else {
                unreachable!()
            };
            let carried_slot = body.slot_of[ctx.output(o).1 as usize];
            lw.value[mi] = Some(match carried_slot {
                Some(slot) => Ref::Value(inst, group_of[&(ctx.context_of(o), l)] * n_out + slot),
                None => lw.constant(0.0),
            });
        }
    }

    fn lower_node<K: Field>(&self, ctx: &Graph<K>, lw: &mut Lowering, i: usize) {
        let val = |lw: &Lowering, e: ExprId| self.val(lw, e);
        let (kind, ins): (Kind, Vec<Ref>) = match *ctx.node(self.base[i]) {
            Node::Symbol(s) => {
                lw.value[i] = Some(match self.input(s) {
                    Some(k) => Ref::Input(k),
                    None => lw.constant(f64::NAN),
                });
                return;
            }
            Node::Const(c) => {
                lw.value[i] = Some(lw.constant(ctx.const_val(c).to_f64()));
                return;
            }
            Node::Add(a, b) => {
                let fused = |x: ExprId| self.fused_into[self.pos(x)] == Some(i);
                match (fused(a), fused(b)) {
                    (false, false) => (Kind::Add, vec![val(lw, a), val(lw, b)]),
                    (fa, _) => {
                        let (f, other) = if fa { (a, b) } else { (b, a) };
                        match *ctx.node(f) {
                            Node::Mul(x, y) => {
                                (Kind::MulAdd, vec![val(lw, x), val(lw, y), val(lw, other)])
                            }
                            Node::Neg(x) => (Kind::Sub, vec![val(lw, other), val(lw, x)]),
                            _ => unreachable!("only a Mul or a Neg fuses"),
                        }
                    }
                }
            }
            Node::Mul(a, b) => (Kind::Mul, vec![val(lw, a), val(lw, b)]),
            Node::Neg(a) => (Kind::Neg, vec![val(lw, a)]),
            Node::Pow(a, n) => (Kind::Powi(n as i32), vec![val(lw, a)]),
            Node::Unary(op, a) => (Kind::Unary(op), vec![val(lw, a)]),
            Node::Binary(op, a, b) => (Kind::Binary(op), vec![val(lw, a), val(lw, b)]),
            Node::Cmp(op, a, b) => (Kind::Cmp(op), vec![val(lw, a), val(lw, b)]),
            Node::Select(c, t, e) => (Kind::Select, vec![val(lw, c), val(lw, t), val(lw, e)]),
            Node::Reduce(op, l) => (
                Kind::Reduce(op),
                ctx.args(l).iter().map(|&a| val(lw, a)).collect(),
            ),
            Node::Dot(l) => (
                Kind::Dot(l.len() as u32 / 2),
                ctx.args(l).iter().map(|&a| val(lw, a)).collect(),
            ),
            Node::Call(..) | Node::Solve(..) => unreachable!("calls and solves lower as groups"),
        };
        let inst = lw.push(kind, ins, 1, self.pure[i]);
        lw.value[i] = Some(Ref::Value(inst, 0));
    }
}

#[derive(Default)]
struct CallSets {
    set_of: HashMap<(u32, Site), u32>,
    sets: Vec<Vec<u32>>,
    members: HashMap<(u32, Site), Vec<(usize, u32)>>,
}

// https://github.com/milanofthe/rsdag/issues/241
#[derive(PartialEq, Eq, Hash, Clone)]
enum GroupKey {
    Call {
        func: u32,
        depth: u32,
        pure: bool,
        output_set: u32,
    },
    Gemv {
        vector: Vec<ExprId>,
        depth: u32,
        pure: bool,
    },
    Gemm {
        sorted_rows: Vec<Vec<ExprId>>,
        vectors: Vec<Vec<ExprId>>,
    },
    Solve(ArgList, bool),
    SolveMany {
        matrix: Vec<ExprId>,
        rhs_lists: Vec<ArgList>,
    },
    SolveBatch(Vec<System>),
}

type System = (Vec<ExprId>, Vec<ArgList>);

struct Groups {
    groups: Vec<(GroupKey, Vec<usize>)>,
    kernel_of: Vec<Option<usize>>,
}

struct Lowering {
    p: Program,
    value: Vec<Option<Ref>>,
    template_consts_by_bits: HashMap<u64, Ref>,
}

impl Lowering {
    fn push(
        &mut self,
        kind: Kind,
        ins: impl IntoIterator<Item = Ref>,
        n_out: u32,
        pure: bool,
    ) -> u32 {
        let start = self.p.pool.len() as u32;
        self.p.pool.extend(ins);
        self.p.insts.push(Inst {
            kind,
            ins: (start, self.p.pool.len() as u32 - start),
            n_out,
            pure,
        });
        self.p.insts.len() as u32 - 1
    }

    fn constant(&mut self, v: f64) -> Ref {
        Ref::Value(self.push(Kind::Const(v), [], 1, true), 0)
    }

    fn expand(&mut self, bodies: &mut Bodies, t: &Program, args: &[Ref]) -> Vec<Ref> {
        let bundles: Vec<u32> = t
            .bundles
            .iter()
            .map(|b| bodies.intern(&mut self.p.bundles, b))
            .collect();
        let mut shared_const: Vec<Option<Ref>> = vec![None; t.insts.len()];
        for (i, inst) in t.insts.iter().enumerate() {
            if let Kind::Const(v) = inst.kind {
                shared_const[i] = Some(
                    *self
                        .template_consts_by_bits
                        .entry(v.to_bits())
                        .or_insert_with(|| {
                            let start = self.p.pool.len() as u32;
                            self.p.insts.push(Inst {
                                kind: Kind::Const(v),
                                ins: (start, 0),
                                n_out: 1,
                                pure: true,
                            });
                            Ref::Value(self.p.insts.len() as u32 - 1, 0)
                        }),
                );
            }
        }
        let base = self.p.insts.len() as u32;
        let mut next = base;
        let mut index = vec![0u32; t.insts.len()];
        for (i, slot) in shared_const.iter().enumerate() {
            if slot.is_none() {
                index[i] = next;
                next += 1;
            }
        }
        let map = |r: Ref| match r {
            Ref::Input(k) => args[k as usize],
            Ref::Value(i, o) => {
                shared_const[i as usize].unwrap_or(Ref::Value(index[i as usize], o))
            }
        };
        for (i, inst) in t.insts.iter().enumerate() {
            if shared_const[i].is_some() {
                continue;
            }
            let mut kind = inst.kind.clone();
            if let Kind::Call { bundle, .. } | Kind::CallProlog { bundle, .. } = &mut kind {
                *bundle = bundles[*bundle as usize];
            }
            let ins = t.ins(i).iter().map(|&r| map(r));
            self.push(kind, ins, inst.n_out, inst.pure);
        }
        t.roots.iter().map(|&r| map(r)).collect()
    }
}

#[derive(Default)]
struct Templates {
    expanded_leaf_calls_apart: bool,
}

impl Templates {
    fn get<K: Field>(
        &mut self,
        ctx: &Graph<K>,
        f: u32,
        outs: &[u32],
        mask: Option<Vec<bool>>,
    ) -> Arc<Program> {
        self.expanded_leaf_calls_apart = true;
        let key = (f, outs.to_vec(), mask);
        // https://github.com/milanofthe/rsdag/issues/241
        let cached = ctx.templates.lock().unwrap().get(&key).cloned();
        if let Some(t) = cached {
            return t;
        }
        let func = ctx.func(FuncId(f));
        let mut syms: Vec<SymbolId> = func.params().to_vec();
        syms.extend_from_slice(&func.globals_in(ctx).symbols);
        let exprs: Vec<ExprId> = outs
            .iter()
            .filter_map(|&k| match func.outputs()[k as usize] {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let forest = Forest::analyze(ctx, &exprs, &syms, key.2.as_deref());
        let mut p = forest.lower(ctx, &exprs, self);
        drop(forest);
        let mut roots = std::mem::take(&mut p.roots).into_iter();
        let mut lw = Lowering {
            p,
            value: Vec::new(),
            template_consts_by_bits: HashMap::default(),
        };
        let all: Vec<Ref> = outs
            .iter()
            .map(|&k| match func.outputs()[k as usize] {
                Output::Expr(_) => roots.next().expect("one per expression"),
                _ => lw.constant(0.0),
            })
            .collect();
        lw.p.roots = all;
        let t = Arc::new(lw.p);
        ctx.templates.lock().unwrap().insert(key, t.clone());
        t
    }
}

#[derive(Default)]
struct Bodies {
    of: HashMap<(u32, u32), (u32, Body)>,
    bundle_idx: HashMap<usize, u32>,
}

impl Bodies {
    fn intern(
        &mut self,
        bundles: &mut Vec<Arc<dyn ExternBundle>>,
        bundle: &Arc<dyn ExternBundle>,
    ) -> u32 {
        let ptr = Arc::as_ptr(bundle) as *const () as usize;
        *self.bundle_idx.entry(ptr).or_insert_with(|| {
            bundles.push(bundle.clone());
            bundles.len() as u32 - 1
        })
    }

    fn get<K: Field>(
        &mut self,
        ctx: &Graph<K>,
        bundles: &mut Vec<Arc<dyn ExternBundle>>,
        f: u32,
        set: u32,
        sets: &[Vec<u32>],
    ) -> (u32, &Body) {
        let bundle_idx = &mut self.bundle_idx;
        let (b, body) = self.of.entry((f, set)).or_insert_with(|| {
            let body = ctx.func(FuncId(f)).body_for(ctx, &sets[set as usize]);
            let ptr = Arc::as_ptr(&body.bundle) as *const () as usize;
            let b = *bundle_idx.entry(ptr).or_insert_with(|| {
                bundles.push(body.bundle.clone());
                bundles.len() as u32 - 1
            });
            (b, body)
        });
        (*b, body)
    }
}

impl Program {
    // https://github.com/milanofthe/rsdag/issues/241
    pub(super) fn schedule(&self) -> Vec<u32> {
        let m = self.insts.len();
        let held_by_kept_state =
            |i: usize| self.state.as_ref().is_some_and(|(at, _)| at[i] != u32::MAX);
        let is_const =
            |i: usize| matches!(self.insts[i].kind, Kind::Const(_)) && !held_by_kept_state(i);
        let placed_before_first_consumer = is_const;
        let mut stamp: Vec<u32> = vec![u32::MAX; m];
        let mut dep_start: Vec<u32> = Vec::with_capacity(m + 1);
        let mut dep_list: Vec<u32> = Vec::with_capacity(self.pool.len());
        dep_start.push(0);
        for i in 0..m {
            for r in self.ins(i) {
                if let Ref::Value(j, _) = *r {
                    if stamp[j as usize] != i as u32 {
                        stamp[j as usize] = i as u32;
                        dep_list.push(j);
                    }
                }
            }
            dep_start.push(dep_list.len() as u32);
        }
        let deps = |i: usize| &dep_list[dep_start[i] as usize..dep_start[i + 1] as usize];
        let mut user_count = vec![0u32; m];
        let mut pending = vec![0u32; m];
        for i in 0..m {
            pending[i] = (deps(i).iter())
                .filter(|&&x| !placed_before_first_consumer(x as usize))
                .count() as u32;
            for &x in deps(i) {
                user_count[x as usize] += 1;
            }
        }
        let mut user_start = vec![0u32; m + 1];
        for i in 0..m {
            user_start[i + 1] = user_start[i] + user_count[i];
        }
        let mut users = vec![0u32; user_start[m] as usize];
        let mut fill = user_start.clone();
        for i in 0..m {
            for &x in deps(i) {
                users[fill[x as usize] as usize] = i as u32;
                fill[x as usize] += 1;
            }
        }
        let mut remaining = user_count.clone();
        let kills_of = |i: usize, remaining: &[u32]| -> u32 {
            deps(i)
                .iter()
                .filter(|&&d| remaining[d as usize] == 1)
                .count() as u32
        };
        let mut ready = Ready::default();
        for i in (0..m).rev() {
            if !is_const(i) && pending[i] == 0 {
                ready.push(self.insts[i].pure, kills_of(i, &remaining), i as u32);
            }
        }
        let mut order: Vec<u32> = Vec::with_capacity(m);
        let mut placed = vec![false; m];
        let mut last: Option<usize> = None;
        while let Some((pure, kills, iu)) = ready.pop() {
            let mut i = iu as usize;
            if let (Some(lo), Some(iu2)) = (last, ready.peek_same(pure, kills)) {
                let i2 = iu2 as usize;
                // https://github.com/milanofthe/rsdag/issues/241
                let overlaps_independent_chain =
                    deps(i).contains(&(lo as u32)) && !deps(i2).contains(&(lo as u32));
                if overlaps_independent_chain {
                    ready.swap_top(pure, kills, iu);
                    i = i2;
                }
            }
            last = Some(i);
            for &d in deps(i) {
                let d = d as usize;
                if placed_before_first_consumer(d) && !placed[d] {
                    placed[d] = true;
                    order.push(d as u32);
                }
            }
            placed[i] = true;
            order.push(i as u32);
            for &d in deps(i) {
                remaining[d as usize] -= 1;
            }
            for k in user_start[i]..user_start[i + 1] {
                let u = users[k as usize] as usize;
                pending[u] -= 1;
                if pending[u] == 0 {
                    ready.push(self.insts[u].pure, kills_of(u, &remaining), u as u32);
                }
            }
        }
        for i in 0..m {
            if !placed[i] {
                debug_assert!(is_const(i), "the list scheduler placed every instruction");
                order.push(i as u32);
            }
        }
        debug_assert_eq!(order.len(), m);
        debug_assert!(
            (order.iter())
                .filter(|&&i| !is_const(i as usize))
                .map(|&i| self.insts[i as usize].pure)
                .collect::<Vec<bool>>()
                .windows(2)
                .all(|w| w[0] || !w[1]),
            "every pure instruction precedes every impure one"
        );
        order
    }

    fn regions(&self, order: &[u32], prolog_ops: usize) -> (Vec<u32>, Vec<Region>) {
        use crate::node::ReduceOp;
        let plain = |i: u32| match self.insts[i as usize].kind {
            Kind::Call { .. }
            | Kind::CallProlog { .. }
            | Kind::Gemv { .. }
            | Kind::Gemm { .. }
            | Kind::Solve { .. }
            | Kind::Reduce(ReduceOp::Min | ReduceOp::Max) => false,
            _ => self.insts[i as usize].n_out == 1,
        };
        let is_const = |i: u32| matches!(self.insts[i as usize].kind, Kind::Const(_));
        let mut out = Vec::with_capacity(order.len());
        let mut regions = Vec::new();
        let mut position_in_run = vec![u32::MAX; self.insts.len()];
        let mut parent: Vec<u32> = Vec::new();
        fn root(parent: &mut [u32], mut a: u32) -> u32 {
            while parent[a as usize] != a {
                parent[a as usize] = parent[parent[a as usize] as usize];
                a = parent[a as usize];
            }
            a
        }
        let mut k = 0;
        while k < order.len() {
            let end = if k < prolog_ops {
                prolog_ops
            } else {
                order.len()
            };
            let mut e = k;
            while e < end && plain(order[e]) {
                e += 1;
            }
            if e == k {
                out.push(order[k]);
                k += 1;
                continue;
            }
            let run = &order[k..e];
            k = e;
            let work = run.iter().filter(|&&i| !is_const(i)).count();
            if work < REGION_MIN_OPS {
                out.extend_from_slice(run);
                continue;
            }
            parent.clear();
            parent.extend(0..run.len() as u32);
            for (p, &i) in run.iter().enumerate() {
                position_in_run[i as usize] = p as u32;
            }
            for (p, &i) in run.iter().enumerate() {
                if is_const(i) {
                    continue;
                }
                for r in self.ins(i as usize) {
                    if let Ref::Value(j, _) = *r {
                        let q = position_in_run[j as usize];
                        if q != u32::MAX && !is_const(j) {
                            let (a, b) = (root(&mut parent, p as u32), root(&mut parent, q));
                            let first_op_stays_root = a.min(b);
                            parent[a.max(b) as usize] = first_op_stays_root;
                        }
                    }
                }
            }
            let mut members: Vec<Vec<u32>> = Vec::new();
            let mut part = vec![u32::MAX; run.len()];
            let at = out.len();
            for (p, &i) in run.iter().enumerate() {
                position_in_run[i as usize] = u32::MAX;
                if is_const(i) {
                    out.push(i);
                    continue;
                }
                let r = root(&mut parent, p as u32) as usize;
                if part[r] == u32::MAX {
                    part[r] = members.len() as u32;
                    members.push(Vec::new());
                }
                members[part[r] as usize].push(i);
            }
            let mut starts = Vec::new();
            let mut size = PIECE_OPS;
            for m in &members {
                if size >= PIECE_OPS {
                    starts.push(out.len() as u32);
                    size = 0;
                }
                out.extend_from_slice(m);
                size += m.len();
            }
            let pieces_run_alongside = starts.len() >= 2;
            if pieces_run_alongside {
                regions.push(Region {
                    starts,
                    hi: out.len() as u32,
                });
            } else {
                out.truncate(at);
                out.extend_from_slice(run);
            }
        }
        debug_assert_eq!(out.len(), order.len());
        (out, regions)
    }

    pub(super) fn emit(&self, order: &[u32], split: bool) -> Tape {
        let m = self.insts.len();
        let prolog_ops = if split {
            order
                .iter()
                .position(|&i| !self.insts[i as usize].pure)
                .unwrap_or(m)
        } else {
            0
        };
        let (order, regions) = self.regions(order, prolog_ops);
        let order = &order[..];
        let mut pos = vec![0usize; m];
        for (k, &i) in order.iter().enumerate() {
            pos[i as usize] = k;
        }
        debug_assert!(
            order.iter().enumerate().all(|(k, &i)| {
                self.ins(i as usize).iter().all(|r| match *r {
                    Ref::Value(j, _) => pos[j as usize] < k,
                    Ref::Input(_) => true,
                })
            }),
            "the regions keep a topological order"
        );
        let mut last = vec![0usize; m];
        let mut pinned = vec![false; m];
        for (k, &i) in order.iter().enumerate() {
            for r in self.ins(i as usize) {
                if let Ref::Value(j, _) = *r {
                    last[j as usize] = last[j as usize].max(k);
                }
            }
        }
        for r in &self.roots {
            if let Ref::Value(j, _) = *r {
                pinned[j as usize] = true;
                last[j as usize] = usize::MAX;
            }
        }
        if split {
            for i in 0..m {
                if pos[i] < prolog_ops
                    && last[i] >= prolog_ops
                    && self.placeholder != Some(i as u32)
                {
                    pinned[i] = true;
                    last[i] = usize::MAX;
                }
            }
        }
        let mut state_base = vec![u32::MAX; m];
        let state_block_start: u32 = 0;
        let mut next: u32 = state_block_start;
        match &self.state {
            Some((at, len)) => {
                for i in 0..m {
                    if at[i] != u32::MAX {
                        assert!(pos[i] < prolog_ops, "a kept state value is the prolog's");
                        (state_base[i], pinned[i], last[i]) = (at[i], true, usize::MAX);
                    } else {
                        assert!(
                            !(pinned[i] && pos[i] < prolog_ops && split),
                            "the main phase reads a value the kept state does not hold"
                        );
                    }
                }
                next = *len;
            }
            None => {
                for &i in &order[..prolog_ops] {
                    if pinned[i as usize] {
                        state_base[i as usize] = next;
                        next += self.insts[i as usize].n_out;
                    }
                }
            }
        }
        let state_len = next as usize;
        debug_assert!(
            (state_base.iter()).all(|&s| s == u32::MAX || s < state_len as u32),
            "the state is one block at the start of the work array"
        );
        let mut reserved_in_kernel_run = vec![u32::MAX; m];
        for &i in order {
            let inst = &self.insts[i as usize];
            let runs: Vec<usize> = match inst.kind {
                Kind::Gemv {
                    m: rows,
                    n,
                    ref acc,
                } => {
                    let mut r = vec![(rows * n) as usize, n as usize];
                    if acc.is_some() {
                        r.push(rows as usize);
                    }
                    r
                }
                Kind::Gemm {
                    m: rows,
                    k,
                    n,
                    ref acc,
                } => {
                    let mut r = vec![(rows * k) as usize, (n * k) as usize];
                    if acc.is_some() {
                        r.push((rows * n) as usize);
                    }
                    r
                }
                Kind::Solve { n, k, count } => {
                    vec![(count * n * n) as usize, (count * n * k) as usize]
                }
                _ => continue,
            };
            let ins = self.ins(i as usize);
            let mut at = 0usize;
            for len in runs {
                let operand = &ins[at..at + len];
                at += len;
                let mut s = 0usize;
                while s < operand.len() {
                    let placeable = |r: &Ref| match *r {
                        Ref::Value(j, 0) => {
                            self.insts[j as usize].n_out == 1
                                && reserved_in_kernel_run[j as usize] == u32::MAX
                                && state_base[j as usize] == u32::MAX
                        }
                        _ => false,
                    };
                    if !placeable(&operand[s]) {
                        s += 1;
                        continue;
                    }
                    let mut e = s;
                    while e < operand.len() && placeable(&operand[e]) {
                        e += 1;
                    }
                    if e - s >= PLACE_MIN_RUN {
                        for r in &operand[s..e] {
                            if let Ref::Value(j, _) = *r {
                                reserved_in_kernel_run[j as usize] = next;
                                next += 1;
                            }
                        }
                    }
                    s = e;
                }
            }
        }
        let mut piece_of = vec![u32::MAX; m];
        for (p, k) in regions
            .iter()
            .flat_map(|r| (0..r.starts.len()).map(move |p| r.piece(p)))
            .enumerate()
            .flat_map(|(p, ks)| ks.map(move |k| (p, k)))
        {
            piece_of[order[k] as usize] = p as u32;
        }
        let (mut piece, mut freed_for_this_piece_only, mut freed_for_after_the_region) =
            (u32::MAX, Vec::<u32>::new(), Vec::<u32>::new());
        let mut base = vec![u32::MAX; m];
        let mut free: Vec<u32> = Vec::new();
        let mut ops: Vec<Op> = Vec::with_capacity(m);
        let mut dst: Vec<u32> = Vec::with_capacity(m);
        let mut arg_pool: Vec<u32> = Vec::new();
        let mut max_args = 0usize;
        let mut n_selects = 0usize;
        let slot_of = |r: Ref, base: &[u32]| -> u32 {
            match r {
                Ref::Value(j, k) => base[j as usize] + k,
                Ref::Input(k) => k | INPUT,
            }
        };
        let mut operand_slots_before_freeing: Vec<u32> = Vec::new();
        let mut dying: Vec<u32> = Vec::new();
        for (k, &i) in order.iter().enumerate() {
            let inst = &self.insts[i as usize];
            if piece_of[i as usize] != piece {
                piece = piece_of[i as usize];
                freed_for_after_the_region.append(&mut freed_for_this_piece_only);
                if piece == u32::MAX {
                    free.append(&mut freed_for_after_the_region);
                }
            }
            let ins = self.ins(i as usize);
            operand_slots_before_freeing.clear();
            operand_slots_before_freeing.extend(ins.iter().map(|&r| slot_of(r, &base)));
            dying.clear();
            dying.extend(ins.iter().filter_map(|r| match *r {
                Ref::Value(j, _)
                    if last[j as usize] == k
                        && !pinned[j as usize]
                        && reserved_in_kernel_run[j as usize] == u32::MAX =>
                {
                    Some(j)
                }
                _ => None,
            }));
            dying.sort_unstable();
            dying.dedup();
            for &j in &dying {
                let slots = base[j as usize]..base[j as usize] + self.insts[j as usize].n_out;
                match piece {
                    u32::MAX => free.extend(slots),
                    p if piece_of[j as usize] == p => freed_for_this_piece_only.extend(slots),
                    _ => freed_for_after_the_region.extend(slots),
                }
            }
            let d = if state_base[i as usize] != u32::MAX {
                state_base[i as usize]
            } else if inst.n_out == 1 && reserved_in_kernel_run[i as usize] != u32::MAX {
                reserved_in_kernel_run[i as usize]
            } else if inst.n_out == 1 {
                (freed_for_this_piece_only.pop())
                    .or_else(|| free.pop())
                    .unwrap_or_else(|| {
                        let s = next;
                        next += 1;
                        s
                    })
            } else {
                let s = next;
                next += inst.n_out;
                s
            };
            base[i as usize] = d;
            let gather = |arg_pool: &mut Vec<u32>, max_args: &mut usize, ops: &[u32]| -> u32 {
                let start = arg_pool.len() as u32;
                arg_pool.extend_from_slice(ops);
                *max_args = (*max_args).max(ops.len());
                start
            };
            let dense = |arg_pool: &mut Vec<u32>, max_args: &mut usize, ops: &[u32]| -> Src {
                let consecutive_from =
                    |k0: u32| ops.iter().enumerate().all(|(j, &k)| k == k0 + j as u32);
                match ops.first() {
                    Some(&k0) if consecutive_from(k0) && input_index(k0).is_some() => {
                        Src::Inputs(k0 & !INPUT)
                    }
                    Some(&k0) if consecutive_from(k0) => Src::Slots(k0),
                    _ => Src::Pool(gather(arg_pool, max_args, ops)),
                }
            };
            let o = &operand_slots_before_freeing;
            let op = match inst.kind {
                Kind::Const(v) => Op::Const(v),
                Kind::Add => Op::Add(o[0], o[1]),
                Kind::Mul => Op::Mul(o[0], o[1]),
                Kind::MulAdd => Op::MulAdd(o[0], o[1], o[2]),
                Kind::Sub => Op::Sub(o[0], o[1]),
                Kind::Neg => Op::Neg(o[0]),
                Kind::Powi(n) => Op::Powi(o[0], n),
                Kind::Unary(op) => Op::Unary(op, o[0]),
                Kind::Binary(op) => Op::Binary(op, o[0], o[1]),
                Kind::Cmp(op) => Op::Cmp(op, o[0], o[1]),
                Kind::Select => {
                    n_selects += 1;
                    Op::Select(o[0], o[1], o[2])
                }
                Kind::Reduce(op) => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    Op::Reduce(op, start, o.len() as u32)
                }
                Kind::Dot(n) => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    Op::Dot(start, n)
                }
                Kind::Call {
                    bundle,
                    n_groups,
                    n_args,
                    ref reads,
                    stateful,
                    ..
                } => {
                    let (args, state) = split_state(o, stateful);
                    let start = gather(&mut arg_pool, &mut max_args, args);
                    let whole_args_read_or_not = (n_groups * n_args) as usize;
                    max_args = max_args.max(whole_args_read_or_not);
                    let (n_in, reads) = match reads {
                        None => (n_args, super::ALL_ARGS),
                        Some(r) => {
                            let at = arg_pool.len() as u32;
                            arg_pool.extend_from_slice(r);
                            (r.len() as u32, at)
                        }
                    };
                    Op::Call {
                        bundle,
                        start,
                        n_groups,
                        n_args,
                        n_in,
                        reads,
                        n_out: inst.n_out / n_groups,
                        state,
                        args: 0,
                    }
                }
                Kind::CallProlog {
                    bundle,
                    n_groups,
                    n_pure,
                    ..
                } => {
                    let start = gather(&mut arg_pool, &mut max_args, o);
                    let np = n_pure as usize;
                    let own = arg_pool.len() as u32;
                    let slot_differs_from_first_instance =
                        |j: usize| (1..n_groups as usize).any(|g| o[g * np + j] != o[j]);
                    for j in 0..np {
                        if slot_differs_from_first_instance(j) {
                            arg_pool.push(j as u32);
                        }
                    }
                    Op::CallProlog {
                        bundle,
                        start,
                        n_groups,
                        n_pure,
                        own,
                        n_own: arg_pool.len() as u32 - own,
                        args: 0,
                    }
                }
                Kind::Gemv {
                    m: rows,
                    n,
                    ref acc,
                } => {
                    let (a, rest) = o.split_at((rows * n) as usize);
                    let (x, c) = rest.split_at(n as usize);
                    let a = dense(&mut arg_pool, &mut max_args, a);
                    let x = dense(&mut arg_pool, &mut max_args, x);
                    let acc = acc.as_ref().map(|codes| {
                        let reads = codes.iter().any(|&code| Fold(code).reads_operand());
                        let c = reads.then(|| dense(&mut arg_pool, &mut max_args, c));
                        let start = arg_pool.len() as u32;
                        arg_pool.extend_from_slice(codes);
                        super::Accum { c, codes: start }
                    });
                    // https://github.com/milanofthe/rsdag/issues/241
                    let scratch_holds_every_operand = (rows * n + n + rows) as usize;
                    max_args = max_args.max(scratch_holds_every_operand);
                    Op::Gemv {
                        a,
                        x,
                        m: rows,
                        n,
                        acc,
                    }
                }
                Kind::Gemm {
                    m: rows,
                    k,
                    n,
                    ref acc,
                } => {
                    let (a, rest) = o.split_at((rows * k) as usize);
                    let (b, c) = rest.split_at((n * k) as usize);
                    let a = dense(&mut arg_pool, &mut max_args, a);
                    let b = dense(&mut arg_pool, &mut max_args, b);
                    let acc = acc.as_ref().map(|codes| {
                        let reads = codes.iter().any(|&code| Fold(code).reads_operand());
                        let c = reads.then(|| dense(&mut arg_pool, &mut max_args, c));
                        let start = arg_pool.len() as u32;
                        arg_pool.extend_from_slice(codes);
                        super::Accum { c, codes: start }
                    });
                    max_args = max_args.max((rows * k + n * k + rows * n) as usize);
                    Op::Gemm {
                        a,
                        b,
                        m: rows,
                        k,
                        n,
                        acc,
                    }
                }
                Kind::Solve { n, k, count } => {
                    let (a, b) = o.split_at((count * n * n) as usize);
                    let a = dense(&mut arg_pool, &mut max_args, a);
                    let b = dense(&mut arg_pool, &mut max_args, b);
                    max_args = max_args.max((count * (n * n + n * k)) as usize);
                    Op::Solve { a, b, n, k, count }
                }
            };
            ops.push(op);
            dst.push(d);
        }
        let outputs: Vec<u32> = self.roots.iter().map(|&r| slot_of(r, &base)).collect();
        let solves = ops.iter().map(|op| match *op {
            Op::Solve { n, k, .. } => crate::semantics::solve_scratch_len(n as usize, k as usize),
            _ => 0,
        });
        let lent_to_one_borrower_at_a_time = self
            .bundles
            .iter()
            .map(|b| b.work_len())
            .chain(solves)
            .max()
            .unwrap_or(0);
        let stages = super::plan_stages(&ops, &dst, &arg_pool, &self.bundles, prolog_ops);
        for st in &stages {
            let mut gather_offset = 0u32;
            for op in &mut ops[st.lo as usize..st.hi as usize] {
                let (args, width) = match op {
                    Op::Call {
                        args,
                        n_groups,
                        n_args,
                        ..
                    } => (args, *n_groups * *n_args),
                    Op::CallProlog {
                        args,
                        n_groups,
                        n_pure,
                        ..
                    } => (args, *n_groups * *n_pure),
                    _ => unreachable!("a stage holds calls only"),
                };
                *args = gather_offset;
                gather_offset += width;
            }
            max_args = max_args.max(gather_offset as usize);
        }
        Tape {
            interp: super::Interp::new(&ops, &dst, &arg_pool, &outputs, next as usize, prolog_ops),
            stages,
            regions,
            lent: lent_to_one_borrower_at_a_time,
            ops,
            dst,
            n_selects,
            arg_pool,
            outputs,
            n_work: next as usize,
            max_args,
            bundles: self.bundles.clone(),
            prolog_ops,
            state_len,
            n_inputs: 0,
        }
    }
}

// https://github.com/milanofthe/rsdag/issues/241
#[derive(Default)]
struct Ready {
    buckets: [Vec<Vec<u32>>; 2],
    empty_above: [usize; 2],
    len: usize,
}

impl Ready {
    fn push(&mut self, pure: bool, kills: u32, i: u32) {
        let (p, k) = (pure as usize, kills as usize);
        let level = &mut self.buckets[p];
        if level.len() <= k {
            level.resize_with(k + 1, Vec::new);
        }
        level[k].push(i);
        self.empty_above[p] = self.empty_above[p].max(k);
        self.len += 1;
    }

    fn best(&mut self) -> Option<(usize, usize)> {
        if self.len == 0 {
            return None;
        }
        for p in [1, 0] {
            let level = &self.buckets[p];
            let hi = &mut self.empty_above[p];
            while *hi > 0 && level.get(*hi).is_none_or(Vec::is_empty) {
                *hi -= 1;
            }
            if level.get(*hi).is_some_and(|b| !b.is_empty()) {
                return Some((p, *hi));
            }
        }
        None
    }

    fn pop(&mut self) -> Option<(bool, u32, u32)> {
        let (p, k) = self.best()?;
        let i = self.buckets[p][k].pop()?;
        self.len -= 1;
        Some((p == 1, k as u32, i))
    }

    fn peek_same(&self, pure: bool, kills: u32) -> Option<u32> {
        self.buckets[pure as usize]
            .get(kills as usize)?
            .last()
            .copied()
    }

    fn swap_top(&mut self, pure: bool, kills: u32, i: u32) {
        let b = &mut self.buckets[pure as usize][kills as usize];
        let top = b.len() - 1;
        b[top] = i;
    }
}
