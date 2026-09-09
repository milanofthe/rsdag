//! Choice specialization: shorten a tape against a recorded `Select` trace,
//! keeping guard outputs that detect a region flip. See [`Tape::specialize`].

use super::{BatchTable, Op, Tape};

impl Tape {
    /// Shorten the tape against a recorded choice trace (fidget-style tape
    /// specialization): every `Select` is pinned to its traced arm and drops out
    /// of the instruction stream, ops reachable only through untaken arms die,
    /// and work slots are reallocated for the surviving subsequence. Each live
    /// `Select`'s condition survives as a *guard output*, so
    /// [`SpecializedTape::eval_checked`] can detect a region flip and demand a
    /// re-trace on the full tape.
    ///
    /// Why point guards are sound: take the earliest (op-order) live `Select`
    /// whose condition truth at the new inputs differs from the trace. Its
    /// condition chain contains only earlier ops, all of whose pinned `Select`s
    /// are still valid, so that guard value is computed exactly as the full
    /// tape would compute it -- the first flip is always detected.
    ///
    /// The surviving ops are the identical instructions in the identical order,
    /// so a checked evaluation is bit-exact against [`eval`](Self::eval).
    ///
    /// `choices` must be a trace of this tape from
    /// [`eval_traced`](Self::eval_traced) (`len == n_selects()`).
    pub fn specialize(&self, choices: &[u8]) -> SpecializedTape {
        self.specialize_partial(choices, &vec![true; self.n_selects])
    }

    /// [`specialize`](Self::specialize) with per-`Select` control: `pin[k]`
    /// pins select `k` to its traced arm (guarded); an unpinned select stays a
    /// real `Select` in the shortened tape -- both arms live, no guard. A
    /// caller that observes a select flipping across respecializations unpins
    /// it, so periodically region-hopping devices (a driven transient) stop
    /// costing a respecialization per period while the stable bias structure
    /// stays shortened.
    pub fn specialize_partial(&self, choices: &[u8], pin: &[bool]) -> SpecializedTape {
        assert_eq!(
            choices.len(),
            self.n_selects,
            "choice trace length mismatch"
        );
        assert_eq!(pin.len(), self.n_selects, "pin mask length mismatch");
        let m = self.ops.len();

        // Forward pass: op-level def-use. `deps` records, per op, the producing
        // op of every operand *at execution time* (slots are liveness-reused, so
        // this is only recoverable in execution order). A `Select` records its
        // condition/taken-arm producers instead and vacates the value chain:
        // `vsrc[i]` is the op whose slot actually carries op `i`'s value (the
        // pinned-select alias chain collapsed; `vsrc[i] == i` for real ops).
        let mut dep_start: Vec<u32> = Vec::with_capacity(m + 1);
        let mut dep_pool: Vec<u32> = Vec::new();
        let mut prod = vec![u32::MAX; self.n_work];
        let mut bprod = vec![u32::MAX; self.bundle_scratch_len];
        let mut vsrc: Vec<u32> = Vec::with_capacity(m);
        // Per select: (condition producer, taken-arm producer, traced choice).
        let mut sel: Vec<(u32, u32, u8)> = Vec::with_capacity(self.n_selects);
        let mut sel_at = vec![u32::MAX; m];
        let mut n_sel_seen = 0usize;
        for i in 0..m {
            dep_start.push(dep_pool.len() as u32);
            let p = |s: u32| prod[s as usize];
            match self.ops[i] {
                Op::Const(_) | Op::Input(_) => {}
                Op::Add(a, b)
                | Op::Mul(a, b)
                | Op::Sub(a, b)
                | Op::Cmp(_, a, b)
                | Op::Binary(_, a, b) => {
                    dep_pool.extend([p(a), p(b)]);
                }
                Op::MulAdd(a, b, c) | Op::Fma(a, b, c) => {
                    dep_pool.extend([p(a), p(b), p(c)]);
                }
                Op::Neg(a) | Op::Powi(a, _) | Op::Unary(_, a) => dep_pool.push(p(a)),
                Op::Select(c, t, e) => {
                    let k = n_sel_seen;
                    n_sel_seen += 1;
                    if pin[k] {
                        let arm = if choices[k] != 0 { p(t) } else { p(e) };
                        sel_at[i] = sel.len() as u32;
                        sel.push((p(c), arm, choices[k]));
                    } else {
                        // stays a real select: value chain keeps all three deps
                        dep_pool.extend([p(c), p(t), p(e)]);
                    }
                }
                Op::Reduce(_, s, l) => {
                    dep_pool.extend((0..l).map(|k| p(self.arg_pool[(s + k) as usize])));
                }
                Op::Dot(s, l) => {
                    dep_pool.extend((0..2 * l).map(|k| p(self.arg_pool[(s + k) as usize])));
                }
                Op::BundleCall(bidx, s, l, base) => {
                    dep_pool.extend((0..l).map(|k| p(self.arg_pool[(s + k) as usize])));
                    let n = self.bundles[bidx as usize].n_outputs();
                    for k in 0..n {
                        bprod[base as usize + k] = i as u32;
                    }
                }
                Op::BundleBatch(_, t) => {
                    let tbl = self.batches[t as usize];
                    let flat = tbl.n_groups * tbl.n_args;
                    dep_pool.extend((0..flat).map(|k| p(self.arg_pool[(tbl.start + k) as usize])));
                    for k in 0..(tbl.n_groups * tbl.n_out) as usize {
                        bprod[tbl.base0 as usize + k] = i as u32;
                    }
                }
                Op::BundlePick(idx) => dep_pool.push(bprod[idx as usize]),
            }
            vsrc.push(match sel_at[i] {
                u32::MAX => i as u32,
                k => vsrc[sel[k as usize].1 as usize], // arm < i: already resolved
            });
            prod[self.dst[i] as usize] = i as u32;
        }
        dep_start.push(dep_pool.len() as u32);
        let deps = |i: usize| &dep_pool[dep_start[i] as usize..dep_start[i + 1] as usize];

        // Backward liveness under pinning: a live select keeps its condition
        // (the guard) and its taken arm; everything else keeps all operands.
        let mut live = vec![false; m];
        let mut stack: Vec<u32> = self.outputs.iter().map(|&s| prod[s as usize]).collect();
        while let Some(i) = stack.pop() {
            let i = i as usize;
            if live[i] {
                continue;
            }
            live[i] = true;
            match sel_at[i] {
                u32::MAX => stack.extend_from_slice(deps(i)),
                k => {
                    let (cond, arm, _) = sel[k as usize];
                    stack.push(cond);
                    stack.push(arm);
                }
            }
        }

        // Guard outputs: each live select's condition, value-resolved and
        // deduped (hash-consing makes shared region conditions common). Two
        // selects sharing a condition source necessarily traced the same truth.
        let mut guard_srcs: Vec<u32> = Vec::new();
        let mut expected: Vec<u8> = Vec::new();
        let mut guard_seen = vec![false; m];
        for i in 0..m {
            if !live[i] || sel_at[i] == u32::MAX {
                continue;
            }
            let (cond, _, choice) = sel[sel_at[i] as usize];
            let src = vsrc[cond as usize] as usize;
            if !guard_seen[src] {
                guard_seen[src] = true;
                guard_srcs.push(src as u32);
                expected.push(choice);
            }
        }
        let out_srcs: Vec<u32> = self
            .outputs
            .iter()
            .map(|&s| vsrc[prod[s as usize] as usize])
            .collect();

        // Slot-read deps of an emitted op: all value deps, resolved through the
        // pinned selects. A BundlePick reads the bundle-scratch region (not a
        // slot), and a BundleCall writes only the shared never-read sink.
        let sdeps = |i: usize| -> &[u32] {
            match self.ops[i] {
                Op::BundlePick(_) => &[],
                _ => deps(i),
            }
        };

        // Last use per (resolved) value producer, over the emitted subsequence;
        // outputs and guards are pinned so their slots survive to the end.
        let mut last = vec![0usize; m];
        let mut pinned = vec![false; m];
        for i in 0..m {
            if !live[i] || sel_at[i] != u32::MAX {
                continue;
            }
            for &d in sdeps(i) {
                last[vsrc[d as usize] as usize] = i;
            }
        }
        for &o in out_srcs.iter().chain(guard_srcs.iter()) {
            pinned[o as usize] = true;
            last[o as usize] = usize::MAX;
        }
        // The specialization inherits the source tape's prolog split: surviving
        // ops keep their order, so the boundary is where original indices cross
        // `self.prolog_ops`. Prolog values read by the main phase are pinned,
        // exactly like `compile_split` pins them -- repeated main passes must
        // not clobber a prolog result they will read again.
        if self.prolog_ops > 0 {
            for i in self.prolog_ops..m {
                if !live[i] || sel_at[i] != u32::MAX {
                    continue;
                }
                for &d in sdeps(i) {
                    let v = vsrc[d as usize] as usize;
                    if v < self.prolog_ops {
                        pinned[v] = true;
                        last[v] = usize::MAX;
                    }
                }
            }
        }

        // Emit the surviving subsequence with fresh slots (same LIFO free-list
        // scheme as `compile`); pinned selects emit nothing.
        let mut ops: Vec<Op> = Vec::new();
        let mut dst: Vec<u32> = Vec::new();
        let mut arg_pool: Vec<u32> = Vec::new();
        let mut new_slot = vec![u32::MAX; m];
        let mut free: Vec<u32> = Vec::new();
        let mut next: u32 = 0;
        let mut max_args = 0usize;
        let mut sink: Option<u32> = None;
        let mut batches: Vec<BatchTable> = Vec::new();
        // Surviving-op count from the source prolog region: order is preserved,
        // so this is the specialized tape's own prolog length.
        let mut spec_prolog_ops = 0usize;
        for i in 0..m {
            if !live[i] || sel_at[i] != u32::MAX {
                continue;
            }
            if i < self.prolog_ops {
                spec_prolog_ops += 1;
            }
            let ds = |k: usize| new_slot[vsrc[deps(i)[k] as usize] as usize];
            let gather = |arg_pool: &mut Vec<u32>, max_args: &mut usize, n: usize| -> u32 {
                let start = arg_pool.len() as u32;
                arg_pool.extend((0..n).map(|k| new_slot[vsrc[deps(i)[k] as usize] as usize]));
                *max_args = (*max_args).max(n);
                start
            };
            let op = match self.ops[i] {
                Op::Const(v) => Op::Const(v),
                Op::Input(k) => Op::Input(k),
                Op::Add(..) => Op::Add(ds(0), ds(1)),
                Op::Mul(..) => Op::Mul(ds(0), ds(1)),
                Op::MulAdd(..) => Op::MulAdd(ds(0), ds(1), ds(2)),
                Op::Fma(..) => Op::Fma(ds(0), ds(1), ds(2)),
                Op::Sub(..) => Op::Sub(ds(0), ds(1)),
                Op::Neg(..) => Op::Neg(ds(0)),
                Op::Powi(_, n) => Op::Powi(ds(0), n),
                Op::Unary(op, _) => Op::Unary(op, ds(0)),
                Op::Cmp(op, ..) => Op::Cmp(op, ds(0), ds(1)),
                Op::Binary(op, ..) => Op::Binary(op, ds(0), ds(1)),
                // Only *pinned* selects vanish; an unpinned one survives as a
                // real select over its (resolved) three operands.
                Op::Select(..) => Op::Select(ds(0), ds(1), ds(2)),
                Op::Reduce(op, _, l) => {
                    Op::Reduce(op, gather(&mut arg_pool, &mut max_args, l as usize), l)
                }
                Op::Dot(_, l) => Op::Dot(gather(&mut arg_pool, &mut max_args, 2 * l as usize), l),
                Op::BundleCall(bidx, _, l, base) => Op::BundleCall(
                    bidx,
                    gather(&mut arg_pool, &mut max_args, l as usize),
                    l,
                    base,
                ),
                Op::BundleBatch(bidx, t) => {
                    let tbl = self.batches[t as usize];
                    let flat = (tbl.n_groups * tbl.n_args) as usize;
                    let start = gather(&mut arg_pool, &mut max_args, flat);
                    let tidx = batches.len() as u32;
                    batches.push(BatchTable { start, ..tbl });
                    Op::BundleBatch(bidx, tidx)
                }
                Op::BundlePick(idx) => Op::BundlePick(idx),
            };
            // Free distinct operand slots that die at this step, then pick dst
            // (so an op can reuse a dying operand's slot, as in `compile`).
            let mut dying: Vec<u32> = sdeps(i)
                .iter()
                .map(|&d| vsrc[d as usize] as usize)
                .filter(|&v| last[v] == i && !pinned[v])
                .map(|v| new_slot[v])
                .collect();
            dying.sort_unstable();
            dying.dedup();
            free.extend(dying);
            let d = if matches!(self.ops[i], Op::BundleCall(..) | Op::BundleBatch(..)) {
                // All bundle calls share one never-read, never-freed sink slot.
                *sink.get_or_insert_with(|| {
                    let s = next;
                    next += 1;
                    s
                })
            } else {
                free.pop().unwrap_or_else(|| {
                    let s = next;
                    next += 1;
                    s
                })
            };
            new_slot[i] = d;
            ops.push(op);
            dst.push(d);
        }

        let n_real = out_srcs.len();
        let outputs: Vec<u32> = out_srcs
            .iter()
            .chain(guard_srcs.iter())
            .map(|&o| new_slot[o as usize])
            .collect();
        // Guards whose (value-resolved) condition lives in the prolog: checked
        // once after a prolog pass, by slot (their pinned slots stay valid
        // across every later main pass).
        let prolog_guards: Vec<(u32, u8)> = guard_srcs
            .iter()
            .zip(&expected)
            .filter(|(&src, _)| (src as usize) < self.prolog_ops)
            .map(|(&src, &e)| (new_slot[src as usize], e))
            .collect();
        let n_selects_out = ops.iter().filter(|o| matches!(o, Op::Select(..))).count();
        SpecializedTape {
            tape: Tape {
                ops,
                dst,
                n_selects: n_selects_out,
                arg_pool,
                outputs,
                n_work: next as usize,
                max_args,
                bundles: self.bundles.clone(),
                bundle_scratch_len: self.bundle_scratch_len,
                batches,
                batch_args_len: self.batch_args_len,
                // Inherited from the source tape: surviving ops keep their
                // order, so the boundary is the surviving prefix.
                prolog_ops: spec_prolog_ops,
            },
            n_real,
            expected,
            prolog_guards,
        }
    }
}

/// A [`Tape`] shortened against a choice trace ([`Tape::specialize`]): all
/// `Select`s pinned, untaken arms removed, plus guard outputs that re-validate
/// the trace at every evaluation.
pub struct SpecializedTape {
    tape: Tape,
    /// The first `n_real` outputs are the original tape's; the rest are guards.
    n_real: usize,
    /// Expected truth (`1`/`0`) of each guard output.
    expected: Vec<u8>,
    /// Guards whose condition lives in the inherited prolog, as `(work slot,
    /// expected)` -- checked right after a prolog pass (their pinned slots stay
    /// valid across every later main pass). Empty for unsplit tapes.
    prolog_guards: Vec<(u32, u8)>,
}

impl SpecializedTape {
    /// Instruction count of the shortened tape (vs. [`Tape::n_ops`]).
    pub fn n_ops(&self) -> usize {
        self.tape.ops.len()
    }

    /// The shortened tape itself, for alternative backends (the chunked JIT
    /// compiles it like any other tape; its outputs are the real outputs
    /// followed by the guards).
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// Number of real (non-guard) outputs.
    pub fn n_real(&self) -> usize {
        self.n_real
    }

    /// Expected truth (`1`/`0`) of each guard output, for a caller that
    /// evaluates [`tape`](Self::tape) through another backend and re-implements
    /// the [`eval_checked`](Self::eval_checked) guard test.
    pub fn expected(&self) -> &[u8] {
        &self.expected
    }

    /// Evaluate the shortened tape. Returns `true` if every pinned choice still
    /// holds, in which case `out` is bit-exact against the full tape. On
    /// `false` a region flipped and `out` is NOT valid -- re-trace on the full
    /// tape ([`Tape::eval_traced`]) and respecialize.
    pub fn eval_checked(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) -> bool {
        self.tape.eval(inputs, work, out);
        let ok = out[self.n_real..]
            .iter()
            .zip(&self.expected)
            .all(|(&v, &e)| (v != 0.0) == (e != 0));
        out.truncate(self.n_real);
        ok
    }

    /// Evaluate the inherited prolog prefix into `work` and check the guards
    /// that live in it. `false` means a *parameter* change flipped a pinned
    /// region -- respecialize before running any main pass.
    pub fn eval_prolog_checked(&self, inputs: &[f64], work: &mut Vec<f64>) -> bool {
        self.tape.eval_prolog(inputs, work);
        self.check_prolog_guards(work)
    }

    /// The prolog-resident guards as `(work slot, expected)`, for a backend
    /// with a non-scalar work layout (e.g. lane-interleaved) that must
    /// re-implement [`check_prolog_guards`](Self::check_prolog_guards).
    pub fn prolog_guards(&self) -> &[(u32, u8)] {
        &self.prolog_guards
    }

    /// Check the prolog-resident guards against a `work` buffer some backend
    /// (interpreted or native) filled with this tape's prolog pass.
    pub fn check_prolog_guards(&self, work: &[f64]) -> bool {
        self.prolog_guards
            .iter()
            .all(|&(s, e)| (work[s as usize] != 0.0) == (e != 0))
    }

    /// Evaluate the main phase over a buffer prepared by
    /// [`eval_prolog_checked`](Self::eval_prolog_checked), guard-checked like
    /// [`eval_checked`](Self::eval_checked) (all guards: prolog guard slots are
    /// pinned, so their outputs remain valid and cost nothing extra).
    pub fn eval_main_checked(&self, inputs: &[f64], work: &mut [f64], out: &mut Vec<f64>) -> bool {
        self.tape.eval_main(inputs, work, out);
        let ok = out[self.n_real..]
            .iter()
            .zip(&self.expected)
            .all(|(&v, &e)| (v != 0.0) == (e != 0));
        out.truncate(self.n_real);
        ok
    }

    /// Verify guard outputs produced by an alternative backend's full or main
    /// evaluation of [`tape`](Self::tape) (`out` = real outputs ++ guards);
    /// truncates `out` to the real outputs. Same contract as
    /// [`eval_checked`](Self::eval_checked).
    pub fn check_outputs(&self, out: &mut Vec<f64>) -> bool {
        let ok = out[self.n_real..]
            .iter()
            .zip(&self.expected)
            .all(|(&v, &e)| (v != 0.0) == (e != 0));
        out.truncate(self.n_real);
        ok
    }
}
