//! Choice specialization: shorten a tape against a recorded `Select` trace,
//! keeping guard outputs that detect a region flip. See [`Tape::specialize`]
//! and [`Tape::specialize_main`].

use rustc_hash::FxHashSet;

use super::compile::{Inst, Kind, Program, Ref};
use super::{input_index, Op, Program as _, Tape};
use crate::node::{CmpOp, ReduceOp};

#[derive(Clone, Copy)]
struct PinnedSelect {
    cond: Ref,
    taken_arm: Ref,
    choice: u8,
}

/// How a shortened tape reports its guards, after its real outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guards {
    /// One output per distinct condition: which one failed is known.
    Each,
    /// One output, the number of conditions that do not hold: zero exactly
    /// when every one does. One value per evaluation to check, however
    /// many guards, for a batch of instances; which select flipped a trace
    /// tells.
    Count,
}

impl Tape {
    /// A shortened tape for the region a choice trace describes: `choices`
    /// is one byte per `Select` as recorded by [`Tape::eval_with`] with a
    /// `Vec<u8>` sink (`len == n_selects()`); `pin[k]` pins select `k` to
    /// its traced arm, guarded, and an unpinned select stays a real `Select`
    /// in the shortened tape, both arms live, no guard. A guarded tape
    /// reports through its checks whether the region still holds.
    ///
    /// The tape's program is transformed and compiled again: a pinned
    /// select becomes the arm it took, the condition of every one the
    /// outputs still reach becomes a guard output, and what nothing reads
    /// any more is dropped before the program is scheduled and given slots
    /// like a fresh compilation (the prolog split kept).
    pub fn specialize(&self, choices: &[u8], pin: &[bool]) -> SpecializedTape {
        let _s = rsprobe::scope!("specialize");
        assert_eq!(
            choices.len(),
            self.n_selects,
            "choice trace length mismatch"
        );
        assert_eq!(pin.len(), self.n_selects, "pin mask length mismatch");
        let p = self.lift();
        let pins = Self::pins_per_inst(&p, 0, choices, pin);
        self.shorten(p, pins, false, Guards::Each)
    }

    /// The main phase shortened for the region a trace of it describes,
    /// over the state as this tape's prolog lays it out: `choices` one byte
    /// per `Select` of the main phase, as [`Tape::eval_main_with`] records
    /// them (`len == n_main_selects()`), `pin` as for
    /// [`specialize`](Self::specialize), the guards reported as `guards`
    /// says. The prolog and the state are this tape's, so the shortened
    /// main phase and this one run over the same prolog result, either of
    /// them at any pass: a pass whose guards do not hold is run again on
    /// this tape's main phase, with nothing else to redo.
    pub fn specialize_main(&self, choices: &[u8], pin: &[bool], guards: Guards) -> SpecializedTape {
        let _s = rsprobe::scope!("specialize");
        let n = self.n_main_selects();
        assert_eq!(choices.len(), n, "choice trace length mismatch");
        assert_eq!(pin.len(), n, "pin mask length mismatch");
        let mut p = self.lift();
        let pins = Self::pins_per_inst(&p, self.prolog_ops, choices, pin);
        let mut state_slot_of_inst = vec![u32::MAX; p.insts.len()];
        for (i, a) in state_slot_of_inst
            .iter_mut()
            .enumerate()
            .take(self.prolog_ops)
        {
            if (self.dst[i] as usize) < self.state_len {
                *a = self.dst[i];
            }
        }
        p.state = Some((state_slot_of_inst, self.state_len as u32));
        self.shorten(p, pins, true, guards)
    }

    /// The selects of the main phase, the length of its trace.
    pub fn n_main_selects(&self) -> usize {
        self.ops[self.prolog_ops..]
            .iter()
            .filter(|op| matches!(op, Op::Select(..)))
            .count()
    }

    fn pins_per_inst(
        p: &Program,
        first_traced_inst: usize,
        choices: &[u8],
        pin: &[bool],
    ) -> Vec<Option<PinnedSelect>> {
        let mut pinned: Vec<Option<PinnedSelect>> = vec![None; p.insts.len()];
        let mut k = 0usize;
        for (i, slot) in pinned.iter_mut().enumerate().skip(first_traced_inst) {
            if matches!(p.insts[i].kind, Kind::Select) {
                if pin[k] {
                    let ins = p.ins(i);
                    let arm = if choices[k] != 0 { ins[1] } else { ins[2] };
                    *slot = Some(PinnedSelect {
                        cond: ins[0],
                        taken_arm: arm,
                        choice: choices[k],
                    });
                }
                k += 1;
            }
        }
        pinned
    }

    fn shorten(
        &self,
        mut p: Program,
        pinned: Vec<Option<PinnedSelect>>,
        trace_is_main_phase: bool,
        form: Guards,
    ) -> SpecializedTape {
        let m = p.insts.len();
        let mut arm: Vec<Option<Ref>> = pinned.iter().map(|q| q.map(|q| q.taken_arm)).collect();
        let mut reached_from_outputs = vec![false; m];
        let mut stack: Vec<Ref> = p.roots.clone();
        while let Some(r) = stack.pop() {
            let Ref::Value(i, _) = r else { continue };
            if std::mem::replace(&mut reached_from_outputs[i as usize], true) {
                continue;
            }
            match pinned[i as usize] {
                Some(q) => stack.extend([q.cond, q.taken_arm]),
                None => stack.extend_from_slice(p.ins(i as usize)),
            }
        }
        let mut distinct_guard_conds: FxHashSet<Ref> = FxHashSet::default();
        let mut guards: Vec<Ref> = Vec::new();
        let mut expected: Vec<u8> = Vec::new();
        let first = if trace_is_main_phase {
            self.prolog_ops
        } else {
            0
        };
        for i in first..m {
            if let (Kind::Select, true, Some(PinnedSelect { cond, choice, .. })) =
                (&p.insts[i].kind, reached_from_outputs[i], pinned[i])
            {
                let cond = decided(&p, &arm, cond, false);
                if distinct_guard_conds.insert(cond) {
                    guards.push(cond);
                    expected.push(choice);
                }
            }
        }
        let n_real = p.roots.len();
        match form {
            Guards::Each => p.roots.extend(guards),
            Guards::Count => {
                let count = count_unexpected_conds(&mut p, &guards, &expected);
                p.roots.push(count);
                if let Some((state_slot_of_inst, _)) = &mut p.state {
                    state_slot_of_inst.resize(p.insts.len(), u32::MAX);
                }
            }
        }
        arm.resize(p.insts.len(), None);
        let resolve =
            |v: &[Ref]| -> Vec<Ref> { v.iter().map(|&r| decided(&p, &arm, r, false)).collect() };
        (p.pool, p.roots) = (resolve(&p.pool), resolve(&p.roots));
        let tape = self.emit_again(&mut p, self.prolog_ops > 0);
        let n_individual_guards = if form == Guards::Each {
            expected.len()
        } else {
            0
        };
        let prolog_guards: Vec<(u32, u8)> = (0..n_individual_guards)
            .filter(|&g| match p.roots[n_real + g] {
                Ref::Input(_) => true,
                Ref::Value(i, _) => p.insts[i as usize].pure && tape.prolog_ops > 0,
            })
            .map(|g| (tape.outputs[n_real + g], expected[g]))
            .collect();
        SpecializedTape {
            tape,
            n_real,
            expected_truth: expected,
            prolog_guards,
            guards_as_count: form == Guards::Count,
        }
    }
}

impl Tape {
    fn emit_again(&self, p: &mut Program, keep_prolog_split: bool) -> Tape {
        p.retain_reachable();
        let order = p.schedule();
        let mut tape = p.emit(&order, keep_prolog_split);
        tape.n_inputs = self.n_inputs;
        rsprobe::add!("ops", tape.n_ops());
        tape
    }
}

fn count_unexpected_conds(p: &mut Program, conds: &[Ref], expected: &[u8]) -> Ref {
    let push = |p: &mut Program, kind: Kind, ins: &[Ref]| -> Ref {
        let start = p.pool.len() as u32;
        p.pool.extend_from_slice(ins);
        p.insts.push(Inst {
            kind,
            ins: (start, ins.len() as u32),
            n_out: 1,
            pure: false,
        });
        Ref::Value(p.insts.len() as u32 - 1, 0)
    };
    let zero = push(p, Kind::Const(0.0), &[]);
    let (mut expected_true, mut expected_false) = (Vec::new(), Vec::new());
    for (&c, &e) in conds.iter().zip(expected) {
        let is_comparison = match c {
            Ref::Value(i, 0) => matches!(p.insts[i as usize].kind, Kind::Cmp(_)),
            _ => false,
        };
        let truth_as_select_reads_it = if is_comparison {
            c
        } else {
            push(p, Kind::Cmp(CmpOp::Ne), &[c, zero])
        };
        if e != 0 {
            expected_true.push(truth_as_select_reads_it);
        } else {
            expected_false.push(truth_as_select_reads_it);
        }
    }
    let sum = |p: &mut Program, v: &[Ref]| match v.len() {
        0 => None,
        1 => Some(v[0]),
        _ => Some(push(p, Kind::Reduce(ReduceOp::Sum), v)),
    };
    let n_expected_true = push(p, Kind::Const(expected_true.len() as f64), &[]);
    let true_of_expected_false_plus_n_expected_true = match sum(p, &expected_false) {
        Some(s) => push(p, Kind::Add, &[s, n_expected_true]),
        None => n_expected_true,
    };
    match sum(p, &expected_true) {
        Some(s) => push(
            p,
            Kind::Sub,
            &[true_of_expected_false_plus_n_expected_true, s],
        ),
        None => true_of_expected_false_plus_n_expected_true,
    }
}

/// A [`Tape`] shortened against a choice trace ([`Tape::specialize`]): the
/// pinned `Select`s gone, their untaken arms removed, plus guard outputs
/// that re-validate the trace at every evaluation.
pub struct SpecializedTape {
    tape: Tape,
    n_real: usize,
    expected_truth: Vec<u8>,
    prolog_guards: Vec<(u32, u8)>,
    guards_as_count: bool,
}

impl SpecializedTape {
    /// Instruction count of the shortened tape (vs. [`Tape::n_ops`]).
    pub fn n_ops(&self) -> usize {
        self.tape.ops.len()
    }

    /// The shortened tape itself, for other backends (its outputs are the
    /// real outputs followed by the guards).
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// The shortened tape, owned.
    pub fn into_tape(self) -> Tape {
        self.tape
    }

    /// The original tape's outputs, the first of the shortened tape's.
    pub fn n_real(&self) -> usize {
        self.n_real
    }

    /// Whether the guard outputs `guards` (the shortened tape's outputs
    /// after the real ones) hold: the real outputs are then the full
    /// tape's, bit for bit.
    pub fn holds(&self, guards: &[f64]) -> bool {
        if self.guards_as_count {
            return guards.len() == 1 && guards[0] == 0.0;
        }
        guards
            .iter()
            .zip(&self.expected_truth)
            .all(|(&v, &e)| (v != 0.0) == (e != 0))
    }

    /// Evaluate the shortened tape. Returns `true` if every pinned choice still
    /// holds, in which case `out` is bit-exact against the full tape. On
    /// `false` a region flipped and `out` is NOT valid: re-trace on the full
    /// tape ([`Tape::eval_with`]) and respecialize.
    pub fn eval_checked(&self, inputs: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) -> bool {
        self.tape.eval(inputs, work, out);
        self.check_outputs(out)
    }

    /// Evaluate the inherited prolog prefix into `work` and check the guards
    /// that live in it. `false` means a *parameter* change flipped a pinned
    /// region: respecialize before running any main pass.
    pub fn eval_prolog_checked(&self, inputs: &[f64], work: &mut Vec<f64>) -> bool {
        self.tape.eval_prolog(inputs, work);
        self.check_prolog_guards(inputs, work)
    }

    /// The prolog-resident guards as `(operand, expected)`, for a backend
    /// that re-implements [`check_prolog_guards`](Self::check_prolog_guards).
    pub fn prolog_guards(&self) -> &[(u32, u8)] {
        &self.prolog_guards
    }

    /// Check the prolog-resident guards against a `work` buffer some backend
    /// filled with this tape's prolog pass.
    pub fn check_prolog_guards(&self, inputs: &[f64], work: &[f64]) -> bool {
        self.prolog_guards.iter().all(|&(k, e)| {
            let v = match input_index(k) {
                Some(i) => inputs.get(i as usize).copied().unwrap_or(f64::NAN),
                None => work[k as usize],
            };
            (v != 0.0) == (e != 0)
        })
    }

    /// Evaluate the main phase over a buffer prepared by
    /// [`eval_prolog_checked`](Self::eval_prolog_checked), guard-checked like
    /// [`eval_checked`](Self::eval_checked).
    pub fn eval_main_checked(
        &self,
        inputs: &[f64],
        work: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) -> bool {
        self.tape.eval_main(inputs, work, out);
        self.check_outputs(out)
    }

    /// Verify guard outputs produced by another backend's evaluation of
    /// [`tape`](Self::tape) (`out` = real outputs ++ guards); truncates
    /// `out` to the real outputs.
    pub fn check_outputs(&self, out: &mut Vec<f64>) -> bool {
        let ok = self.holds(&out[self.n_real..]);
        out.truncate(self.n_real);
        ok
    }
}

/// The `Select`s of a tape split over its parameters whose condition is
/// parameter-pure (a prolog value, or a pure input): the choices a
/// parameter binding makes once for every evaluation after it (see
/// [`Tape::param_selects`]).
pub struct ParamSelects {
    distinct_conds: Tape,
    select_inst_and_cond_output: Vec<(u32, u32)>,
}

impl ParamSelects {
    fn decided_arm_per_inst(&self, p: &Program, pattern: &[bool]) -> Vec<Option<Ref>> {
        let mut arm: Vec<Option<Ref>> = vec![None; p.insts.len()];
        for &(i, c) in &self.select_inst_and_cond_output {
            let ins = p.ins(i as usize);
            arm[i as usize] = Some(if pattern[c as usize] { ins[1] } else { ins[2] });
        }
        arm
    }

    /// The number of distinct conditions, the length of a pattern.
    pub fn n_conds(&self) -> usize {
        self.distinct_conds.outputs.len()
    }

    /// The pattern of the conditions at the inputs `args` (the tape's
    /// inputs; only the pure ones are read): one truth per condition.
    pub fn pattern(&self, args: &[f64], work: &mut Vec<f64>, out: &mut Vec<f64>) -> Vec<bool> {
        self.distinct_conds.eval(args, work, out);
        out.iter().map(|&c| c != 0.0).collect()
    }

    /// The tape of the conditions, an output each, over the tape's inputs:
    /// what a backend compiles to compute [`pattern`](Self::pattern).
    pub fn tape(&self) -> &Tape {
        &self.distinct_conds
    }
}

impl Tape {
    /// Whether the tape has a select a parameter binding decides: any of the
    /// prolog's, or one of the main phase whose condition is a prolog value
    /// or a pure input (`pure_inputs`). A scan of the ops, for deciding
    /// cheaply whether [`param_selects`](Self::param_selects) has anything
    /// to find.
    pub fn has_param_selects(&self, pure_inputs: &[bool]) -> bool {
        self.ops.iter().enumerate().any(|(k, op)| match *op {
            super::Op::Select(_, _, _) if k < self.prolog_ops => true,
            super::Op::Select(c, _, _) => match input_index(c) {
                Some(i) => pure_inputs.get(i as usize).copied().unwrap_or(false),
                None => (c as usize) < self.state_len,
            },
            _ => false,
        })
    }

    /// The selects a parameter binding decides: those whose condition is
    /// computed in the prolog or is a pure input (`pure_inputs`, one flag
    /// per input), in the prolog or the main phase. `None` for a tape
    /// without a prolog split or without such a select.
    pub fn param_selects(&self, pure_inputs: &[bool]) -> Option<ParamSelects> {
        if self.prolog_ops == 0 && !pure_inputs.iter().any(|&p| p) {
            return None;
        }
        let mut p = self.lift();
        let pure = |r: Ref, p: &Program| match r {
            Ref::Input(k) => pure_inputs.get(k as usize).copied().unwrap_or(false),
            Ref::Value(j, _) => p.insts[j as usize].pure,
        };
        let mut conds: Vec<Ref> = Vec::new();
        let mut of = Vec::new();
        for i in 0..p.insts.len() {
            let inst = &p.insts[i];
            if !matches!(inst.kind, Kind::Select) {
                continue;
            }
            let c = p.ins(i)[0];
            if !pure(c, &p) {
                continue;
            }
            let k = match conds.iter().position(|&d| d == c) {
                Some(k) => k,
                None => {
                    conds.push(c);
                    conds.len() - 1
                }
            };
            of.push((i as u32, k as u32));
        }
        if of.is_empty() {
            return None;
        }
        p.roots = conds;
        Some(ParamSelects {
            distinct_conds: self.emit_again(&mut p, false),
            select_inst_and_cond_output: of,
        })
    }

    /// The instructions, undecided and decided by each of `patterns` (see
    /// [`decide`](Self::decide)), counted on the tape's program without
    /// compiling it again: what deciding would remove.
    pub fn ops_decided(&self, ps: &ParamSelects, patterns: &[Vec<bool>]) -> (usize, Vec<usize>) {
        let p = self.lift();
        let count = |pattern: Option<&[bool]>| -> usize {
            let arm = match pattern {
                Some(pattern) => ps.decided_arm_per_inst(&p, pattern),
                None => vec![None; p.insts.len()],
            };
            let mut seen = vec![false; p.insts.len()];
            let mut stack: Vec<Ref> = p
                .roots
                .iter()
                .map(|&r| decided(&p, &arm, r, true))
                .collect();
            let mut n = 0;
            while let Some(r) = stack.pop() {
                let Ref::Value(i, _) = r else { continue };
                let i = i as usize;
                if std::mem::replace(&mut seen[i], true) {
                    continue;
                }
                n += 1;
                let main = !p.insts[i].pure;
                stack.extend(p.ins(i).iter().map(|&r| decided(&p, &arm, r, main)));
            }
            n
        };
        let decided = patterns.iter().map(|q| count(Some(q))).collect();
        (count(None), decided)
    }

    /// This tape with the selects of `ps` decided by `pattern` (one truth
    /// per condition, see [`ParamSelects::pattern`]):
    /// each the arm its condition picks, what only the other arms read
    /// dropped. Bit for bit the tape's outputs wherever the conditions take
    /// `pattern`; the prolog split kept, its state laid out anew, the main
    /// phase reading no more than the tape's (see `decided`). No guards:
    /// the caller computes the pattern.
    pub fn decide(&self, ps: &ParamSelects, pattern: &[bool]) -> Tape {
        let _s = rsprobe::scope!("decide");
        let mut p = self.lift();
        assert_eq!(pattern.len(), ps.n_conds(), "pattern length mismatch");
        let arm = ps.decided_arm_per_inst(&p, pattern);
        let mut operand_read_by_main = vec![false; p.pool.len()];
        for inst in &p.insts {
            let (s, l) = inst.ins;
            operand_read_by_main[s as usize..(s + l) as usize].fill(!inst.pure);
        }
        let pool: Vec<Ref> = (0..p.pool.len())
            .map(|k| decided(&p, &arm, p.pool[k], operand_read_by_main[k]))
            .collect();
        let roots: Vec<Ref> = p
            .roots
            .iter()
            .map(|&r| decided(&p, &arm, r, true))
            .collect();
        (p.pool, p.roots) = (pool, roots);
        self.emit_again(&mut p, self.prolog_ops > 0)
    }
}

fn decided(p: &Program, arm: &[Option<Ref>], r: Ref, read_by_main: bool) -> Ref {
    let mut r = r;
    while let Ref::Value(i, _) = r {
        let main_reads_prolog_select_from_state = read_by_main && p.insts[i as usize].pure;
        match arm[i as usize] {
            Some(Ref::Input(_)) if main_reads_prolog_select_from_state => break,
            Some(a) => r = a,
            None => break,
        }
    }
    r
}
