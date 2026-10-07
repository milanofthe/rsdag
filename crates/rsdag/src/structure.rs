//! The structure a parameter binding decides: a system reduced by what it
//! fixes.
//!
//! A frontend that lowers a model exactly keeps every branch on a
//! parameter. Some of those branches change the structure of the system:
//! a resistance that is a branch for `R > 0` and a short for `R = 0` is a
//! switch, an extra unknown (its current) and a row that is `V(a) - V(b)`
//! on one side of the condition and `i - flow` on the other. Within a
//! binding the condition is fixed, and so is the structure: the row of the
//! short is an alias, the current is a cut, and both leave the system.
//!
//! [`Structure`] analyses a system once; [`Structure::plan`] finds, at a
//! binding, the reductions it allows, by two exact rules:
//!
//! - **Alias.** A row without charge whose current, under the binding, is
//!   `c_a x_a + c_b x_b + g` (no other state; `c_a`, `c_b` and `g` free of
//!   the states and of time; `c_a` nonzero): `x_a` is
//!   `-(c_b x_b + g) / c_a`, and the row goes.
//! - **Cut.** A state that every row reads linearly, with a coefficient
//!   free of the states and of time, and no charge reads; a pivot row whose
//!   coefficient is nonzero. Every other row less its coefficient's share
//!   of the pivot row no longer reads the state; the pivot row and the
//!   state go. Only where that adds no more entries to the system than it
//!   takes away.
//!
//! The steps make a [`Plan`]. Equal plans reduce to the same system, which
//! [`Structure::reduce`] builds over the system's own expressions, its
//! branches kept: it is exact at every binding whose plan it is, so the
//! values a binding moves within one plan never ask for another reduction.
//! The reduced system says how to get every state of the full one back
//! ([`Reduced::expand`]) and which rows of the full one each of its rows
//! combines ([`Reduced::combination`]).
//!
//! Which states may go is the consumer's to say ([`System::eliminable`]):
//! one it limits, reports in a way the reduction cannot, or feeds a delay
//! with stays. Rows pair with states by index, and a step drops only the
//! row of an eliminable state: a state that stays keeps its own row.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::field::Field;
use crate::func::{FuncId, Output};
use crate::graph::{Graph, Join, Set};
use crate::node::{ArgList, ExprId, Node, SymbolId};
use crate::role::ParamRole;
use crate::tape::Tape;

/// A system in charge form: row `r` reads `currents[r] + d/dt charges[r]`.
pub struct System {
    /// The unknowns.
    pub states: Vec<SymbolId>,
    /// The independent variable, where the system reads it.
    pub time: Option<SymbolId>,
    /// The parameters, in the order a binding gives their values.
    pub params: Vec<SymbolId>,
    pub currents: Vec<ExprId>,
    pub charges: Vec<ExprId>,
    /// Per state, whether a reduction may eliminate it, and drop its row
    /// (row `k` is state `k`'s).
    pub eliminable: Vec<bool>,
}

/// One step of a [`Plan`], by full row and state indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    /// Row `row` makes `elim` a function of `keep` (or of the parameters
    /// alone).
    Alias {
        row: u32,
        elim: u32,
        keep: Option<u32>,
    },
    /// Row `row` is the pivot that eliminates `elim` from every other row.
    Cut { row: u32, elim: u32 },
}

/// The reductions a binding allows, in order: a key for the reduced
/// system (see the module docs).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Plan {
    pub steps: Vec<Step>,
}

/// A system reduced by a [`Plan`].
pub struct Reduced {
    /// The states kept, as indices of the full system's, ascending.
    pub states: Vec<usize>,
    /// Per reduced row, the full row it continues; reduced row `k` pairs
    /// with state `states[k]`, that state's own row where it survives.
    pub rows: Vec<usize>,
    pub currents: Vec<ExprId>,
    pub charges: Vec<ExprId>,
    /// Per reduced row, the full rows it sums and their coefficients:
    /// expressions free of the states at every binding of the plan.
    pub combination: Vec<Vec<(usize, ExprId)>>,
    /// Per full state, its value: over the kept states, their rates
    /// ([`rates`](Self::rates)), the parameters and time.
    pub expand: Vec<ExprId>,
    /// Per kept state, the symbol of its time derivative in `expand` (only
    /// a state a cut eliminated reads one, through the charge of its pivot
    /// row).
    pub rates: Vec<SymbolId>,
    /// The expressions [`Structure::reduce`] was asked to carry along,
    /// over the kept states.
    pub others: Vec<ExprId>,
}

/// A function's select conditions over its pure operands: its `Param`
/// parameters, and the globals its body reads that are the system's
/// parameters.
struct Conds {
    /// Per pattern position, the condition.
    index: HashMap<ExprId, u32>,
    /// The conditions as a program over the operands.
    tape: Tape,
    /// Per operand, whether it is pure.
    pure: Vec<bool>,
    /// The operand each symbol of the body is.
    operand: HashMap<SymbolId, u32>,
    /// Per pattern, the operands each node of the body reads that are not
    /// pure, the conditions decided by the pattern.
    reads: Mutex<HashMap<Box<[bool]>, HashMap<ExprId, Set>>>,
}

/// A call of a function with conditions: where its pure operands are.
struct Site {
    f: FuncId,
    /// Per operand, where its value is among `Structure::pure_values`'s
    /// outputs; `None` for an operand that is not pure.
    at: Vec<Option<u32>>,
}

/// A system analysed for its structure (see the module docs).
pub struct Structure {
    n_states: usize,
    states: Vec<SymbolId>,
    state_of: HashMap<SymbolId, u32>,
    time: Option<SymbolId>,
    params: Vec<SymbolId>,
    eliminable: Vec<bool>,
    currents: Vec<ExprId>,
    charges: Vec<ExprId>,
    /// `d currents / d states` and `d charges / d states`, by row.
    jac_i: crate::autodiff::SparseRows,
    jac_q: crate::autodiff::SparseRows,
    funcs: HashMap<FuncId, Conds>,
    sites: Vec<Site>,
    site_of: HashMap<(FuncId, u32, ArgList), u32>,
    /// The sites' pure operands, as a program over the parameters.
    pure_values: Tape,
    /// The conditions of the system's own selects that read parameters
    /// only, by position, and as a program over the parameters.
    top: HashMap<ExprId, u32>,
    top_tape: Tape,
}

/// What a binding decides of the system: per row, the states (and time, as
/// state `n_states`) its current and its charge read, and per Jacobian
/// entry whether it is free of them and, if so, its value.
struct Facts {
    cur: Vec<BTreeSet<u32>>,
    chg: Vec<BTreeSet<u32>>,
    /// Per row, its current's entries: column, free of the states, value.
    gi: Vec<BTreeMap<u32, (bool, f64)>>,
    /// Per row, its charge's entries: column, free of the states.
    gq: Vec<BTreeMap<u32, bool>>,
}

impl Structure {
    /// Analyse `sys`: its Jacobians, and the conditions on parameters of
    /// its own selects and of the functions it calls.
    pub fn new<K: Field>(ctx: &mut Graph<K>, sys: &System) -> Structure {
        let n = sys.states.len();
        assert_eq!(sys.currents.len(), sys.charges.len(), "a charge per row");
        assert_eq!(sys.currents.len(), n, "a row per state");
        assert_eq!(sys.eliminable.len(), n, "a flag per state");
        let jac_i = crate::sparse_jacobian(ctx, &sys.currents, &sys.states);
        let jac_q = crate::sparse_jacobian(ctx, &sys.charges, &sys.states);
        let params: HashSet<SymbolId> = sys.params.iter().copied().collect();
        let roots = roots_of(&sys.currents, &sys.charges, &jac_i, &jac_q);
        let mut funcs: HashMap<FuncId, Option<Conds>> = HashMap::default();
        let (mut sites, mut site_of) = (Vec::new(), HashMap::default());
        let (mut top, mut top_conds) = (HashMap::default(), Vec::new());
        let (mut pure_roots, mut pure_at) = (Vec::new(), HashMap::default());
        for e in ctx.cone_sorted(&roots, true) {
            match *ctx.node(e) {
                Node::Select(c, _, _) if !top.contains_key(&c) => {
                    if reads_only(ctx, c, &params) {
                        top.insert(c, top_conds.len() as u32);
                        top_conds.push(c);
                    }
                }
                Node::Call(o, l) => {
                    let (f, _) = ctx.output(o);
                    let key = (f, ctx.context_of(o), l);
                    if site_of.contains_key(&key) {
                        continue;
                    }
                    let Some(conds) = funcs.entry(f).or_insert_with(|| Conds::of(ctx, f, &params))
                    else {
                        continue;
                    };
                    let mut operands = ctx.full_args(o, l).into_owned();
                    operands.extend(ctx.globals(f).iter().copied());
                    let at = (operands.iter().zip(&conds.pure))
                        .map(|(&a, &pure)| {
                            pure.then(|| {
                                *pure_at.entry(a).or_insert_with(|| {
                                    pure_roots.push(a);
                                    pure_roots.len() as u32 - 1
                                })
                            })
                        })
                        .collect();
                    site_of.insert(key, sites.len() as u32);
                    sites.push(Site { f, at });
                }
                _ => {}
            }
        }
        Structure {
            n_states: n,
            states: sys.states.clone(),
            state_of: sys
                .states
                .iter()
                .enumerate()
                .map(|(k, &s)| (s, k as u32))
                .collect(),
            time: sys.time,
            params: sys.params.clone(),
            eliminable: sys.eliminable.clone(),
            currents: sys.currents.clone(),
            charges: sys.charges.clone(),
            jac_i,
            jac_q,
            funcs: funcs
                .into_iter()
                .filter_map(|(f, c)| Some((f, c?)))
                .collect(),
            sites,
            site_of,
            pure_values: Tape::compile(ctx, &pure_roots, &sys.params),
            top,
            top_tape: Tape::compile(ctx, &top_conds, &sys.params),
        }
    }

    /// The reductions the binding `p` (one value per parameter) allows.
    pub fn plan<K: Field>(&self, ctx: &Graph<K>, p: &[f64]) -> Plan {
        assert_eq!(p.len(), self.params.len(), "a value per parameter");
        plan_of(self.facts(ctx, p), &self.eliminable, self.n_states)
    }

    /// Reduce the system by `plan` (one [`plan`](Self::plan) found),
    /// carrying `others` along.
    pub fn reduce<K: Field>(&self, ctx: &mut Graph<K>, plan: &Plan, others: &[ExprId]) -> Reduced {
        reduce(self, ctx, plan, others)
    }

    /// Per site its pattern at `p`, and per top-level condition its truth.
    fn patterns(&self, p: &[f64]) -> (Vec<Box<[bool]>>, Vec<bool>) {
        let (mut w, mut vals, mut top) = (Vec::new(), Vec::new(), Vec::new());
        self.pure_values.eval(p, &mut w, &mut vals);
        self.top_tape.eval(p, &mut w, &mut top);
        // The instances of one function with the same pure operands (a
        // card's) share a pattern.
        let mut seen: HashMap<(FuncId, Vec<u64>), Box<[bool]>> = HashMap::default();
        let (mut args, mut out) = (Vec::new(), Vec::new());
        let pats = (self.sites.iter())
            .map(|s| {
                let key =
                    s.at.iter()
                        .flatten()
                        .map(|&k| vals[k as usize].to_bits())
                        .collect();
                seen.entry((s.f, key))
                    .or_insert_with(|| {
                        args.clear();
                        args.extend(
                            s.at.iter()
                                .map(|a| a.map_or(f64::NAN, |k| vals[k as usize])),
                        );
                        self.funcs[&s.f].tape.eval(&args, &mut w, &mut out);
                        out.iter().map(|&c| c != 0.0).collect()
                    })
                    .clone()
            })
            .collect();
        (pats, top.iter().map(|&c| c != 0.0).collect())
    }

    /// What `p` decides of the system (see [`Facts`]).
    fn facts<K: Field>(&self, ctx: &Graph<K>, p: &[f64]) -> Facts {
        let (pats, top) = self.patterns(p);
        let roots = roots_of(&self.currents, &self.charges, &self.jac_i, &self.jac_q);
        let reads = self.reads(ctx, &roots, &pats, &top);
        let set = |e: ExprId| -> BTreeSet<u32> { reads[&e].iter().collect() };
        let free = |e: ExprId| reads[&e].is_bottom();
        // The values of the current entries free of the states, at `p`.
        let wanted: Vec<ExprId> = (self.jac_i.iter().flatten())
            .map(|&(_, e)| e)
            .filter(|&e| free(e))
            .collect();
        let env: std::collections::HashMap<SymbolId, f64> =
            self.params.iter().copied().zip(p.iter().copied()).collect();
        let values = crate::eval::<f64, K>(ctx, &wanted, &env);
        let vals: HashMap<ExprId, f64> = wanted.into_iter().zip(values).collect();
        Facts {
            cur: self.currents.iter().map(|&e| set(e)).collect(),
            chg: self.charges.iter().map(|&e| set(e)).collect(),
            gi: (self.jac_i.iter())
                .map(|row| {
                    let v = |e: ExprId| (free(e), vals.get(&e).copied().unwrap_or(f64::NAN));
                    row.iter().map(|&(c, e)| (c as u32, v(e))).collect()
                })
                .collect(),
            gq: (self.jac_q.iter())
                .map(|row| row.iter().map(|&(c, e)| (c as u32, free(e))).collect())
                .collect(),
        }
    }

    /// Per node under `roots`, the states (and time, as `n_states`) it
    /// reads, a select whose condition the binding decides reading its arm:
    /// `pats` per site, `top` per top-level condition.
    fn reads<K: Field>(
        &self,
        ctx: &Graph<K>,
        roots: &[ExprId],
        pats: &[Box<[bool]>],
        top: &[bool],
    ) -> HashMap<ExprId, Set> {
        let n = self.n_states + 1;
        let mut at: HashMap<ExprId, Set> = HashMap::default();
        for e in ctx.cone_sorted(roots, true) {
            let v = match *ctx.node(e) {
                Node::Symbol(s) => match self.state_of.get(&s) {
                    Some(&k) => Set::one(k, n),
                    None if Some(s) == self.time => Set::one(self.n_states as u32, n),
                    None => Set::bottom(),
                },
                Node::Select(c, a, b) if self.top.contains_key(&c) => {
                    let arm = if top[self.top[&c] as usize] { a } else { b };
                    at.get(&arm).cloned().unwrap_or_default()
                }
                Node::Call(o, l) => {
                    let (f, out) = ctx.output(o);
                    let mut operands = ctx.full_args(o, l).into_owned();
                    operands.extend(ctx.globals(f).iter().copied());
                    let site = self.site_of.get(&(f, ctx.context_of(o), l));
                    let read: Vec<u32> = match (site, ctx.func(f).outputs()[out as usize]) {
                        (_, Output::Zero) => Vec::new(),
                        (Some(&s), Output::Expr(body)) => {
                            self.funcs[&f].reads(ctx, body, &pats[s as usize])
                        }
                        // An extern body, or a function without conditions:
                        // whatever it is passed.
                        _ => (0..operands.len() as u32).collect(),
                    };
                    Set::join_all(read.iter().filter_map(|&k| at.get(&operands[k as usize])))
                }
                _ => {
                    let ops = ctx.operands(e);
                    Set::join_all(ops.iter().filter_map(|x| at.get(x)))
                }
            };
            at.insert(e, v);
        }
        at
    }
}

/// The expressions a system's facts are about: its rows and their
/// Jacobians.
fn roots_of(
    currents: &[ExprId],
    charges: &[ExprId],
    jac_i: &crate::autodiff::SparseRows,
    jac_q: &crate::autodiff::SparseRows,
) -> Vec<ExprId> {
    (currents.iter().chain(charges).copied())
        .chain(jac_i.iter().flatten().map(|&(_, e)| e))
        .chain(jac_q.iter().flatten().map(|&(_, e)| e))
        .collect()
}

/// Whether `e` reads symbols, and those of `syms` only.
fn reads_only<K: Field>(ctx: &Graph<K>, e: ExprId, syms: &HashSet<SymbolId>) -> bool {
    let free = ctx.free_symbols_in(&[e]);
    !free.is_empty() && free.iter().all(|s| syms.contains(s))
}

impl Conds {
    /// The conditions of `f`'s selects that read its pure operands only;
    /// `None` without any, or for an extern function.
    fn of<K: Field>(ctx: &Graph<K>, f: FuncId, params: &HashSet<SymbolId>) -> Option<Conds> {
        let func = ctx.func(f);
        let mut syms: Vec<SymbolId> = func.params().to_vec();
        let mut pure: Vec<bool> = func
            .param_roles()
            .iter()
            .map(|r| matches!(r, ParamRole::Param))
            .collect();
        for &g in ctx.globals(f).iter() {
            let Node::Symbol(s) = *ctx.node(g) else {
                unreachable!("a global is a symbol")
            };
            syms.push(s);
            pure.push(params.contains(&s));
        }
        let pure_syms: HashSet<SymbolId> = syms
            .iter()
            .zip(&pure)
            .filter(|(_, &p)| p)
            .map(|(&s, _)| s)
            .collect();
        let outs: Vec<ExprId> = (func.outputs().iter())
            .filter_map(|o| match *o {
                Output::Expr(e) => Some(e),
                _ => None,
            })
            .collect();
        let (mut conds, mut index) = (Vec::new(), HashMap::default());
        for e in ctx.cone_sorted(&outs, true) {
            if let Node::Select(c, _, _) = *ctx.node(e) {
                if !index.contains_key(&c) && reads_only(ctx, c, &pure_syms) {
                    index.insert(c, conds.len() as u32);
                    conds.push(c);
                }
            }
        }
        if conds.is_empty() {
            return None;
        }
        Some(Conds {
            tape: Tape::compile(ctx, &conds, &syms),
            index,
            pure,
            operand: syms
                .iter()
                .enumerate()
                .map(|(k, &s)| (s, k as u32))
                .collect(),
            reads: Mutex::default(),
        })
    }

    /// The operands that `body` (an output of the function) reads and that
    /// are not pure, its conditions decided by `pattern`.
    fn reads<K: Field>(&self, ctx: &Graph<K>, body: ExprId, pattern: &[bool]) -> Vec<u32> {
        let mut memo = self.reads.lock().unwrap();
        let at = memo.entry(pattern.into()).or_default();
        if !at.contains_key(&body) {
            let n = self.pure.len();
            for e in ctx.cone_sorted(&[body], true) {
                if at.contains_key(&e) {
                    continue;
                }
                let v = match *ctx.node(e) {
                    Node::Symbol(s) => match self.operand.get(&s) {
                        Some(&k) if !self.pure[k as usize] => Set::one(k, n),
                        _ => Set::bottom(),
                    },
                    Node::Select(c, a, b) if self.index.contains_key(&c) => {
                        let arm = if pattern[self.index[&c] as usize] {
                            a
                        } else {
                            b
                        };
                        at.get(&arm).cloned().unwrap_or_default()
                    }
                    // A call in the body: whatever it is passed and the
                    // globals it reads, its own conditions undecided.
                    Node::Call(o, _) => {
                        let ops = ctx.operands(e);
                        let globals = ctx.globals(ctx.output(o).0);
                        let mut v = Set::join_all(ops.iter().filter_map(|x| at.get(x)));
                        for &g in globals.iter() {
                            if let Node::Symbol(s) = *ctx.node(g) {
                                if let Some(&k) = self.operand.get(&s) {
                                    if !self.pure[k as usize] {
                                        v.join(&Set::one(k, n));
                                    }
                                }
                            }
                        }
                        v
                    }
                    _ => {
                        let ops = ctx.operands(e);
                        Set::join_all(ops.iter().filter_map(|x| at.get(x)))
                    }
                };
                at.insert(e, v);
            }
        }
        at[&body].iter().collect()
    }
}

/// The steps the facts allow (see the module docs), deterministic in them.
fn plan_of(mut f: Facts, eliminable: &[bool], n: usize) -> Plan {
    let time = n as u32;
    let mut alive_row = vec![true; f.cur.len()];
    let mut alive_state = vec![true; n];
    // The states the charges of the cuts' pivot rows read: their rates make
    // the cut states' values, so no later cut may take one.
    let mut pivot_chg: BTreeSet<u32> = BTreeSet::new();
    let mut steps = Vec::new();
    let usable = |v: f64| v.is_finite() && v != 0.0;
    loop {
        // Aliases, row by row.
        let mut found = None;
        for r in (0..f.cur.len()).filter(|&r| alive_row[r] && eliminable[r]) {
            let cur = &f.cur[r];
            if !f.chg[r].is_empty() || cur.contains(&time) || cur.is_empty() || cur.len() > 2 {
                continue;
            }
            let entry = |s: u32| f.gi[r].get(&s).copied().filter(|&(free, _)| free);
            let Some(cs) = cur.iter().map(|&s| entry(s)).collect::<Option<Vec<_>>>() else {
                continue;
            };
            // The state to eliminate: the last eliminable one with a
            // nonzero coefficient.
            let vars: Vec<u32> = cur.iter().copied().collect();
            let Some(k) = (0..vars.len())
                .rev()
                .find(|&k| eliminable[vars[k] as usize] && usable(cs[k].1))
            else {
                continue;
            };
            let keep = (vars.len() == 2).then(|| (vars[1 - k], cs[1 - k].1));
            found = Some((r, vars[k], cs[k].1, keep));
            break;
        }
        if let Some((r, a, ca, keep)) = found {
            alive_row[r] = false;
            alive_state[a as usize] = false;
            steps.push(Step::Alias {
                row: r as u32,
                elim: a,
                keep: keep.map(|(b, _)| b),
            });
            // `x_a = rho x_b + offset` in every other row.
            let rho = keep.map(|(_, cb)| -cb / ca);
            for s in (0..f.cur.len()).filter(|&s| alive_row[s]) {
                if f.cur[s].remove(&a) {
                    let (free_a, va) = f.gi[s].remove(&a).unwrap_or((false, f64::NAN));
                    if let (Some((b, _)), Some(rho)) = (keep, rho) {
                        f.cur[s].insert(b);
                        let e = f.gi[s].entry(b).or_insert((true, 0.0));
                        *e = (e.0 && free_a, e.1 + va * rho);
                    }
                }
                if f.chg[s].remove(&a) {
                    let free_a = f.gq[s].remove(&a).unwrap_or(false);
                    if let Some((b, _)) = keep {
                        f.chg[s].insert(b);
                        let e = f.gq[s].entry(b).or_insert(true);
                        *e = *e && free_a;
                    }
                }
            }
            if pivot_chg.remove(&a) {
                if let Some((b, _)) = keep {
                    pivot_chg.insert(b);
                }
            }
            continue;
        }
        // A cut, the one that fills least.
        let mut best: Option<(i64, u32, usize)> = None;
        for i in (0..n as u32).filter(|&i| alive_state[i as usize] && eliminable[i as usize]) {
            if pivot_chg.contains(&i) {
                continue;
            }
            let rows: Vec<usize> = (0..f.cur.len())
                .filter(|&s| alive_row[s] && f.cur[s].contains(&i))
                .collect();
            let linear = rows
                .iter()
                .all(|&s| f.gi[s].get(&i).is_some_and(|&(free, _)| free));
            let in_charge = (0..f.chg.len()).any(|s| alive_row[s] && f.chg[s].contains(&i));
            if rows.is_empty() || !linear || in_charge {
                continue;
            }
            let count = |set: &BTreeSet<u32>| set.iter().filter(|&&c| c != time).count() as i64;
            for &r in &rows {
                if !eliminable[r] || !usable(f.gi[r][&i].1) {
                    continue;
                }
                let added: i64 = (rows.iter().filter(|&&s| s != r))
                    .map(|&s| {
                        let cur = f.cur[r].iter().filter(|&&c| c != i && c != time);
                        let chg = f.chg[r].iter().filter(|&&c| c != time);
                        cur.filter(|c| !f.cur[s].contains(c)).count() as i64
                            + chg.filter(|c| !f.chg[s].contains(c)).count() as i64
                    })
                    .sum();
                let removed = count(&f.cur[r]) + count(&f.chg[r]) + rows.len() as i64 - 1;
                let fill = added - removed;
                if fill <= 0 && best.is_none_or(|(bf, bi, br)| (fill, i, r) < (bf, bi, br)) {
                    best = Some((fill, i, r));
                }
            }
        }
        let Some((_, i, r)) = best else {
            break;
        };
        alive_row[r] = false;
        alive_state[i as usize] = false;
        steps.push(Step::Cut {
            row: r as u32,
            elim: i,
        });
        let vr = f.gi[r][&i].1;
        let (cur_r, chg_r) = (f.cur[r].clone(), f.chg[r].clone());
        let (gi_r, gq_r) = (f.gi[r].clone(), f.gq[r].clone());
        let combined: Vec<usize> = (0..f.cur.len())
            .filter(|&s| alive_row[s] && f.cur[s].contains(&i))
            .collect();
        for s in combined {
            let k = f.gi[s][&i].1 / vr;
            f.cur[s].extend(cur_r.iter().copied());
            f.cur[s].remove(&i);
            f.gi[s].remove(&i);
            for (&c, &(free_rc, vrc)) in gi_r.iter().filter(|(&c, _)| c != i) {
                let e = f.gi[s].entry(c).or_insert((true, 0.0));
                *e = (e.0 && free_rc, e.1 - k * vrc);
            }
            f.chg[s].extend(chg_r.iter().copied());
            for (&c, &free_rc) in &gq_r {
                let e = f.gq[s].entry(c).or_insert(true);
                *e = *e && free_rc;
            }
        }
        pivot_chg.extend(chg_r.iter().copied().filter(|&c| c != time));
    }
    Plan { steps }
}

/// The value of an eliminated state, over the states alive when it went.
enum Def {
    /// An alias: its value.
    Value(ExprId),
    /// A cut: `-(current + d/dt charge) / coefficient`, of its pivot row
    /// with the state zero.
    Cut {
        current: ExprId,
        charge: ExprId,
        coef: ExprId,
    },
}

/// [`Structure::reduce`].
fn reduce<K: Field>(st: &Structure, ctx: &mut Graph<K>, plan: &Plan, others: &[ExprId]) -> Reduced {
    let (n, sym) = (st.n_states, st.states.clone());
    let mut cur = st.currents.clone();
    let mut chg = st.charges.clone();
    let mut comb: Vec<BTreeMap<usize, ExprId>> = (0..n)
        .map(|r| {
            let one = ctx.one();
            BTreeMap::from([(r, one)])
        })
        .collect();
    // Per row, the states it may read (through every branch).
    let state_set = |ctx: &Graph<K>, es: &[ExprId]| -> BTreeSet<u32> {
        ctx.free_symbols_in(es)
            .iter()
            .filter_map(|s| st.state_of.get(s).copied())
            .collect()
    };
    let mut support: Vec<BTreeSet<u32>> =
        (0..n).map(|r| state_set(ctx, &[cur[r], chg[r]])).collect();
    let mut alive = vec![true; n];
    let mut alive_state = vec![true; n];
    let mut defs: Vec<(u32, Def)> = Vec::new();
    let mut steps = plan.steps.iter().peekable();
    while steps.peek().is_some() {
        // A round: steps that share no row and read nothing another one
        // eliminates, applied with one substitution.
        let (mut used, mut gone): (HashSet<usize>, HashSet<u32>) = Default::default();
        // What the rows take for the round's states (a cut's, zero: its
        // terms cancel), and the states those values read.
        let mut rowmap: HashMap<SymbolId, ExprId> = HashMap::default();
        let mut read: HashSet<u32> = HashSet::default();
        while let Some(&&step) = steps.peek() {
            let (row, elim) = match step {
                Step::Alias { row, elim, .. } | Step::Cut { row, elim } => (row as usize, elim),
            };
            let mut rows = vec![row];
            if let Step::Cut { .. } = step {
                rows.extend((0..n).filter(|&s| s != row && alive[s] && support[s].contains(&elim)));
            }
            let keep = match step {
                Step::Alias { keep, .. } => keep,
                Step::Cut { .. } => None,
            };
            let clash = rows.iter().any(|r| used.contains(r))
                || rows
                    .iter()
                    .any(|&r| support[r].iter().any(|s| gone.contains(s)))
                || keep.is_some_and(|b| gone.contains(&b))
                || read.contains(&elim);
            if clash && !used.is_empty() {
                break;
            }
            steps.next();
            used.extend(rows.iter().copied());
            gone.insert(elim);
            alive[row] = false;
            alive_state[elim as usize] = false;
            match step {
                Step::Alias { keep, .. } => {
                    let ca = crate::differentiate(ctx, cur[row], sym[elim as usize]);
                    let mut at_zero: HashMap<SymbolId, ExprId> = HashMap::default();
                    let z = ctx.zero();
                    at_zero.insert(sym[elim as usize], z);
                    let mut lin = z;
                    if let Some(b) = keep {
                        let cb = crate::differentiate(ctx, cur[row], sym[b as usize]);
                        at_zero.insert(sym[b as usize], z);
                        let xb = ctx.symbol_expr(sym[b as usize]);
                        lin = ctx.mul(cb, xb);
                    }
                    let g = crate::substitute(ctx, &[cur[row]], &at_zero)[0];
                    let sum = ctx.add(lin, g);
                    let neg = ctx.neg(sum);
                    let value = ctx.div(neg, ca);
                    read.extend(state_set(ctx, &[value]));
                    rowmap.insert(sym[elim as usize], value);
                    defs.push((elim, Def::Value(value)));
                }
                Step::Cut { .. } => {
                    let coef = crate::differentiate(ctx, cur[row], sym[elim as usize]);
                    for &s in &rows[1..] {
                        let js = crate::differentiate(ctx, cur[s], sym[elim as usize]);
                        let k = ctx.div(js, coef);
                        let ki = ctx.mul(k, cur[row]);
                        cur[s] = ctx.sub(cur[s], ki);
                        let kq = ctx.mul(k, chg[row]);
                        chg[s] = ctx.sub(chg[s], kq);
                        let pivot = comb[row].clone();
                        for (r, c) in pivot {
                            let kc = ctx.mul(k, c);
                            let prev = comb[s].get(&r).copied().unwrap_or_else(|| ctx.zero());
                            let v = ctx.sub(prev, kc);
                            comb[s].insert(r, v);
                        }
                        let add = support[row].clone();
                        support[s].extend(add);
                    }
                    let z = ctx.zero();
                    let at_zero = HashMap::from_iter([(sym[elim as usize], z)]);
                    let current = crate::substitute(ctx, &[cur[row]], &at_zero)[0];
                    rowmap.insert(sym[elim as usize], z);
                    defs.push((
                        elim,
                        Def::Cut {
                            current,
                            charge: chg[row],
                            coef,
                        },
                    ));
                }
            }
        }
        // The round's substitution in every row alive and in the
        // combinations' coefficients.
        let live: Vec<usize> = (0..n).filter(|&r| alive[r]).collect();
        let mut roots: Vec<ExprId> = live.iter().flat_map(|&r| [cur[r], chg[r]]).collect();
        let coefs: Vec<(usize, usize)> = live
            .iter()
            .flat_map(|&r| comb[r].keys().map(move |&c| (r, c)))
            .collect();
        roots.extend(coefs.iter().map(|&(r, c)| comb[r][&c]));
        let new = crate::substitute(ctx, &roots, &rowmap);
        for (j, &r) in live.iter().enumerate() {
            cur[r] = new[2 * j];
            chg[r] = new[2 * j + 1];
        }
        for (j, &(r, c)) in coefs.iter().enumerate() {
            comb[r].insert(c, new[2 * live.len() + j]);
        }
        for &r in &live {
            if support[r].iter().any(|s| gone.contains(s)) {
                support[r] = state_set(ctx, &[cur[r], chg[r]]);
            }
        }
    }
    // The kept states and their rates.
    let states: Vec<usize> = (0..n).filter(|&k| alive_state[k]).collect();
    let rates: Vec<SymbolId> = states
        .iter()
        .map(|&k| {
            let name = format!("d/dt {}", ctx.symbol_name(sym[k]));
            let e = ctx.sym(&name);
            match *ctx.node(e) {
                Node::Symbol(s) => s,
                _ => unreachable!("a symbol"),
            }
        })
        .collect();
    // The eliminated states' values, the last eliminated first, each over
    // the kept states once the later ones are in.
    let mut value: HashMap<SymbolId, ExprId> = HashMap::default();
    for (elim, def) in defs.into_iter().rev() {
        let v = match def {
            Def::Value(v) => crate::substitute(ctx, &[v], &value)[0],
            Def::Cut {
                current,
                charge,
                coef,
            } => {
                let r = crate::substitute(ctx, &[current, charge, coef], &value);
                // d/dt charge, through the kept states' rates and time.
                let mut terms = Vec::new();
                for (j, &k) in states.iter().enumerate() {
                    let d = crate::differentiate(ctx, r[1], sym[k]);
                    if ctx.const_f64(d) != Some(0.0) {
                        let rate = ctx.symbol_expr(rates[j]);
                        terms.push(ctx.mul(d, rate));
                    }
                }
                if let Some(t) = st.time {
                    terms.push(crate::differentiate(ctx, r[1], t));
                }
                let total = match terms.is_empty() {
                    true => r[0],
                    false => {
                        let dq = ctx.reduce(crate::node::ReduceOp::Sum, terms);
                        ctx.add(r[0], dq)
                    }
                };
                let neg = ctx.neg(total);
                ctx.div(neg, r[2])
            }
        };
        value.insert(sym[elim as usize], v);
    }
    let expand: Vec<ExprId> = (0..n)
        .map(|k| match value.get(&sym[k]) {
            Some(&v) => v,
            None => ctx.symbol_expr(sym[k]),
        })
        .collect();
    let others = crate::substitute(ctx, others, &value);
    // The rows, each paired with a kept state: its own where it survives.
    let mut rows: Vec<usize> = Vec::with_capacity(states.len());
    let mut spare: Vec<usize> = (0..n).filter(|&r| alive[r] && !alive_state[r]).collect();
    spare.reverse();
    for &k in &states {
        rows.push(if alive[k] {
            k
        } else {
            spare.pop().expect("a row per kept state")
        });
    }
    Reduced {
        currents: rows.iter().map(|&r| cur[r]).collect(),
        charges: rows.iter().map(|&r| chg[r]).collect(),
        combination: rows
            .iter()
            .map(|&r| comb[r].iter().map(|(&c, &e)| (c, e)).collect())
            .collect(),
        rows,
        states,
        expand,
        rates,
        others,
    }
}
