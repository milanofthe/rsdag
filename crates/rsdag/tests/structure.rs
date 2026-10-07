//! A system reduced by the structure a binding decides solves to the full
//! system's solution: a collapsible resistance (a switch row, a short for
//! `R = 0`) in a device body, its current and internal node eliminated or
//! kept as the binding says, the full states got back by `expand`, and a
//! reduction built at one binding exact at every other of its plan.

use rsdag::structure::{Plan, Structure, System};
use rsdag::{CmpOp, ExprId, Graph, Node, ParamRole, SymbolId, F64};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// A source `V` behind a conductance `Gs`, a capacitance `Ca` on node `a`,
/// and a device from `a` to ground: a collapsible resistance `R` from `a`
/// to its internal node `m` (its current `i`), a nonlinear branch
/// `tanh(m)` and a capacitance `C` from `m` to ground. States `a, m, i`,
/// parameters `V, R, C, Gs, Ca`; `m` and `i` may go.
struct Circuit {
    g: Graph<F64>,
    sys: System,
}

fn circuit() -> Circuit {
    circuits(1)
}

/// `n` of [`circuit`]'s, each on its own node with its own parameters, one
/// device function for all.
fn circuits(n: usize) -> Circuit {
    let mut g: Graph<F64> = Graph::new();
    let (fa, fas) = sym(&mut g, "cr.a");
    let (fm, fms) = sym(&mut g, "cr.m");
    let (fi, fis) = sym(&mut g, "cr.i");
    let (fr, frs) = sym(&mut g, "cr.R");
    let (fc, fcs) = sym(&mut g, "cr.C");
    let zero = g.zero();
    let t = g.tanh(fm);
    let ni = g.neg(fi);
    let o_m = g.add(ni, t);
    let q_m = g.mul(fc, fm);
    // the switch: `i - (a - m) / R` for `R > 0`, else the short `a - m`
    let pos = g.cmp(CmpOp::Gt, fr, zero);
    let d = g.sub(fa, fm);
    let flow = g.div(d, fr);
    let open = g.sub(fi, flow);
    let sw = g.select(pos, open, d);
    let f = g.define_func("cr", vec![fas, fms, fis, frs, fcs], vec![fi, o_m, q_m, sw]);
    g.set_param_role(f, 3, ParamRole::Param);
    g.set_param_role(f, 4, ParamRole::Param);
    let mut sys = System {
        states: Vec::new(),
        time: None,
        params: Vec::new(),
        currents: Vec::new(),
        charges: Vec::new(),
        eliminable: Vec::new(),
    };
    for k in 0..n {
        let s = |g: &mut Graph<F64>, name: &str| sym(g, &format!("{name}{k}"));
        let (a, as_) = s(&mut g, "a");
        let (m, ms) = s(&mut g, "m");
        let (i, is) = s(&mut g, "i");
        let (v, vs) = s(&mut g, "V");
        let (r, rs) = s(&mut g, "R");
        let (c, cs) = s(&mut g, "C");
        let (gs, gss) = s(&mut g, "Gs");
        let (ca, cas) = s(&mut g, "Ca");
        let args = [a, m, i, r, c];
        let into_a = g.call(f, 0, &args);
        let into_m = g.call(f, 1, &args);
        let charge_m = g.call(f, 2, &args);
        let switch = g.call(f, 3, &args);
        let av = g.sub(a, v);
        let source = g.mul(gs, av);
        let row_a = g.add(into_a, source);
        let charge_a = g.mul(ca, a);
        let z = g.zero();
        sys.states.extend([as_, ms, is]);
        sys.params.extend([vs, rs, cs, gss, cas]);
        sys.currents.extend([row_a, into_m, switch]);
        sys.charges.extend([charge_a, charge_m, z]);
        sys.eliminable.extend([false, true, true]);
    }
    Circuit { sys, g }
}

/// `exprs` at `env`.
fn eval(g: &Graph<F64>, exprs: &[ExprId], env: &[(SymbolId, f64)]) -> Vec<f64> {
    rsdag::eval::<f64, F64>(g, exprs, &env.iter().copied().collect())
}

/// A root of `f` from `x`, by Newton with a central-difference Jacobian.
fn solve(f: &dyn Fn(&[f64]) -> Vec<f64>, mut x: Vec<f64>) -> Vec<f64> {
    let n = x.len();
    for _ in 0..60 {
        let r = f(&x);
        if r.iter().all(|v| v.abs() < 1e-14) {
            break;
        }
        let mut jac = vec![vec![0.0; n + 1]; n];
        for k in 0..n {
            let h = 1e-6 * (1.0 + x[k].abs());
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            let (fp, fm) = (f(&xp), f(&xm));
            for row in 0..n {
                jac[row][k] = (fp[row] - fm[row]) / (2.0 * h);
            }
        }
        for row in 0..n {
            jac[row][n] = -r[row];
        }
        for col in 0..n {
            let p = (col..n)
                .max_by(|&a, &b| jac[a][col].abs().total_cmp(&jac[b][col].abs()))
                .unwrap();
            jac.swap(col, p);
            for row in col + 1..n {
                let k = jac[row][col] / jac[col][col];
                for c in col..=n {
                    jac[row][c] -= k * jac[col][c];
                }
            }
        }
        let mut dx = vec![0.0; n];
        for row in (0..n).rev() {
            let s: f64 = (row + 1..n).map(|c| jac[row][c] * dx[c]).sum();
            dx[row] = (jac[row][n] - s) / jac[row][row];
        }
        for k in 0..n {
            x[k] += dx[k];
        }
    }
    x
}

/// The full system's DC solution at `p`, and the reduced one's expanded.
fn both(c: &mut Circuit, st: &Structure, plan: &Plan, p: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let red = st.reduce(&mut c.g, plan, &[]);
    let g = &c.g;
    let params: Vec<(SymbolId, f64)> = c
        .sys
        .params
        .iter()
        .copied()
        .zip(p.iter().copied())
        .collect();
    let full = |x: &[f64]| {
        let mut env = params.clone();
        env.extend(c.sys.states.iter().copied().zip(x.iter().copied()));
        eval(g, &c.sys.currents, &env)
    };
    let x = solve(&full, vec![0.1; c.sys.states.len()]);
    let kept: Vec<SymbolId> = red.states.iter().map(|&k| c.sys.states[k]).collect();
    let at = |y: &[f64]| {
        let mut env = params.clone();
        env.extend(kept.iter().copied().zip(y.iter().copied()));
        env.extend(red.rates.iter().map(|&s| (s, 0.0)));
        env
    };
    let reduced = |y: &[f64]| eval(g, &red.currents, &at(y));
    let y = solve(&reduced, vec![0.1; kept.len()]);
    (x, eval(g, &red.expand, &at(&y)))
}

fn close(got: &[f64], want: &[f64], what: &str) {
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-9 * (1.0 + w.abs()),
            "{what}: state {k}: {g} vs {w}"
        );
    }
}

/// `V, R, C, Gs, Ca`.
const OPEN: [f64; 5] = [1.5, 2.0, 1e-3, 2.0, 3e-3];
const SHORT: [f64; 5] = [1.5, 0.0, 1e-3, 2.0, 3e-3];

#[test]
fn a_collapsible_resistance_reduces_by_its_binding() {
    let mut c = circuit();
    let st = Structure::new(&mut c.g, &c.sys);
    let (open, short) = (st.plan(&c.g, &OPEN), st.plan(&c.g, &SHORT));
    // Open: the current goes, the nonlinear node stays; short: both go.
    assert_eq!(st.reduce(&mut c.g, &open, &[]).states, vec![0, 1]);
    assert_eq!(st.reduce(&mut c.g, &short, &[]).states, vec![0]);
    for (plan, p) in [(&open, OPEN), (&short, SHORT)] {
        let (x, back) = both(&mut c, &st, plan, &p);
        close(&back, &x, &format!("{plan:?}"));
    }
    // A plan holds for every binding that finds it: one reduction serves
    // them all, exactly.
    let other = [0.7, 0.25, 2e-3, 0.5, 1e-3];
    assert_eq!(st.plan(&c.g, &other), open);
    let (x, back) = both(&mut c, &st, &open, &other);
    close(&back, &x, "another binding of the open plan");
}

/// Many instances, each shorted or open by its own binding: each reduces
/// by its own, in rounds that take many steps at once, and a plan found at
/// one binding serves another that finds it.
#[test]
fn instances_reduce_each_by_its_own_binding() {
    let n = 12;
    let mut c = circuits(n);
    let st = Structure::new(&mut c.g, &c.sys);
    let binding = |shorted: u32, scale: f64| -> Vec<f64> {
        (0..n)
            .flat_map(|k| {
                let r = if (shorted >> k) & 1 == 1 {
                    0.0
                } else {
                    1.0 + k as f64 * scale
                };
                [1.0 + 0.1 * k as f64, r, 1e-3, 2.0, 3e-3]
            })
            .collect()
    };
    let p = binding(0b1010_0110_1001, 0.5);
    let plan = st.plan(&c.g, &p);
    let red = st.reduce(&mut c.g, &plan, &[]);
    let shorted = (0..n)
        .filter(|k| (0b1010_0110_1001u32 >> k) & 1 == 1)
        .count();
    assert_eq!(
        red.states.len(),
        2 * n - shorted,
        "a state per node and open device"
    );
    let (x, back) = both(&mut c, &st, &plan, &p);
    close(&back, &x, "the instances");
    let q = binding(0b1010_0110_1001, 0.25);
    assert_eq!(st.plan(&c.g, &q), plan, "the values move within the plan");
    let (x, back) = both(&mut c, &st, &plan, &q);
    close(&back, &x, "another binding of the plan");
    assert_ne!(st.plan(&c.g, &binding(0b1, 0.5)), plan);
}

/// A cut current is its pivot row's balance, its charge's rate included:
/// shorted, `m` is `a`, and the current through the short is what `m`'s
/// node passes on, `i = tanh(a) + C da/dt`, which at a consistent point is
/// also what the source and `a`'s capacitance leave, `-(Gs (a - V) + Ca
/// da/dt)`.
#[test]
fn a_cut_current_carries_its_pivot_rows_charge_rate() {
    let mut c = circuit();
    let st = Structure::new(&mut c.g, &c.sys);
    let plan = st.plan(&c.g, &SHORT);
    let red = st.reduce(&mut c.g, &plan, &[]);
    assert_eq!(red.states, vec![0]);
    let [v, _, cap, gs, ca] = SHORT;
    let a = 0.3;
    // The one reduced row, `Gs (a - V) + tanh(a) + (Ca + C) da/dt = 0`.
    let rate = -(gs * (a - v) + a.tanh()) / (ca + cap);
    let mut env: Vec<(SymbolId, f64)> = c.sys.params.iter().copied().zip(SHORT).collect();
    env.extend([(c.sys.states[0], a), (red.rates[0], rate)]);
    let row = eval(&c.g, &red.currents, &env)[0] + (ca + cap) * rate;
    assert!(row.abs() < 1e-12, "a consistent point: {row}");
    let back = eval(&c.g, &red.expand, &env);
    assert_eq!(back[1], a, "m = a");
    let through_m = a.tanh() + cap * rate;
    let from_a = -(gs * (a - v) + ca * rate);
    assert!((through_m - from_a).abs() < 1e-12);
    assert!(
        (back[2] - through_m).abs() < 1e-12,
        "i: {} vs {through_m}",
        back[2]
    );
}
