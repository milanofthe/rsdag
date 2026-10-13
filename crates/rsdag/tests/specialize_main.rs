//! A main phase shortened against a trace of it (`Tape::specialize_main`)
//! runs over the full prolog's state: wherever its guards hold, its outputs
//! are the full main phase's, bit for bit; where they do not, the selects
//! it reports flipped are the ones that took the other arm. Its own prolog
//! is the full one, the state the same.

use rsdag::node::CmpOp;
use rsdag::{ExprId, Graph, Guards, Node, SymbolId, Tape, F64};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    (e, s)
}

/// Selects on the varying inputs and on prolog values, nested, sharing
/// conditions, an arm a prolog value.
fn tape() -> Tape {
    let mut g: Graph<F64> = Graph::new();
    let (a, sa) = sym(&mut g, "a");
    let (b, sb) = sym(&mut g, "b");
    let (p, sp) = sym(&mut g, "p");
    let (q, sq) = sym(&mut g, "q");
    let pq = g.mul(p, q);
    let e = g.exp(pq);
    let ab = g.mul(a, b);
    let x = g.add(ab, e);
    let c1 = g.cmp(CmpOp::Lt, a, q);
    let t1 = g.mul(b, pq);
    let t1 = g.add(t1, x);
    let sx = g.sin(x);
    let e1 = g.mul(sx, e);
    let s1 = g.select(c1, t1, e1);
    let zero = g.zero();
    let c2 = g.cmp(CmpOp::Gt, x, zero);
    let sq1 = g.mul(s1, s1);
    let ns1 = g.neg(s1);
    let s2 = g.select(c2, sq1, ns1);
    let c3 = g.cmp(CmpOp::Gt, b, p);
    let eb = g.exp(b);
    let half = g.ratio(1, 2);
    let a2 = g.mul(a, a);
    let a2 = g.add(a2, half);
    let la = g.ln(a2);
    let s3 = g.select(c3, eb, la);
    // the same condition again, an arm a prolog value
    let s4 = g.select(c1, e, s3);
    let inner = g.select(c3, s1, x);
    let s5 = g.select(c2, inner, s4);
    let o1 = g.add(s2, s3);
    let o2 = g.mul(s4, s5);
    // a condition that is a value, not a comparison (a NaN takes `then`)
    let raw = g.add(a, b);
    let s6 = g.select(raw, s2, e1);
    Tape::compile_split(
        &g,
        &[o1, o2, s1, x, s6],
        &[sa, sb, sp, sq],
        &[false, false, true, true],
    )
}

fn rng(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// The full main phase at `x` over `w`, and its trace.
fn full(t: &Tape, x: &[f64], w: &mut [f64]) -> (Vec<f64>, Vec<u8>) {
    let mut out = vec![0.0; t.out_len()];
    let mut trace: Vec<u8> = Vec::new();
    t.eval_main_with(x, w, &mut out, &mut trace);
    (out, trace)
}

#[test]
fn a_shortened_main_phase_is_the_full_one_where_its_guards_hold() {
    let t = tape();
    let n = t.n_main_selects();
    assert!(n >= 5, "the main phase keeps its selects ({n})");
    let sl = t.state_len();
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let (mut held, mut flipped) = (0, 0);
    for _ in 0..200 {
        // a binding, and a trace at one point of it
        let (p, q) = (rng(&mut s), rng(&mut s));
        let x0 = [rng(&mut s), rng(&mut s), p, q];
        let mut w = Vec::new();
        t.eval_prolog(&x0, &mut w);
        let (_, trace) = full(&t, &x0, &mut w);
        assert_eq!(trace.len(), n);
        let sp = t.specialize_main(&trace, &vec![true; n], Guards::Each);
        let counted = t.specialize_main(&trace, &vec![true; n], Guards::Count);
        assert_eq!(counted.n_guards(), 1);
        let st = sp.tape();
        assert!(sp.n_ops() < t.n_ops(), "the trace shortens the tape");
        assert_eq!(st.state_len(), sl, "the state is kept");
        // its prolog is the full one
        let mut ws = Vec::new();
        st.eval_prolog(&x0, &mut ws);
        assert!(same(&ws[..sl], &w[..sl]), "the same state");
        ws.resize(st.work_len().max(ws.len()), 0.0);
        // the main phase at other points of the binding, over the full
        // prolog's state
        for _ in 0..20 {
            let mut x = [rng(&mut s), rng(&mut s), p, q];
            if rng(&mut s) > 1.8 {
                x[(s % 2) as usize] = f64::NAN;
            }
            let (want, now) = full(&t, &x, &mut w);
            ws[..sl].copy_from_slice(&w[..sl]);
            let mut got = vec![0.0; st.out_len()];
            st.eval_main_into(&x, &mut ws, &mut got);
            let (real, guards) = got.split_at(sp.n_real());
            assert_eq!(guards.len(), sp.n_guards());
            let flips: Vec<usize> = sp.flipped(guards).collect();
            // the counted guard holds where every guard does
            let ct = counted.tape();
            let mut wc = vec![0.0; ct.work_len()];
            wc[..sl].copy_from_slice(&w[..sl]);
            let mut gc = vec![0.0; ct.out_len()];
            ct.eval_main_into(&x, &mut wc, &mut gc);
            let (real_c, count) = gc.split_at(counted.n_real());
            assert_eq!(counted.holds(count), sp.holds(guards), "count {count:?}");
            if counted.holds(count) {
                assert!(same(real_c, &want), "counted guard holds: the full outputs");
            }
            if sp.holds(guards) {
                held += 1;
                assert!(same(real, &want), "guards hold: the full outputs");
                assert!(flips.is_empty());
            } else {
                flipped += 1;
                assert!(!flips.is_empty(), "a failing guard names a select");
                for &k in &flips {
                    assert_ne!(trace[k], now[k], "select {k} reported flipped");
                }
            }
        }
    }
    assert!(
        held > 100 && flipped > 100,
        "both cases met: {held} held, {flipped} flipped"
    );
}

#[test]
fn unpinned_selects_stay_and_need_no_guard() {
    let t = tape();
    let n = t.n_main_selects();
    let sl = t.state_len();
    let x0 = [0.3, -0.7, 0.2, 0.9];
    let mut w = Vec::new();
    t.eval_prolog(&x0, &mut w);
    let (_, trace) = full(&t, &x0, &mut w);
    let none = t.specialize_main(&trace, &vec![false; n], Guards::Each);
    assert_eq!(none.n_guards(), 0);
    let mut s = 7u64;
    let mut ws = Vec::new();
    none.tape().eval_prolog(&x0, &mut ws);
    ws.resize(none.tape().work_len().max(ws.len()), 0.0);
    for _ in 0..100 {
        let x = [rng(&mut s), rng(&mut s), 0.2, 0.9];
        let (want, _) = full(&t, &x, &mut w);
        ws[..sl].copy_from_slice(&w[..sl]);
        let mut got = vec![0.0; none.tape().out_len()];
        none.tape().eval_main_into(&x, &mut ws, &mut got);
        assert!(
            same(&got, &want),
            "nothing pinned: the full tape everywhere"
        );
    }
}
