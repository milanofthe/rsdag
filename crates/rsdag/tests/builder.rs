//! A model written once against `Builder` computes and records the same
//! thing: the numeric path and the recorded-then-compiled path agree to the
//! bit on the whole vocabulary.

use rsdag::synth::{inputs, Spec};
use rsdag::{Builder, Graph, Numeric, Scope, Tape, F64};

/// A model with a bit of everything: ring, elementary functions, a clamp,
/// a comparison, a reduction and a dot.
fn model<B: Builder>(b: &mut B, x: [B::N; 4], k: B::N) -> [B::N; 3] {
    let vt = b.cst(0.025);
    let is = b.cst(1e-14);
    let diode = {
        let q = b.div(x[0], vt);
        let e = b.exp(q);
        let one = b.cst(1.0);
        let m = b.sub(e, one);
        b.mul(is, m)
    };
    let clamp = {
        let v = b.mul(k, x[1]);
        let hi = b.cst(1.0);
        let lo = b.cst(-1.0);
        let over = b.gt(v, hi);
        let capped = b.select(over, hi, v);
        let under = b.lt(capped, lo);
        b.select(under, lo, capped)
    };
    let mix = {
        let s = b.sum(&[x[0], x[1], x[2], x[3]]);
        let d = b.dot(&[x[0], x[1]], &[x[2], x[3]]);
        let r = b.sqrt(d);
        let t = b.tanh(s);
        let a = b.atan2(r, t);
        let m = b.min(a, k);
        b.hypot(m, diode)
    };
    [diode, clamp, mix]
}

fn record(x: [f64; 4], k: f64) -> Vec<f64> {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "model");
    let xs = [s.param("x0"), s.param("x1"), s.param("x2"), s.param("x3")];
    let kk = s.param("k");
    let outs = model(&mut *s, xs, kk);
    let f = s.close(outs.to_vec());
    let params = g.func(f).params.clone();
    let tape = Tape::compile(&g, &outs, &params);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    tape.eval(&[x[0], x[1], x[2], x[3], k], &mut w, &mut o);
    o
}

#[test]
fn computing_and_recording_agree_to_the_bit() {
    let mut rng = Spec::new(3).rng();
    for _ in 0..200 {
        let v = inputs(&mut rng, 5);
        let x = [v[0], v[1], v[2], v[3]];
        let k = v[4];
        let computed = model(&mut Numeric, x, k);
        let recorded = record(x, k);
        for (i, (a, b)) in computed.iter().zip(&recorded).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "output {i} at {x:?}, k = {k}: computed {a:?}, recorded {b:?}"
            );
        }
    }
}

/// The recorded twin is a real graph: it differentiates.
#[test]
fn the_recorded_model_differentiates() {
    let mut g: Graph<F64> = Graph::new();
    let mut s = Scope::new(&mut g, "m");
    let xs = [s.param("x0"), s.param("x1"), s.param("x2"), s.param("x3")];
    let kk = s.param("k");
    let outs = model(&mut *s, xs, kk);
    let f = s.close(outs.to_vec());
    let params = g.func(f).params.clone();
    // d(diode)/dx0 = is/vt * exp(x0/vt), checked against a finite difference
    // of the numeric twin.
    let d = rsdag::differentiate(&mut g, outs[0], params[0]);
    let tape = Tape::compile(&g, &[d], &params);
    let (mut w, mut o) = (Vec::new(), Vec::new());
    let at = [0.6, 0.2, 0.3, 0.4, 2.0];
    tape.eval(&at, &mut w, &mut o);
    let h = 1e-7;
    let up = model(&mut Numeric, [0.6 + h, 0.2, 0.3, 0.4], 2.0)[0];
    let dn = model(&mut Numeric, [0.6 - h, 0.2, 0.3, 0.4], 2.0)[0];
    let fd = (up - dn) / (2.0 * h);
    assert!(
        (o[0] - fd).abs() <= 1e-6 * fd.abs(),
        "symbolic {} vs fd {fd}",
        o[0]
    );
}
