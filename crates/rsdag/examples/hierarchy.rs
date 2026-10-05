//! One instance of a wide body, as a circuit's top-level subcircuit is: a
//! body over n nodes (a ring of n devices and a capacitor per node, the
//! current into each node an output) called once, every output its own call
//! over the one argument list. The cost of each stage a solver runs over it,
//! by n: building the calls, the Jacobian, the derivatives set to zero, the
//! calls specialized to that, and the program (`Tape::compose`: the body
//! inlined, its device calls batched). Each grows with the body, not with
//! its outputs times its width.
//!
//!     cargo run --release -p rsdag --example hierarchy

use std::time::Instant;

use rsdag::{sparse_jacobian, ExprId, Graph, Node, ParamRole, Scope, SymbolId, Tape, F64};

fn sym(g: &mut Graph<F64>, name: &str) -> (ExprId, SymbolId) {
    let e = g.sym(name);
    match *g.node(e) {
        Node::Symbol(s) => (e, s),
        _ => unreachable!(),
    }
}

fn main() {
    println!("n,build_ms,jacobian_ms,substitute_ms,specialize_ms,program_ms,nodes");
    for n in [1000usize, 2000, 4000, 8000, 16000, 32000] {
        let mut g: Graph<F64> = Graph::new();
        // d(va, vb, p) = p tanh(va - vb): the current from a to b
        let mut s = Scope::new(&mut g, "d");
        let va = s.param_with_role("va", ParamRole::State { id: 0 });
        let vb = s.param_with_role("vb", ParamRole::State { id: 1 });
        let p = s.param_with_role("p", ParamRole::Param);
        let dv = s.sub(va, vb);
        let th = s.tanh(dv);
        let i = s.mul(p, th);
        let d = s.close(vec![i]);
        // body(v, vdot, p, c): node k takes c_k vdot_k, the device from k to
        // k + 1 and the one from k - 1 to k
        let mut s = Scope::new(&mut g, "body");
        let v: Vec<ExprId> = (0..n)
            .map(|k| s.param_with_role(&format!("v{k}"), ParamRole::State { id: k as u32 }))
            .collect();
        let vd: Vec<ExprId> = (0..n)
            .map(|k| s.param_with_role(&format!("vd{k}"), ParamRole::StateDot { id: k as u32 }))
            .collect();
        let p: Vec<ExprId> = (0..n)
            .map(|k| s.param_with_role(&format!("p{k}"), ParamRole::Param))
            .collect();
        let c: Vec<ExprId> = (0..n)
            .map(|k| s.param_with_role(&format!("c{k}"), ParamRole::Param))
            .collect();
        let currents: Vec<ExprId> = (0..n)
            .map(|k| s.call(d, 0, &[v[k], v[(k + 1) % n], p[k]]))
            .collect();
        let outs: Vec<ExprId> = (0..n)
            .map(|k| {
                let q = s.mul(c[k], vd[k]);
                let out = s.add(q, currents[k]);
                s.sub(out, currents[(k + n - 1) % n])
            })
            .collect();
        let body = s.close(outs);

        let names = ["x", "xd", "p", "c"];
        let mut args = Vec::with_capacity(4 * n);
        let mut syms: Vec<Vec<SymbolId>> = vec![Vec::new(); 4];
        for (j, name) in names.iter().enumerate() {
            for k in 0..n {
                let (e, s) = sym(&mut g, &format!("{name}{k}"));
                args.push(e);
                syms[j].push(s);
            }
        }

        let t = Instant::now();
        let outs: Vec<u32> = (0..n as u32).collect();
        let rows = g.calls(body, &outs, &args);
        let build = t.elapsed();

        let t = Instant::now();
        let wrt: Vec<SymbolId> = syms[0].iter().chain(&syms[1]).copied().collect();
        let jac = sparse_jacobian(&mut g, &rows, &wrt);
        let jacobian = t.elapsed();
        let mut roots = rows.clone();
        roots.extend(jac.iter().flatten().map(|&(_, e)| e));

        let t = Instant::now();
        let zero = g.zero();
        let at_rest = syms[1].iter().map(|&s| (s, zero)).collect();
        let rest = rsdag::substitute(&mut g, &roots, &at_rest);
        let substitute = t.elapsed();

        let t = Instant::now();
        let spec = g.specialize_calls(&rest);
        let specialize = t.elapsed();

        let t = Instant::now();
        let inputs: Vec<SymbolId> = syms.iter().flatten().copied().collect();
        let tape = Tape::compose(&mut g, &spec, &inputs);
        let tape_t = t.elapsed();
        std::hint::black_box(&tape);

        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        println!(
            "{n},{:.2},{:.2},{:.2},{:.2},{:.2},{}",
            ms(build),
            ms(jacobian),
            ms(substitute),
            ms(specialize),
            ms(tape_t),
            g.len()
        );
    }
}
