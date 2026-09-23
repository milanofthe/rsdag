//! A function's roles as the signature of the programs over it: one input
//! order, groups contiguous, parameters pure.

use rsdag::{Graph, Node, ParamRole, Signature, SymbolId, F64};

fn sym(g: &mut Graph<F64>, name: &str) -> SymbolId {
    let e = g.sym(name);
    let Node::Symbol(s) = *g.node(e) else {
        unreachable!()
    };
    s
}

#[test]
fn the_signature_orders_by_role_and_index() {
    let mut g: Graph<F64> = Graph::new();
    let names = ["p1", "x1", "t", "xd0", "h0", "x0", "p0", "u", "f"];
    let s: Vec<SymbolId> = names.iter().map(|n| sym(&mut g, n)).collect();
    let roles = [
        ParamRole::Param,
        ParamRole::State { id: 1 },
        ParamRole::Time,
        ParamRole::StateDot { id: 0 },
        ParamRole::History { id: 0 },
        ParamRole::State { id: 0 },
        ParamRole::Param,
        ParamRole::Input { port: 0, elem: 0 },
        ParamRole::Free,
    ];
    let x = g.sym("x0");
    let f = g.define_func("sys", s.clone(), vec![x]);
    for (i, r) in roles.iter().enumerate() {
        g.set_param_role(f, i as u32, *r);
    }
    let sig = Signature::of(g.func(f));
    let order: Vec<&str> = sig.syms.iter().map(|&s| g.symbol_name(s)).collect();
    // States by id, derivatives, inputs, parameters in declaration order,
    // time, histories, the rest.
    assert_eq!(order, ["x0", "x1", "xd0", "u", "p1", "p0", "t", "h0", "f"]);
    assert_eq!(sig.range(|r| matches!(r, ParamRole::State { .. })), 0..2);
    assert_eq!(sig.range(|r| matches!(r, ParamRole::Param)), 4..6);
    assert_eq!(
        sig.range(|r| matches!(r, ParamRole::Memory { .. })).len(),
        0
    );
    let pure: Vec<usize> = sig
        .pure_mask()
        .iter()
        .enumerate()
        .filter(|(_, &p)| p)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(pure, [4, 5]);
}
