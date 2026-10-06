"""CasADi and JAX on the DAE modules rsdag's `modules` example measures.

    python docs/bench/modules.py <values dir> <module.json>...

A module (as SANE's `export_module` writes it) is rebuilt in each tool from
the rsdag graph, in the tool's idiom: every device body a function, CasADi
calling it with SX arguments (its SX virtual machine evaluates the result),
JAX mapping it over all instances with `vmap` and tracing the circuit
equations around the calls. The states are the inputs, the parameters are
constants, so a branch on a parameter is decided while the model is built
(CasADi also runs with the parameters as inputs, rsdag's default). The residual and its Jacobian are checked against rsdag's values
(`<values dir>/<name>.values.json` from the example) at `x0 + 0.01`, off
the solution where the residual is not zero, then timed at `x0`: the setup (differentiation and compilation; the tool's graph is
built first and not counted, as rsdag's comes from the exporter) and the
time per call, on one core: CasADi through its buffer interface (the
arguments bound once), JAX a jitted call from Python. CasADi builds the sparse Jacobian,
JAX the dense one (`jacfwd`). JAX runs per module in a child process under
a time and a memory limit (JAX_SECONDS, JAX_MEMORY); a module it does not
compile within them is reported with the limit it hit. One CSV row per
tool and module; needs casadi, jax and psutil.
"""
import json
import os
import struct
import sys
import time

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"] = "1"

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.setrecursionlimit(100000)

EXP_LIMIT = 80.0
LN_FLOOR = 1e-30
# JAX runs per module in a child process under these limits.
JAX_SECONDS = 60
JAX_MEMORY = 16 * 2**30


class Module:
    """The graph of a module JSON: nodes, constants, argument lists,
    functions, and the system function's inputs."""

    def __init__(self, path):
        d = json.load(open(path))
        m = d["module"]
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.nodes = m["nodes"]
        self.consts = [struct.unpack("<d", struct.pack("<Q", b))[0] for b in m["consts"]]
        self.pool = m["arg_pool"]
        self.symbols = m["symbols"]
        self.funcs = m["funcs"]
        self.call_outputs = m["call_outputs"]
        # per call output, the parameters its calls have bound and the list
        # of their expressions (a model card bound to its device function)
        self.contexts = m.get("call_contexts") or [None] * len(self.call_outputs)
        self.params = d["params"]
        self.x0 = np.array(d["x0"])
        f = self.funcs[d["circuit"]]
        self.states, self.bound = [], {}
        for s, role in zip(f["params"], f["param_roles"]):
            if isinstance(role, dict) and "State" in role:
                self.states.append((role["State"]["id"], s))
            elif role == "Param":
                self.bound[s] = self.params[self.symbols[s]]
            else:
                self.bound[s] = 0.0  # derivatives and time: the DC residual
        self.states.sort()
        self.residuals = [o["Expr"] for o, r in zip(f["outputs"], f["output_roles"])
                          if isinstance(r, dict) and "Residual" in r]
        # A symbol a body reads without it being a parameter is a global.
        for s, name in enumerate(self.symbols):
            if name in self.params:
                self.bound.setdefault(s, self.params[name])

    def args(self, lst):
        return self.pool[lst["start"]:lst["start"] + lst["len"]]

    def call_args(self, o, lst):
        """The operands of a call of output `o` over `lst` in its function's
        parameter order: the arguments, and where its context binds a
        parameter the bound expression."""
        args = self.args(lst)
        if self.contexts[o] is None:
            return args
        at, exprs = self.contexts[o]
        bound = dict(zip(at, self.args(exprs)))
        rest = iter(args)
        return [bound[p] if p in bound else next(rest) for p in range(len(args) + len(bound))]

    def operands(self, i):
        k, v = next(iter(self.nodes[i].items()))
        if k in ("Add", "Mul"):
            return v
        if k in ("Neg",):
            return [v]
        if k == "Pow":
            return [v[0]]
        if k == "Unary":
            return [v[1]]
        if k in ("Cmp", "Binary"):
            return v[1:]
        if k == "Select":
            return v
        if k == "Reduce":
            return self.args(v[1])
        if k == "Call":
            return self.call_args(v[0], v[1])
        return []

    def cone(self, roots):
        """The nodes `roots` read, ascending (a topological order); a call's
        body is not part of it."""
        seen, stack = set(), list(roots)
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(self.operands(i))
        return sorted(seen)


class Ops:
    """One tool's arithmetic with rsdag's semantics (its domain guards)."""

    def __init__(self, lib):
        self.lib = lib
        if lib == "casadi":
            self.where = lambda c, a, b: ca.if_else(c, a, b)
            self.exp, self.log, self.sqrt = ca.exp, ca.log, ca.sqrt
            self.atan, self.floor = ca.atan, ca.floor
            self.fmin, self.fmax = ca.fmin, ca.fmax
        else:
            self.where = jnp.where
            self.exp, self.log, self.sqrt = jnp.exp, jnp.log, jnp.sqrt
            self.atan, self.floor = jnp.arctan, jnp.floor
            self.fmin, self.fmax = jnp.minimum, jnp.maximum

    def const(self, x):
        """The value of `x` when it is a constant, else None: a model
        written for the tool decides a branch on a constant in Python."""
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        if self.lib == "casadi":
            return float(x) if isinstance(x, ca.SX) and x.is_constant() else None
        return None if isinstance(x, jax.core.Tracer) else float(x)

    def unary(self, op, x):
        c = self.const(x)
        if c is not None:
            return float(unary_np(op, c))
        w = self.where
        if op == "Exp":
            return w(x > EXP_LIMIT, np.exp(EXP_LIMIT) * (1.0 + (x - EXP_LIMIT)), self.exp(self.fmin(x, EXP_LIMIT)))
        if op == "Ln":
            return w(x <= LN_FLOOR, np.log(LN_FLOOR), self.log(self.fmax(x, LN_FLOOR)))
        if op == "Sqrt":
            return w(x <= 0.0, 0.0, self.sqrt(w(x <= 0.0, 1.0, x)))
        if op == "Atan":
            return self.atan(x)
        if op == "Floor":
            return self.floor(x)
        raise NotImplementedError(op)

    def cmp(self, op, a, b):
        ca_, cb = self.const(a), self.const(b)
        if ca_ is not None and cb is not None:
            a, b = ca_, cb
        r = {"Gt": a > b, "Ge": a >= b, "Lt": a < b, "Le": a <= b, "Eq": a == b, "Ne": a != b}[op]
        return float(r) if isinstance(r, bool) else self.where(r, 1.0, 0.0)


def unary_np(op, x):
    """rsdag's semantics of a unary op on a number."""
    if op == "Exp":
        return np.exp(EXP_LIMIT) * (1.0 + (x - EXP_LIMIT)) if x > EXP_LIMIT else np.exp(x)
    if op == "Ln":
        return np.log(LN_FLOOR) if x <= LN_FLOOR else np.log(x)
    if op == "Sqrt":
        return 0.0 if x <= 0.0 else np.sqrt(x)
    return {"Atan": np.arctan, "Floor": np.floor}[op](x)


def evaluate(mod, ops, roots, env, call):
    """The values of `roots` with the symbols bound by `env` (a dict, falling
    back to the module's bound constants); `call(func, out, args, node)`
    evaluates a call."""
    val = {}
    for i in mod.cone(roots):
        k, v = next(iter(mod.nodes[i].items()))
        if k == "Const":
            r = mod.consts[v]
        elif k == "Symbol":
            r = env[v] if v in env else mod.bound[v]
        elif k == "Add":
            r = val[v[0]] + val[v[1]]
        elif k == "Mul":
            r = val[v[0]] * val[v[1]]
        elif k == "Neg":
            r = -val[v]
        elif k == "Pow":
            r = val[v[0]] ** int(v[1])
        elif k == "Unary":
            r = ops.unary(v[0], val[v[1]])
        elif k == "Cmp":
            r = ops.cmp(v[0], val[v[1]], val[v[2]])
        elif k == "Select":
            c = ops.const(val[v[0]])
            if c is not None:
                r = val[v[1]] if c != 0.0 else val[v[2]]
            else:
                r = ops.where(val[v[0]] != 0.0, val[v[1]], val[v[2]])
        elif k == "Reduce":
            assert v[0] == "Sum", v[0]
            xs = [val[a] for a in mod.args(v[1])]
            r = xs[0]
            for x in xs[1:]:
                r = r + x
        elif k == "Call":
            f, out = mod.call_outputs[v[0]]
            r = call(f, out, [val[a] for a in mod.call_args(v[0], v[1])], i)
        else:
            raise NotImplementedError(k)
        val[i] = r
    return [val[r] for r in roots]


def body_outputs(mod, f):
    fd = mod.funcs[f]
    return fd["params"], [o["Expr"] if isinstance(o, dict) and "Expr" in o else None for o in fd["outputs"]]


# ---- CasADi -----------------------------------------------------------------

def casadi_model(mod, params_as_inputs=False):
    """The residual over SX states, and the SX parameters (empty unless
    `params_as_inputs`, where the parameters are inputs rather than
    constants: rsdag's condition, nothing decided at build time)."""
    ops = Ops("casadi")
    bound = mod.bound
    psyms = []
    if params_as_inputs:
        names = sorted(mod.params)
        by_name = {n: ca.SX.sym(n) for n in names}
        psyms = [by_name[n] for n in names]
        mod.bound = {s: by_name.get(mod.symbols[s], v) for s, v in bound.items()}
    bodies = {}

    def body(f):
        if f not in bodies:
            params, outs = body_outputs(mod, f)
            args = [ca.SX.sym(f"a{k}") for k in range(len(params))]
            roots = [o for o in outs if o is not None]
            vals = evaluate(mod, ops, roots, dict(zip(params, args)), call)
            it = iter(vals)
            res = [next(it) if o is not None else ca.SX(0) for o in outs]
            bodies[f] = ca.Function(f"f{f}", args, res)  # a name CasADi takes
        return bodies[f]

    def call(f, out, args, _node):
        return body(f)(*args)[out] if len(mod.funcs[f]["outputs"]) > 1 else body(f)(*args)

    x = ca.SX.sym("x", len(mod.states))
    env = {s: x[k] for k, (_, s) in enumerate(mod.states)}
    res = ca.vertcat(*evaluate(mod, ops, mod.residuals, env, call))
    mod.bound = bound
    return x, ca.vertcat(*psyms), res


# ---- JAX --------------------------------------------------------------------

def jax_model(mod):
    ops = Ops("jax")
    fns = {}

    def body(f, wanted):
        """The body of `f` as a function of its argument vector, the
        outputs `wanted` (a zero output as 0) and no others."""
        key = (f, tuple(wanted))
        if key not in fns:
            params, outs = body_outputs(mod, f)
            roots = [outs[o] for o in wanted if outs[o] is not None]

            def fn(argv):
                vals = evaluate(mod, ops, roots, {p: argv[k] for k, p in enumerate(params)}, direct)
                it = iter(vals)
                return jnp.stack([jnp.asarray(next(it), dtype=jnp.float64) if outs[o] is not None
                                  else jnp.float64(0.0) for o in wanted])

            fns[key] = fn
        return fns[key]

    def direct(f, out, args, _node):
        return body(f, [out])(jnp.stack([jnp.asarray(a, dtype=jnp.float64) for a in args]))[0]

    # The circuit's calls, grouped by body: one vmap per body.
    calls = {}
    for i in mod.cone(mod.residuals):
        n = mod.nodes[i]
        if "Call" in n:
            f, out = mod.call_outputs[n["Call"][0]]
            calls.setdefault(f, []).append((i, out, tuple(mod.call_args(n["Call"][0], n["Call"][1]))))

    def residual(x):
        env = {s: x[k] for k, (_, s) in enumerate(mod.states)}
        done = {}
        for f, sites in calls.items():
            lists = sorted({a for _, _, a in sites})
            index = {a: k for k, a in enumerate(lists)}
            flat = sorted({a for lst in lists for a in lst})
            vals = dict(zip(flat, evaluate(mod, ops, flat, env, direct)))
            argm = jnp.stack([jnp.stack([jnp.asarray(vals[a], dtype=jnp.float64) for a in lst]) for lst in lists])
            wanted = sorted({o for _, o, _ in sites})
            col = {o: k for k, o in enumerate(wanted)}
            out = jax.vmap(body(f, wanted))(argm)
            for node, o, lst in sites:
                done[node] = out[index[lst], col[o]]

        def call(f, o, args, node):
            return done[node]

        return jnp.stack([jnp.asarray(r, dtype=jnp.float64)
                          for r in evaluate(mod, ops, mod.residuals, env, call)])

    return residual


# ---- measuring --------------------------------------------------------------

def per_call(fn, x):
    reps = 1
    while True:
        t = time.perf_counter()
        for _ in range(reps):
            fn(x)
        dt = time.perf_counter() - t
        if dt > 0.05:
            break
        reps *= 4
    best = dt / reps
    for _ in range(4):
        t = time.perf_counter()
        for _ in range(reps):
            fn(x)
        best = min(best, (time.perf_counter() - t) / reps)
    return best


def check(name, tool, f, j, ref):
    fr = np.asarray(ref["f"])
    jr = np.zeros((len(fr), len(fr)))
    jr[ref["j_rows"], ref["j_cols"]] = ref["j"]
    scale_f = max(np.abs(fr).max(), 1e-300)
    scale_j = max(np.abs(jr).max(), 1e-300)
    ef = np.abs(np.asarray(f) - fr).max() / scale_f
    ej = np.abs(np.asarray(j) - jr).max() / scale_j
    assert ef < 1e-9 and ej < 1e-9, f"{name} {tool}: residual {ef:.2e}, Jacobian {ej:.2e}"


def casadi_row(mod, ref, params_as_inputs=False):
    x0 = mod.x0[[i for i, _ in mod.states]]
    x, p, res = casadi_model(mod, params_as_inputs)
    pv = [mod.params[n] for n in sorted(mod.params)] if params_as_inputs else []
    t = time.perf_counter()
    F0 = ca.Function("F", [x, p], [res])
    F0(x0, pv)
    sf = time.perf_counter() - t
    t = time.perf_counter()
    J0 = ca.Function("J", [x, p], [ca.jacobian(res, x)])
    J0(x0, pv)
    sj = time.perf_counter() - t
    xc = x0 + 0.01
    check(mod.name, "CasADi", np.array(F0(xc, pv)).ravel(), np.array(J0(xc, pv)), ref)
    return [sf, sj, buffer_call(F0, x0, pv), buffer_call(J0, x0, pv), "ok"]


def buffer_call(fn, x, p):
    """Seconds per numeric evaluation through CasADi's buffer interface: the
    arguments and results bound once, no conversion per call."""
    buf, trigger = fn.buffer()
    args = [np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(p, dtype=float)]
    out = np.zeros(fn.sparsity_out(0).nnz())
    for k, a in enumerate(args):
        buf.set_arg(k, memoryview(a))
    buf.set_res(0, memoryview(out))
    return per_call(lambda _: trigger(), None)


def jax_child(values_dir, path, out):
    """JAX on one module, in its own process: each result into `out` as it
    is measured (the right-hand side first), so a Jacobian that does not
    compile still leaves the right-hand side."""
    mod = Module(path)
    ref = json.load(open(os.path.join(values_dir, f"{mod.name}.values.json")))
    x0 = jnp.asarray(mod.x0[[i for i, _ in mod.states]])
    r = jax_model(mod)
    Fj, Jj = jax.jit(r), jax.jit(jax.jacfwd(r))
    t = time.perf_counter()
    Fj(x0).block_until_ready()
    sf = time.perf_counter() - t
    cf = per_call(lambda y: Fj(y).block_until_ready(), x0)
    with open(out, "a") as f:
        print("F", sf, cf, file=f)
    t = time.perf_counter()
    Jj(x0).block_until_ready()
    sj = time.perf_counter() - t
    check(mod.name, "JAX", np.asarray(Fj(x0 + 0.01)), np.asarray(Jj(x0 + 0.01)), ref)
    cj = per_call(lambda y: Jj(y).block_until_ready(), x0)
    with open(out, "a") as f:
        print("J", sj, cj, file=f)


def jax_row(path, values_dir):
    """JAX in a child process under the time and memory limits; what it
    measured before a limit, and the limit it hit."""
    import psutil
    import subprocess
    import tempfile

    out = tempfile.mktemp(suffix=".jax")
    child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--jax-child", values_dir, path, out],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc, start, status = psutil.Process(child.pid), time.perf_counter(), "ok"
    while child.poll() is None:
        try:
            rss = sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)])
        except psutil.NoSuchProcess:
            break
        if rss > JAX_MEMORY:
            status = f"over {JAX_MEMORY // 2**30} GB"
        elif time.perf_counter() - start > JAX_SECONDS:
            status = f"over {JAX_SECONDS // 60} min"
        if status != "ok":
            for p in [*proc.children(recursive=True), proc]:
                p.kill()
            break
        time.sleep(0.5)
    if status == "ok" and child.wait() != 0:
        status = "failed"
    got = {}
    if os.path.exists(out):
        for line in open(out):
            k, setup, call = line.split()
            got[k] = (float(setup), float(call))
        os.remove(out)
    f, j = got.get("F", (None, None)), got.get("J", (None, None))
    return [f[0], j[0], f[1], j[1], status]


def run(path, values_dir):
    mod = Module(path)
    ref = json.load(open(os.path.join(values_dir, f"{mod.name}.values.json")))
    rows = [
        ("CasADi", casadi_row(mod, ref)),
        ("CasADi (parameter inputs)", casadi_row(mod, ref, params_as_inputs=True)),
        ("JAX", jax_row(path, values_dir)),
    ]
    fmt = lambda v, k=1.0: "" if v is None else f"{v * k:.6g}"
    for tool, (sf, sj, cf, cj, status) in rows:
        print(f"{mod.name},{len(mod.states)},{tool},{fmt(sf)},{fmt(sj)},{fmt(cf, 1e6)},{fmt(cj, 1e6)},{status}")
        sys.stdout.flush()


def main():
    if sys.argv[1] == "--jax-child":
        jax_child(*sys.argv[2:5])
        return
    values_dir, paths = sys.argv[1], sys.argv[2:]
    print("module,states,tool,setup_f_s,setup_j_s,call_f_us,call_j_us,status")
    for p in paths:
        run(p, values_dir)


if __name__ == "__main__":
    main()
