//! Python frontend: operator-overloading trace of a Python function into an
//! rsdag graph, then differentiation and compilation.
//!
//! A `Scope` owns a `Graph<F64>`; `Tracer` values are expression handles
//! into it and overload Python arithmetic, comparisons and the numpy ufunc
//! method protocol (an object array of tracers under `np.sin` calls each
//! element's `sin`), so plain numpy code traces without changes. A closed
//! trace is a `Program`: a tape run by the interpreter or, once compiled,
//! by native code. Derivatives are programs of their own
//! (`Scope::jacobian`, `Scope::gradient`). A program keeps its symbolic
//! form, so it composes: called with tracers inside another trace it is a
//! function of that trace's graph, called as one instance, and the program
//! traced around it is compiled as one (`Tape::compile`). `Dispatch` keeps a traced
//! function's programs by argument shapes; a call reads numpy arrays
//! through the buffer protocol and evaluates on per-thread buffers without
//! leaving Rust.

// pyo3's method expansion trips clippy's `useless_conversion` on every
// `PyResult` method; the conversions are the macro's, not ours.
#![allow(clippy::useless_conversion)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use pyo3::basic::CompareOp;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList, PyTuple};

use rsdag::{BinOp, CmpOp, ExprId, FuncId, Graph, Node, ReduceOp, SymbolId, Tape, UnaryOp, F64};

/// A trace's graph, and the programs called in it by program id, as the
/// functions of the graph they were imported as.
struct Traced {
    graph: RefCell<Graph<F64>>,
    imported: RefCell<HashMap<u64, FuncId>>,
}

type Shared = Rc<Traced>;

/// An open graph: hands out input tracers and closes into programs.
#[pyclass(unsendable)]
pub struct Scope {
    g: Shared,
    inputs: Vec<SymbolId>,
}

/// An expression handle into a scope's graph.
#[pyclass(unsendable, skip_from_py_object)]
#[derive(Clone)]
pub struct Tracer {
    g: Shared,
    id: ExprId,
}

/// The result of a binary operator on a tracer: a traced value, or Python's
/// `NotImplemented` when the other operand is neither a tracer nor a number.
/// Returning `NotImplemented` instead of raising is what lets the
/// interpreter try the other operand's reflected method, so a numpy array of
/// tracers on the right of a tracer (`k * np.exp(x)`) broadcasts elementwise
/// instead of failing.
enum BinOut {
    Value(Tracer),
    NotImplemented,
}

impl<'py> IntoPyObject<'py> for BinOut {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            BinOut::Value(t) => Ok(Bound::new(py, t)?.into_any()),
            BinOut::NotImplemented => Ok(py.NotImplemented().into_bound(py)),
        }
    }
}

impl Tracer {
    fn wrap(&self, id: ExprId) -> Tracer {
        Tracer {
            g: self.g.clone(),
            id,
        }
    }
    /// Coerce a Python operand (tracer, int, float) into an expression;
    /// `None` when it is neither, so a binary operator can hand the pair
    /// back to Python (see [`BinOut`]).
    fn operand_opt(&self, other: &Bound<'_, PyAny>) -> PyResult<Option<ExprId>> {
        if let Ok(t) = other.cast::<Tracer>() {
            let t = t.borrow();
            if !Rc::ptr_eq(&t.g, &self.g) {
                return Err(PyValueError::new_err("tracers from different scopes"));
            }
            return Ok(Some(t.id));
        }
        if let Ok(v) = other.extract::<f64>() {
            return Ok(Some(self.g.graph.borrow_mut().konst_f64(v)));
        }
        Ok(None)
    }
    /// As [`Self::operand_opt`], raising for an operand that cannot be
    /// traced.
    fn operand(&self, other: &Bound<'_, PyAny>) -> PyResult<ExprId> {
        self.operand_opt(other)?.ok_or_else(|| unsupported(other))
    }
    fn unary(&self, op: UnaryOp) -> Tracer {
        let id = self.g.graph.borrow_mut().unary(op, self.id);
        self.wrap(id)
    }
    /// `f(self, other)`, or `f(other, self)` when `reflected`.
    fn binary(
        &self,
        other: &Bound<'_, PyAny>,
        reflected: bool,
        f: impl FnOnce(&mut Graph<F64>, ExprId, ExprId) -> ExprId,
    ) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let (a, b) = if reflected {
            (o, self.id)
        } else {
            (self.id, o)
        };
        let id = f(&mut self.g.graph.borrow_mut(), a, b);
        Ok(BinOut::Value(self.wrap(id)))
    }
}

fn unsupported(other: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "unsupported operand for a traced value: {}",
        other
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_default()
    ))
}

/// Python's `a // b`.
fn floordiv(g: &mut Graph<F64>, a: ExprId, b: ExprId) -> ExprId {
    let q = g.div(a, b);
    g.unary(UnaryOp::Floor, q)
}

/// Python's floored modulo: `a - b * floor(a / b)`.
fn floormod(g: &mut Graph<F64>, a: ExprId, b: ExprId) -> ExprId {
    let f = floordiv(g, a, b);
    let bf = g.mul(b, f);
    g.sub(a, bf)
}

fn powf(g: &mut Graph<F64>, a: ExprId, b: ExprId) -> ExprId {
    g.binary(BinOp::Powf, a, b)
}

/// Node budget of a tracer's `repr`.
const REPR_NODES: usize = 200;

/// Whether the expression under `id` written out as a tree has at most
/// `budget` nodes; stops counting once over, so the cost is the budget.
fn fits(g: &Graph<F64>, id: ExprId, budget: &mut usize) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    g.operands(id).iter().all(|&c| fits(g, c, budget))
}

/// An elementwise function a tracer answers to by name: numpy's ufuncs
/// call it on each element of an object array (`np.sin(a)` calls
/// `a[i].sin()`, `np.arctan2(a, b)` calls `a[i].arctan2(b[i])`).
#[derive(Clone, Copy)]
enum Ufunc {
    Unary(UnaryOp),
    Binary(BinOp),
    Reduce(ReduceOp),
}

/// numpy's names where they differ from rsdag's; every other name of
/// [`rsdag::node::UNARY_OPS`] and [`rsdag::node::BINARY_OPS`] resolves as
/// it is.
const NUMPY_NAMES: &[(&str, Ufunc)] = &[
    ("arcsin", Ufunc::Unary(UnaryOp::Asin)),
    ("arccos", Ufunc::Unary(UnaryOp::Acos)),
    ("arctan", Ufunc::Unary(UnaryOp::Atan)),
    ("arcsinh", Ufunc::Unary(UnaryOp::Asinh)),
    ("arccosh", Ufunc::Unary(UnaryOp::Acosh)),
    ("arctanh", Ufunc::Unary(UnaryOp::Atanh)),
    ("log", Ufunc::Unary(UnaryOp::Ln)),
    ("fabs", Ufunc::Unary(UnaryOp::Abs)),
    ("absolute", Ufunc::Unary(UnaryOp::Abs)),
    ("rint", Ufunc::Unary(UnaryOp::Round)),
    ("gammaln", Ufunc::Unary(UnaryOp::Lgamma)),
    ("gamma", Ufunc::Unary(UnaryOp::Tgamma)),
    ("arctan2", Ufunc::Binary(BinOp::Atan2)),
    ("fmod", Ufunc::Binary(BinOp::Mod)),
    ("power", Ufunc::Binary(BinOp::Powf)),
    ("maximum", Ufunc::Reduce(ReduceOp::Max)),
    ("minimum", Ufunc::Reduce(ReduceOp::Min)),
];

fn ufunc(name: &str) -> Option<Ufunc> {
    NUMPY_NAMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, f)| f)
        .or_else(|| UnaryOp::from_name(name).map(Ufunc::Unary))
        .or_else(|| BinOp::from_name(name).map(Ufunc::Binary))
}

/// A tracer's elementwise function, bound to it (`t.sin`, `t.arctan2`).
#[pyclass(unsendable)]
struct Method {
    t: Tracer,
    f: Ufunc,
}

#[pymethods]
impl Method {
    #[pyo3(signature = (*args))]
    fn __call__(&self, args: &Bound<'_, PyTuple>) -> PyResult<Tracer> {
        let t = &self.t;
        let out = match (self.f, args.len()) {
            (Ufunc::Unary(op), 0) => return Ok(t.unary(op)),
            (Ufunc::Binary(BinOp::Powf), 1) => t.__pow__(&args.get_item(0)?, None)?,
            (Ufunc::Binary(op), 1) => {
                t.binary(&args.get_item(0)?, false, |g, x, y| g.binary(op, x, y))?
            }
            (Ufunc::Reduce(op), 1) => {
                let o = t.operand(&args.get_item(0)?)?;
                let id = t.g.graph.borrow_mut().reduce(op, vec![t.id, o]);
                BinOut::Value(t.wrap(id))
            }
            (_, n) => {
                return Err(PyTypeError::new_err(format!(
                    "wrong number of arguments: {n}"
                )))
            }
        };
        match out {
            BinOut::Value(v) => Ok(v),
            BinOut::NotImplemented => Err(unsupported(&args.get_item(0)?)),
        }
    }
}

#[pymethods]
impl Tracer {
    /// The elementwise functions by name (see [`NUMPY_NAMES`]).
    fn __getattr__(&self, name: &str) -> PyResult<Method> {
        let f = ufunc(name).ok_or_else(|| {
            PyAttributeError::new_err(format!("'Tracer' object has no attribute '{name}'"))
        })?;
        Ok(Method { t: self.clone(), f })
    }
    /// The expression as text, or a summary when written out as a tree it
    /// would exceed [`REPR_NODES`] nodes (a shared subexpression is written
    /// once per use, so the text of a DAG can be exponential in its size).
    fn __repr__(&self) -> String {
        let g = self.g.graph.borrow();
        let mut budget = REPR_NODES;
        if fits(&g, self.id, &mut budget) {
            format!("Tracer({})", rsdag::to_string(&g, self.id))
        } else {
            format!("Tracer(#{}, over {REPR_NODES} nodes)", self.id.0)
        }
    }
    fn __add__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, false, Graph::add)
    }
    fn __radd__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, true, Graph::add)
    }
    fn __sub__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, false, Graph::sub)
    }
    fn __rsub__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, true, Graph::sub)
    }
    fn __mul__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, false, Graph::mul)
    }
    fn __rmul__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, true, Graph::mul)
    }
    fn __truediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, false, Graph::div)
    }
    fn __rtruediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, true, Graph::div)
    }
    fn __floordiv__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, false, floordiv)
    }
    fn __rfloordiv__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, true, floordiv)
    }
    fn __mod__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, false, floormod)
    }
    fn __rmod__(&self, o: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.binary(o, true, floormod)
    }
    fn __pow__(
        &self,
        o: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<BinOut> {
        match o.extract::<i64>() {
            Ok(n) => {
                let id = self.g.graph.borrow_mut().pow_i(self.id, n);
                Ok(BinOut::Value(self.wrap(id)))
            }
            Err(_) => self.binary(o, false, powf),
        }
    }
    fn __rpow__(
        &self,
        o: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<BinOut> {
        self.binary(o, true, powf)
    }
    fn __richcmp__(&self, o: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(o)? else {
            return Ok(BinOut::NotImplemented);
        };
        let c = match op {
            CompareOp::Lt => CmpOp::Lt,
            CompareOp::Le => CmpOp::Le,
            CompareOp::Gt => CmpOp::Gt,
            CompareOp::Ge => CmpOp::Ge,
            CompareOp::Eq => CmpOp::Eq,
            CompareOp::Ne => CmpOp::Ne,
        };
        let id = self.g.graph.borrow_mut().cmp(c, self.id, o);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __neg__(&self) -> Tracer {
        let id = self.g.graph.borrow_mut().neg(self.id);
        self.wrap(id)
    }
    fn __pos__(&self) -> Tracer {
        self.clone()
    }
    fn __abs__(&self) -> Tracer {
        self.unary(UnaryOp::Abs)
    }
    // `math.floor` and friends, which numpy's object loops call.
    fn __floor__(&self) -> Tracer {
        self.unary(UnaryOp::Floor)
    }
    fn __ceil__(&self) -> Tracer {
        self.unary(UnaryOp::Ceil)
    }
    fn __trunc__(&self) -> Tracer {
        self.unary(UnaryOp::Trunc)
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "a traced value has no truth value: data-dependent control flow is not traceable, use rsdag.where",
        ))
    }
    fn __float__(&self) -> PyResult<f64> {
        Err(PyTypeError::new_err(
            "a traced value cannot be converted to float during tracing",
        ))
    }
    /// `cond != 0 ? self : other` (the select node).
    fn select(&self, cond: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let c = self.operand(cond)?;
        let o = self.operand(other)?;
        let id = self.g.graph.borrow_mut().select(c, self.id, o);
        Ok(self.wrap(id))
    }
    /// The expression as text.
    fn expr(&self) -> String {
        rsdag::to_string(&self.g.graph.borrow(), self.id)
    }
}

#[pymethods]
impl Scope {
    #[new]
    fn new() -> Self {
        Scope {
            g: Rc::new(Traced {
                graph: RefCell::new(Graph::new()),
                imported: RefCell::default(),
            }),
            inputs: Vec::new(),
        }
    }
    /// A new positional input.
    #[pyo3(signature = (name = None))]
    fn input(&mut self, name: Option<&str>) -> Tracer {
        let k = self.inputs.len();
        let mut g = self.g.graph.borrow_mut();
        let name = name
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("in{k}"));
        let e = g.sym(&name);
        let s = match *g.node(e) {
            Node::Symbol(s) => s,
            _ => unreachable!(),
        };
        self.inputs.push(s);
        drop(g);
        Tracer {
            g: self.g.clone(),
            id: e,
        }
    }
    /// A constant of this scope.
    fn constant(&self, v: f64) -> Tracer {
        let id = self.g.graph.borrow_mut().konst_f64(v);
        Tracer {
            g: self.g.clone(),
            id,
        }
    }
    /// Number of inputs handed out.
    fn n_inputs(&self) -> usize {
        self.inputs.len()
    }
    /// Close the scope over `outputs` into a program.
    fn compile(&self, outputs: &Bound<'_, PyList>) -> PyResult<Program> {
        let ids = self.output_ids(outputs)?;
        Ok(self.program(&ids))
    }
    /// The Jacobian `d outputs / d inputs[wrt]` as a program with
    /// `len(outputs) * len(wrt)` outputs (row-major); `wrt` are input
    /// indices (all inputs when empty).
    #[pyo3(signature = (outputs, wrt = vec![]))]
    fn jacobian(&self, outputs: &Bound<'_, PyList>, wrt: Vec<usize>) -> PyResult<Program> {
        let ids = self.output_ids(outputs)?;
        let wrt = self.wrt_symbols(&wrt)?;
        let mut g = self.g.graph.borrow_mut();
        let zero = g.zero();
        let rows = rsdag::sparse_jacobian(&mut g, &ids, &wrt);
        let mut flat: Vec<ExprId> = Vec::with_capacity(ids.len() * wrt.len());
        for row in rows {
            let mut dense = vec![zero; wrt.len()];
            for (j, e) in row {
                dense[j] = e;
            }
            flat.extend(dense);
        }
        drop(g);
        Ok(self.program(&flat))
    }
    /// The Jacobian `d outputs / d inputs[wrt]` in its nonzeros: a program
    /// whose outputs are the structurally nonzero entries, row by row in
    /// ascending columns, with that pattern (`Program.pattern`).
    #[pyo3(signature = (outputs, wrt = vec![]))]
    fn sparse_jacobian(&self, outputs: &Bound<'_, PyList>, wrt: Vec<usize>) -> PyResult<Program> {
        let ids = self.output_ids(outputs)?;
        let wrt = self.wrt_symbols(&wrt)?;
        let mut g = self.g.graph.borrow_mut();
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for (i, row) in rsdag::sparse_jacobian(&mut g, &ids, &wrt)
            .into_iter()
            .enumerate()
        {
            for (j, e) in row {
                rows.push(i);
                cols.push(j);
                vals.push(e);
            }
        }
        drop(g);
        let mut p = self.program(&vals);
        p.pattern = Some((rows, cols));
        Ok(p)
    }
    /// The gradient of one scalar output with respect to `inputs[wrt]`
    /// (reverse mode; all inputs when empty).
    #[pyo3(signature = (output, wrt = vec![]))]
    fn gradient(&self, output: &Tracer, wrt: Vec<usize>) -> PyResult<Program> {
        let wrt = self.wrt_symbols(&wrt)?;
        let mut g = self.g.graph.borrow_mut();
        let grad = rsdag::gradient(&mut g, output.id, &wrt);
        drop(g);
        Ok(self.program(&grad))
    }
    /// Number of nodes in the graph.
    fn n_nodes(&self) -> usize {
        self.g.graph.borrow().len()
    }
}

impl Scope {
    /// The program of `roots` over the inputs: their function, kept in a
    /// graph of its own so the program composes into other traces, and its
    /// tape over the composition (the programs called in this trace inlined
    /// where they call others, batched where they are leaves).
    fn program(&self, roots: &[ExprId]) -> Program {
        let mut g = self.g.graph.borrow_mut();
        let f = g.define_func("program", self.inputs.clone(), roots.to_vec());
        let mut own = Graph::new();
        let func = own.import(&g, f, &mut Default::default());
        let tape = Tape::compile(&g, roots, &self.inputs);
        let mut p = Program::new(tape, self.inputs.len(), roots.len());
        p.symbolic = Some(Arc::new(Symbolic {
            graph: own,
            func,
            id: NEXT_PROGRAM.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }));
        p
    }

    fn wrt_symbols(&self, wrt: &[usize]) -> PyResult<Vec<SymbolId>> {
        if wrt.is_empty() {
            return Ok(self.inputs.clone());
        }
        wrt.iter()
            .map(|&k| {
                self.inputs
                    .get(k)
                    .copied()
                    .ok_or_else(|| PyValueError::new_err(format!("input index {k} out of range")))
            })
            .collect()
    }
    fn output_ids(&self, outputs: &Bound<'_, PyList>) -> PyResult<Vec<ExprId>> {
        let mut ids = Vec::with_capacity(outputs.len());
        for o in outputs.iter() {
            if let Ok(t) = o.cast::<Tracer>() {
                ids.push(t.borrow().id);
            } else if let Ok(v) = o.extract::<f64>() {
                ids.push(self.g.graph.borrow_mut().konst_f64(v));
            } else {
                return Err(PyTypeError::new_err("outputs must be tracers or numbers"));
            }
        }
        Ok(ids)
    }
}

/// A compiled function: the tape, evaluated by the interpreter or, after
/// `compile_native()`, by the native backend.
#[pyclass(frozen)]
pub struct Program {
    tape: Tape,
    native: std::sync::OnceLock<rsdag_jit::NativeTape>,
    n_in: usize,
    n_out: usize,
    /// `(rows, cols)` of the outputs of a sparse Jacobian.
    pattern: Option<(Vec<usize>, Vec<usize>)>,
    /// The program as a function, for calling it inside another trace.
    symbolic: Option<Arc<Symbolic>>,
}

/// A program's function in a graph of its own, and the id a trace that
/// imports it keeps it by.
struct Symbolic {
    graph: Graph<F64>,
    func: FuncId,
    id: u64,
}

static NEXT_PROGRAM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Evaluation buffers of one thread, shared by all programs: an evaluation
/// takes them and puts them back, so once grown a call allocates nothing,
/// and a reentrant call (an input whose conversion runs Python) gets its
/// own.
#[derive(Default)]
struct Scratch {
    ins: Vec<f64>,
    /// The shapes of the gathered arguments (see [`gather`]).
    key: Vec<usize>,
    work: Vec<f64>,
    out: Vec<f64>,
}

/// Programs from this many ops on release the GIL while they run; below,
/// releasing and retaking it costs more than the evaluation.
const DETACH_OPS: usize = 256;

/// Append the values of `arg` to `s.ins` and its shape to `s.key` (the
/// number of dimensions, then the extents; a number has none): a number,
/// a flat list or tuple of numbers, an object exporting a buffer of doubles
/// (a float64 numpy array, in any layout, flattened in C order), or
/// anything numpy converts to one.
fn gather(arg: &Bound<'_, PyAny>, s: &mut Scratch) -> PyResult<()> {
    let py = arg.py();
    if let Ok(x) = arg.cast::<PyFloat>() {
        s.ins.push(x.value());
        s.key.push(0);
        return Ok(());
    }
    if arg.is_instance_of::<PyList>() || arg.is_instance_of::<PyTuple>() {
        if let Ok(xs) = arg.extract::<Vec<f64>>() {
            s.key.extend([1, xs.len()]);
            s.ins.extend(xs);
            return Ok(());
        }
    } else if let Some(buf) = doubles(arg) {
        s.key.push(buf.dimensions());
        s.key.extend(buf.shape());
        let n = s.ins.len();
        s.ins.resize(n + buf.item_count(), 0.0);
        return buf.copy_to_slice(py, &mut s.ins[n..]);
    } else if let Ok(x) = arg.extract::<f64>() {
        s.ins.push(x);
        s.key.push(0);
        return Ok(());
    }
    // A float64 array now, which the buffer branch takes.
    let arr = py
        .import("numpy")?
        .call_method1("asarray", (arg, "float64"))?;
    gather(&arr, s)
}

/// The buffer of `arg` if it holds doubles in the host's byte order.
/// pyo3 0.29 also takes a big-endian `>d` for `f64` on a little-endian
/// host, so the order is checked here.
fn doubles(arg: &Bound<'_, PyAny>) -> Option<PyBuffer<f64>> {
    let buf = PyBuffer::<f64>::get(arg).ok()?;
    let host = if cfg!(target_endian = "little") {
        b'<'
    } else {
        b'>'
    };
    match buf.format().to_bytes() {
        [b'd'] | [b'@' | b'=', b'd'] => Some(buf),
        [c, b'd'] if *c == host => Some(buf),
        _ => None,
    }
}

/// The shapes in a [`Scratch::key`] as Python sees them: `None` for a
/// scalar, else a tuple of extents.
fn shapes<'py>(py: Python<'py>, key: &[usize]) -> PyResult<Bound<'py, PyTuple>> {
    let mut out = Vec::new();
    let mut rest = key;
    while let Some((&nd, tail)) = rest.split_first() {
        let (dims, tail) = tail.split_at(nd);
        out.push(match nd {
            0 => py.None().into_bound(py),
            _ => PyTuple::new(py, dims)?.into_any(),
        });
        rest = tail;
    }
    PyTuple::new(py, out)
}

/// `vals` into `out` (any writable float64 buffer of their length), which
/// is returned, or as a new list.
fn emit<'py>(
    py: Python<'py>,
    vals: &[f64],
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match out {
        Some(o) => {
            doubles(&o)
                .ok_or_else(|| PyTypeError::new_err("out takes a buffer of float64"))?
                .copy_from_slice(py, vals)?;
            Ok(o)
        }
        None => Ok(PyList::new(py, vals)?.into_any()),
    }
}

impl Program {
    fn new(tape: Tape, n_in: usize, n_out: usize) -> Self {
        Program {
            tape,
            native: std::sync::OnceLock::new(),
            n_in,
            n_out,
            pattern: None,
            symbolic: None,
        }
    }
    fn backend(&self) -> &dyn rsdag::Program {
        match self.native.get() {
            Some(n) => n,
            None => &self.tape,
        }
    }
    /// Evaluate on `s.ins` into `s.out`.
    fn run(&self, py: Python<'_>, s: &mut Scratch) -> PyResult<()> {
        if s.ins.len() != self.n_in {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.n_in,
                s.ins.len()
            )));
        }
        let p = self.backend();
        if s.work.len() < p.work_len() {
            s.work.resize(p.work_len(), 0.0);
        }
        s.out.resize(self.n_out, 0.0);
        let Scratch {
            ins,
            work,
            out: vals,
            ..
        } = s;
        let mut run = || p.eval_into(ins, work, vals);
        if self.tape.n_ops() >= DETACH_OPS {
            py.detach(run);
        } else {
            run();
        }
        Ok(())
    }
}

/// Clear `s` and gather `args` into it.
fn gather_all(args: &Bound<'_, PyTuple>, s: &mut Scratch) -> PyResult<()> {
    s.ins.clear();
    s.key.clear();
    args.iter().try_for_each(|a| gather(&a, s))
}

/// A traced function's programs by argument shapes, the base of the Python
/// `Compiled`. A call gathers the arguments, looks the program up by their
/// shapes and evaluates it, all here; a new shape asks the subclass's
/// `_trace(shapes)` for the program and the shape of its result (`None`
/// for a scalar).
#[pyclass(subclass, frozen)]
struct Dispatch {
    programs: Mutex<HashMap<Box<[usize]>, Entry>>,
    /// `numpy.empty`, for the results.
    empty: Py<PyAny>,
}

struct Entry {
    program: Py<Program>,
    shape: Option<Py<PyTuple>>,
}

impl Dispatch {
    /// The entry for the shapes of the arguments gathered into `s`.
    fn entry(slf: &Bound<'_, Self>, s: &Scratch) -> PyResult<(Py<Program>, Option<Py<PyTuple>>)> {
        let py = slf.py();
        let copy = |e: &Entry| {
            (
                e.program.clone_ref(py),
                e.shape.as_ref().map(|t| t.clone_ref(py)),
            )
        };
        let this = slf.get();
        if let Some(e) = this.programs.lock().unwrap().get(&s.key[..]) {
            return Ok(copy(e));
        }
        let (program, shape) = slf
            .call_method1("_trace", (shapes(py, &s.key)?,))?
            .extract::<(Py<Program>, Option<Py<PyTuple>>)>()?;
        let e = Entry { program, shape };
        let out = copy(&e);
        this.programs
            .lock()
            .unwrap()
            .insert(s.key.as_slice().into(), e);
        Ok(out)
    }
}

#[pymethods]
impl Dispatch {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        py: Python<'_>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Dispatch {
            programs: Mutex::default(),
            empty: py.import("numpy")?.getattr("empty")?.unbind(),
        })
    }
    #[pyo3(signature = (*args))]
    fn __call__<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        rsdag::scratch::with(|s: &mut Scratch| {
            if let Err(e) = gather_all(args, s) {
                // Called with tracers inside another trace: a composition.
                if slf.call_method1("_traced", (args,))?.is_truthy()? {
                    return slf.call_method1("_compose", args);
                }
                return Err(e);
            }
            let (program, shape) = Self::entry(slf, s)?;
            program.get().run(py, s)?;
            match shape {
                None => Ok(PyFloat::new(py, s.out[0]).into_any()),
                Some(shape) => {
                    let o = slf.get().empty.bind(py).call1((shape,))?;
                    emit(py, &s.out, Some(o))
                }
            }
        })
    }
    /// The program traced for the shapes of `args`.
    #[pyo3(signature = (*args))]
    fn program(slf: &Bound<'_, Self>, args: &Bound<'_, PyTuple>) -> PyResult<Py<Program>> {
        let mut s = Scratch::default();
        gather_all(args, &mut s)?;
        Ok(Self::entry(slf, &s)?.0)
    }
}

#[pymethods]
impl Program {
    /// This program called inside another trace: `args` are its inputs in
    /// order, tracers of that trace or numbers; the result is one tracer per
    /// output, the outputs of one instance of the program in the trace's
    /// graph. The program comes into a graph once, however often it is
    /// called there; a program it calls comes along.
    #[pyo3(signature = (*args))]
    fn compose(&self, args: &Bound<'_, PyTuple>) -> PyResult<Vec<Tracer>> {
        let sym = self
            .symbolic
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("this program has no symbolic form"))?;
        if args.len() != self.n_in {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.n_in,
                args.len()
            )));
        }
        let mut shared: Option<Shared> = None;
        for a in args.iter() {
            if let Ok(t) = a.cast::<Tracer>() {
                let g = t.borrow().g.clone();
                match &shared {
                    Some(s) if !Rc::ptr_eq(s, &g) => {
                        return Err(PyValueError::new_err("inputs from different traces"))
                    }
                    _ => shared = Some(g),
                }
            }
        }
        let g =
            shared.ok_or_else(|| PyTypeError::new_err("compose takes the tracers of a trace"))?;
        let mut ids = Vec::with_capacity(args.len());
        for a in args.iter() {
            if let Ok(t) = a.cast::<Tracer>() {
                ids.push(t.borrow().id);
            } else {
                let v: f64 = a
                    .extract()
                    .map_err(|_| PyTypeError::new_err("inputs must be tracers or numbers"))?;
                ids.push(g.graph.borrow_mut().konst_f64(v));
            }
        }
        let f = *g.imported.borrow_mut().entry(sym.id).or_insert_with(|| {
            g.graph
                .borrow_mut()
                .import(&sym.graph, sym.func, &mut Default::default())
        });
        let outs: Vec<u32> = (0..self.n_out as u32).collect();
        let calls = g.graph.borrow_mut().calls(f, &outs, &ids);
        Ok(calls
            .into_iter()
            .map(|id| Tracer { g: g.clone(), id })
            .collect())
    }

    /// Evaluate on the inputs `args` hold back to back (numbers, float64
    /// arrays, anything numpy converts). The outputs go into `out` when
    /// given, a writable float64 buffer of `n_outputs` values that is
    /// returned, else into a new list. A program is shared, not borrowed:
    /// threads evaluate it at once, each on its own buffers, and a large one
    /// releases the GIL while it runs.
    #[pyo3(signature = (*args, out = None))]
    fn eval<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        out: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        rsdag::scratch::with(|s: &mut Scratch| {
            gather_all(args, s)?;
            self.run(py, s)?;
            emit(py, &s.out, out)
        })
    }
    /// Evaluate `len(inputs) / n_inputs` input vectors laid back to back;
    /// their outputs back to back go into `out` or a new list, as in
    /// [`Self::eval`]. Natively the instances run in parallel; the GIL is
    /// released throughout.
    #[pyo3(signature = (inputs, out = None))]
    fn eval_many<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
        out: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut s = Scratch::default();
        gather(inputs, &mut s)?;
        let ins = s.ins;
        let n_in = self.n_in.max(1);
        if !ins.len().is_multiple_of(n_in) {
            return Err(PyValueError::new_err(format!(
                "{} inputs are not a whole number of vectors of {}",
                ins.len(),
                self.n_in
            )));
        }
        let all = py.detach(|| {
            let mut all = vec![0.0; ins.len() / n_in * self.n_out];
            match self.native.get() {
                Some(n) => n.eval_many(&ins, n_in, &mut all),
                None => {
                    let mut work = vec![0.0; self.tape.work_len()];
                    rsdag::Program::eval_many_into(&self.tape, &ins, n_in, &mut work, &mut all);
                }
            }
            all
        });
        emit(py, &all, out)
    }
    /// Compile the tape to native code; evaluation switches over. A large
    /// batch of function-body calls runs on the thread pool.
    fn compile_native(&self, py: Python<'_>) -> PyResult<()> {
        if self.native.get().is_some() {
            return Ok(());
        }
        let opts = rsdag_jit::Options {
            batch: rsdag_jit::Batch::Parallel { min_ops: 1 << 16 },
            ..rsdag_jit::Options::default()
        };
        let c = py
            .detach(|| rsdag_jit::NativeTape::compile_opts(&self.tape, &opts, &[]))
            .map_err(|e| PyValueError::new_err(format!("native compile failed: {e:?}")))?;
        let _ = self.native.set(c);
        Ok(())
    }
    #[getter]
    fn n_inputs(&self) -> usize {
        self.n_in
    }
    #[getter]
    fn n_outputs(&self) -> usize {
        self.n_out
    }
    /// `(rows, cols)` of a sparse Jacobian's outputs, `None` for any other
    /// program.
    #[getter]
    fn pattern(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        self.pattern.clone()
    }
    #[getter]
    fn n_ops(&self) -> usize {
        self.tape.n_ops()
    }
    fn dump(&self) -> String {
        self.tape.dump()
    }
}

/// The items of an iterable.
fn items<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    obj.try_iter()?.collect()
}

/// The first tracer among `items`: the scope the numbers among them join.
fn anchor<'a, 'py: 'a>(mut items: impl Iterator<Item = &'a Bound<'py, PyAny>>) -> PyResult<Tracer> {
    items
        .find_map(|x| x.cast::<Tracer>().ok().map(|t| t.borrow().clone()))
        .ok_or_else(|| PyTypeError::new_err("a traced operand is needed"))
}

fn exprs(t: &Tracer, items: &[Bound<'_, PyAny>]) -> PyResult<Vec<ExprId>> {
    items.iter().map(|x| t.operand(x)).collect()
}

/// `where(cond, a, b)` over tracers and numbers of one scope.
#[pyfunction]
fn select(cond: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Tracer> {
    let t = anchor([cond, a, b].into_iter())?;
    let (c, x, y) = (t.operand(cond)?, t.operand(a)?, t.operand(b)?);
    let id = t.g.graph.borrow_mut().select(c, x, y);
    Ok(t.wrap(id))
}

/// `a @ b` for the `m * k` entries of `a` and the `k * n` of `b`, both
/// row-major: the `m * n` entries of the product, each one `Dot` node. The
/// rows against one column fuse into a `Gemv` kernel once compiled, against
/// several columns into a `Gemm`.
#[pyfunction]
fn matmul(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<Tracer>> {
    let (a, b) = (items(a)?, items(b)?);
    let t = anchor(a.iter().chain(&b))?;
    let (a, b) = (exprs(&t, &a)?, exprs(&t, &b)?);
    let k = b.len().checked_div(n).unwrap_or(0);
    if k == 0 || b.len() != k * n || !a.len().is_multiple_of(k) {
        return Err(PyValueError::new_err(
            "matmul takes the m*k entries of a and the k*n of b",
        ));
    }
    let mut g = t.g.graph.borrow_mut();
    let mut out = Vec::with_capacity(a.len() / k * n);
    for row in a.chunks(k) {
        for j in 0..n {
            let col = b[j..].iter().step_by(n).copied().collect();
            out.push(g.dot(row.to_vec(), col));
        }
    }
    drop(g);
    Ok(out.into_iter().map(|id| t.wrap(id)).collect())
}

/// A reduction (`"sum"`, `"product"`, `"min"`, `"max"`) over an iterable,
/// one `Reduce` node in the reference fold order.
#[pyfunction]
fn reduce(op: &str, xs: &Bound<'_, PyAny>) -> PyResult<Tracer> {
    let rop = match op {
        "sum" => ReduceOp::Sum,
        "product" => ReduceOp::Product,
        "min" => ReduceOp::Min,
        "max" => ReduceOp::Max,
        _ => return Err(PyValueError::new_err(format!("unknown reduction '{op}'"))),
    };
    let xs = items(xs)?;
    let t = anchor(xs.iter())?;
    let ids = exprs(&t, &xs)?;
    let id = t.g.graph.borrow_mut().reduce(rop, ids);
    Ok(t.wrap(id))
}

/// The solution of the dense system `A x = b`, `a` the `n*n` entries
/// row-major and `b` the `n` right-hand sides: one pivoting kernel in the
/// compiled program, differentiable.
#[pyfunction]
fn solve(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Vec<Tracer>> {
    let (a, b) = (items(a)?, items(b)?);
    let t = anchor(a.iter().chain(&b))?;
    let (a, b) = (exprs(&t, &a)?, exprs(&t, &b)?);
    if a.len() != b.len() * b.len() {
        return Err(PyValueError::new_err(
            "solve takes the n*n entries of a and the n of b",
        ));
    }
    let xs = t.g.graph.borrow_mut().solve_dense(a, b);
    Ok(xs.into_iter().map(|id| t.wrap(id)).collect())
}

#[pymodule]
fn _rsdag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scope>()?;
    m.add_class::<Tracer>()?;
    m.add_class::<Program>()?;
    m.add_class::<Dispatch>()?;
    m.add_function(wrap_pyfunction!(select, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(reduce, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
