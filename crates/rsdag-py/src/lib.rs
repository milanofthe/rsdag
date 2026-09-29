//! Python frontend: operator-overloading trace of a Python function into an
//! rsdag graph, then differentiation and compilation.
//!
//! A `Scope` owns a `Graph<F64>`; `Tracer` values are expression handles
//! into it and overload Python arithmetic, comparisons and the numpy ufunc
//! method protocol (an object array of tracers under `np.sin` calls each
//! element's `sin`), so plain numpy code traces without changes. A closed
//! trace is a `Program`: a tape run by the interpreter or, once compiled,
//! by native code. Derivatives are programs of their own
//! (`Scope::jacobian`, `Scope::gradient`).

// pyo3's method expansion trips clippy's `useless_conversion` on every
// `PyResult` method; the conversions are the macro's, not ours.
#![allow(clippy::useless_conversion)]

use std::cell::RefCell;
use std::rc::Rc;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use rsdag::{BinOp, CmpOp, ExprId, Graph, Node, ReduceOp, SymbolId, Tape, UnaryOp, F64};

type Shared = Rc<RefCell<Graph<F64>>>;

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
            return Ok(Some(self.g.borrow_mut().konst_f64(v)));
        }
        Ok(None)
    }
    /// As [`Self::operand_opt`], raising for an operand that cannot be
    /// traced.
    fn operand(&self, other: &Bound<'_, PyAny>) -> PyResult<ExprId> {
        self.operand_opt(other)?.ok_or_else(|| unsupported(other))
    }
    fn unary(&self, op: UnaryOp) -> Tracer {
        let id = self.g.borrow_mut().unary(op, self.id);
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
        let id = f(&mut self.g.borrow_mut(), a, b);
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
                let id = t.g.borrow_mut().reduce(op, vec![t.id, o]);
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
        let g = self.g.borrow();
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
                let id = self.g.borrow_mut().pow_i(self.id, n);
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
        let id = self.g.borrow_mut().cmp(c, self.id, o);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __neg__(&self) -> Tracer {
        let id = self.g.borrow_mut().neg(self.id);
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
        let id = self.g.borrow_mut().select(c, self.id, o);
        Ok(self.wrap(id))
    }
    /// The expression as text.
    fn expr(&self) -> String {
        rsdag::to_string(&self.g.borrow(), self.id)
    }
}

#[pymethods]
impl Scope {
    #[new]
    fn new() -> Self {
        Scope {
            g: Rc::new(RefCell::new(Graph::new())),
            inputs: Vec::new(),
        }
    }
    /// A new positional input.
    #[pyo3(signature = (name = None))]
    fn input(&mut self, name: Option<&str>) -> Tracer {
        let k = self.inputs.len();
        let mut g = self.g.borrow_mut();
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
        let id = self.g.borrow_mut().konst_f64(v);
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
        let g = self.g.borrow();
        let tape = Tape::compile(&g, &ids, &self.inputs);
        Ok(Program::new(tape, self.inputs.len(), ids.len()))
    }
    /// The Jacobian `d outputs / d inputs[wrt]` as a program with
    /// `len(outputs) * len(wrt)` outputs (row-major); `wrt` are input
    /// indices (all inputs when empty).
    #[pyo3(signature = (outputs, wrt = vec![]))]
    fn jacobian(&self, outputs: &Bound<'_, PyList>, wrt: Vec<usize>) -> PyResult<Program> {
        let ids = self.output_ids(outputs)?;
        let wrt = self.wrt_symbols(&wrt)?;
        let mut g = self.g.borrow_mut();
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
        let tape = Tape::compile(&g, &flat, &self.inputs);
        Ok(Program::new(tape, self.inputs.len(), flat.len()))
    }
    /// The gradient of one scalar output with respect to `inputs[wrt]`
    /// (reverse mode; all inputs when empty).
    #[pyo3(signature = (output, wrt = vec![]))]
    fn gradient(&self, output: &Tracer, wrt: Vec<usize>) -> PyResult<Program> {
        let wrt = self.wrt_symbols(&wrt)?;
        let mut g = self.g.borrow_mut();
        let grad = rsdag::gradient(&mut g, output.id, &wrt);
        let tape = Tape::compile(&g, &grad, &self.inputs);
        Ok(Program::new(tape, self.inputs.len(), grad.len()))
    }
    /// Number of nodes in the graph.
    fn n_nodes(&self) -> usize {
        self.g.borrow().len()
    }
}

impl Scope {
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
                ids.push(self.g.borrow_mut().konst_f64(v));
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
}

impl Program {
    fn new(tape: Tape, n_in: usize, n_out: usize) -> Self {
        Program {
            tape,
            native: std::sync::OnceLock::new(),
            n_in,
            n_out,
        }
    }
}

#[pymethods]
impl Program {
    /// Evaluate on a flat list of inputs; returns the flat outputs. The
    /// GIL is released while the program runs, and a program is shared,
    /// not borrowed: threads evaluate it at once, each on its own buffers.
    fn eval(&self, py: Python<'_>, inputs: Vec<f64>) -> PyResult<Vec<f64>> {
        if inputs.len() != self.n_in {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.n_in,
                inputs.len()
            )));
        }
        let (mut work, mut out) = (Vec::new(), Vec::new());
        py.detach(|| match self.native.get() {
            Some(n) => n.eval(&inputs, &mut work, &mut out),
            None => self.tape.eval(&inputs, &mut work, &mut out),
        });
        Ok(out)
    }
    /// Evaluate `len(inputs) / n_inputs` input vectors laid back to back;
    /// returns their outputs back to back. Natively the instances run in
    /// parallel; the GIL is released throughout.
    fn eval_many(&self, py: Python<'_>, inputs: Vec<f64>) -> PyResult<Vec<f64>> {
        let n_in = self.n_in.max(1);
        if !inputs.len().is_multiple_of(n_in) {
            return Err(PyValueError::new_err(format!(
                "{} inputs are not a whole number of vectors of {}",
                inputs.len(),
                self.n_in
            )));
        }
        Ok(py.detach(|| {
            let mut all = Vec::with_capacity(inputs.len() / n_in * self.n_out);
            match self.native.get() {
                Some(n) => n.eval_many(&inputs, n_in, &mut all),
                None => {
                    let (mut work, mut out) = (Vec::new(), Vec::new());
                    for ins in inputs.chunks(n_in) {
                        self.tape.eval(ins, &mut work, &mut out);
                        all.extend_from_slice(&out);
                    }
                }
            }
            all
        }))
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
    let id = t.g.borrow_mut().select(c, x, y);
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
    let mut g = t.g.borrow_mut();
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
    let id = t.g.borrow_mut().reduce(rop, ids);
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
    let xs = t.g.borrow_mut().solve_dense(a, b);
    Ok(xs.into_iter().map(|id| t.wrap(id)).collect())
}

#[pymodule]
fn _rsdag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scope>()?;
    m.add_class::<Tracer>()?;
    m.add_class::<Program>()?;
    m.add_function(wrap_pyfunction!(select, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(reduce, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
