//! Python frontend: operator-overloading trace of a Python function into an
//! rsdag graph, then differentiation and compilation.
//!
//! A `Scope` owns a `Graph<F64>`; `Tracer` values are expression handles
//! into it and overload Python arithmetic, comparisons and the numpy ufunc
//! method protocol (an object array of tracers under `np.sin` calls each
//! element's `sin`), so plain numpy code traces without changes. A closed
//! trace is a `Program`: a tape with an interpreter, an optional native
//! native form, symbolic derivatives, and C source.

// pyo3 0.22's method expansion trips clippy's `useless_conversion` on every
// `PyResult` method; the conversions are the macro's, not ours.
#![allow(clippy::useless_conversion)]

use std::cell::RefCell;
use std::rc::Rc;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyTypeError, PyValueError};
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
    /// traced. For the named ufunc methods, which numpy calls elementwise
    /// with scalar operands, so there is nothing to defer to.
    fn operand(&self, other: &Bound<'_, PyAny>) -> PyResult<ExprId> {
        self.operand_opt(other)?.ok_or_else(|| {
            PyTypeError::new_err(format!(
                "unsupported operand for a traced value: {}",
                other
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            ))
        })
    }
    fn unary(&self, op: UnaryOp) -> Tracer {
        let id = self.g.borrow_mut().unary(op, self.id);
        self.wrap(id)
    }
}

#[pymethods]
impl Tracer {
    // numpy's unary ufunc methods on object arrays, and the plain names.
    fn sin(&self) -> Tracer {
        self.unary(UnaryOp::Sin)
    }
    fn cos(&self) -> Tracer {
        self.unary(UnaryOp::Cos)
    }
    fn tan(&self) -> Tracer {
        self.unary(UnaryOp::Tan)
    }
    fn arcsin(&self) -> Tracer {
        self.unary(UnaryOp::Asin)
    }
    fn arccos(&self) -> Tracer {
        self.unary(UnaryOp::Acos)
    }
    fn arctan(&self) -> Tracer {
        self.unary(UnaryOp::Atan)
    }
    fn sinh(&self) -> Tracer {
        self.unary(UnaryOp::Sinh)
    }
    fn cosh(&self) -> Tracer {
        self.unary(UnaryOp::Cosh)
    }
    fn tanh(&self) -> Tracer {
        self.unary(UnaryOp::Tanh)
    }
    fn arcsinh(&self) -> Tracer {
        self.unary(UnaryOp::Asinh)
    }
    fn arccosh(&self) -> Tracer {
        self.unary(UnaryOp::Acosh)
    }
    fn arctanh(&self) -> Tracer {
        self.unary(UnaryOp::Atanh)
    }
    fn exp(&self) -> Tracer {
        self.unary(UnaryOp::Exp)
    }
    fn expm1(&self) -> Tracer {
        self.unary(UnaryOp::Expm1)
    }
    fn log(&self) -> Tracer {
        self.unary(UnaryOp::Ln)
    }
    fn log10(&self) -> Tracer {
        self.unary(UnaryOp::Log10)
    }
    fn log2(&self) -> Tracer {
        self.unary(UnaryOp::Log2)
    }
    fn log1p(&self) -> Tracer {
        self.unary(UnaryOp::Log1p)
    }
    fn sqrt(&self) -> Tracer {
        self.unary(UnaryOp::Sqrt)
    }
    fn cbrt(&self) -> Tracer {
        self.unary(UnaryOp::Cbrt)
    }
    fn fabs(&self) -> Tracer {
        self.unary(UnaryOp::Abs)
    }
    fn absolute(&self) -> Tracer {
        self.unary(UnaryOp::Abs)
    }
    fn sign(&self) -> Tracer {
        self.unary(UnaryOp::Sign)
    }
    fn floor(&self) -> Tracer {
        self.unary(UnaryOp::Floor)
    }
    fn ceil(&self) -> Tracer {
        self.unary(UnaryOp::Ceil)
    }
    fn rint(&self) -> Tracer {
        self.unary(UnaryOp::Round)
    }
    fn trunc(&self) -> Tracer {
        self.unary(UnaryOp::Trunc)
    }
    fn erf(&self) -> Tracer {
        self.unary(UnaryOp::Erf)
    }
    fn erfc(&self) -> Tracer {
        self.unary(UnaryOp::Erfc)
    }
    fn gammaln(&self) -> Tracer {
        self.unary(UnaryOp::Lgamma)
    }
    fn gamma(&self) -> Tracer {
        self.unary(UnaryOp::Tgamma)
    }
    fn digamma(&self) -> Tracer {
        self.unary(UnaryOp::Digamma)
    }
    fn rand_uniform(&self) -> Tracer {
        self.unary(UnaryOp::RandUniform)
    }
    fn __repr__(&self) -> String {
        format!("Tracer({})", rsdag::to_string(&self.g.borrow(), self.id))
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let id = self.g.borrow_mut().add(self.id, o);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.__add__(other)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let id = self.g.borrow_mut().sub(self.id, o);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let id = self.g.borrow_mut().sub(o, self.id);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let id = self.g.borrow_mut().mul(self.id, o);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        self.__mul__(other)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let id = self.g.borrow_mut().div(self.id, o);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let id = self.g.borrow_mut().div(o, self.id);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let mut g = self.g.borrow_mut();
        let q = g.div(self.id, o);
        let id = g.unary(UnaryOp::Floor, q);
        drop(g);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<BinOut> {
        // Python's floored modulo: a - b * floor(a / b).
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let mut g = self.g.borrow_mut();
        let q = g.div(self.id, o);
        let f = g.unary(UnaryOp::Floor, q);
        let bf = g.mul(o, f);
        let id = g.sub(self.id, bf);
        drop(g);
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
    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<BinOut> {
        if let Ok(n) = other.extract::<i64>() {
            let id = self.g.borrow_mut().pow_i(self.id, n);
            return Ok(BinOut::Value(self.wrap(id)));
        }
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let id = self.g.borrow_mut().binary(BinOp::Powf, self.id, o);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
            return Ok(BinOut::NotImplemented);
        };
        let id = self.g.borrow_mut().binary(BinOp::Powf, o, self.id);
        Ok(BinOut::Value(self.wrap(id)))
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<BinOut> {
        let Some(o) = self.operand_opt(other)? else {
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
    // numpy's binary ufunc methods on object arrays.
    fn arctan2(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().binary(BinOp::Atan2, self.id, o);
        Ok(self.wrap(id))
    }
    fn hypot(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().binary(BinOp::Hypot, self.id, o);
        Ok(self.wrap(id))
    }
    fn fmod(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().binary(BinOp::Mod, self.id, o);
        Ok(self.wrap(id))
    }
    fn maximum(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().reduce(ReduceOp::Max, vec![self.id, o]);
        Ok(self.wrap(id))
    }
    fn minimum(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().reduce(ReduceOp::Min, vec![self.id, o]);
        Ok(self.wrap(id))
    }
    fn power(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        match self.__pow__(other, None)? {
            BinOut::Value(t) => Ok(t),
            BinOut::NotImplemented => Err(PyTypeError::new_err(format!(
                "unsupported operand for a traced value: {}",
                other
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            ))),
        }
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
    /// Compile the tape to native code; evaluation switches over.
    fn compile_native(&self, py: Python<'_>) -> PyResult<()> {
        if self.native.get().is_some() {
            return Ok(());
        }
        let c = py
            .detach(|| rsdag_jit::NativeTape::compile(&self.tape))
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

/// `where(cond, a, b)` over tracers and numbers of one scope.
#[pyfunction]
fn select(cond: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Tracer> {
    let anchor = [cond, a, b]
        .into_iter()
        .find_map(|x| x.cast::<Tracer>().ok().map(|t| t.borrow().clone()))
        .ok_or_else(|| PyTypeError::new_err("select needs at least one traced operand"))?;
    let (c, x, y) = (
        anchor.operand(cond)?,
        anchor.operand(a)?,
        anchor.operand(b)?,
    );
    let id = anchor.g.borrow_mut().select(c, x, y);
    Ok(anchor.wrap(id))
}

/// The tracer among a list of operands, and the operands as expressions.
fn traced_list(items: &Bound<'_, PyAny>) -> PyResult<(Tracer, Vec<ExprId>)> {
    let items: Vec<Bound<'_, PyAny>> = items.try_iter()?.collect::<PyResult<_>>()?;
    let anchor = items
        .iter()
        .find_map(|x| x.cast::<Tracer>().ok().map(|t| t.borrow().clone()))
        .ok_or_else(|| PyTypeError::new_err("a traced operand is needed"))?;
    let ids = items
        .iter()
        .map(|x| anchor.operand(x))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((anchor, ids))
}

/// The inner product of two equal-length lists, one `Dot` node: rows of
/// one vector fuse into a matrix-vector kernel in the compiled program.
#[pyfunction]
fn dot(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Tracer> {
    let both = a
        .py()
        .eval(c"lambda a, b: list(a) + list(b)", None, None)?
        .call1((a, b))?;
    let (anchor, ids) = traced_list(&both)?;
    let n = ids.len() / 2;
    let (x, y) = ids.split_at(n);
    let id = anchor.g.borrow_mut().dot(x.to_vec(), y.to_vec());
    Ok(anchor.wrap(id))
}

/// A reduction (`"sum"`, `"product"`, `"min"`, `"max"`) over a list, one
/// `Reduce` node in the reference fold order.
#[pyfunction]
fn reduce(op: &str, items: &Bound<'_, PyAny>) -> PyResult<Tracer> {
    let rop = match op {
        "sum" => rsdag::ReduceOp::Sum,
        "product" => rsdag::ReduceOp::Product,
        "min" => rsdag::ReduceOp::Min,
        "max" => rsdag::ReduceOp::Max,
        _ => return Err(PyValueError::new_err(format!("unknown reduction '{op}'"))),
    };
    let (anchor, ids) = traced_list(items)?;
    let id = anchor.g.borrow_mut().reduce(rop, ids);
    Ok(anchor.wrap(id))
}

/// The solution of the dense system `A x = b`, `a` the `n*n` entries
/// row-major and `b` the `n` right-hand sides: one pivoting kernel in the
/// compiled program, differentiable.
#[pyfunction]
fn solve(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Vec<Tracer>> {
    let both = a
        .py()
        .eval(c"lambda a, b: list(a) + list(b)", None, None)?
        .call1((a, b))?;
    let (anchor, ids) = traced_list(&both)?;
    let n = rsdag::Graph::<rsdag::F64>::solve_n(ids.len());
    let (m, rhs) = ids.split_at(n * n);
    let xs = anchor.g.borrow_mut().solve_dense(m.to_vec(), rhs.to_vec());
    Ok(xs.into_iter().map(|id| anchor.wrap(id)).collect())
}

#[pymodule]
fn _rsdag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scope>()?;
    m.add_class::<Tracer>()?;
    m.add_class::<Program>()?;
    m.add_function(wrap_pyfunction!(select, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(reduce, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    let _ = PyTuple::empty(m.py());
    Ok(())
}
