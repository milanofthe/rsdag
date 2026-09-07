//! Python frontend: operator-overloading trace of a Python function into an
//! rsgb graph, then differentiation and compilation.
//!
//! A `Scope` owns a `Graph<F64>`; `Tracer` values are expression handles
//! into it and overload Python arithmetic, comparisons and the numpy ufunc
//! method protocol (an object array of tracers under `np.sin` calls each
//! element's `sin`), so plain numpy code traces without changes. A closed
//! trace is a `Program`: a tape with an interpreter, an optional native
//! (Cranelift) form, symbolic derivatives, and C source.

use std::cell::RefCell;
use std::rc::Rc;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use rsgb::{BinOp, CmpOp, ExprId, Graph, Node, ReduceOp, SymbolId, Tape, UnaryOp, F64};

type Shared = Rc<RefCell<Graph<F64>>>;

/// An open graph: hands out input tracers and closes into programs.
#[pyclass(unsendable)]
pub struct Scope {
    g: Shared,
    inputs: Vec<SymbolId>,
}

/// An expression handle into a scope's graph.
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Tracer {
    g: Shared,
    id: ExprId,
}

impl Tracer {
    fn wrap(&self, id: ExprId) -> Tracer {
        Tracer {
            g: self.g.clone(),
            id,
        }
    }
    /// Coerce a Python operand (tracer, int, float) into an expression.
    fn operand(&self, other: &Bound<'_, PyAny>) -> PyResult<ExprId> {
        if let Ok(t) = other.downcast::<Tracer>() {
            let t = t.borrow();
            if !Rc::ptr_eq(&t.g, &self.g) {
                return Err(PyValueError::new_err("tracers from different scopes"));
            }
            return Ok(t.id);
        }
        if let Ok(v) = other.extract::<f64>() {
            return Ok(self.g.borrow_mut().konst_f64(v));
        }
        Err(PyTypeError::new_err(format!(
            "unsupported operand for a traced value: {}",
            other.get_type().name()?
        )))
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
        format!("Tracer({})", rsgb::to_string(&self.g.borrow(), self.id))
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().add(self.id, o);
        Ok(self.wrap(id))
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        self.__add__(other)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().sub(self.id, o);
        Ok(self.wrap(id))
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().sub(o, self.id);
        Ok(self.wrap(id))
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().mul(self.id, o);
        Ok(self.wrap(id))
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        self.__mul__(other)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().div(self.id, o);
        Ok(self.wrap(id))
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().div(o, self.id);
        Ok(self.wrap(id))
    }
    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let mut g = self.g.borrow_mut();
        let q = g.div(self.id, o);
        let id = g.unary(UnaryOp::Floor, q);
        drop(g);
        Ok(self.wrap(id))
    }
    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tracer> {
        // Python's floored modulo: a - b * floor(a / b).
        let o = self.operand(other)?;
        let mut g = self.g.borrow_mut();
        let q = g.div(self.id, o);
        let f = g.unary(UnaryOp::Floor, q);
        let bf = g.mul(o, f);
        let id = g.sub(self.id, bf);
        drop(g);
        Ok(self.wrap(id))
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
    ) -> PyResult<Tracer> {
        if let Ok(n) = other.extract::<i64>() {
            let id = self.g.borrow_mut().pow_i(self.id, n);
            return Ok(self.wrap(id));
        }
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().binary(BinOp::Powf, self.id, o);
        Ok(self.wrap(id))
    }
    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let id = self.g.borrow_mut().binary(BinOp::Powf, o, self.id);
        Ok(self.wrap(id))
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Tracer> {
        let o = self.operand(other)?;
        let c = match op {
            CompareOp::Lt => CmpOp::Lt,
            CompareOp::Le => CmpOp::Le,
            CompareOp::Gt => CmpOp::Gt,
            CompareOp::Ge => CmpOp::Ge,
            CompareOp::Eq => CmpOp::Eq,
            CompareOp::Ne => CmpOp::Ne,
        };
        let id = self.g.borrow_mut().cmp(c, self.id, o);
        Ok(self.wrap(id))
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "a traced value has no truth value: data-dependent control flow is not traceable, use rsgb.where",
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
        self.__pow__(other, None)
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
        rsgb::to_string(&self.g.borrow(), self.id)
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
        let rows = rsgb::jacobian(&mut g, &ids, &wrt);
        let flat: Vec<ExprId> = rows.into_iter().flatten().collect();
        let tape = Tape::compile(&g, &flat, &self.inputs);
        Ok(Program::new(tape, self.inputs.len(), flat.len()))
    }
    /// The gradient of one scalar output with respect to `inputs[wrt]`
    /// (reverse mode; all inputs when empty).
    #[pyo3(signature = (output, wrt = vec![]))]
    fn gradient(&self, output: &Tracer, wrt: Vec<usize>) -> PyResult<Program> {
        let wrt = self.wrt_symbols(&wrt)?;
        let mut g = self.g.borrow_mut();
        let grad = rsgb::gradient(&mut g, output.id, &wrt);
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
            if let Ok(t) = o.downcast::<Tracer>() {
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
/// `compile_native()`, by the Cranelift backend.
#[pyclass(unsendable)]
pub struct Program {
    tape: Tape,
    native: Option<rsgb_jit::ChunkedTape>,
    n_in: usize,
    n_out: usize,
    work: Vec<f64>,
    out: Vec<f64>,
}

impl Program {
    fn new(tape: Tape, n_in: usize, n_out: usize) -> Self {
        Program {
            tape,
            native: None,
            n_in,
            n_out,
            work: Vec::new(),
            out: Vec::new(),
        }
    }
}

#[pymethods]
impl Program {
    /// Evaluate on a flat list of inputs; returns the flat outputs.
    fn eval(&mut self, inputs: Vec<f64>) -> PyResult<Vec<f64>> {
        if inputs.len() != self.n_in {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.n_in,
                inputs.len()
            )));
        }
        match &self.native {
            Some(n) => n.eval(&inputs, &mut self.work, &mut self.out),
            None => self.tape.eval(&inputs, &mut self.work, &mut self.out),
        }
        Ok(self.out.clone())
    }
    /// Compile the tape to native code (Cranelift); evaluation switches over.
    fn compile_native(&mut self) -> PyResult<()> {
        let c = rsgb_jit::ChunkedTape::compile(&self.tape)
            .map_err(|e| PyValueError::new_err(format!("native compile failed: {e:?}")))?;
        self.native = Some(c);
        Ok(())
    }
    /// The program as a C function of the given name.
    fn c_source(&self, name: &str) -> PyResult<String> {
        rsgb_c::emit(&self.tape, name).map_err(|e| PyValueError::new_err(e.to_string()))
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
        .find_map(|x| x.downcast::<Tracer>().ok().map(|t| t.borrow().clone()))
        .ok_or_else(|| PyTypeError::new_err("select needs at least one traced operand"))?;
    let (c, x, y) = (
        anchor.operand(cond)?,
        anchor.operand(a)?,
        anchor.operand(b)?,
    );
    let id = anchor.g.borrow_mut().select(c, x, y);
    Ok(anchor.wrap(id))
}

#[pymodule]
fn _rsgb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scope>()?;
    m.add_class::<Tracer>()?;
    m.add_class::<Program>()?;
    m.add_function(wrap_pyfunction!(select, m)?)?;
    let _ = PyTuple::empty_bound(m.py());
    Ok(())
}
