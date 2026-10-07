# Structure per binding

Status: the backend part is built (`rsdag::structure`); SANE is its first
consumer.

## Problem

A frontend that lowers a model exactly keeps every branch on a
parameter. Most such branches change values only, and per-binding body
variants (`rsdag::variant`) remove their cost: each instance runs the body
its binding decides. Some change the *structure* of the system: a
resistance that is a branch for `R > 0` and a short for `R = 0` (PSP103's
`CollapsableR`, BSIM4's `rgateMod`, any `if (p) V(a,b) <+ 0`). Lowered
exactly, such a branch is a switch: an extra unknown (the branch current)
and a row that is `V(a) - V(b)` on one side of the condition and
`i - flow` on the other. The system carries both topologies in every
binding.

Measured on a 9-stage PSP103 ring (18 instances, 7 collapsible branches
each): 299 unknowns exact against 47 with the shorts folded, `nnz(A)` 1309
against 238, the same steps and Newton iterations, and twice the time per
iteration. Variants cannot help: the cost is the size of the linear
system, not the bodies.

Folding the condition in the frontend at the instance's values gets the
size back, but makes the model the frontend's guess: a binding across the
condition is either wrong or refused. That is not exact.

## Principle

The frontend lowers exactly and stays a thin translation. The backend
specializes per binding: the values a binding decides (body variants) and
the structure it decides (this). Both are exact: what runs for a binding
is the system the frontend lowered, at that binding, with what the binding
fixes taken out.

## The system and what a binding decides

`System`: the states, the parameters, time, and per row a current and a
charge, `F_r = i_r + d/dt q_r`. A binding's values fix every select whose
condition reads parameters only, at every call site (from the instance's
parameter arguments, through each function's conditions compiled once)
and in the system's own expressions. Under those decisions the analysis
finds, per row, which states its current and its charge read, and per
Jacobian entry whether it is free of the states and time, and its value
if so.

## Rules

Two exact rules, applied to a fixpoint; a [`Plan`] lists the steps.

1. **Alias.** A row without charge whose current reads one or two states,
   both entries free of the states: `c_a x_a + c_b x_b + g`. The row goes,
   `x_a = -(c_b x_b + g) / c_a` (`c_a` nonzero at the binding).
2. **Cut.** A state every row reads linearly (entries free of the states),
   no charge reads, and a pivot row with a nonzero entry. Every other row
   less its entry's share of the pivot row (current and charge alike) no
   longer reads it; the pivot row and the state go. Applied only where it
   adds no more entries (current and charge) than it removes, so a
   reduction never makes the system denser.

Which states may go is the consumer's (`System::eliminable`), and rows
pair with states by index: a step drops only the row of an eliminable
state, so a state that stays keeps its own row (a solver's shunt, a
homotopy's companion stamp, stay on it). A state that
a stored pivot charge reads is not cut later: a cut state's value carries
that charge's rate, which would otherwise need second derivatives.

## Exactness and reuse

The reduced system is built over the system's own expressions, its
branches kept: an alias substitutes `x_a`'s value, a cut combines every
row that reads the state through any branch with the symbolic coefficient
`J_si / J_ri` (zero where the binding's branch does not read it), and the
cut state is zero in the combined rows, where its terms cancel. So the
reduced system is exact at every binding whose plan is the same; the
plan, not the binding's pattern, keys the consumer's cache, and a value
that flips a branch without changing the structure (a binning boundary)
costs the plan's search, not a new system.

Values agree with the full system's to the solver's tolerance, not bit
for bit: the reduced Newton iteration is a different, equivalent one.

## What the consumer gets

`Reduced`:

- `states`: the kept states (full indices); `rows`: per reduced row the
  full row it continues, paired with its kept state where that state's own
  row survives.
- `currents`, `charges`: the reduced rows, over the kept states.
- `combination`: per reduced row, the full rows it sums and their
  coefficients (expressions free of the states at the plan's bindings),
  for quantities that live in the rows (a noise injection, an excitation).
- `expand`: every full state over the kept states, their rates (`rates`,
  one symbol per kept state), the parameters and time. A cut state is its
  pivot row solved for it, `-(i_r + d/dt q_r) / J_ri`, the rate of the
  charge through the kept states' rates: zero at DC, `jw` times the phasor
  in a small-signal analysis, the integrator's rate in a transient.
- `others`: expressions the consumer asked to carry along (guards,
  observers), over the kept states.

Each step removes one row and one state, so the reduced system stays
square. A reduction runs in rounds: steps that share no row and read
nothing another eliminates go through one substitution, so a circuit with
many instances is not reduced one step at a time.

## SANE as the first consumer

- `Model` keeps the full DAE as its public face: its unknowns, raw
  residual and Jacobian evaluation, assertions. Per binding it finds the
  plan, the reduced system of that plan (built and compiled on first use,
  cached), and runs every analysis on it.
- Results (operating points, waveforms, phasors, sensitivities) come back
  in the full layout through `expand`; inputs in the full layout
  (`x0`, initial conditions, nodesets) map onto the kept states.
- Device limits name unknowns: a limit on an aliased node follows its
  alias; one on an eliminated branch goes with the branch. Noise sources
  are injected into rows, mapped through `combination`.
- What SANE lets go at first: device-internal unknowns (internal nodes,
  switch and probe currents). Node voltages and source currents of the
  netlist stay.

## Consequences

- SANE's Verilog-A lowering drops the topology decisions with their
  assertions and the two-pass static node collapse: a short at top level
  is an alias row like any.
- Integer parameters can stay parameters (a follow-up): their mode selects
  are decided per binding like any.
