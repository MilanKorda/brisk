# Brisk.jl: BRISK from Julia and JuMP

`Brisk.jl` calls BRISK through its shared library `libbrisk` (C interface `capi.c`). The
solver is the same code as the command line: the same presolve, method choice and
fallbacks, and bit-identical results on the same problem.

## Install

1. Build the shared library in the BRISK source directory:

   ```sh
   make libbrisk            # libbrisk.so (Linux) or libbrisk.dylib (macOS)
   make capitest            # optional: the C interface against the file path
   ```

   It takes the same options as the command-line build, e.g. `make libbrisk OMP=bundled`
   or `BLASLIB=...`. On macOS it uses Accelerate and BRISK's bundled OpenMP runtime, so
   there is nothing else to install.

2. Add the package to your Julia environment (Julia ≥ 1.9):

   ```julia
   using Pkg
   Pkg.develop(path = "/path/to/brisk/julia/Brisk")
   ```

   This is done once; afterwards a session only needs `using JuMP, Brisk`. Use the full
   path (a relative one is read from the directory Julia was started in), and keep the BRISK
   directory in place: Julia records the path and the package loads the library from there.

   The package finds the library in the BRISK directory that contains `julia/Brisk`. To use
   a library somewhere else, set `ENV["BRISK_LIBRARY"] = "/path/to/libbrisk.so"` before
   `using Brisk`, or call `Brisk.load_library(path)`.

3. Check the installation: `using Brisk; Brisk.version()`, then `Pkg.test("Brisk")`. The
   tests take about 12 minutes on 2 cores, most of it MathOptInterface's test suite (3 698
   checks in all); set `ENV["BRISK_TEST_MOI"] = "0"` to skip that part (40 s).

## JuMP

```julia
using JuMP, Brisk, LinearAlgebra
C = [2.0 1 0; 1 3 1; 0 1 1]
model = Model(Brisk.Optimizer)
@variable(model, X[1:3, 1:3], PSD)
@constraint(model, tr(X) == 1)
@objective(model, Max, dot(C, X))
optimize!(model)
objective_value(model), value.(X)
```

Options:

| what | how |
|---|---|
| any BRISK option ([OPTIONS.md](../../OPTIONS.md); `./brisk` without arguments prints the list) | `set_attribute(model, "acc", "high")`, `set_attribute(model, "chordal", 0)` (chordal decomposition off; -1 auto, 1 force), `set_attribute(model, "sym", "none")`, `set_attribute(model, "fom", 1)`; a flag takes `true` or 1: `set_attribute(model, "nohsd", true)`; an underscore stands for a dash. In the low-level calls the same as keywords: `Brisk.solve_sdpa(file; chordal = 0, acc = "high")` |
| no output | `set_silent(model)` |
| time limit | `set_time_limit_sec(model, 60)` (BRISK's `-timelimit`) |
| threads | `MOI.set(model, MOI.NumberOfThreads(), 2)` (BRISK's `-threads`) |
| where the output goes | `set_attribute(model, "output", :julia)` (the default: Julia's `stdout`, so it works in notebooks and with `redirect_stdout`), `:stdout` (written by C, as on the command line), `:silent` |
| the SDPA file of the model | `set_attribute(model, "write_sdpa", "model.dat-s")`: written before each solve, so `./brisk model.dat-s` reproduces it |
| the SDPA form the model is passed in | `set_attribute(model, "form", "auto")` (the default), `"kernel"` or `"image"`: see "How the model is passed" |

After the solve, `MOI.get(model, Brisk.DIMACSErrors())` returns BRISK's six DIMACS
errors, and `MOI.get(model, Brisk.RawResult())` returns the `Brisk.Result` in BRISK's
convention.

Supported: linear objectives, and affine functions in `Zeros`, `Nonnegatives` and
`PositiveSemidefiniteConeTriangle`. Through MOI's bridges this covers equalities,
inequalities, variable bounds, `PSDCone()` in both square and triangle forms,
`SecondOrderCone`, `RotatedSecondOrderCone`, the nuclear and spectral norm cones, and the
Hermitian PSD cone. There are no integer variables, and no exponential or power cones.

SumOfSquares.jl works (`SOSModel(Brisk.Optimizer)`, see `examples/sos.jl`). Its Gram
matrices reach BRISK as the X blocks of the kernel form.

Measured on 2 cores, with warm timings and the time inside BRISK:

| model | form, rows | time |
|---|---|---|
| max-cut, 200 × 200 PSD variable | kernel, 200 | 0.10 s |
| max-cut, 400 × 400 PSD variable | kernel, 400 | 0.23 s |
| max-cut, 800 × 800 PSD variable | kernel, 800 | 1.7 s (`optimize!` 2.3 s) |
| SOS lower bound of a quartic in 8 variables | kernel, 495 | 0.7 s (image form: 2.3 s) |
| SOS lower bound of a quartic in 12 variables | kernel, 1 820 | 2.6 s (image form: 8.4 s) |

## Without JuMP

```julia
r = Brisk.solve_sdpa("theta3.dat-s"; acc = "high", threads = 2)
r.status, r.primal_objective, r.y, r.X, r.Z, r.dimacs

# the numbers of an SDPA file, in memory (nothing written to disk)
r = Brisk.solve_sdpa_data(m, blocksizes, c, mat, blk, i, j, v; output = :silent)
Brisk.write_sdpa("same.dat-s", m, blocksizes, c, mat, blk, i, j, v)
```

`Brisk.Result` uses BRISK's convention: (P) min ⟨C,X⟩ s.t. ⟨A_i,X⟩ = b_i, X ∈ K, and
(D) max b'y s.t. C − Σ y_i A_i = Z ∈ K, with C = −F0, A_i = F_i and b = c of the SDPA
file. The SDPA x is −y, and "primal/dual infeasible" refer to (P)/(D).

**Very large chordal problems.** When BRISK decomposes a large sparse block into cliques
(AC-OPF relaxations), the dense `X` and `Z` of that block are returned only if they fit in
memory. Above 30% of the memory, `X` is `nothing`, the large blocks of `Z` are empty matrices,
and the point is verified on the cliques; the status and `dimacs` are as usual. The option
`returnx = 0` never forms the dense X (fastest), `returnx = 1` always does.

**High precision.** The option `prec` solves in double-double (`"dd"`, about 32 digits),
quad-double (`"qd"`, about 64) or a variable precision (a number of digits):

```julia
r = Brisk.solve_sdpa("theta1.dat-s"; prec = 100)           # with JuMP: set_attribute(model, "prec", "qd")
h = Brisk.hp_solution()                                     # all the digits, as BigFloat
h.digits, h.pobj, h.dobj, h.y, h.X[1], h.Z[1]
```

`r` holds the solution rounded to `Float64` and the errors measured in the working
precision. With `solve_sdpa` on a file the numbers are read exactly as written; data passed
as arrays (`solve_sdpa_data`, JuMP) are `Float64` and are taken exactly as given. See
OPTIONS.md (`prec`, `hptol`, `hpext`) for the tolerance, the extension levels and the status PRECISION LIMIT.
With `bound = "d"` (or `"p"`) the certificate is built and checked in the precision of
`prec`: `Brisk.hp_solution().bound` is the rigorous bound with all the digits (`nothing`
without a certificate), and `y` (or `X`) is the certificate.

## How the model is passed (for developers)

**The forms.** After MOI's bridges a model is `min a0'x s.t. A_k x + b_k ∈ K_k`, with
K_k ∈ {Zeros, Nonnegatives, PSDConeTriangle}. The wrapper (`src/MOI_wrapper.jl`) hands it
to BRISK in one of two SDPA forms:

- **Image form.** x is free: the SDPA primal `Σ x_j F_j − F0 ⪰ 0`.
  - Each PSD constraint is an SDP block, and the Nonnegatives rows go into one LP block.
  - A Zeros row becomes a ± pair in the LP block: split free variables of BRISK's (P),
    which its presolve eliminates.
  - This is the natural form of moment relaxations (free moments in LMIs). BRISK's rows are
    the variables.
- **Kernel form.** BRISK's (P), `min ⟨C,X⟩ s.t. ⟨A_i,X⟩ = b_i, X ∈ K`.
  - A *variable block* is a cone constraint whose rows are distinct variables with
    coefficient 1 and no constant, which is what JuMP creates for
    `@variable(model, X[1:n,1:n], PSD)` and for `x >= 0`. Its variables are entries of X.
  - Every other cone constraint gets a slack block S = A x + b, with one row per entry.
  - Every Zeros row is a row.
  - Free variables become split pairs.
  - This is the natural form of SOS programs and of SDPs over matrix variables. BRISK's
    rows are the equalities plus the slack entries.
- **`form = "auto"`** (the default) counts the rows left once the split pairs are
  eliminated, which BRISK does at one row per pair. It takes the form with fewer pairs
  unless the other has fewer than half its rows, the rule of BRISK's own dual form. JuMP's
  PSD-variable models therefore go in the kernel form and moment relaxations in the image
  form. `set_attribute(model, "form", "kernel")` or `"image"` forces a form.
  - Forcing the form is for experiments. MOI's suite passes with every model in the image
    form. With every model in the kernel form, a few bridged problems whose variables are
    all free (a geometric-mean cone, an SOC) lose accuracy: their variables are then
    differences of split pairs, accurate to about 1e-4.
- **Why the choice is made here.** BRISK's automatic dual form would find the cheap side of
  either file, but only after paying for the presolve of the expensive one. Also, its
  automatic first-order rule looks at the row count as read. A max-cut over a
  400 × 400 PSD variable in the image form has 80 200 rows as read, so BRISK went to the
  first-order engine (337 s); in the kernel form it is 400 rows.
- Variables that appear in no constraint are left out. If one has an objective
  coefficient, the model is reported unbounded when the rest is feasible.

**The results.** BRISK returns X of (P), and y and Z = C − A'y of (D). MOI's
`PSDConeTriangle` is self-dual under `set_dot`, which doubles off-diagonal products, so the
dual of entry (i, j) is entry (i, j) of the dual matrix.

| | variable values | duals |
|---|---|---|
| image form | x = −y | PSD row (i, j): X_ij; Nonnegatives row: X_lp; Zeros row: X_lp(+) − X_lp(−) |
| kernel form | x from X; a free variable is X_lp(+) − X_lp(−) | variable-block row: Z at the variable's entry; slack row: Z at the slack entry; Zeros row: y_i |

In both forms the constraint primals A x + b are computed in Julia from the data, and the
dual objective is computed from the duals.

**Infeasibility.** BRISK's statuses refer to its (P) and (D).

- In the image form, (P) is MOI's dual. BRISK's `PRIMAL INFEASIBLE` is `DUAL_INFEASIBLE`
  with the primal ray −y, and `DUAL INFEASIBLE` is `INFEASIBLE` with a dual ray from X.
- In the kernel form, (P) is MOI's primal, so the two are swapped. The dual ray comes from
  y and Z − C.
- BRISK returns the diverging iterate (for example an X of norm 1e8 with A(X) = c), so
  certificates are scaled to unit max-norm. After scaling they satisfy the ray equations to
  about 1e-8.

**Statuses.**

| BRISK status | MOI status |
|---|---|
| OPTIMAL | `OPTIMAL` |
| SOLVED TO REDUCED ACCURACY | `ALMOST_OPTIMAL` |
| ITERATION LIMIT | `ITERATION_LIMIT` |
| NUMERICAL DIFFICULTIES | `NUMERICAL_ERROR` |
| TIME LIMIT | `TIME_LIMIT` |

The primal and dual result statuses come from the DIMACS errors of the side that is MOI's
primal and of the side that is MOI's dual. Errors 1 and 2 belong to (P) and errors 3 and 4
to (D); which of them is MOI's primal depends on the form.

**The C interface** (`capi.c`, `brisk.h`):

- `brisk_solve_data` and `brisk_solve_file` take plain C arrays and option strings, and
  fill an opaque result that is read through accessor functions. Callers never depend on
  the layout of a struct.
- The in-memory path (`brisk_run_data`, `problem_from_sdpa_data`) feeds the same pipeline
  as the file. Retries rebuild the problem from the same arrays, so the Julia wrapper keeps
  them alive during the solve.
- The library exports only the `brisk_*` symbols, through a version script (Linux) or an
  exported-symbols list (macOS).
- On Linux the package opens it with `RTLD_DEEPBIND`. Julia's `libblastrampoline` exports
  the LP64 BLAS names (`dgemm_`, ...) and forwards them to an ILP64 backend, so without
  deep binding BRISK's calls would go there. With deep binding they reach the BLAS it was
  linked with.
- `OPENBLAS_NUM_THREADS=1` is set while that BLAS loads, as the command line does for its
  OpenMP threads.

**Limits.**

- One solve at a time per process: the solver has global state, and the package holds a
  lock.
- A solve cannot be interrupted with Ctrl-C (use a time limit).
- While BRISK runs, the Julia thread that called it doesn't reach a GC safepoint, so other
  Julia threads that need a garbage collection wait for the solve to finish.
- The bridged SOC tests of MOI's suite need `acc = "high"` for 1e-4 duals, because on a
  curved face the duals are accurate to about √tol. `crossover = 1` makes them exact.
- Windows is not supported (neither is the command line).

The tests are in `test/runtests.jl`:

- the low-level interface against the file path (bit-identical results);
- small JuMP models with known answers in each form (`auto`, `image`, `kernel`): both
  senses, a PSD variable, the moment form, a mixed model, SOC through bridges, infeasible
  and unbounded models;
- the options and the SDPA dump;
- `MOI.Test.runtests`.
