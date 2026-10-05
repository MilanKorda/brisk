# BRISK from Python and CVXPY

The `brisk` Python package runs the BRISK SDP solver from Python. It has two layers:

- **low level**: SDPA files or the numbers of an SDPA problem (`brisk.solve_file`,
  `brisk.solve_sdpa`), with numpy arrays in and out;
- **CVXPY**: a conic solver, `prob.solve(solver=brisk.BRISK())`, for models with linear,
  second-order cone and semidefinite constraints.

The solver is BRISK's compiled C library, `libbrisk`, the same code as the `./brisk` command
line. Python calls it through `ctypes` (the standard library's foreign-function interface): the
problem data are handed over as pointers to numpy arrays and the whole solve runs in C, so the
Python layer costs milliseconds per solve. `pip install` builds `libbrisk` with BRISK's Makefile
and puts it inside the package; no Python extension module is compiled.

## Contents

1. [Installation on Linux](#installation-on-linux)
2. [Installation on macOS](#installation-on-macos)
3. [A first example](#a-first-example)
4. [CVXPY](#cvxpy)
5. [Low level: SDPA data](#low-level-sdpa-data)
6. [Interrupting a solve](#interrupting-a-solve)
7. [Reference](#reference)
8. [Troubleshooting](#troubleshooting)

## Installation on Linux

**1. Compiler, make, BLAS/LAPACK.** BRISK needs a C compiler with OpenMP (gcc), `make`, and
BLAS/LAPACK with their development links. OpenBLAS is recommended: the reference BLAS works,
but it is several times slower (theta3: 2.1 s against 0.4 s).

| distribution | command |
|---|---|
| Debian, Ubuntu | `sudo apt install build-essential liblapack-dev libopenblas-dev python3-venv` |
| Fedora, RHEL | `sudo dnf install gcc make openblas-devel` (then build with `BRISK_MAKEFLAGS="BLASLIB=-lopenblas"`) |
| Arch | `sudo pacman -S base-devel openblas` (then build with `BRISK_MAKEFLAGS="BLASLIB=-lopenblas"`) |
| no root access | nothing: build against SciPy's bundled OpenBLAS with `BRISK_MAKEFLAGS="BLAS=scipy"` (step 3) |

On Debian and Ubuntu, `libopenblas-dev` registers OpenBLAS as the system `libblas`/`liblapack`, so
the default `-llapack -lblas` links OpenBLAS.

**2. Python.** Python 3.9 or later, numpy and scipy; CVXPY 1.9 or later for the CVXPY
interface. A virtual environment keeps things separate:

```sh
tar xzf brisk-1.2.tar.gz
cd brisk-1.2
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

**3. Install.** From the BRISK source directory:

```sh
pip install "./python[cvxpy]"         # the package, libbrisk, and CVXPY
```

`pip` runs `make libbrisk` in the source directory; that takes about a minute. Extra `make`
arguments go through `BRISK_MAKEFLAGS`, for example:

```sh
BRISK_MAKEFLAGS="BLASLIB=-lopenblas" pip install "./python[cvxpy]"   # Fedora
pip install numpy scipy setuptools                                   # no system BLAS: SciPy's
BRISK_MAKEFLAGS="BLAS=scipy" pip install --no-build-isolation "./python[cvxpy]"
BRISK_MAKEFLAGS="-j4" pip install "./python[cvxpy]"                  # parallel build
```

`BLAS=scipy` links the OpenBLAS of the SciPy installed in the environment. The build must see
that SciPy, which pip's isolated build environment hides, hence `--no-build-isolation`. If you
later upgrade or remove SciPy, reinstall `brisk`. A build whose BLAS cannot be linked fails at
link time (`-Wl,--no-undefined`), not at import.

**4. Check.**

```sh
python -c "import brisk; print(brisk.version(), brisk.library_path())"
python python/examples/quickstart.py
pip install "./python[test]" && python -m pytest python/tests     # 46 tests, about 5 s
```

## Installation on macOS

**1. Compiler.** Apple's command-line tools provide clang and make:

```sh
xcode-select --install
```

Nothing else is needed. BLAS and LAPACK come from Apple's Accelerate framework, and threads
from BRISK's own OpenMP runtime (`omp/`), since Apple's clang ships no OpenMP library. No
Homebrew packages are required.

**2. Python.** Python 3.9 or later: from python.org, Homebrew (`brew install python`) or conda.

```sh
tar xzf brisk-1.2.tar.gz
cd brisk-1.2
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

**3. Install and check.**

```sh
pip install "./python[cvxpy]"         # builds libbrisk.dylib (Accelerate, bundled OpenMP)
python -c "import brisk; print(brisk.version(), brisk.library_path())"
python python/examples/quickstart.py
pip install "./python[test]" && python -m pytest python/tests
```

The build is for the machine's own processor (`-mcpu=native` on Apple silicon, `-march=native`
on Intel).

Status: the macOS build of `libbrisk` has not yet been run on a Mac. If the
build fails, run `make omptest` in the source directory and report its output. A link error
naming a `__kmpc_*` symbol means the bundled OpenMP runtime lacks an entry point that Apple's
clang uses.

## A first example

`python/examples/quickstart.py`, the max-cut relaxation of a random graph:

```python
import cvxpy as cp
import numpy as np
import brisk

rng = np.random.default_rng(0)
n = 20
W = np.triu(rng.random((n, n)) < 0.3, 1).astype(float)
W = W + W.T                                     # adjacency matrix
L = np.diag(W.sum(1)) - W                       # Laplacian

X = cp.Variable((n, n), symmetric=True)
prob = cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), [X >> 0, cp.diag(X) == 1])
prob.solve(solver=brisk.BRISK(), threads=2)     # any ./brisk option as a keyword

print(prob.status, prob.value)                  # optimal 38.822515...
print(prob.solver_stats.extra_stats["dimacs"])  # DIMACS errors, all below 1e-9
print(prob.constraints[1].dual_value[:5])       # duals of diag(X) == 1
```

The other examples:

- `examples/sos_univariate.py`: a sum-of-squares lower bound on the minimum of
  x⁴ − 3x² + x (exact: −3.5139050).
- `examples/sdpa_lowlevel.py`: an SDPA problem given as numbers and as a file, without CVXPY.

## CVXPY

```python
prob.solve(solver=brisk.BRISK(), verbose=True, acc="high", timelimit=600, threads=4)
```

Pass an instance of `brisk.BRISK` as the solver; CVXPY 1.9 or later is required.

**Constraints.** Supported: equalities, inequalities, second-order cones (including everything
CVXPY reduces to them: norms, quadratic objectives and constraints) and semidefinite constraints
(`X >> 0`, `PSD=True` variables). Not supported: exponential and power cones; integer variables.

**Options.** Every keyword that is not one of the interface's own options below is passed to
BRISK as a command-line option: `acc="high"` becomes `-acc high`, `threads=4` becomes
`-threads 4`, `chordal=0` turns the chordal decomposition off, `sym="none"` the symmetry
reduction. A flag takes `True` (or 1): `nohsd=True`. An underscore stands for a dash. Every
option is listed and explained in [OPTIONS.md](../OPTIONS.md); `./brisk` with no arguments
prints the same list. The interface's own:

| option | default | meaning |
|---|---|---|
| `form` | `"auto"` | how the model is handed to BRISK: `"image"`, `"kernel"` or `"auto"` (below) |
| `soc` | `"auto"` | second-order cones as `"arrow"` blocks, as a `"tree"` of 2×2 blocks, or `"auto"` (arrow in the image form, tree in the kernel form) |
| `soc_arrow_max` | none | with `soc="arrow"`: cones larger than this use the tree |
| `options` | none | raw command-line strings, e.g. `options=["-acc", "high"]` |
| `write_sdpa` | none | a file name: write the SDPA problem handed to BRISK, for `./brisk` or other solvers |
| `bound` | none | `"primal"` or `"sos"` (a feasible point of the model, e.g. the SOS Gram matrices: a guaranteed bound on the value), `"dual"` (a feasible dual). The bound, on CVXPY's minimisation form min c'x + offset, is in `extra_stats["bound"]` (`value`, `kind` upper/lower, `valid`); for a Maximize model CVXPY minimises −f, so the sign and kind flip |
| `inaccurate_tol` | `1e-4` | after an iteration or time limit, an interrupt or numerical difficulties: `OPTIMAL_INACCURATE` when the largest DIMACS error is at most this, else `USER_LIMIT` / `SOLVER_ERROR` |

**Forms.** CVXPY gives the solver `min c'x  s.t.  b − A x ∈ K`. The interface writes this as an
SDPA problem in one of two ways, as the Julia package does:

- **image form**: CVXPY's variables become the SDPA variables, and each cone becomes a block of
  the matrix `Σ xᵢ Fᵢ − F₀ ⪰ 0`. Equalities become pairs of inequalities, which BRISK's
  presolve eliminates. This is the natural form of moment relaxations (free moments in LMIs).
- **kernel form**: the cones become BRISK's matrix variable `X`, with one equality per entry.
  A cone entry that is exactly one variable (a `PSD=True` or symmetric matrix variable in
  `X >> 0`, or `x >= 0`) *is* that entry of `X`, and needs no equality. This is the natural form of
  SOS programs and SDPs over matrix variables.

`form="auto"` takes the form that leaves BRISK fewer rows after its presolve, using the same rule
as the Julia package and BRISK's own dual form. The form used is reported in
`prob.solver_stats.extra_stats["form"]`.

**Results.**

- `prob.value`, `x.value` and `constraint.dual_value` follow CVXPY's conventions. The dual of a
  PSD constraint is the dual matrix.
- `prob.solver_stats.extra_stats` holds BRISK's status string, the DIMACS errors, the exit code,
  the form, the block sizes, and whether the solve was interrupted.

**Statuses.**

| BRISK | CVXPY |
|---|---|
| OPTIMAL | `OPTIMAL` |
| REDUCED ACCURACY | `OPTIMAL_INACCURATE` |
| infeasible | `INFEASIBLE`, with a Farkas certificate in the duals (`A'z = 0`, `b'z = −1`, `z ∈ K*`) |
| unbounded | `UNBOUNDED` |
| iteration or time limit, interrupt, numerical difficulties | `OPTIMAL_INACCURATE` if the DIMACS errors are at most `inaccurate_tol`, else `USER_LIMIT` or `SOLVER_ERROR` |

BRISK's statuses refer to its own (P) and (D). The interface translates them, since in the image
form BRISK's (P) is CVXPY's dual problem.

## Low level: SDPA data

```python
import brisk
r = brisk.solve_file("theta3.dat-s", {"acc": "high"})           # as ./brisk theta3.dat-s -acc high
r = brisk.solve_sdpa(blocksizes, c, mat, blk, i, j, v, options=["-threads", "2"], verbose=False)
brisk.write_sdpa("problem.dat-s", blocksizes, c, mat, blk, i, j, v)
```

The problem is the numbers of an SDPA sparse file:

    min c'x   s.t.   x₁F₁ + … + x_mF_m − F₀ ⪰ 0.

- `blocksizes` lists the block sizes; LP (diagonal) blocks are negative.
- The entries are `(mat[q], blk[q], i[q], j[q], v[q])`, 1-based as in the file: matrix
  (0 = F₀), block, row, column, value. An entry with `i > j` is swapped; only the upper
  triangle is needed.
- Nothing is copied: numpy arrays of any integer and float types are accepted.

`examples/sdpa_lowlevel.py` shows a complete call. The result, a `brisk.Result`:

| field | contents |
|---|---|
| `status`, `status_str` | 0 `OPTIMAL`, 1 `PRIMAL INFEASIBLE`, 2 `DUAL INFEASIBLE`, 3 iteration limit, 4 numerical difficulties, 5 reduced accuracy, 6 time limit (`INTERRUPTED` after an interrupt) |
| `x`, `y` | the SDPA variable x, and BRISK's y = −x |
| `X`, `Z` | lists of numpy arrays per block (n×n, or a vector for an LP block): X is the SDPA dual matrix, Z = F(x) |
| `pobj`, `dobj` | BRISK's ⟨C, X⟩ and b'y (the SDPA value c'x is −dobj) |
| `dimacs` | the six DIMACS errors on the data as given |
| `iterations`, `time`, `exit_code`, `blocksizes`, `interrupted` | |
| `bound` | with the option `bound` (`-bound p\|d`): `side`, `kind` (upper/lower bound on BRISK's (P) value), `value`, `resid`, `lammin`, `valid` (2 with the margin, 1 PSD, 0 not) |
| `n_resolves`, `t_resolves`, `cause` | re-solves run and their time; the likely cause when the tolerance is missed |

Conventions are those of the command line (`brisk.h`): BRISK solves (P) min ⟨C, X⟩ s.t.
⟨Aᵢ, X⟩ = bᵢ, X ⪰ 0 and (D) max b'y s.t. C − Σ yᵢAᵢ = Z ⪰ 0, with C = −F₀, Aᵢ = Fᵢ and b = c.

Options are those of the command line ([OPTIONS.md](../OPTIONS.md)), given as a dict
(`{"acc": "high", "tol": 1e-9, "chordal": 0}`; `True` or 1 gives a bare flag, `None`/`False`
drops the option; an underscore stands for a dash), a list of strings, or one string. Bad
options raise `RuntimeError`. `verbose=True` sends the solver's output through `sys.stdout`,
so it shows in Jupyter.

**Very large chordal problems.** When BRISK decomposes a large sparse block into cliques
(AC-OPF relaxations), the dense `X` and `Z` of that block are returned only if they fit in
memory: above 30% of the memory, `r.X` and `r.Z` are `None`, the point is verified on the
cliques, and the status and `r.dimacs` are as usual. `options={"returnx": 0}` never forms the
dense X (fastest), `"returnx": 1` always does. `r.y` is always returned.

## SeDuMi format, linear and second-order cone programs

```python
import numpy as np, scipy.sparse as sp, brisk
# min c'x  s.t.  A x = b,  x in K;   K: f free, l nonnegative, q second-order cones (x0 >= |x(1:)|),
# r rotated cones (2 x0 x1 >= |x(2:)|^2), s semidefinite blocks (d*d entries by columns), in this order
A = sp.csc_matrix([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
r = brisk.solve_sedumi(A, b=[2.0, 1.0], c=[1.0, 0.0, 0.0, 1.0], K={"l": 1, "q": [3]})
r.status, r.status_str, r.pobj, r.dobj, r.x, r.y, r.z, r.dimacs, r.iterations, r.time
r = brisk.solve_file("problem.mat")          # a MAT-file with A (or At), b, c, K
```

Without semidefinite blocks the problem is solved by BRISK's cone solver (a separate
interior-point code for linear and second-order cone programs); its options are `tol`, `acc`,
`maxit`, `timelimit`, `threads`. With semidefinite blocks, or with `conesolver=0`, `prec`,
`bound`, `certify`, `fom`, `mfipm`, `lralm`, the semidefinite solver is used (second-order
cones as arrow blocks) with all its options. The result is a `brisk.ConeResult`: `x`, `y`,
`z = c - A'y` in the SeDuMi order; for an infeasible problem `y` (status 1, b'y = 1) or `x`
(status 2, c'x = −1) is the certificate.

In CVXPY nothing changes: a model without PSD constraints (LP, QP, SOCP) goes to the cone
solver, `prob.solve(solver=brisk.BRISK(), conesolver=0)` forces the semidefinite solver.

## Interrupting a solve

A call into C cannot be interrupted by Python. So a solve started from the main thread runs in
a worker thread, and the main thread waits for it:

- **Ctrl-C** (or `brisk.interrupt()` from any thread) stops the solver as if its time limit had
  been reached. You get the current iterate with `status_str == "INTERRUPTED"`, and CVXPY gets
  `USER_LIMIT` or `OPTIMAL_INACCURATE`. The solver checks at every iteration.
- **A second Ctrl-C** abandons the solve and raises `KeyboardInterrupt`. The solver finishes its
  current step in the background, and the next solve waits for it. The symmetry search and the
  presolve do not check for interrupts, so on very large problems the first Ctrl-C can take a
  while.
- `brisk.set_interruptible(False)` runs solves in the calling thread instead; Ctrl-C then waits
  until the solve ends. Solves started from other threads always run in their own thread.

The library is not reentrant, so solves are serialized by a lock. Solves started from several
threads run one after the other.

## Reference

| function | |
|---|---|
| `brisk.BRISK()` | the CVXPY solver |
| `brisk.solve_file(path, options=None, verbose=True)` | solve an SDPA file |
| `brisk.solve_sdpa(blocksizes, c, mat, blk, i, j, v, options=None, verbose=True)` | solve SDPA data |
| `brisk.write_sdpa(path, blocksizes, c, mat, blk, i, j, v)` | write an SDPA file |
| `brisk.hp_solution(convert=None)` | all the digits of the last high-precision solve (see below) |
| `brisk.interrupt()` | stop the running solve |
| `brisk.set_interruptible(flag)` | run main-thread solves in a worker thread (default `True`) |
| `brisk.version()`, `brisk.library_path()`, `brisk.load_library(path)` | the loaded library |
| `brisk.options_to_argv(options)` | the command-line strings for an options dict or list |
| `brisk.cvxpy_solver.cone_program_to_sdpa(A, b, c, dims, form, soc)` | CVXPY's conic data as SDPA data (for inspection) |

**Where the library is found.** The first match wins:

1. `$BRISK_LIBRARY`;
2. the copy inside the installed package;
3. the source tree (`python/brisk/../../libbrisk.so`).

On Linux it is opened with `RTLD_DEEPBIND`, so that its BLAS/LAPACK calls go to the library it
was linked with, not to one that numpy or another host library has already loaded. This was
tested with a host library exporting `dgemm_`/`dpotrf_`: with `RTLD_DEEPBIND` the solve is correct;
without it the host's `dpotrf_` is called. macOS's two-level namespaces make this automatic.

**Development install.** `make libbrisk` in the source directory, then
`pip install -e "./python[test]"`; the editable package uses `../libbrisk.so`. `make pytest` runs
the tests.

**Build variables.**

| variable | effect |
|---|---|
| `BRISK_MAKEFLAGS` | extra `make` arguments (`BLAS=scipy`, `BLASLIB=...`, `OMP=0`, `-j4`) |
| `BRISK_NO_BUILD=1` | do not run `make`: use an existing `libbrisk` in the source directory |
| `MAKE` | the make program |

The library is built for the build machine's processor (`-march=native`). For a wheel meant for
other machines, set `BRISK_MAKEFLAGS="ARCHFLAG=-march=x86-64-v2"`.

## High precision

The option `prec` solves in double-double (`"dd"`, about 32 digits), quad-double (`"qd"`,
about 64) or a variable precision (a number of digits):

```python
r = brisk.solve_file("theta1.dat-s", {"prec": 100})        # or prob.solve(solver=brisk.BRISK(), prec="qd")
h = brisk.hp_solution()                                    # all the digits, as decimal.Decimal
h["digits"], h["pobj"], h["dobj"], h["y"], h["X"][0], h["Z"][0]
h = brisk.hp_solution(mpmath.mpf)                          # or any type built from a string
```

`r` holds the solution rounded to doubles and the errors measured in the working precision.
With `solve_file` the numbers of the file are read exactly as written; data passed as
arrays (`solve_sdpa`, CVXPY) are doubles and are taken exactly as given. See OPTIONS.md
(`prec`, `hptol`, `hpext`) for the tolerance, the extension levels and the status PRECISION LIMIT.

Rigorous bounds in high precision: with `bound` the certificate is built and checked in the
precision of `prec`.

```python
r = brisk.solve_file("theta1.dat-s", {"prec": "dd", "bound": "d"})
r.bound["certified"], r.bound["rigorous"]                  # the bound rounded outward to a double
h = brisk.hp_solution()
h["bound"]                                                 # the rigorous lower bound, all digits
c = brisk.certify("theta1.dat-s", "d", y=h["y"], prec="dd")   # checks a certificate again
```

## Troubleshooting

- **`libbrisk not found`**: install with `pip install ./python` from the source directory, run
  `make libbrisk` for an editable install, or set `BRISK_LIBRARY` to the library's path.
- **`cannot find -llapack` / `-lblas`**: the development packages are missing (step 1).
  Alternatively, build with `BRISK_MAKEFLAGS="BLASLIB=-lopenblas"` or `"BLAS=scipy"`.
- **The solver is slow**: check which BLAS the library uses (`ldd $(python -c "import brisk;
  print(brisk.library_path())")` on Linux). Reference BLAS is several times slower than OpenBLAS.
  Set `threads=` explicitly.
- **`SolverError: ... bad options`**: a keyword is not a BRISK option; `./brisk` lists them.
- **CVXPY says `Solution may be inaccurate`**: BRISK stopped at reduced accuracy or at a limit.
  The DIMACS errors in `prob.solver_stats.extra_stats["dimacs"]` show how far off it is. Try
  `acc="high"` or a longer `timelimit`.
