# Installing BRISK

BRISK is a C program on top of BLAS/LAPACK. The command line `brisk` and the shared library
`libbrisk` are built with `make`. The Python, Julia and MATLAB/Octave interfaces call the same
C code, so every interface gives the same results as the command line.

Everything else is bundled: the AMD ordering (`amd/`, BSD 3-clause) and an OpenMP runtime for
compilers that lack one (`omp/`). Nothing has to be downloaded during the build.

Contents:
1. [Requirements](#1-requirements)
2. [The command line](#2-the-command-line)
3. [Python and CVXPY](#3-python-and-cvxpy)
4. [Julia and JuMP](#4-julia-and-jump)
5. [MATLAB and GNU Octave](#5-matlab-and-gnu-octave)
6. [Troubleshooting](#6-troubleshooting)
7. [Reporting a problem](#7-reporting-a-problem)

All commands below are run from the top directory of the unpacked package (`brisk-1.3.2/`).

## 1. Requirements

A C compiler, `make`, and BLAS/LAPACK with their development files. OpenBLAS is recommended:
the reference BLAS works but is several times slower.

| system | command |
|---|---|
| Debian, Ubuntu | `sudo apt install build-essential liblapack-dev libopenblas-dev` |
| Fedora, RHEL | `sudo dnf install gcc make openblas-devel`, then build with `make BLASLIB=-lopenblas` |
| Arch | `sudo pacman -S base-devel openblas`, then build with `make BLASLIB=-lopenblas` |
| macOS (Apple silicon or Intel) | `xcode-select --install`. BLAS/LAPACK come from Apple's Accelerate, and the OpenMP runtime is the bundled one. |
| Linux without root access | Python with SciPy: `make BLAS=scipy` links SciPy's bundled OpenBLAS. |

On Debian and Ubuntu, `libopenblas-dev` registers OpenBLAS as the system `libblas`/`liblapack`,
so the default `-llapack -lblas` links OpenBLAS.

## 2. The command line

```sh
make                   # builds ./brisk (about a minute)
./brisk examples/theta1.dat-s
```

The last lines of the output should read `status: OPTIMAL` and the optimal value
`2.2999999997e+01`. To check all examples:

```sh
for f in examples/*.dat-s; do ./brisk "$f" -q; echo "$f: exit code $?"; done
```

The other input formats have an example each: a linear program in MPS format, and a
second-order cone program in CBF and in SeDuMi format (optimal value 1.41421356):

```sh
./brisk examples/afiro.mps
./brisk examples/soc_small.cbf
./brisk examples/soc_small.mat
```

Exit code 0 means optimal. The other codes are listed in SOLVER.md ("Exit codes").

Build options (combine as needed):

| option | effect |
|---|---|
| `make BLASLIB="-lopenblas"` | a specific BLAS/LAPACK (also MKL etc.) |
| `make BLAS=scipy` | SciPy's bundled OpenBLAS |
| `make OMP=0` | no threads of BRISK's own (BLAS threads only) |
| `make OMP=bundled` | BRISK's own OpenMP runtime (needs clang; the default on macOS) |
| `make clean` | needed before switching options, e.g. after a `BLAS=scipy` build |

`./brisk` without arguments lists all solver options; [OPTIONS.md](OPTIONS.md) explains them and
shows how to pass them from Python, Julia and MATLAB (e.g. chordal decomposition off:
`-chordal 0`, `options={"chordal": 0}`, `set_attribute(model, "chordal", 0)`, `opts.chordal = 0`). `OMP_NUM_THREADS` sets the thread count
(default: all cores).

The shared library used by Python and Julia, and its self-test:

```sh
make libbrisk          # libbrisk.so (Linux) or libbrisk.dylib (macOS)
make capitest          # a small problem through the C interface; ends with "capi: all checks passed"
```

The C interface is `capi.c` (`brisk_solve_file`, `brisk_solve_data` and the `brisk_result_*` accessors; `tools/capi_test.c` is a worked example).

## 3. Python and CVXPY

Requirements: Python 3.9 or later. `pip` builds `libbrisk` with the Makefile and installs it
inside the package. No Python extension module is compiled.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install "./python[cvxpy]"
python -c "import brisk; print(brisk.version(), brisk.library_path())"
python python/examples/quickstart.py
```

Optional tests (a few minutes):

```sh
pip install "./python[test]"
python -m pytest python/tests
```

Build options are passed in `BRISK_MAKEFLAGS`, for example
`BRISK_MAKEFLAGS="BLASLIB=-lopenblas" pip install ./python`, or `BRISK_MAKEFLAGS="BLAS=scipy"`
without root access.

Use from CVXPY: `prob.solve(solver=brisk.BRISK())`. On SDPA data: `brisk.solve_file("problem.dat-s")`.
Full documentation: [python/README.md](python/README.md).

## 4. Julia and JuMP

Requirements: Julia 1.9 or later.

1. In a terminal, build the library (section 2) in the `brisk-1.3.2` directory: `make libbrisk`.
2. In Julia, add the package by its **full path**:

   ```julia
   using Pkg
   Pkg.develop(path = "/path/to/brisk-1.3.2/julia/Brisk")
   Pkg.add("JuMP")                        # for the JuMP interface
   using Brisk
   Brisk.version()
   ```

   - These steps are done once. Afterwards a Julia session only needs `using JuMP, Brisk`.
   - A relative path such as `julia/Brisk` is read from the directory Julia was started in,
     and fails elsewhere.
   - Julia records the path and does not copy the package, and the package loads `libbrisk`
     from the `brisk-1.3.2` directory: keep that directory in place. If you move it, run
     `Pkg.develop` once more with the new path.
   - The installation belongs to the active Julia environment; in another environment, run
     the two `Pkg` lines again.
   - After updating BRISK, run `make libbrisk` again; no Julia step is needed.
   - To use a library elsewhere, set `ENV["BRISK_LIBRARY"] = "/path/to/libbrisk.so"` before
     `using Brisk`.

3. A first model:

   ```julia
   using JuMP, Brisk, LinearAlgebra
   model = Model(Brisk.Optimizer)
   @variable(model, X[1:3, 1:3], PSD)
   @constraint(model, tr(X) == 1)
   @objective(model, Max, dot([2.0 1 0; 1 3 1; 0 1 1], X))
   optimize!(model)
   objective_value(model)
   ```

4. Optional tests:

   ```julia
   ENV["BRISK_TEST_MOI"] = "0"   # skips MathOptInterface's own test suite: about 40 s instead of 12 min
   Pkg.test("Brisk")
   ```

Examples are in `julia/Brisk/examples/`. Full documentation:
[julia/Brisk/README.md](julia/Brisk/README.md).

## 5. MATLAB and GNU Octave

The MEX interface is compiled from MATLAB or Octave by `build_brisk_mex`. It compiles the C
sources directly, so `make` is not needed for this interface.

**macOS (MATLAB).** Install the Xcode command-line tools (`xcode-select --install`) and run
`mex -setup C` once in MATLAB.

**Linux (MATLAB).** gcc and BLAS/LAPACK (section 1). MATLAB must have a supported gcc
(`mex -setup C`).

**GNU Octave.** Octave with its development files (Debian/Ubuntu: `sudo apt install octave
liboctave-dev`) and BLAS/LAPACK (section 1).

Then, in MATLAB or Octave:

```matlab
cd brisk-1.3.2/matlab
build_brisk_mex            % about a minute
test_brisk                 % every line should say PASS
addpath(pwd); savepath     % keeps it on the path
```

Use:
- SeDuMi data: `[x, y, info] = brisk_sedumi(A, b, c, K, opts)`.
- SDPA data: `[objVal, x, X, Y, INFO] = brisk_sdpa('problem.dat-s')`, or SDPA-M cells, or triplets.

Options go in a struct, e.g. `opts.timelimit = 60`. Full documentation:
[matlab/README.md](matlab/README.md).

## 6. Troubleshooting

- **`cannot find -llapack` / `-lblas`:** the BLAS/LAPACK development package is missing
  (section 1), or the library has another name: `make BLASLIB="-lopenblas"`.
- **Slow solves:** check which BLAS is linked (`ldd ./brisk | grep -i blas`). The reference
  BLAS is several times slower than OpenBLAS.
- **Threads on a busy machine:** with the default thread count BRISK uses only the cores
  that are free when the solve starts (Linux). If you fix the count (`-threads`,
  `OMP_NUM_THREADS`) and other processes occupy cores, spin-waiting OpenMP barriers stall:
  use fewer threads or set `OMP_WAIT_POLICY=passive`.
- **Build options changed but nothing happened:** `make clean` first.
- **macOS link error naming a `__kmpc_*` symbol:** run `make omptest` and send its output
  (the bundled OpenMP runtime lacks an entry point that this clang uses). The macOS build has
  not yet been tested on a Mac. Please report how it goes.
- **Python: `libbrisk not found`:** reinstall with `pip install --force-reinstall ./python`
  and watch the `building libbrisk:` line for compiler errors.
- **Julia: `libbrisk not found at ...`:** run `make libbrisk` in the top directory, or set
  `ENV["BRISK_LIBRARY"]`.

## 7. Reporting a problem

Please send:
- the command and the full output of `./brisk problem.dat-s -v`;
- the problem file, or how to generate it;
- the system (`uname -a`), the compiler (`cc --version`) and the BLAS (`ldd ./brisk | grep -i blas`).

The verbose output records every automatic decision BRISK made, so it usually suffices to
reproduce the run.
