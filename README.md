# BRISK

[![CI](https://github.com/MilanKorda/brisk/actions/workflows/ci.yml/badge.svg)](https://github.com/MilanKorda/brisk/actions/workflows/ci.yml)

BRISK is an interior-point solver for semidefinite programs (SDPs) with linear blocks:

```
(P)  min <C,X>  s.t. <A_i,X> = b_i,  X in K        (D)  max b'y  s.t.  sum y_i A_i + Z = C,  Z in K
```

It is written in C on BLAS/LAPACK, reads SDPA sparse files (`.dat-s`), and is called from the
command line, Python (CVXPY), Julia (JuMP), MATLAB and GNU Octave. All interfaces run the same
code.

**Website and documentation:** https://homepages.laas.fr/mkorda/brisk/

## Quick start

```sh
make                                  # needs a C compiler and BLAS/LAPACK (Linux: libopenblas-dev)
./brisk examples/theta1.dat-s
```

| interface | install | use |
|---|---|---|
| Command line | `make` | `./brisk problem.dat-s [options]` |
| Python / CVXPY | `pip install "./python[cvxpy]"` | `prob.solve(solver=brisk.BRISK())` |
| Julia / JuMP | `make libbrisk`, then in Julia `Pkg.develop(path = "/full/path/to/brisk/julia/Brisk")` | `Model(Brisk.Optimizer)` |
| MATLAB / Octave | `build_brisk_mex` in `matlab/` | `brisk_sedumi(A, b, c, K)`, `brisk_sdpa(...)` |
| C | `make libbrisk` | `capi.c`, `tools/capi_test.c` |

Full instructions: [INSTALL.md](INSTALL.md). Every option, and how to pass it from each
interface: [OPTIONS.md](OPTIONS.md).

## What it does

- A primal-dual path-following method (Mehrotra predictor-corrector, HKM or NT direction) in
  two variants, the homogeneous self-dual embedding and the infeasible start. Each is the
  other's fallback.
- The returned X, y and Z, the status and the DIMACS errors refer to the problem as given:
  presolve reductions are undone and the errors are measured on the original data.
- Structure used automatically: chordal sparsity, sign and permutation symmetry, facial
  reduction, free variables, the dual form, sparse and low-rank constraint matrices, sparse
  Schur complements.
- A bound mode (`-bound p|d`) that returns a feasible point on one side, with a rigorous
  check in interval arithmetic (`-certify`).
- High precision (`-prec dd | qd | <digits>`): double-double, quad-double or any number of
  digits, with the solver's own arithmetic and from every interface; rigorous bounds with all
  the digits.
- For problems whose Schur complement does not fit in memory: a first-order engine (`-fom`),
  a matrix-free interior-point method (`-mfipm`) and an experimental low-rank method (`-lralm`).

Details: [SOLVER.md](SOLVER.md) (methods, automatic choices, exit codes) and
[CHANGES.md](CHANGES.md).

## Status

Version 1.1. Tested on Linux at every release on about 350 instances (SDPLIB, Mittelmann's
benchmark, AC optimal power flow and moment-SOS relaxations) and through the test suites of
the three interfaces. The macOS build has not yet been validated on a Mac. Please report
problems through the issue tracker, with the output of `./brisk problem.dat-s -v`.

## Author

Milan Korda, LAAS-CNRS, Toulouse. The AMD ordering in `amd/` is from SuiteSparse (BSD 3-clause,
`amd/LICENSE_AMD.txt`).

## License

BRISK is released under the [Apache License 2.0](LICENSE). The bundled AMD ordering keeps
its BSD 3-clause licence (`amd/LICENSE_AMD.txt`).
