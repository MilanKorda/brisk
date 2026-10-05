# BRISK for MATLAB (and GNU Octave)

A MEX interface to the BRISK SDP solver. It runs the same code as the command-line solver
(the whole pipeline: presolve, dual form, free-variable handling, the fallbacks, and the
verification of the returned point on the data as given).

## Build (macOS, Apple Silicon M1–M4)

1. Install the Xcode command-line tools: `xcode-select --install`.
2. In MATLAB, once: `mex -setup C` (it should find Xcode's clang).
3. In MATLAB:
   ```matlab
   cd brisk-1.2/matlab
   build_brisk_mex            % about a minute; produces brisk_mex.mexmaca64
   test_brisk                 % every line should say PASS
   addpath(pwd); savepath     % keep it on the path
   ```

Nothing else needs to be installed. BLAS and LAPACK come from Apple's Accelerate framework,
the AMD ordering is bundled (`../amd`, BSD 3-clause), and so is the OpenMP runtime (`../omp`): Apple's clang compiles the OpenMP pragmas but ships no OpenMP library, so BRISK
brings its own — a small runtime (`omp/brisk_omp.c`, POSIX threads) compiled into the MEX
file. BRISK's threads then run the Schur assembly, the sparse Cholesky, the block algebra
and the correctors in parallel, alongside Accelerate's BLAS threads; there is no dylib to
find at run time and no clash with an OpenMP runtime MATLAB may load itself (the runtime's
symbols are internal to the MEX file). The thread count follows `OMP_NUM_THREADS` (set before
starting MATLAB) or `opts.threads = 8`; the default is every core.

The default target is `-mcpu=apple-m1`, which runs on every Apple Silicon chip (M1 to M4);
`build_brisk_mex('cpu', 'apple-m3')` targets the M3 if your Xcode's clang knows that name.
`build_brisk_mex('openmp', 'system')` uses Homebrew's libomp instead (`brew install libomp`),
`build_brisk_mex('openmp', false)` builds without threads of BRISK's own.

Linux: `build_brisk_mex` with gcc and a system BLAS/LAPACK (OpenBLAS); gcc's own OpenMP
runtime (libgomp, part of gcc) is used. GNU Octave: the same script (it uses `mkoctfile
--mex`); with `CC=clang` in the environment, `build_brisk_mex('openmp', 'bundled')` works on
Linux too (that is how the bundled runtime was tested there).

### Linux: MATLAB's BLAS

On Linux, MATLAB's own BLAS/LAPACK (MKL, 64-bit integers) is what every MEX file ends up calling,
whatever library it was linked with. `build_brisk_mex` therefore compiles BRISK through an
integer-widening layer (`../blas64.c`) and links `-lmwblas -lmwlapack` by default
(`'blas', 'matlab'`); nothing has to be installed, and it is faster than a system OpenBLAS.
Octave links the system BLAS/LAPACK with 32-bit integers (`'blas', 'system'`); an Octave built
for a 64-bit-integer BLAS takes `'blas', 'system64'`. (Before version 1.2 the Linux MATLAB build linked the
system libraries and crashed on the first BLAS call.)

## Use

### SeDuMi format

```matlab
[x, y, info] = brisk_sedumi(A, b, c, K)          % same call as sedumi(A, b, c, K)
[x, y, info] = brisk_sedumi(A, b, c, K, opts)
[x, y, info, z] = brisk_sedumi(...)              % z = c - A'*y
```

`min c'x  s.t.  A x = b, x in K`, with `K.f` (free), `K.l` (nonnegative), `K.q` (second-order
cones `x(1) >= norm(x(2:end))`), `K.r` (rotated cones `2*x(1)*x(2) >= norm(x(3:end))^2`) and
`K.s` (PSD blocks, `vec(X)` column-major, the symmetric part of the data is used — as in
SeDuMi). A problem without `K.s` is solved by BRISK's cone solver for linear and second-order
cone programs (`opts.conesolver = 0` forces the semidefinite solver, which takes the cones as
arrow blocks). Complex data are not supported. `info` has SeDuMi's
`pinf`, `dinf`, `numerr` (0 solved, 1 reduced accuracy, 2 failure), `iter`, `cpusec`, plus
`pobj`, `dobj`, `dimacs` (the six DIMACS errors, measured on your data), `status`.

A problem exported by NCTSSOS/TSSOS/YALMIP in SeDuMi format can be passed as is (the free
variables, e.g. the bound of a polynomial optimization problem, go in `K.f`).

### SDPA format

```matlab
[objVal, x, X, Y, INFO] = brisk_sdpa('problem.dat-s')                    % a file
[objVal, x, X, Y, INFO] = brisk_sdpa(mDIM, nBLOCK, bLOCKsTRUCT, c, F)    % SDPA-M data
[objVal, x, X, Y, INFO] = brisk_sdpa(mDIM, nBLOCK, bLOCKsTRUCT, c, T)    % triplets
... (..., opts)
```

The SDPA pair: `min c'x s.t. X = sum_i F_i x_i - F_0 psd` and `max F_0.Y s.t. F_i.Y = c_i,
Y psd`. `F` is the SDPA-M cell array `F{k, i+1}` (block `k` of `F_i`; LP blocks as vectors),
`T` an `nnz x 5` array `[i k r s value]` (the body of a `.dat-s` file). Outputs follow SDPA-M:
`objVal = [c'x, F_0.Y]`, `X`, `Y` cell arrays (LP blocks as column vectors).

### Options

```matlab
opts.verbose   = 0;          % 0 quiet (default), 1 summary, 2 iteration log
opts.acc       = 'high';     % 'low' | 'default' | 'high'  (1e-6 / 1e-8 / 1e-10)
opts.tol       = 1e-9;       % overrides acc
opts.maxit     = 200;
opts.timelimit = 3600;       % seconds
opts.threads   = 8;          % threads (default: every core)
opts.prec      = 'qd';       % high precision: 'dd', 'qd' or a number of digits (results returned as doubles;
                             % all the digits: command line with -x / -y / -z)
opts.mfipm     = 1;          % matrix-free interior-point method, for problems with a low-rank optimal side
opts.sym       = 'none';     % symmetry reduction: 'auto' (default) finds the sign and
                             % permutation symmetries of the data and uses them exactly; 'none' (or false) off
opts.symalg    = 0;          % block structure shared by all data matrices of a block (matrix-algebra splitting):
                             % -1 auto (default, blocks up to opts.symalgmax = 1000), 0 off, 1 every block
opts.chordal   = 0;          % chordal decomposition: -1 automatic (default), 0 off, 1 force
opts.nohsd     = true;       % a flag: true (or 1) sets it, false (or 0) leaves it out
opts.fom       = 1;          % first-order engine; opts.lralm = 1: low-rank ALM (experimental)
opts.returnx   = 0;          % X of a chordal-decomposed block: -1 returned unless its dense form
                             % exceeds 30% of the memory (default), 0 never (fastest), 1 always.
                             % Blocks that are not returned are empty; info.have_x says which case
opts.args      = '-nofr';    % raw command-line options (string or cell array)
```

Every option of the command line can be a field: `opts.<name> = value` gives
`-<name> value`, and an underscore in the name stands for a dash. The list with explanations is
[OPTIONS.md](../OPTIONS.md). An unknown field is an error, not silently ignored. Output files
(`opts.x`, `opts.y`, `opts.z`) are refused: the solution is returned.

The low-level gateway is `[y, X, Z, info] = brisk_mex(m, blocksizes, b, T, args)` or
`brisk_mex(filename, args)`, in BRISK's own convention: `min <C,X> s.t. <A_i,X> = b_i`,
`max b'y s.t. Z = C - sum y_i A_i psd`, with `C = -F_0`, `A_i = F_i`.

## Notes

- The data is handed to the solver through a temporary SDPA file in `$TMPDIR` (written
  with 17 significant digits, deleted after the solve); for large problems that costs a
  fraction of a second per million nonzeros.
- Ctrl-C does not interrupt a running solve; use `opts.timelimit`.
- An invalid option or an out-of-memory condition inside the solver raises a MATLAB error
  (`brisk:aborted`) instead of ending MATLAB; the memory of that aborted solve is not
  reclaimed until MATLAB is restarted (`clear mex` does not help).
- Tested with GNU Octave 8.4 on Linux (the MEX API), including a comparison with SeDuMi, with
  gcc/libgomp and with clang and the bundled OpenMP runtime (threaded solves, `clear mex`
  with the thread pool alive, reloads); the macOS build itself could not be run where this
  package was made — run `test_brisk` after building.
