#!/bin/bash
# release_check.sh: every interface and every guard of RELEASE_CHECKS.md, on the tree it is run in (for a release: the
# unpacked user package). A release goes out only when this ends with "release_check: PASSED" (the owner, 19 October
# 2026: "We cannot afford to have public releases with interface bugs"). An interface that cannot be run (Octave or
# Julia not installed, libopenblas64-dev or clang missing) is a FAIL, not a skip: install it (tools/setup_env.sh
# --octave --julia) and run again.
#   tools/release_check.sh            all checks (about 30 minutes; Julia's MathOptInterface suite is most of it)
#   JULIA=/path/to/julia tools/release_check.sh
cd "$(dirname "$0")/.."
LOG=${LOG:-release_check.log}; : > "$LOG"
pass=(); fail=()
run() { local name=$1; shift; printf "%-44s " "$name"; if ( "$@" ) >> "$LOG" 2>&1; then echo ok; pass+=("$name"); else echo FAIL; fail+=("$name"); fi; }
have() { command -v "$1" > /dev/null 2>&1; }
JULIA=${JULIA:-$(command -v julia || ls "$HOME"/tools/julia-*/bin/julia 2>/dev/null | head -1)}
echo "release_check: $(pwd), log in $LOG"
run "build (solver, library)"                make -s brisk libbrisk
run "examples (as the public CI)"             bash -c 'for f in examples/*.dat-s examples/*.mps examples/*.cbf examples/*.mat; do [ "$f" = examples/pop_wood_20.dat-s ] && continue; ./brisk "$f" -q || { echo "$f failed"; exit 1; }; done'
run "command-line suites (R1, R4)"           make hpredtest restoretest advicetest frtest bindtest
run "C API (R5)"                              make capitest
run "Python and CVXPY"                        make pytest
run "MATLAB's ILP64 BLAS layer (R2, R3)"      make blas64test
run "library under AddressSanitizer (R5)"     make asantest
run "clang + bundled OpenMP (R6)"             bash -c 'have() { command -v "$1" > /dev/null; }; have clang && make omptest'
run "Octave: MEX build and test_brisk"        bash -c 'command -v octave-cli > /dev/null && cd matlab && octave-cli --eval "build_brisk_mex; test_brisk" | tee /dev/stderr | grep -q "test_brisk: all checks passed"'
run "Julia + JuMP + full MOI suite (R7)"      bash -c "[ -x '$JULIA' ] && cd julia/Brisk && BRISK_TEST_MOI=1 '$JULIA' --project=. -e 'import Pkg; Pkg.test(\"Brisk\")'"
echo "passed: ${#pass[@]}, failed: ${#fail[@]}"
if [ ${#fail[@]} -gt 0 ]; then printf '  FAILED: %s\n' "${fail[@]}"; echo "release_check: FAILED (see $LOG) - no release"; exit 1; fi
echo "release_check: PASSED"
