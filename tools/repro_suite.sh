#!/bin/bash
# The reproducibility test (CLAUDE.md, "Reproducible results"; ISSUES 708; RELEASE_CHECKS R16): the same input gives
# the same bits (status, iterations, y, X) at any thread count, with thread changes during a solve, and whatever was
# solved before in the same process. Part of the pre-release testing (tools/release_check.sh).
#   tools/repro_suite.sh            the command line and the C library with the BLAS the tree is built with
#   tools/repro_suite.sh --mkl      the same with MKL and Intel's OpenMP runtime preloaded (BRISK_MKL_LIB: the folder of
#                                  libmkl_core and libiomp5; default: the Python package mkl's, tools/setup_env.sh --mkl)
# Run from the tree's root after `make brisk libbrisk`. Ends with "repro_suite: PASSED" or "repro_suite: FAILED".
cd "$(dirname "$0")/.." || exit 2
MKL=0; [ "${1:-}" = "--mkl" ] && MKL=1
T=$(nproc 2>/dev/null || echo 4); [ "$T" -gt 8 ] && T=8; [ "$T" -lt 2 ] && T=2
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
PRE=""
if [ $MKL = 1 ]; then
  L=${BRISK_MKL_LIB:-}
  if [ -z "$L" ]; then   # where pip puts the packages' libraries: the prefix, /usr/local for the system's pip, the user's
    for d in $(python3 -c 'import sys, site; print(sys.prefix + "/lib", site.getuserbase() + "/lib")' 2>/dev/null) /usr/local/lib; do
      ls "$d"/libmkl_core.so.* > /dev/null 2>&1 && { L=$d; break; }
    done
  fi
  IOMP=$(ls "$L"/libiomp5.so 2>/dev/null | head -1)
  LP64=$(ls "$L"/libmkl_intel_lp64.so.* 2>/dev/null | head -1)
  THR=$(ls "$L"/libmkl_intel_thread.so.* 2>/dev/null | head -1)
  CORE=$(ls "$L"/libmkl_core.so.* 2>/dev/null | head -1)
  if [ -z "$IOMP" ] || [ -z "$LP64" ] || [ -z "$THR" ] || [ -z "$CORE" ]; then
    echo "repro_suite: MKL or Intel's OpenMP runtime not found in '$L' (pip install mkl intel-openmp, or set BRISK_MKL_LIB)"
    echo "repro_suite: FAILED"; exit 1
  fi
  PRE="$IOMP $LP64 $THR $CORE"; export LD_LIBRARY_PATH="$L${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "repro_suite: with MKL ($LP64)"
fi
BIN=${BRISK_BIN:-./brisk}    # (another build to test: the test must fail on the tree before the fixes of ISSUES 708)
run() { if [ -n "$PRE" ]; then LD_PRELOAD="$PRE" "$@"; else "$@"; fi; }
fail=0
# the problems: every cone and method the command line routes, small enough for seconds (SDP, SDP with second-order
# cones, LP, QP); the options: the default route, the self-dual embedding (its polishing steps assemble block by block
# with a team of threads), the bound mode
FILES="examples/theta3.dat-s examples/control1.dat-s examples/truss1.dat-s examples/arch0.dat-s examples/mcp250-1.dat-s
examples/pop_wood_20.dat-s examples/hidden_symmetry23.dat-s examples/soc_small.mat tests/repro/cutlike_20.mat
examples/afiro.mps examples/hs21.qps"
# a user's four PolyOpt cut SDPs (private: in the development tree only), where all three defects of ISSUES 708 showed
# (cutsdp_4 at 4 threads gave a different X in every run before the fix): run whenever the benchmark tree has them
PRIV=${BRISK_BENCH:-benchmarks}/user_marek_cutsdp
if [ -d "$PRIV" ]; then FILES="$FILES $PRIV/cutsdp_1.mat $PRIV/cutsdp_2.mat $PRIV/cutsdp_3.mat $PRIV/cutsdp_4.mat"
else echo "repro_suite: (the user's cut SDPs are not in this tree: run the test in the development tree as well)"; fi
# generated files on the threaded paths of the SOCP and QP solvers (not committed; run when present): a group-sparse
# control SOCP (python3 -I bench/socp_gen/gen_socp3.py socp_gctrl), a TV denoising SOCP (bench/adversarial_5.19/scripts/
# generate.py socp_tv_150) and an MPC QP (python3 -I bench/qp/to_qps.py benchmarks/qp_mpc/orig/QUADCMPC4.npz
# benchmarks/qp_mpc/x); and CBLIB's chained singular function (SOCP, in the benchmark tree). ken-07 (LP) runs the LP
# interior-point method alone (-lpmethod 1: the default route races the dual simplex method beside it by the clock at two
# threads or more since 27 October 2026, the LP developer's change, which the rule exempts; so every LP file, afiro
# too) and
# the simplex method's deterministic automatic route (-lpmethod 5 with BRISK_SXDETERMINISTIC=1: the route through the
# dual problem and the interior-point method beside it, by counted work, the LP developer's suggestion); the race by
# the clock (-lpmethod 5 without the variable) is exempt from the rule and not tested. db-shear-wall (CBLIB) and the DIMACS
# sched_100_100_orig and sched_200_100_orig (SOCP): the SOCP solver's factorization by parts of the elimination tree on
# the threads (the SOCP developer's; sched_200_100_orig moved from OPTIMAL to numerical difficulties under an earlier
# version that changed the order of a sum)
GEN="bench/socp_gen/problems/socp_gctrl_2000x12x4/socp_gctrl_2000x12x4.cbf
bench/adversarial_5.19/problems/socp_tv_150/socp_tv_150.cbf ${BRISK_BENCH:-benchmarks}/qp_mpc/x/QUADCMPC4.QPS
${BRISK_BENCH:-benchmarks}/socp/cblib/chainsing-10000-1.cbf.gz
${BRISK_BENCH:-benchmarks}/kennington/ken-07.mps
${BRISK_BENCH:-benchmarks}/socp/cblib/db-shear-wall.cbf.gz ${BRISK_BENCH:-benchmarks}/dimacs/mat/sched_100_100_orig.mat
${BRISK_BENCH:-benchmarks}/dimacs/mat/sched_200_100_orig.mat"
for f in $GEN; do
  if [ -f "$f" ]; then FILES="$FILES $f"; else echo "repro_suite: (not generated here: $f)"; fi
done
for f in $FILES; do
  [ -f "$f" ] || { echo "repro_suite: missing $f"; fail=1; continue; }
  case $f in */ken-07.mps) OPTS="'-lpmethod 1' '-lpmethod 5'";; */db-shear-wall.cbf.gz|*/sched_*_orig.mat) OPTS="''";; *.mps) OPTS="'-lpmethod 1' '-lpmethod 5'";; *.qps|*.QPS|*.cbf|*.cbf.gz) OPTS="''";; */user_marek_cutsdp/*) OPTS="'-hsd -bound p -certify -boundprior 2' '-hsd'";;
                *) OPTS="'' '-hsd' '-bound p'";; esac
  eval "set -- $OPTS"
  for o in "$@"; do
    if [ "$o" = "-lpmethod 5" ]; then export BRISK_SXDETERMINISTIC=1; else unset BRISK_SXDETERMINISTIC; fi
    ref=""; k=0
    for spec in "1" "2" "$T" "$T:7" "$T:13"; do
      t=${spec%%:*}; seed=""; [ "$spec" != "$t" ] && seed=${spec#*:}
      k=$((k + 1))
      if [ -n "$seed" ]; then
        BRISK_THREADTEST=$seed run "$BIN" $o -threads $t -q -y "$W/y" -x "$W/x" "$f" > "$W/log" 2>&1
      else
        run "$BIN" $o -threads $t -q -y "$W/y" -x "$W/x" "$f" > "$W/log" 2>&1
      fi
      h=$( { grep -m1 "^status" "$W/log"; cat "$W/y" "$W/x" 2>/dev/null; } | md5sum | cut -c1-12)
      if [ -z "$ref" ]; then ref=$h; elif [ "$h" != "$ref" ]; then
        echo "repro_suite: FAIL $f [$o]: threads $spec differs from threads 1"; fail=1
      fi
    done
    echo "repro_suite: $f [${o:-default}]: one result at 1, 2 and $T threads, with thread changes"
  done
done
unset BRISK_SXDETERMINISTIC
# the C library: problems of the same dimensions alternated in one process (a library host), at varied thread counts.
# Each SDPA problem is paired with a copy whose constraints are in reverse order: the same m and blocks, other
# constraint lists (a cache keyed by the dimensions or by an address would mix the two)
rev() {   # an SDPA file with its constraints in reverse order (the same problem, y reversed)
  awk 'BEGIN { st = 0 } /^[*"]/ { print; next }
       st == 0 { m = $1 + 0; print; st = 1; next }
       st == 1 { print; st = 2; next }
       st == 2 { print; st = 3; next }
       st == 3 { n = split($0, c, /[ ,{}()]+/); k = 0; for (i = 1; i <= n; i++) if (c[i] != "") v[++k] = c[i];
                 s = ""; for (i = k; i >= 1; i--) s = s (i < k ? " " : "") v[i]; print s; st = 4; next }
       { if ($1 + 0 > 0) $1 = m + 1 - $1; print }' "$1" > "$2"
}
CC=${CC:-cc}
if ! $CC -O2 -I. -o "$W/multisolve" tools/repro_multisolve.c -L. -lbrisk -Wl,-rpath,"$PWD" 2> "$W/cc.log"; then
  cat "$W/cc.log"; echo "repro_suite: FAIL: the C library's driver did not build (make libbrisk first)"; fail=1
else
  rev examples/pop_wood_20.dat-s "$W/pop_wood_20_rev.dat-s"
  rev examples/theta3.dat-s "$W/theta3_rev.dat-s"
  args=()
  for r in 1 2 3; do
    for t in 1 "$T" 2; do
      for p in examples/pop_wood_20.dat-s "$W/pop_wood_20_rev.dat-s" tests/repro/cutlike_20.mat tests/repro/cutlike_20b.mat \
               examples/theta3.dat-s "$W/theta3_rev.dat-s"; do
        args+=("ENV=BRISK_THREADTEST=$([ $r = 2 ] && echo $t || echo -)" "$p" -hsd -q -threads "$t" --)
      done
    done
  done
  run "$W/multisolve" "${args[@]}" > "$W/ms.log" 2>&1
  bad=$(grep -E "^ *[0-9]+ " "$W/ms.log" | awk '{ f = $2; $1 = ""; h[f] = h[f] " " $NF; n[f]++ }
        END { for (f in h) { split(h[f], a, " "); for (i in a) if (a[i] != a[1]) { print f; break } } }')
  nl=$(grep -cE "^ *[0-9]+ " "$W/ms.log")
  if [ -n "$bad" ] || [ "$nl" -lt 54 ]; then
    echo "repro_suite: FAIL: the C library gave different results for: ${bad:-(only $nl of 54 solves ran)}"; tail -5 "$W/ms.log"; fail=1
  else
    echo "repro_suite: the C library: one result per problem in 54 solves (6 problems, 1, 2 and $T threads, alternated)"
  fi
fi
[ $fail = 0 ] && echo "repro_suite: PASSED" || echo "repro_suite: FAILED"
exit $fail
