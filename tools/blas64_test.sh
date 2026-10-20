#!/bin/bash
# blas64_test.sh: the MATLAB build's BLAS layer (blas64.c) without MATLAB. MATLAB on Linux links MKL with 64-bit
# integers (ILP64); build_brisk_mex('blas', 'matlab') compiles the solver with -DBLAS_PREFIX=bk_ and links blas64.c,
# which widens the integers. A routine the solver calls that blas64.c does not wrap breaks that build ("undefined
# reference to bk_dcopy_": 1.5.3, found by a user; four routines of the tridiagonalization before, 14 October 2026,
# ISSUES 87). This test (until 19 October 2026 a manual one against the scipy-openblas64 wheel) builds the same sources the same way against an ILP64
# OpenBLAS (Debian/Ubuntu: libopenblas64-dev, plain names with 64-bit integers, as MKL) and checks:
#  1. no object of the solver calls a BLAS/LAPACK routine by its plain name (it would get 32-bit integers);
#  2. the link with --no-undefined succeeds (every bk_ routine is wrapped);
#  3. the ILP64 build gives the results of the normal build on a few files.
set -e
cd "$(dirname "$0")/.."
OB=${OPENBLAS64_DIR:-/usr/lib/x86_64-linux-gnu/openblas64-pthread}
if [ ! -f "$OB/libopenblas64.so" ]; then
  echo "blas64_test: no ILP64 OpenBLAS in $OB (apt-get install libopenblas64-dev, or set OPENBLAS64_DIR)"; exit 2
fi
# 0. the static check (every BL(name) has its BW(name); ISSUES 87): no compiler, no library
tools/blas64_check.sh || exit 1
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
# the sources of the MEX file (matlab/build_brisk_mex.m: names, amd/*.c), without the MEX gateway
SRC=$(sed -n "/^names = {/,/};/p" matlab/build_brisk_mex.m | grep -o "'[a-z0-9_]*\.c'" | tr -d "'")
F="-O2 -fopenmp -std=gnu11 -ffp-contract=off -DHAVE_AMD -DNDEBUG -DBLAS_PREFIX=bk_ -DBLAS64_INT=ptrdiff_t"
for f in $SRC amd/*.c; do
  X=""; [ "$f" = boundcert.c ] && X="-frounding-math -fno-fast-math"
  ${CC:-gcc} $F $X -c "$f" -o "$D/$(echo "$f" | tr / _).o" &
  while [ $(jobs -r | wc -l) -ge ${JOBS:-4} ]; do sleep 0.2; done
done; wait
${CC:-gcc} $F -c blas64.c -o "$D/blas64.o.x"
# 1. plain names: the routines the ILP64 library exports, called strongly ("U") by the solver's objects
nm -D --defined-only "$OB/libopenblas64.so" | awk '{print $3}' | sort -u > "$D/lib.txt"
nm -u "$D"/*.o | awk '$1 == "U" {print $2}' | sort -u > "$D/und.txt"
plain=$(comm -12 "$D/lib.txt" "$D/und.txt" | grep -vE '^(omp_|GOMP_|__|_ITM|malloc|free|memcpy|memset|memmove|calloc|realloc|exp|log|sqrt|pow|fabs)' || true)
if [ -n "$plain" ]; then echo "blas64_test: FAIL: BLAS/LAPACK called by their plain names (not through BL()):"; echo "$plain"; exit 1; fi
echo "  ok    no plain BLAS/LAPACK names in the solver's objects"
# 2. the link: every bk_ routine wrapped
if ! ${CC:-gcc} -fopenmp -o "$D/brisk64" "$D"/*.o "$D/blas64.o.x" -L"$OB" -Wl,-rpath,"$OB" -lopenblas64 -lm -Wl,--no-undefined 2> "$D/link.txt"; then
  echo "blas64_test: FAIL: blas64.c does not wrap:"; grep -o "undefined reference to \`[a-zA-Z_0-9]*'" "$D/link.txt" | sort -u; exit 1
fi
echo "  ok    the ILP64 link (--no-undefined)"
# 3. the same results as the normal build
[ -x ./brisk ] || make -s brisk
fail=0
for f in examples/theta1.dat-s examples/control1.dat-s examples/arch0.dat-s examples/truss1.dat-s examples/mcp250-1.dat-s \
         examples/soc_small.mat examples/afiro.mps; do
  [ -f "$f" ] || continue
  # the status equal, the value to 1e-7 relative (other compiler flags than the normal build: other roundings)
  a=$(./brisk "$f" -threads 2 2>&1 || true); b=$("$D/brisk64" "$f" -threads 2 2>&1 || true)
  sa=$(echo "$a" | grep '^status' | awk '{print $2}'); sb=$(echo "$b" | grep '^status' | awk '{print $2}')
  val() { echo "$1" | grep -E '^(optimal|objective) value|primal obj' | head -1 | sed 's/.*[:=] *\([-+0-9.eE]*\).*/\1/'; }
  va=$(val "$a"); vb=$(val "$b")
  if [ -n "$sa" ] && [ "$sa" = "$sb" ] && awk -v x="$va" -v y="$vb" 'BEGIN { d = x - y; if (d < 0) d = -d; a = x < 0 ? -x : x; exit !(d <= 1e-7 * (1 + a)) }'; then
    echo "  ok    $(basename "$f"): $sa, $va and $vb"
  else echo "  FAIL  $(basename "$f"): $sa $va // $sb $vb"; fail=1; fi
done
# the routines the examples do not reach: the high-precision solver, the first-order engine, the bound certificate,
# and a block of order 300 (the threaded tridiagonalization brisk_dsytrd_low: dsytd2, dlarfg, dsyr2k, dscal)
awk 'BEGIN { n = 300; print n; print 1; print n; for (i = 1; i <= n; i++) printf "1 "; print "";
  for (i = 1; i <= n; i++) { printf "0 1 %d %d %g\n", i, i, 2.0; if (i < n) printf "0 1 %d %d %g\n", i, i + 1, -1.0 + 0.5 * sin(i); if (i + 7 <= n) printf "0 1 %d %d %g\n", i, i + 7, 0.3 * cos(i); }
  for (i = 1; i <= n; i++) printf "%d 1 %d %d 1\n", i, i, i; }' > "$D/band300.dat-s"
cp "$D/band300.dat-s" "$D/b300.dat-s"
for run in "examples/theta1.dat-s -prec dd" "examples/theta1.dat-s -fom 1" "examples/theta1.dat-s -bound d -certify" "$D/b300.dat-s"; do
  # the status of the normal build and of the ILP64 build the same (and for -bound pd both sides certified by both)
  oa=$(./brisk $run -threads 2 2>&1 || true); ob=$("$D/brisk64" $run -threads 2 2>&1 || true)
  sa=$(echo "$oa" | grep '^status' | sed 's/ *iterations.*//; s/ (max.*//'); sb=$(echo "$ob" | grep '^status' | sed 's/ *iterations.*//; s/ (max.*//')
  ca=$(echo "$oa" | grep -c '^CERTIFIED' || true); cb=$(echo "$ob" | grep -c '^CERTIFIED' || true)
  name="$(basename "${run%% *}")${run#"${run%% *}"}"
  if [ -n "$sb" ] && [ "$sa" = "$sb" ] && [ "$ca" = "$cb" ]; then c=""; [ "$cb" -gt 0 ] && c=", $cb side(s) certified"; echo "  ok    $name: ${sb#status: }$c"
  else echo "  FAIL  $name: normal build '$sa' ($ca certified), ILP64 build '$sb' ($cb certified)"; fail=1; fi
done
[ $fail = 0 ] && echo "blas64_test: all checks passed" || { echo "blas64_test: FAILED"; exit 1; }
