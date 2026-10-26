#!/bin/bash
# asan_test.sh: libbrisk built with AddressSanitizer (a copy of the sources in a temporary directory), the C API test
# and the bound-mode handle test on examples; any memory error or leak fails (RELEASE_CHECKS.md R5: two leaks of 1.5.3
# found this way, ISSUES 121).
set -e
SRC=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
tar -C "$SRC" --exclude=./benchmarks --exclude=./bench --exclude=./.git -cf - . | tar -C "$D" -xf -
cd "$D"; make -s clean > /dev/null 2>&1 || true
# (26 October 2026) without -march=native: on an AVX-512 host GCC 13 copies large local structs with 64-byte aligned
# moves (vmovdqa64), and AddressSanitizer's frames for its use-after-return check are only 32-byte aligned, so the copy
# itself faulted (the QP solver object's copy of its problem, qpas.c, in the C API test) where no memory error exists;
# the generic x86-64 code keeps that check
make -s -j${JOBS:-4} libbrisk ARCHFLAG= CC="gcc -fsanitize=address -fno-omit-frame-pointer -g" > build.log 2>&1 || { tail -20 build.log; exit 1; }
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=0:exitcode=99
gcc -fsanitize=address -g -std=gnu11 -I. -o capi_test tools/capi_test.c -L. -lbrisk -Wl,-rpath,"$D" -lm
gcc -fsanitize=address -g -std=gnu11 -o capi_bound_test tools/capi_bound_test.c -L. -lbrisk -Wl,-rpath,"$D" -lm
fail=0
./capi_test > capi.log 2>&1 || { echo "  FAIL  capi_test under ASan"; grep -E "ERROR|SUMMARY" capi.log | head -5; fail=1; }
[ $fail = 0 ] && echo "  ok    capi_test under ASan"
for f in theta1 control1 arch0 truss1; do
  if ./capi_bound_test "$SRC/examples/$f.dat-s" > b.log 2>&1 && ! grep -q "Sanitizer" b.log; then echo "  ok    bound handle test, $f"
  else echo "  FAIL  bound handle test, $f"; grep -E "ERROR|SUMMARY" b.log | head -3; fail=1; fi
done
[ $fail = 0 ] && echo "asan_test: all checks passed" || { echo "asan_test: FAILED"; exit 1; }
