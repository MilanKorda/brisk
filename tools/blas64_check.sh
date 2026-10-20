#!/bin/bash
# Checks that every BLAS/LAPACK routine the solver calls through BL(name) has its 64-bit-integer wrapper
# BW(name) in blas64.c (the MEX build against MATLAB's libraries links through that layer: a missing
# wrapper is an undefined reference there and nowhere else; a user's report of 14 October 2026, four
# routines of the tridiagonalization missing since 5.9). The OpenBLAS thread calls are weak symbols.
# usage: tools/blas64_check.sh      exit 1 and the names on stdout if any is missing
SRC=$(cd "$(dirname "$0")/.." && pwd)
cd "$SRC" || exit 2
missing=$(comm -23 <(grep -oh "BL([a-z0-9_]*)" *.c *.h | sed 's/BL(//; s/)//' | grep -v "^openblas_" | sort -u) \
                  <(grep -o "BW([a-z0-9_]*)" blas64.c | sed 's/BW(//; s/)//' | sort -u))
if [ -n "$missing" ]; then echo "blas64_check: no 64-bit wrapper in blas64.c for:"; echo "$missing"; exit 1; fi
echo "blas64_check: every BLAS/LAPACK call has its wrapper"
