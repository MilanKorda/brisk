#!/bin/bash
# The reductions of the high-precision solver (hpred.inc) on a problem with a known value: the
# theta number of the odd cycle C_n is n cos(pi/n) / (1 + cos(pi/n)).  Checks that
#   1. the symmetry (a dihedral group) and the block structure are used and the value has 20 digits,
#   2. the value without the reductions agrees,
#   3. a symmetry that holds only for the data rounded to doubles (one entry of C is
#      1.000000000000000001) is detected and not used, and the value is that of the solve
#      without reductions.
# usage: tools/hpred_test.sh [brisk]
B=${1:-./brisk}; T=$(mktemp -d); fail=0
python3 - "$T" <<'PY'
import sys
T = sys.argv[1]; n = 41
def w(name, pert):
    L = [f"{n + 1}", "1", f"{n}", " ".join(["1"] + ["0"] * n)]
    for i in range(1, n + 1):
        for j in range(i, n + 1): L.append(f"0 1 {i} {j} {pert if (i, j) == (3, 17) else '1'}")
    for i in range(1, n + 1): L.append(f"1 1 {i} {i} 1")
    for i in range(1, n + 1):
        j = i % n + 1; L.append(f"{i + 1} 1 {min(i, j)} {max(i, j)} 1")
    open(f"{T}/{name}.dat-s", "w").write("\n".join(L) + "\n")
w("sym", "1"); w("pert", "1.000000000000000001")
from decimal import Decimal, getcontext
getcontext().prec = 60
def atan_inv(x):
    x = Decimal(x); s = Decimal(0); t = 1 / x; k = 0
    while abs(t) > Decimal(10) ** -58:
        s += t / (2 * k + 1) if k % 2 == 0 else -t / (2 * k + 1); t /= x * x; k += 1
    return s
pi = 4 * (4 * atan_inv(5) - atan_inv(239)); x = pi / n; c = Decimal(1); t = Decimal(1); k = 0
while abs(t) > Decimal(10) ** -58:
    k += 2; t = -t * x * x / (k * (k - 1)); c += t
open(f"{T}/exact.txt", "w").write(str(n * c / (1 + c)))
PY
val() { grep "^optimal value" | sed 's/.*: //'; }
close() { python3 -c "
import sys
from decimal import Decimal
a, b, tol = Decimal(sys.argv[1]), Decimal(sys.argv[2]), Decimal(sys.argv[3])
sys.exit(0 if abs(a - b) <= tol * (1 + abs(b)) else 1)" "$1" "$2" "$3"; }
o1=$(BRISK_HPREDSMALL=1 "$B" $T/sym.dat-s -prec dd -threads 1 2>&1); v1=$(echo "$o1" | val)
o2=$(BRISK_HPNORED=all "$B" $T/sym.dat-s -prec dd -threads 1 2>&1); v2=$(echo "$o2" | val)
o3=$(BRISK_HPREDSMALL=1 "$B" $T/pert.dat-s -prec dd -threads 1 2>&1); v3=$(echo "$o3" | val)
o4=$(BRISK_HPNORED=all "$B" $T/pert.dat-s -prec dd -threads 1 2>&1); v4=$(echo "$o4" | val)
ex=$(cat $T/exact.txt)
chk() { if eval "$2"; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=1; fi; }
echo "high-precision reductions test ($B): theta(C_41) = ${ex:0:32}"
chk "the group and the block structure are used" 'echo "$o1" | grep -q "symmetry reduction: .*group of order 82" && echo "$o1" | grep -q "^presolve: block structure"'
chk "status OPTIMAL with the reductions" 'echo "$o1" | grep -q "^status: OPTIMAL"'
chk "value with the reductions: 20 digits ($v1)" 'close "$v1" "$ex" 1e-20'
chk "value without them: 20 digits ($v2)" 'close "$v2" "$ex" 1e-20'
chk "a symmetry of the rounded data only is not used" 'echo "$o3" | grep -q "does not hold in the working precision" && ! echo "$o3" | grep -q "^presolve: block structure"'
chk "its value is that of the solve without reductions ($v3)" 'close "$v3" "$v4" 1e-22'
chk "and differs from the symmetric problem's" '! close "$v3" "$ex" 1e-22'
rm -rf "$T"
exit $fail
