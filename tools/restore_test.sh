#!/bin/bash
# The restoration of the primal feasibility at the end of a solve (sqrt_restore in solver.c).
# A tolerance that double precision cannot reach (1e-14) sends well-posed examples through the
# end-of-solve corrections. Checks, on three shipped examples, that
#   1. the restoration is tried by default and not with -nopolishr,
#   2. the returned point is not worse than without it (largest DIMACS error within a factor 10,
#      and below 1e-9), and the optimal values agree to 1e-9,
#   3. the options of the first-method rule are accepted and do not change a result that the
#      rule does not concern (-hsdfirstasm 0 on a max-cut problem).
# usage: tools/restore_test.sh [brisk]
B=${1:-./brisk}; D=$(dirname "$0")/../examples; fail=0
err() { grep "DIMACS errors" | tail -1 | awk '{m=0; for(i=3;i<=NF;i++){v=$i+0; if(v<0)v=-v; if(v>m)m=v}; printf "%.3e", m}'; }
val() { grep "^optimal value" | tail -1 | sed 's/.*: //'; }
ok() { python3 -c "import sys; sys.exit(0 if ($1) else 1)"; }
for n in control1 truss1 arch0; do
    a=$($B $D/$n.dat-s -threads 1 -tol 1e-14 -nodd -v 2>&1); b=$($B $D/$n.dat-s -threads 1 -tol 1e-14 -nodd -nopolishr -v 2>&1)
    ea=$(echo "$a" | err); eb=$(echo "$b" | err); va=$(echo "$a" | val); vb=$(echo "$b" | val)
    ta=$(echo "$a" | grep -c "square-root-metric polishing"); tb=$(echo "$b" | grep -c "square-root-metric polishing")
    if [ "$ta" -ge 1 ] && [ "$tb" -eq 0 ]; then echo "ok   $n: the restoration is tried ($ta), not with -nopolishr"; else echo "FAIL $n: tried $ta times by default, $tb with -nopolishr"; fail=1; fi
    if ok "$ea <= 10 * $eb and $ea <= 1e-9 and abs($va - ($vb)) <= 1e-9 * (1 + abs($vb))"; then echo "ok   $n: error $ea (without: $eb), values agree"; else echo "FAIL $n: error $ea against $eb, values $va $vb"; fail=1; fi
done
a=$($B $D/mcp250-1.dat-s -threads 1 2>&1); b=$($B $D/mcp250-1.dat-s -threads 1 -hsdfirstasm 0 2>&1)
ia=$(echo "$a" | grep "^status" | tail -1); ib=$(echo "$b" | grep "^status" | tail -1)
if [ -n "$ia" ] && [ "$ia" = "$ib" ] && [ "$(echo "$a" | val)" = "$(echo "$b" | val)" ]; then echo "ok   mcp250-1: -hsdfirstasm 0 changes nothing ($ia)"; else echo "FAIL mcp250-1: '$ia' against '$ib'"; fail=1; fi
[ $fail = 0 ] && echo "restore_test: all checks passed" || { echo "restore_test: FAILED"; exit 1; }
