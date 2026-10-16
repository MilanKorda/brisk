#!/bin/bash
# The dual after a facial reduction (ISSUES 93, the owner's benchmark of 15 October 2026). On a sparse SOS
# relaxation (pop_wood_20: chained Wood, correlative sparsity, order 2) the reduction removes 312 constraints
# whose duals cannot be rebuilt from the reduced solve. Checks that
#   1. by default the problem is solved again without the reduction and the returned dual is feasible on
#      the problem as read (dual error at most 1e-6, status OPTIMAL),
#   2. with -frfallback 0 the reduced solve is returned with a status that is not OPTIMAL and a note that
#      y is not dual feasible for the removed constraints,
#   3. the values of the two agree to 1e-6.
# usage: tools/fr_test.sh [brisk]
B=${1:-./brisk}; D=$(dirname "$0")/../examples; fail=0
dual() { grep "DIMACS errors:" | head -1 | awk '{print $6+0}'; }
val() { grep "^optimal value" | tail -1 | sed 's/.*: //'; }
st() { grep "^status:" | tail -1 | sed 's/status: *//; s/ *iterations.*//'; }
ok() { python3 -c "import sys; sys.exit(0 if ($1) else 1)"; }
a=$($B $D/pop_wood_20.dat-s -threads 1 2>&1); b=$($B $D/pop_wood_20.dat-s -threads 1 -frfallback 0 2>&1)
da=$(echo "$a" | dual); db=$(echo "$b" | dual); va=$(echo "$a" | val); vb=$(echo "$b" | val); sa=$(echo "$a" | st); sb=$(echo "$b" | st)
if echo "$a" | grep -q "re-solving without facial reduction for the dual" && [ "$sa" = "OPTIMAL" ] && ok "$da <= 1e-6"; then
    echo "ok   pop_wood_20: re-solved without the reduction, $sa, dual error $da on the problem as read"
else echo "FAIL pop_wood_20: status '$sa', dual error $da (expected the re-solve, OPTIMAL and <= 1e-6)"; fail=1; fi
if [ "$sb" != "OPTIMAL" ] && [ -n "$sb" ] && echo "$b" | grep -q "not dual feasible for the" && ok "$db >= 1e-3"; then
    echo "ok   pop_wood_20 -frfallback 0: the reduced solve returned as '$sb', dual error $db, with the note"
else echo "FAIL pop_wood_20 -frfallback 0: status '$sb', dual error $db (expected not OPTIMAL, a dual error, the note)"; fail=1; fi
if ok "abs($va - ($vb)) <= 1e-6 * (1 + abs($vb))"; then echo "ok   the values agree ($va, $vb)"; else echo "FAIL values $va against $vb"; fail=1; fi
[ $fail = 0 ] && echo "fr_test: all checks passed" || { echo "fr_test: FAILED"; exit 1; }
