#!/bin/bash
# The dual after a facial reduction (ISSUES 93, the owner's benchmark of 15 October 2026; the policy of 16 October).
# On a sparse SOS relaxation (pop_wood_20: chained Wood, correlative sparsity, order 2) the reduction removes 312
# constraints whose duals cannot be rebuilt from the reduced solve. Checks that
#   1. by default the reduced solve is returned with a status that is not OPTIMAL, no re-solve, and the log says
#      that the recovery failed and how to solve again without the reduction (-nofr; the dual error about 1),
#   2. that advice works: with -nofr the status is OPTIMAL and the dual error on the problem as read at most 1e-6,
#   3. with -frfallback 4 the problem is solved again without the reduction and that solve returned (OPTIMAL),
#   4. with BRISK_FRFIX=1 the removed duals are corrected through complementarity with the primal (dual error
#      below 1e-5, a note saying so),
#   5. the values agree to 1e-6.
# usage: tools/fr_test.sh [brisk]
B=${1:-./brisk}; D=$(dirname "$0")/../examples; fail=0
dual() { grep "DIMACS errors:" | head -1 | awk '{print $6+0}'; }
val() { grep "^optimal value" | tail -1 | sed 's/.*: //'; }
st() { grep "^status:" | tail -1 | sed 's/status: *//; s/ *iterations.*//'; }
ok() { python3 -c "import sys; sys.exit(0 if ($1) else 1)"; }
a=$($B $D/pop_wood_20.dat-s -threads 1 2>&1); b=$($B $D/pop_wood_20.dat-s -threads 1 -nofr 2>&1)
c=$($B $D/pop_wood_20.dat-s -threads 1 -frfallback 4 2>&1); d=$(BRISK_FRFIX=1 $B $D/pop_wood_20.dat-s -threads 1 2>&1)
da=$(echo "$a" | dual); db=$(echo "$b" | dual); dc=$(echo "$c" | dual); dd=$(echo "$d" | dual)
va=$(echo "$a" | val); vb=$(echo "$b" | val); vc=$(echo "$c" | val); vd=$(echo "$d" | val)
sa=$(echo "$a" | st); sb=$(echo "$b" | st); sc=$(echo "$c" | st); sd=$(echo "$d" | st)
if [ "$sa" != "OPTIMAL" ] && [ -n "$sa" ] && ! echo "$a" | grep -q "re-solving without facial reduction" \
   && echo "$a" | grep -q "solve again without facial reduction: -nofr" && echo "$a" | grep -q "recovery of the duals of the 312 removed" && ok "$da >= 1e-3"; then
    echo "ok   pop_wood_20: no re-solve, '$sa', dual error $da on the problem as read, the failure and -nofr in the log"
else echo "FAIL pop_wood_20: status '$sa', dual error $da (expected no re-solve, not OPTIMAL, the failure and -nofr in the log)"; fail=1; fi
if [ "$sb" = "OPTIMAL" ] && ok "$db <= 1e-6"; then
    echo "ok   pop_wood_20 -nofr: $sb, dual error $db"
else echo "FAIL pop_wood_20 -nofr: status '$sb', dual error $db (expected OPTIMAL and <= 1e-6)"; fail=1; fi
if echo "$c" | grep -q "re-solving without facial reduction for the dual" && [ "$sc" = "OPTIMAL" ] && ok "$dc <= 1e-6"; then
    echo "ok   pop_wood_20 -frfallback 4: re-solved without the reduction, $sc, dual error $dc"
else echo "FAIL pop_wood_20 -frfallback 4: status '$sc', dual error $dc (expected the re-solve, OPTIMAL and <= 1e-6)"; fail=1; fi
if [ "$sd" != "OPTIMAL" ] && [ -n "$sd" ] && echo "$d" | grep -q "corrected through complementarity" && ok "$dd <= 1e-5"; then
    echo "ok   pop_wood_20 BRISK_FRFIX=1: '$sd', dual error $dd after the correction, with the note"
else echo "FAIL pop_wood_20 BRISK_FRFIX=1: status '$sd', dual error $dd (expected not OPTIMAL, a dual error below 1e-5, the note)"; fail=1; fi
if ok "max(abs($va - ($vb)), abs($va - ($vc)), abs($va - ($vd))) <= 1e-6 * (1 + abs($vb))"; then echo "ok   the values agree ($va, $vb, $vc, $vd)"; else echo "FAIL values $va, $vb, $vc, $vd"; fail=1; fi
[ $fail = 0 ] && echo "fr_test: all checks passed" || { echo "fr_test: FAILED"; exit 1; }
