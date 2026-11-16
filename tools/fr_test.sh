#!/bin/bash
# (ISSUES 725, 31 October 2026: the lines read below are those of the diagnosis log, the default log until then)
export BRISK_LOGDIAG=${BRISK_LOGDIAG-1}
# The dual after a facial reduction (ISSUES 93, the owner's benchmark of 15 October 2026; the policy of 16 October).
# On a sparse SOS relaxation (pop_wood_20: chained Wood, correlative sparsity, order 2) the reduction removes 312
# constraints whose duals cannot be rebuilt from the reduced solve. Checks that
#   1. by default (7 November 2026, the owner: the duals the reduced problem does not determine are not returned) the
#      reduced solve is returned OPTIMAL without a re-solve, y is NaN on the undetermined constraints (in the -y file
#      too), the dual's errors are those of the face, and the log says how to get a dual of the problem as read
#      (-frdual 1, -nofr),
#   2. that advice works: with -nofr the status is OPTIMAL and the dual error on the problem as read at most 1e-6,
#   3. with -frfallback 4 the problem is solved again without the reduction and that solve returned (OPTIMAL),
#   4. with -frdual 2 (the level-by-level rebuild, as up to 1.6.5) and BRISK_FRFIX=1 the removed duals are corrected through complementarity with the primal (dual error
#      below 1e-5, a note saying so),
#   5. the values agree to 1e-6.
# usage: tools/fr_test.sh [brisk]
B=${1:-./brisk}; D=$(dirname "$0")/../examples; fail=0
dual() { grep "DIMACS errors:" | head -1 | awk '{print $6+0}'; }
val() { grep "^optimal value" | tail -1 | sed 's/.*: //'; }
st() { grep "^status:" | tail -1 | sed 's/status: *//; s/ *iterations.*//'; }
ok() { python3 -c "import sys; sys.exit(0 if ($1) else 1)"; }
YF=$(mktemp); a=$($B $D/pop_wood_20.dat-s -threads 1 -y $YF 2>&1); nnan=$(grep -ci nan $YF); rm -f $YF; b=$($B $D/pop_wood_20.dat-s -threads 1 -nofr 2>&1)
c=$($B $D/pop_wood_20.dat-s -threads 1 -frfallback 4 2>&1); d=$(BRISK_FRFIX=1 $B $D/pop_wood_20.dat-s -threads 1 -frdual 2 2>&1)
da=$(echo "$a" | dual); db=$(echo "$b" | dual); dc=$(echo "$c" | dual); dd=$(echo "$d" | dual)
va=$(echo "$a" | val); vb=$(echo "$b" | val); vc=$(echo "$c" | val); vd=$(echo "$d" | val)
sa=$(echo "$a" | st); sb=$(echo "$b" | st); sc=$(echo "$c" | st); sd=$(echo "$d" | st)
if [ "$sa" = "OPTIMAL" ] && ! echo "$a" | grep -q "re-solving without facial reduction" \
   && echo "$a" | grep -qE "^  -nofr +solves without the facial reduction" && echo "$a" | grep -qE "^  -frdual 1 +a second solve that computes a dual for the optimal value found here" \
   && echo "$a" | grep -q "Dual: *NaN for [0-9]* constraints, not determined after the facial reduction" && [ "$nnan" -ge 312 ] && ok "$da <= 1e-6"; then
    echo "ok   pop_wood_20: no re-solve, '$sa', $nnan undetermined duals NaN in the -y file, dual error $da on the face, -frdual 1 and -nofr in the log"
else echo "FAIL pop_wood_20: status '$sa', dual error $da, $nnan NaN (expected no re-solve, OPTIMAL, at least 312 NaN, -frdual 1 and -nofr in the log)"; fail=1; fi
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
