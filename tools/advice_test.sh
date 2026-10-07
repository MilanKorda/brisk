#!/bin/bash
# The advice at the end of the log on the command line: how to get more accuracy and a
# guaranteed bound, in the syntax of the command line (the interfaces check their own syntax
# in their test suites). Checks that
#   1. a default solve advises -acc high, -prec dd, -bound d and -certify,
#   2. what was asked for is not advised again (-acc high, -bound d, -bound d -certify),
#   3. the high-precision solver advises the bound only, and nothing with -bound,
#   4. a linear program is advised the tolerance only,
#   5. second-order cone input gets the same advice from the cone solver and from the SDP
#      solver (-conesolver 0), with one summary each,
#   6. -q prints no advice.
# usage: tools/advice_test.sh [brisk]
B=${1:-./brisk}; D=$(dirname "$0")/../examples; fail=0
has() { echo "$out" | grep -qF -- "$1"; }
chk() { # chk <name> <strings that must appear, separated by |> <strings that must not appear>
    local IFS='|' bad=""
    for s in $2; do has "$s" || bad="$bad missing '$s';"; done
    for s in $3; do has "$s" && bad="$bad unexpected '$s';"; done
    if [ -z "$bad" ]; then echo "ok   $1"; else echo "FAIL $1:$bad"; fail=1; fi
}
A="For higher accuracy:"; G="For a guaranteed bound on the optimal value"; R="For a rigorous check of the bound:"
out=$($B $D/truss1.dat-s -threads 1 2>&1);                 chk "default: accuracy, bound and certificate" "$A|  -acc high |  -prec dd |$G|  -bound d |  -certify " "$R|as keywords|set_attribute"
out=$($B $D/truss1.dat-s -threads 1 -acc high 2>&1);       chk "-acc high is not advised again" "$A|  -prec dd |$G" "  -acc high "
out=$($B $D/truss1.dat-s -threads 1 -bound d 2>&1);        chk "-bound d: the rigorous check is advised" "$A|$R|  -certify " "$G|  -bound d "
out=$($B $D/truss1.dat-s -threads 1 -bound d -certify 2>&1); chk "-bound d -certify: no bound advice" "$A" "$G|$R"
out=$($B $D/truss1.dat-s -threads 1 -prec dd 2>&1);        chk "-prec dd: the bound only" "$G|  -bound d " "$A|  -certify "
out=$($B $D/truss1.dat-s -threads 1 -prec dd -bound d 2>&1); chk "-prec dd -bound d: nothing" "certif" "$A|$G|$R"
out=$($B $D/afiro.mps -threads 1 2>&1);                    chk "linear program: the tolerance only" "$A|  -tol <t> " "  -prec dd |$G"
out=$($B $D/soc_small.mat -threads 1 2>&1);                chk "second-order cones, cone solver" "cone solver|$A|  -prec dd |$G" ""
[ "$(echo "$out" | grep -c '^Summary')" = 1 ] || { echo "FAIL cone solver: not exactly one summary"; fail=1; }
out=$($B $D/soc_small.mat -threads 1 -conesolver 0 2>&1);  chk "second-order cones, SDP solver" "$A|  -prec dd |$G" ""
[ "$(echo "$out" | grep -c '^Summary')" = 1 ] || { echo "FAIL -conesolver 0: not exactly one summary"; fail=1; }
out=$($B $D/truss1.dat-s -threads 1 -q 2>&1);              chk "-q: no advice" "" "$A|$G"
[ $fail = 0 ] && echo "advice_test: all checks passed" || { echo "advice_test: FAILED"; exit 1; }
