#!/bin/bash
# Doubly nonnegative programs (ISSUES 288): a k-means relaxation (Peng and Wei) of 96 points in 4 clusters, generated
# here (a fixed integer generator, the same file on every platform), whose 4,560 bounds X_ij >= 0 are rows with an
# extra variable each. Checks that
#   1. by default the first-order engine runs first with the bounds as a cone on X and ends OPTIMAL,
#   2. -fombound 0 gives the interior-point method, OPTIMAL, with the same objective (to 1e-6 relative),
#   3. an attempt that misses the tolerance (-fommaxit 30) hands over to the interior-point method: OPTIMAL, both solves
#      in the Summary, and the advice names -fombound 0,
#   4. -acc high makes no first-order attempt,
#   5. -fom 1 takes the bounds as well.
# usage: tools/dnn_test.sh [brisk]
B=${1:-./brisk}; fail=0
F=$(mktemp /tmp/dnn_test_XXXXXX.dat-s)
trap 'rm -f "$F"' EXIT
awk -v n=96 -v k=4 'BEGIN {
    s = 12345; for (i = 1; i <= n; i++) { c = i % k; cx = (c % 2 ? 3 : -3); cy = (c >= 2 ? 3 : -3);
        s = (s * 48271) % 2147483647; px[i] = cx + 3.0 * (s / 2147483647.0 - 0.5);
        s = (s * 48271) % 2147483647; py[i] = cy + 3.0 * (s / 2147483647.0 - 0.5); }
    np = n * (n - 1) / 2; m = n + 1 + np;
    print "\"k-means relaxation, " n " points, " k " clusters (tools/dnn_test.sh)"; print m; print 2; print n " " (-np);
    line = ""; for (r = 1; r <= n; r++) line = line "1 "; line = line k; for (p = 1; p <= np; p++) line = line " 0"; print line;
    for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) { d = (px[i] - px[j]) ^ 2 + (py[i] - py[j]) ^ 2; printf "0 1 %d %d %.17g\n", i, j, -d; }
    for (r = 1; r <= n; r++) { printf "%d 1 %d %d 1\n", r, r, r; for (j = 1; j <= n; j++) if (j != r) printf "%d 1 %d %d 0.5\n", r, (r < j ? r : j), (r < j ? j : r); }
    for (i = 1; i <= n; i++) printf "%d 1 %d %d 1\n", n + 1, i, i;
    p = 0; for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) { p++; printf "%d 1 %d %d 0.5\n%d 2 %d %d -1\n", n + 1 + p, i, j, n + 1 + p, p, p; }
}' > "$F"
has() { echo "$out" | grep -qF -- "$1"; }
chk() { # chk <name> <strings that must appear, separated by |> <strings that must not appear>
    local IFS='|' bad=""
    for s in $2; do has "$s" || bad="$bad missing '$s';"; done
    for s in $3; do has "$s" && bad="$bad unexpected '$s';"; done
    if [ -z "$bad" ]; then echo "ok   $1"; else echo "FAIL $1:$bad"; fail=1; fi
}
obj() { echo "$out" | sed -n 's/^optimal value[^:]*: *\([-+0-9.e]*\).*/\1/p' | head -1; }
E="first-order engine first: most constraints are bounds on single entries of X"; W="entrywise bounds on X"
out=$($B "$F" -threads 2 2>&1);                  chk "default: the engine with the bounds first, OPTIMAL" "$E|$W|status: OPTIMAL" "continuing with the interior-point method"
o1=$(obj)
out=$($B "$F" -threads 2 -fombound 0 2>&1);      chk "-fombound 0: the interior-point method" "status: OPTIMAL" "$E|$W"
o2=$(obj)
awk -v a="$o1" -v b="$o2" 'BEGIN { d = a - b; if (d < 0) d = -d; s = (a < 0 ? -a : a); exit !(a != "" && b != "" && d <= 1e-6 * (1 + s)) }' \
    && echo "ok   the same objective ($o1, $o2)" || { echo "FAIL objectives differ: $o1 against $o2"; fail=1; }
out=$($B "$F" -threads 2 -fommaxit 30 2>&1);     chk "an attempt that misses: the interior-point method follows" "$E|continuing with the interior-point method|status: OPTIMAL|Solve 1:|Solve 2:|-fombound 0" ""
out=$($B "$F" -threads 2 -acc high 2>&1);        chk "-acc high: no first-order attempt" "" "$E"
out=$($B "$F" -threads 2 -fom 1 2>&1);           chk "-fom 1: the bounds as well" "$W|status: OPTIMAL" ""
[ $fail = 0 ] && echo "dnn_test: all passed" || echo "dnn_test: FAILED"
exit $fail
