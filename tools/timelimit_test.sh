#!/bin/bash
# Every method keeps the time limit (24 October 2026, after ISSUES 274: the matrix-free interior-point method gave the
# whole of the time left to each of its two conjugate gradient solves, and ran 1,290 s of a 900 s limit).
# A generated problem that none of the methods solves in 3 s (the theta function of a random graph: n = 250, edge
# probability 0.3, seed 1, about 9,300 equality constraints) is solved by each method with -timelimit 3; the wall clock of
# the whole command (reading, presolve, solve, postsolve) must stay within the limit plus 2 s and 25 % for the interior-point
# methods, which check the limit once an iteration (about a second here), and within the limit plus 0.5 s and 10 % for the
# methods that check it inside their inner loops (the matrix-free method's conjugate gradient solves, the first-order
# engine, the low-rank augmented Lagrangian, the high-precision solver's factorization); the exit code is that of a
# finished run (0, 10-15: 15 is the time limit); and the total time the run reports matches the wall clock.
# usage: tools/timelimit_test.sh [brisk]
B=${1:-./brisk}; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; fail=0
python3 - "$T/theta.dat-s" <<'PY'
import sys, random
random.seed(1); n = 250; p = 0.3
E = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1) if random.random() < p]
with open(sys.argv[1], 'w') as f:
    f.write('"theta function of a random graph (timelimit_test.sh)"\n%d\n1\n%d\n' % (1 + len(E), n))
    f.write(' '.join(['1'] + ['0'] * len(E)) + '\n')
    for i in range(1, n + 1):
        for j in range(i, n + 1): f.write('0 1 %d %d 1\n' % (i, j))
    for i in range(1, n + 1): f.write('1 1 %d %d 1\n' % (i, i))
    for k, (i, j) in enumerate(E): f.write('%d 1 %d %d 1\n' % (k + 2, i, j))
PY
check() {   # slack (s), slack (fraction of the limit), name, options...
    local sa=$1 sf=$2 name=$3; shift 3
    local t0=$(date +%s.%N); local out; out=$($B "${PROB:-$T/theta.dat-s}" -threads 2 -timelimit 3 "$@" 2>&1); local rc=$?
    local t1=$(date +%s.%N)
    local w=$(python3 -c "print('%.2f' % ($t1 - $t0))"); local st=$(echo "$out" | grep -m1 "^status" | cut -c1-60)
    if python3 -c "import sys; sys.exit(0 if $w <= 3 + $sa + $sf * 3 else 1)"; then echo "ok   $name: $w s, exit code $rc ($st)"; else echo "FAIL $name: $w s for a 3 s limit ($st)"; fail=1; fi
    # (24 October 2026, after ISSUES 278 and 280: the high-precision solver's total counted its double solve twice, and a
    #  missed first attempt was left out of the total) the total the run reports, the Summary's "Time:" or the "time:
    #  total" line, lies within the wall clock of the whole command and at most 0.5 s and 10 % below it (reading the file
    #  and starting the process are outside it)
    local rt; rt=$(echo "$out" | sed -n -e 's/^  Time: *\([0-9.]*\) s.*/\1/p' -e 's/^time: total \([0-9.]*\)s.*/\1/p' | tail -1)
    if [ -z "$rt" ]; then echo "FAIL $name: no total time in the log"; fail=1
    elif python3 -c "import sys; sys.exit(0 if $w - 0.5 - 0.1 * $w <= $rt <= $w + 0.05 else 1)"; then echo "ok   $name: reported total $rt s"
    else echo "FAIL $name: reported total $rt s, wall clock $w s"; fail=1; fi
    case $rc in 0|10|11|12|13|14|15) ;; *) echo "FAIL $name: exit code $rc"; fail=1;; esac
}
check 2 0.25 "the default (the automatic method choice)"
check 2 0.25 "the infeasible-start interior-point method" -nohsd
check 2 0.25 "the self-dual embedding" -hsd
check 0.5 0.1 "the matrix-free interior-point method" -mfipm 1
check 0.5 0.1 "the matrix-free method, every solve to its deadline" -mfipm 1 -mfproj 1 -mfkmax 0 -mfrmax 0 -mfcgtol 1e-14 -mfcgtolmax 1e-14 -mfcgmax 1000000
check 0.5 0.1 "the first-order engine" -fom 1
check 0.5 0.1 "the low-rank augmented Lagrangian" -lralm 1
# (the high-precision solver starts with a double-precision interior-point solve, which may pass the limit by one of its
#  iterations as the interior-point methods above do, and reads and presolves the problem in its own precision before its
#  first check: their allowance and 0.5 s more. Before ISSUES 278 it took 80 s, and 5.5 s while a level of its ladder could
#  still start after the limit. 24 October 2026, the evening: 1.5 s and 10 % until then, less than the interior-point
#  methods' own allowance; an iteration here takes about 2 s on two threads of the Emerald Rapids host, so a second
#  iteration that started just before the limit ended at 4.9-5.4 s, with the builds before and after ISSUES 282 alike)
check 2.5 0.25 "the high-precision solver" -prec dd
# (ISSUES 280) a max-cut problem of order 500 (a random graph, edge probability 0.1, seed 2): the Burer-Monteiro method
# goes first; at a tolerance it cannot certify it misses after its half of the limit, and the interior-point method
# follows; the reported total includes the attempt
python3 - "$T/maxcut.dat-s" <<'PY2'
import sys, random
random.seed(2); n = 500; p = 0.1
E = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1) if random.random() < p]
deg = [0] * (n + 1)
for i, j in E: deg[i] += 1; deg[j] += 1
with open(sys.argv[1], 'w') as f:
    f.write('"max-cut of a random graph (timelimit_test.sh)"\n%d\n1\n%d\n' % (n, n))
    f.write(' '.join(['1'] * n) + '\n')
    for i in range(1, n + 1): f.write('0 1 %d %d %.2f\n' % (i, i, deg[i] / 4))
    for i, j in E: f.write('0 1 %d %d -0.25\n' % (i, j))
    for i in range(1, n + 1): f.write('%d 1 %d %d 1\n' % (i, i, i))
PY2
PROB="$T/maxcut.dat-s" check 2 0.25 "a missed Burer-Monteiro attempt, then the interior-point method" -tol 1e-17
[ $fail = 0 ] && echo "timelimit_test: all checks passed" || { echo "timelimit_test: FAILED"; exit 1; }
