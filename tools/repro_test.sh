#!/bin/bash
# Results independent of the thread count: each file is solved with 1, 2, ..., T threads
# (BRISK_NOBUSY=1, so the count is exactly the one asked for) and the runs are compared bit for
# bit: y as written by -y (%.17g), the status line, and the objective and DIMACS error lines.
#   tools/repro_test.sh <brisk> [options --] file...
# environment: TMAX (default: the number of cores), REP (runs per count, default 1: 2 also
#              checks run-to-run reproducibility at a fixed count), TMO (seconds per run, 600),
#              SEEDS (e.g. "1 2 3": also runs at TMAX threads whose count changes at random from
#              iteration to iteration, BRISK_THREADTEST=<seed>, as the busy-machine rule changes it)
# One line per file: "same" or "DIFF" with the counts that differ from one thread. Exit code 1
# if any file differs.
B=$1; shift; opts=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts="$opts $1"; shift; done
[ "$1" = "--" ] && shift
[ -x "$B" ] || { echo "repro_test: no executable $B" >&2; exit 2; }
TMAX=${TMAX:-$(nproc)}; REP=${REP:-1}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
rc=0
for f in "$@"; do
  n=$(basename "$f"); ref=""; bad=""
  for t in $(seq 1 "$TMAX") $(for s in $SEEDS; do echo "s$s"; done); do
    for r in $(seq 1 "$REP"); do
      if [ "${t#s}" != "$t" ]; then
        o=$(BRISK_NOBUSY=1 BRISK_THREADTEST=${t#s} timeout "${TMO:-600}" "$B" "$f" -threads "$TMAX" -y "$W/y" $opts 2>&1)
      else
        o=$(BRISK_NOBUSY=1 timeout "${TMO:-600}" "$B" "$f" -threads "$t" -y "$W/y" $opts 2>&1)
      fi
      key=$( { echo "$o" | grep -E "^status|optimal value|primal obj|DIMACS errors" | sed "s/time: [0-9.]*s//" ; cat "$W/y" 2>/dev/null; } | sha1sum | cut -c1-12)
      it=$(echo "$o" | grep "^status" | tail -1 | sed 's/.*iterations: \([0-9]*\).*/\1/')
      rm -f "$W/y"
      if [ -z "$ref" ]; then ref=$key; ref_it=$it; elif [ "$key" != "$ref" ]; then bad="$bad T$t/$r(its $it)"; fi
    done
  done
  if [ -z "$bad" ]; then printf "same  %-34s its %s\n" "$n" "$ref_it"; else printf "DIFF  %-34s its %s at T1; differ:%s\n" "$n" "$ref_it" "$bad"; rc=1; fi
done
exit $rc
