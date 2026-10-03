#!/bin/bash
# Regenerates the "Full list" section of OPTIONS.md from the command line's own option text
# (./brisk without arguments), between the markers. Run it after any change to usage() in main.c.
set -e
cd "$(dirname "$0")/.."
[ -x ./brisk ] || make -s
tmp=$(mktemp)
./brisk > "$tmp" 2>&1 || true
python3 - "$tmp" <<'PY'
import sys
u = open(sys.argv[1]).read().rstrip("\n")
s = open("OPTIONS.md").read()
a = s.index("<!-- BEGIN GENERATED -->") + len("<!-- BEGIN GENERATED -->")
b = s.index("<!-- END GENERATED -->")
s = s[:a] + "\n```text\n" + u + "\n```\n" + s[b:]
open("OPTIONS.md", "w").write(s)
PY
rm -f "$tmp"
echo "OPTIONS.md: full list regenerated"
