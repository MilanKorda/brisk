#!/bin/sh
# The post-link check (5.15). On macOS the first release of Apple's new linker (Xcode 15.0,
# `ld -v`: dyld-1015) changed an instruction that carries no relocation: `fmov d3, x15` in
# sedumi_solve became an undefined word, and the library crashed with SIGILL. Such a word is in
# the linked file but not in the object the compiler wrote, so the check compares the two: a word
# the disassembler cannot decode in the linked file counts as damage only if the input objects do
# not have the same undecodable bytes at the same offset of the same function. Compilers put data
# in code themselves (clang's switch jump tables inline in __text on x86-64), which the linker
# copies unchanged (1.4.4; a user's report on an Intel Mac).
#
# Both are compared twice: by function and offset, and by the bytes with those of the instruction before.
#
#   tools/link_check.sh <output> -- <link command...>
#       runs the link command; on macOS checks <output> against the objects and archives of the
#       command (its arguments ending in .o or .a); if it is damaged, links again with the classic
#       linker (-Wl,-ld_classic, while the linker has it) and checks again; if it is still
#       damaged, removes <output> and fails with a message
#   tools/link_check.sh --check <file> [objects...]
#       the check alone (the MATLAB build uses it): exit 0 clean (or no check possible), 1 damaged.
#       Without objects every undecodable word counts, so on x86-64, where compilers put data in
#       code, the check is then skipped with a note
#
# Elsewhere than on macOS the link command only runs. BRISK_LINKCHECK=0: no check. The
# disassembler is the toolchain's llvm-objdump (xcrun); without one the check is skipped
# with a note.

# the undecodable words of a disassembly, one line each with two keys separated by a tab: "function offset
# bytes" (the offset from the function's start) and "previous-instruction-bytes | bytes" (which needs no
# symbol: an object may label its first function by the section's start symbol)
undecodable() {
    awk 'function hex(s,   i, v) { v = 0; s = tolower(s); for (i = 1; i <= length(s); i++) v = v * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1; return v }
         /^[0-9a-f]+ <.*>:$/ { fn = $2; base = hex($1); prev = ""; next }
         /^ *[0-9a-f]+:/ {
             a = $1; sub(/:$/, "", a); b = ""
             for (i = 2; i <= NF && $i ~ /^[0-9a-f][0-9a-f]$/; i++) b = b " " $i
             if ($0 ~ /<unknown>/) print fn " " (hex(a) - base) b "\t" prev " |" b
             prev = b
         }'
}

check() {   # $1: file, then its input objects; prints the damaged words (at most 5); 0 clean, 1 not
    [ "${BRISK_LINKCHECK:-1}" = 0 ] && return 0
    [ "$(uname -s)" = Darwin ] || return 0
    dis=$(xcrun --find llvm-objdump 2>/dev/null) || dis=$(command -v llvm-objdump 2>/dev/null) || dis=""
    if [ -z "$dis" ]; then
        echo "link_check: no llvm-objdump found, $1 not checked" >&2
        return 0
    fi
    file=$1; shift
    if [ $# = 0 ] && [ "$(uname -m)" = x86_64 ]; then
        echo "link_check: $file not checked (no objects to compare with; on x86-64 compilers put data in code)" >&2
        return 0
    fi
    tmp=${TMPDIR:-/tmp}/link_check.$$
    "$dis" -d "$file" 2>/dev/null | undecodable > "$tmp.out"
    if [ ! -s "$tmp.out" ]; then rm -f "$tmp.out"; return 0; fi
    : > "$tmp.in"
    for o in "$@"; do "$dis" -d "$o" 2>/dev/null | undecodable >> "$tmp.in"; done
    bad=$(awk -F '\t' 'FILENAME == ARGV[1] { seen[$1] = 1; seen[$2] = 1; next }
                        !($1 in seen) && !($2 in seen) { n++; if (n <= 5) print "  " $1 } END { if (n > 0) exit 1 }' "$tmp.in" "$tmp.out")
    st=$?
    rm -f "$tmp.out" "$tmp.in"
    [ $st = 0 ] && return 0
    echo "link_check: $file holds words that the disassembler cannot decode and that are not in the compiled objects (a damaged link; function, offset, bytes):" >&2
    echo "$bad" >&2
    return 1
}

if [ "$1" = --check ]; then
    shift
    check "$@"
    exit $?
fi

out=$1
shift
[ "$1" = -- ] && shift
"$@" || exit $?
objs=""
for a in "$@"; do
    case "$a" in *.o|*.a) [ -f "$a" ] && objs="$objs $a" ;; esac
done
# shellcheck disable=SC2086
check "$out" $objs && exit 0

case " $* " in *" -Wl,-ld_classic "*) classic=1 ;; *) classic=0 ;; esac
if [ "$classic" = 0 ] && ld -ld_classic -v >/dev/null 2>&1; then
    echo "link_check: linking $out again with the classic linker (-Wl,-ld_classic)" >&2
    "$@" -Wl,-ld_classic || exit $?
    # shellcheck disable=SC2086
    if check "$out" $objs; then
        echo "link_check: $out is clean with the classic linker" >&2
        exit 0
    fi
fi
rm -f "$out"
echo "link_check: $out removed: the linker changed instructions of the compiled objects. Update the" >&2
echo "  command-line tools (System Settings > Software Update, or: softwareupdate --list) and report" >&2
echo "  it with the lines above; BRISK_LINKCHECK=0 make ... keeps the binary unchecked." >&2
exit 1
