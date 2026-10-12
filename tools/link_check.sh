#!/bin/sh
# The post-link check (5.15). On macOS the first release of Apple's new linker (Xcode 15.0,
# `ld -v`: dyld-1015) changed an instruction that carries no relocation: `fmov d3, x15` in
# sedumi_solve became an undefined word, and the library crashed with SIGILL. A linked BRISK
# binary has no word the disassembler cannot decode, so one is taken as a damaged link.
#
#   tools/link_check.sh <output> -- <link command...>
#       runs the link command; on macOS checks <output>; if it is damaged, links again with the
#       classic linker (-Wl,-ld_classic, while the linker has it) and checks again; if it is
#       still damaged, removes <output> and fails with a message
#   tools/link_check.sh --check <file>
#       the check alone (the MATLAB build uses it): exit 0 clean (or no check possible), 1 damaged
#
# Elsewhere than on macOS the link command only runs. BRISK_LINKCHECK=0: no check. The
# disassembler is the toolchain's llvm-objdump (xcrun); without one the check is skipped
# with a note.

check() {   # $1: file; prints the undecodable words (at most 5) with their functions; 0 clean, 1 not
    [ "${BRISK_LINKCHECK:-1}" = 0 ] && return 0
    [ "$(uname -s)" = Darwin ] || return 0
    dis=$(xcrun --find llvm-objdump 2>/dev/null) || dis=$(command -v llvm-objdump 2>/dev/null) || dis=""
    if [ -z "$dis" ]; then
        echo "link_check: no llvm-objdump found, $1 not checked" >&2
        return 0
    fi
    bad=$("$dis" -d "$1" 2>/dev/null | awk '/^[0-9a-f]+ <.*>:$/ { fn = $2 } /<unknown>/ { n++; if (n <= 5) print "  " fn " " $1 " " $2 } END { if (n > 0) exit 1 }')
    [ $? = 0 ] && return 0
    echo "link_check: $1 holds words the disassembler cannot decode (a damaged link):" >&2
    echo "$bad" >&2
    return 1
}

if [ "$1" = --check ]; then
    check "$2"
    exit $?
fi

out=$1
shift
[ "$1" = -- ] && shift
"$@" || exit $?
check "$out" && exit 0

case " $* " in *" -Wl,-ld_classic "*) classic=1 ;; *) classic=0 ;; esac
if [ "$classic" = 0 ] && ld -ld_classic -v >/dev/null 2>&1; then
    echo "link_check: linking $out again with the classic linker (-Wl,-ld_classic)" >&2
    "$@" -Wl,-ld_classic || exit $?
    if check "$out"; then
        echo "link_check: $out is clean with the classic linker" >&2
        exit 0
    fi
fi
rm -f "$out"
echo "link_check: $out removed. Update the command-line tools (System Settings > Software Update," >&2
echo "  or: softwareupdate --list); BRISK_LINKCHECK=0 make ... keeps the binary unchecked." >&2
exit 1
