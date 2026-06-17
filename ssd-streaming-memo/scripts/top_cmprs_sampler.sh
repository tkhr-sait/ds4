#!/usr/bin/env bash
# top_cmprs_sampler.sh — emit per-process *compressed memory* (cmprs) telemetry.
#
# Why this exists: macOS `top` is the only base CLI that exposes per-process
# compressed memory (the `cmprs` stat).  `ps` only has rss/vsz, and rss EXCLUDES
# compressed pages, so a process whose pages got compressed shows a *lower* rss
# — useless for attributing the macOS compressor pool.  `top` reports both the
# system-wide compressor (PhysMem header) and the per-process `cmprs` column.
#
# Used by launch_tmux.sh as a background sampler next to the vm_stat capture, so
# a session records WHO holds the compressor pool (krunkit = the Podman VM that
# hosts opencode, vs ds4-server) on the same time axis as vm_stat / mactop.
#
# Usage (writes to stdout; launch_tmux.sh redirects it to ${PREFIX}-top-cmprs.log):
#   bash top_cmprs_sampler.sh [interval_seconds] [match_regex]
#
# Output format (whitespace-separated, epoch-prefixed, awk/make_plots friendly):
#   <epoch> PHYSMEM PhysMem: 104G used (4G wired, 18G compressor), 24G unused.
#   <epoch> <pid> <command> <cmprs>          # one line per watched process
#
# The PHYSMEM line cross-checks the per-process sum against vm_stat's
# "Pages occupied by compressor".  Default watch set covers the Podman VM host
# (krunkit), the native server (ds4-server), and the network shim (gvproxy).

set -u

INTERVAL="${1:-5}"
MATCH="${2:-krunkit|ds4-server|gvproxy}"

while :; do
    ts=$(date +%s)
    # One `top` sample, columns limited to pid/command/cmprs, sorted by cmprs.
    out=$(top -l 1 -o cmprs -stats pid,command,cmprs -n 25 2>/dev/null) || out=""
    # System-wide compressor (PhysMem header) — cross-check vs vm_stat.
    printf '%s\n' "$out" | awk -v ts="$ts" 'tolower($0) ~ /physmem/ {print ts, "PHYSMEM", $0}'
    # Per-process compressed for the watched set (case-insensitive name match).
    printf '%s\n' "$out" | awk -v ts="$ts" -v re="$MATCH" 'tolower($0) ~ tolower(re) {print ts, $0}'
    sleep "$INTERVAL"
done
