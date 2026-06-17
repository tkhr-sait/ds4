#!/usr/bin/env bash
# Launch the 6-pane tmux layout used for cpu-moe-memo runs.
#
#   +---------------------+---------------------+
#   | vm_stat 5           | top -o cmprs (~12%) |
#   +---------------------+---------------------+
#   | ds4-server log      | mactop TUI          |
#   +---------------------+---------------------+
#   | (free shell)        | mactop headless     |
#   +---------------------+---------------------+
#
# The left-bottom pane is left empty so the operator can run whatever client
# they need (opencode, curl chat completions, ds4 cli, etc.).  mactop is
# captured twice: the right-middle pane runs the TUI for the operator, while
# the right-bottom pane streams headless JSON through scripts/mactop_filter.py
# to ${TEST_PREFIX}-mactop.log (trimmed NDJSON parsed by make_plots.py).
# iostat is dropped because mactop's net_disk already covers disk MB/s.
# The top strip is split: vm_stat (left) and scripts/top_cmprs_sampler.sh
# (right) -> ${TEST_PREFIX}-top-cmprs.log, the per-process `cmprs` capture that
# attributes the macOS compressor pool (krunkit/VM vs ds4-server).
#
# Usage:
#   TEST_PREFIX=chat ./scripts/launch_tmux.sh
#   TEST_PREFIX=opencode ./scripts/launch_tmux.sh
#
# Environment overrides:
#   SESSION         tmux session name (default: ds4mon)
#   TEST_PREFIX     log file prefix (default: chat)
#   DS4_SERVER      path to ds4-server binary (default: ../ds4-server)
#   WORKDIR         cwd inside panes (default: $PWD)
#   DS4_SERVER_ARGS ds4-server command-line args.  Default keeps the original
#                   cpu-moe-memo capture command (--prefill-metal-phases auto
#                   --ctx 100000 --kv-disk-dir /tmp/${TEST_PREFIX}-ds4-kv
#                   --kv-disk-space-mb 8192).  Override to run hot-pin or any
#                   other configuration, e.g.:
#                     DS4_SERVER_ARGS="--cpu-moe --n-hot-experts 8 --imatrix-in ../bin --ctx 100000 --kv-disk-space-mb 8192"
#   DS4_SERVER_ENV  env vars prefixed to the ds4-server command (default:
#                   empty, i.e. use the binary's built-in defaults including
#                   DS4_PREFILL_METAL_PHASES_MIN_TOKENS=1500).  Set e.g.
#                   DS4_SERVER_ENV="DS4_PREFILL_METAL_PHASES_MIN_TOKENS=0"
#                   to force phase split even on short prompts.
#   PURGE_BEFORE    "1" to run `sudo purge` before launching the server (clears
#                   the macOS file cache so each run starts from the same cold
#                   state).  Default "1".  Set to "" or "0" to skip.  `sudo
#                   purge` will prompt for the operator password inside the
#                   ds4-server pane on the first run.
#   KV_DISK_DIR     KV disk cache directory to wipe before launch (default
#                   /tmp/${TEST_PREFIX}-ds4-kv, matching the default
#                   DS4_SERVER_ARGS).  Set to "" to skip the wipe (e.g. if you
#                   are intentionally continuing a previous KV).
#   MACTOP_INTERVAL_MS  mactop headless sample interval in ms (default: 5000)
#   VMSTAT_INTERVAL_S   vm_stat sample interval in seconds (default: 5)
#   TMUX_ATTACH     "1" to tmux attach at the end (interactive, default).
#                   Set to "0" for automation (07-verification/run.sh) so the
#                   script returns immediately after the session is set up.
#
# Requires tmux and mactop (https://github.com/context-labs/mactop).
# Run from a directory where vm_stat / mactop logs should land (e.g.
# cpu-moe-memo/<date-dir>).

set -euo pipefail

SESSION="${SESSION:-ds4mon}"
TEST_PREFIX="${TEST_PREFIX:-chat}"
WORKDIR="${WORKDIR:-$PWD}"
DS4_SERVER="${DS4_SERVER:-ds4-server}"
DS4_SERVER_ARGS="${DS4_SERVER_ARGS:---prefill-metal-phases auto --ctx 100000 --kv-disk-dir /tmp/${TEST_PREFIX}-ds4-kv --kv-disk-space-mb 8192}"
DS4_SERVER_ENV="${DS4_SERVER_ENV-}"
PURGE_BEFORE="${PURGE_BEFORE:-1}"
KV_DISK_DIR="${KV_DISK_DIR-/tmp/${TEST_PREFIX}-ds4-kv}"
MACTOP_INTERVAL_MS="${MACTOP_INTERVAL_MS:-5000}"
VMSTAT_INTERVAL_S="${VMSTAT_INTERVAL_S:-5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
MACTOP_FILTER="$SCRIPT_DIR/mactop_filter.py"
TOP_SAMPLER="$SCRIPT_DIR/top_cmprs_sampler.sh"

# WORKDIR must exist; pin it as absolute so the rest of the resolution is stable.
WORKDIR="$(cd "$WORKDIR" && pwd -P)"

# ds4-server assumes its own directory is the cwd (it loads weights / mmap
# files via relative paths next to the binary).  Resolve to an absolute binary
# path and an absolute log path, then run the pane chdir'd to the binary dir.
case "$DS4_SERVER" in
  /*) ;;
  *)  DS4_SERVER="$(cd "$WORKDIR" && cd "$(dirname "$DS4_SERVER")" && pwd -P)/$(basename "$DS4_SERVER")" ;;
esac
DS4_SERVER_DIR="$(dirname "$DS4_SERVER")"

DS4_LOG="$WORKDIR/${TEST_PREFIX}-ds4-server.log"
VM_LOG="$WORKDIR/${TEST_PREFIX}-vm_stat.log"
MACTOP_LOG="$WORKDIR/${TEST_PREFIX}-mactop.log"
TOP_CMPRS_LOG="$WORKDIR/${TEST_PREFIX}-top-cmprs.log"

# ---- preflight: required tools (see cpu-moe-memo/README.md "Requirements") --
# Spawned across the panes: tmux (layout), vm_stat / mactop (telemetry),
# python3 (mactop_filter.py), and the ds4-server binary.  Fail fast with a
# pointer to the install list instead of leaving half-empty panes.
_lt_missing=""
for _t in tmux mactop vm_stat top python3; do
    command -v "$_t" >/dev/null 2>&1 || _lt_missing="$_lt_missing $_t"
done
[ -x "$DS4_SERVER" ] || _lt_missing="$_lt_missing $DS4_SERVER"
if [ -n "$_lt_missing" ]; then
    echo "launch_tmux.sh: ERROR: missing required tool(s):$_lt_missing" >&2
    echo "  install them (see cpu-moe-memo/README.md \"Requirements\") and retry." >&2
    exit 3
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$WORKDIR"

P_VM=$(tmux list-panes -t "$SESSION:0" -F '#{pane_id}' | head -n 1)

# Top vm_stat is ~12% height; main area takes the remaining 88%.
P_MAIN=$(tmux split-window -t "$P_VM" -v -l 88% -P -F '#{pane_id}' -c "$WORKDIR")

# Split the top telemetry strip: vm_stat (left) | top-cmprs sampler (right), so
# per-process compressed-memory attribution runs foreground in its own pane and
# is killed cleanly with the session (no orphaned background `top` loop).
P_TOPCMPRS=$(tmux split-window -t "$P_VM" -h -l 40% -P -F '#{pane_id}' -c "$WORKDIR")

# Main area: 50/50 left (ds4-server + free shell) | right (mactop TUI + mactop JSON).
P_RIGHT=$(tmux split-window -t "$P_MAIN" -h -l 50% -P -F '#{pane_id}' -c "$WORKDIR")
P_LEFT="$P_MAIN"

# Left column: 50/50 ds4-server (top) / free shell (bottom).
# The ds4-server pane chdir's to the binary's own directory.
P_FREE=$(tmux split-window -t "$P_LEFT" -v -l 50% -P -F '#{pane_id}' -c "$WORKDIR")
P_DSSRV="$P_LEFT"
tmux send-keys -t "$P_DSSRV" "cd \"$DS4_SERVER_DIR\"" Enter

# Right column: mactop TUI (~70%) / mactop --headless JSON capture (~30%).
P_MACTOP_JSON=$(tmux split-window -t "$P_RIGHT" -v -l 30% -P -F '#{pane_id}' -c "$WORKDIR")
P_MACTOP="$P_RIGHT"

# `script -q /dev/null` wraps vm_stat in a pseudo-TTY so its stdout is
# line-buffered.  Without it, vm_stat block-buffers when stdout is a pipe and
# the trailing samples (= the COOLDOWN_S tail in run.sh) sit in libc's buffer
# until tmux kills the pane.  macOS's BSD `script` ships in the base system.
tmux send-keys -t "$P_VM"     "script -q /dev/null vm_stat $VMSTAT_INTERVAL_S 2>&1 | tee \"$VM_LOG\"" Enter
# Per-process compressed-memory (cmprs) sampler — foreground in its own pane so
# it is killed cleanly with the session.  `top` is macOS base (no extra dep).
# Each iteration's awk exits and flushes, so unlike vm_stat it needs no `script`
# PTY wrapper for line buffering.
tmux send-keys -t "$P_TOPCMPRS" "bash \"$TOP_SAMPLER\" \"$VMSTAT_INTERVAL_S\" 2>&1 | tee \"$TOP_CMPRS_LOG\"" Enter
tmux send-keys -t "$P_MACTOP" "mactop" Enter
# `python3 -u` keeps the filter unbuffered so each mactop JSON record reaches
# the log file immediately (otherwise the last several samples sit in the pipe
# buffer and are lost when tmux kills the pane at cleanup time).
tmux send-keys -t "$P_MACTOP_JSON" \
    "mactop --headless --count 0 -i $MACTOP_INTERVAL_MS | python3 -u \"$MACTOP_FILTER\" > \"$MACTOP_LOG\"" Enter
DS4_SERVER_CMD="\"$DS4_SERVER\" $DS4_SERVER_ARGS 2>&1 | tee \"$DS4_LOG\""
if [ -n "$DS4_SERVER_ENV" ]; then
    DS4_SERVER_CMD="$DS4_SERVER_ENV $DS4_SERVER_CMD"
fi
# Pre-launch hygiene so each capture starts from the same cold state:
#   1) wipe the KV disk cache so server reprefills from scratch
#   2) sudo purge to flush the macOS unified buffer cache (model mmap, etc.)
# Both are chained with && so a failure halts the launch and the operator can
# react inside the pane.
PRE_CMDS=""
if [ -n "$KV_DISK_DIR" ]; then
    PRE_CMDS="rm -rf \"$KV_DISK_DIR\""
fi
if [ "$PURGE_BEFORE" = "1" ]; then
    if [ -n "$PRE_CMDS" ]; then
        PRE_CMDS="$PRE_CMDS && sudo purge"
    else
        PRE_CMDS="sudo purge"
    fi
fi
if [ -n "$PRE_CMDS" ]; then
    DS4_SERVER_CMD="$PRE_CMDS && $DS4_SERVER_CMD"
fi
tmux send-keys -t "$P_DSSRV" "$DS4_SERVER_CMD" Enter

# Left-bottom pane: no command sent — operator types whatever client they want.

tmux select-pane -t "$P_FREE"
if [ "${TMUX_ATTACH:-1}" = "1" ]; then
    tmux attach -t "$SESSION"
fi
