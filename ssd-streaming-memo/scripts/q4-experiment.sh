#!/usr/bin/env bash
# q4-experiment.sh — driver for a DeepSeek-V4 opencode run (streaming or full).
#
# Picks the model by TAG (q4/q2), launches run.sh (opencode client) under one of
# the expert-supply routes, and budget-tags the log prefix so re-runs at
# different budgets do not overwrite each other.
#
#   ROUTE = pread  pread expert cache    (--ssd-streaming, mlock'd slab)
#   ROUTE = full   NO --ssd-streaming    (whole model resident; the non-stream
#                  A/B floor — ignores BUDGET_MIB, adds no cache flag)
#   ROUTE = raw    no preset; --ssd-streaming + your own EXTRA_ENV / EXTRA_ARG
#
# ONE budget knob (BUDGET_MIB) sizes the streaming routes via a single ds4 flag,
# --ssd-streaming-cache-experts <GB> (always GB form, so the "32 == 32 experts
# not 32GB" gotcha can't happen).  `full` does not stream, so BUDGET_MIB only
# tags its log prefix.  (raw skips the auto cache flag for full manual control.)
#
# Machine-specific paths (DS4, DS4_SERVER_BIN) go in a gitignored config ONCE so
# the per-run command stays short — copy q4-experiment.conf.example to
# q4-experiment.conf and edit.  Everything is resolved relative to this script
# or via that config; no absolute paths are baked in (runs on macOS).
#
# Usage — run FROM a session/log dir (logs land in the cwd):
#   TAG=q2 BUDGET_MIB=32768  bash /path/to/scripts/q4-experiment.sh pread
#   TAG=q2                   bash /path/to/scripts/q4-experiment.sh full
#   TAG=q2 EXTRA_ARG='...' EXTRA_ENV='K=V' bash .../q4-experiment.sh raw
#
# Envs (besides the config file):
#   TAG         q2 (default) | q4 | q2-pro  (selects model + log prefix)
#   BUDGET_MIB  cache size in MiB -> --ssd-streaming-cache-experts <GB> (default
#               32768 = 32GB).  Drives pread.  Capped by ds4 to ~70% of the Metal
#               working set; use raw + env for a larger budget.  Ignored by full.
#   MODEL       gguf path relative to $DS4  (overrides the TAG default)
#   PREFIX      log-prefix override         (default: pread- / full- / '')
#   EXTRA_ARG   extra ds4-server args       (appended after the cache flag)
#   EXTRA_ENV   extra "K=V K=V" env

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd -P)"

# Machine config (gitignored): sets DS4 and exports DS4_SERVER_BIN for run.sh.
CONFIG="${CONFIG:-$SCRIPTS_DIR/q4-experiment.conf}"
# shellcheck disable=SC1090
[ -f "$CONFIG" ] && . "$CONFIG"

ROUTE="${1:-pread}"
# EXTRA_ENV may be the 2nd positional arg OR the EXTRA_ENV env var (positional
# wins).  Accepting both avoids silently dropping a 2nd-arg "K=V" passed by habit.
EXTRA_ENV="${2:-${EXTRA_ENV:-}}"
TAG="${TAG:-q2}"
BUDGET_MIB="${BUDGET_MIB:-32768}"

# ds4 repo root (holds gguf/ + the model).  From config, or env, or the
# sibling-layout fallback (<scripts>/../ds4).  ds4-server itself is run.sh's
# DS4_SERVER_BIN — may live elsewhere, which is why it is a separate setting.
DS4_REL="${DS4:-$SCRIPTS_DIR/../ds4}"
[ -d "$DS4_REL" ] || { echo "[q4x] ERROR: ds4 root '$DS4_REL' not found — set DS4= in $CONFIG" >&2; exit 2; }
DS4_ABS="$(cd "$DS4_REL" && pwd -P)"

case "$TAG" in
    q4)     DEFAULT_MODEL="gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf" ;;
    q2)     DEFAULT_MODEL="gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" ;;
    q2-pro) DEFAULT_MODEL="gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf" ;;
    *)      DEFAULT_MODEL="" ;;
esac
MODEL_ABS="$DS4_ABS/${MODEL:-$DEFAULT_MODEL}"

RUN_SH="$SCRIPTS_DIR/run.sh"
for f in "$RUN_SH" "$MODEL_ABS"; do
    [ -e "$f" ] || { echo "[q4x] ERROR: missing $f" >&2; exit 2; }
done

# ONE budget knob: BUDGET_MIB -> --ssd-streaming-cache-experts <GB> for the
# streaming routes.  Always GB form (the "32 == 32 experts not 32GB" gotcha can't
# happen).  NOTE: explicit NGB is capped by ds4 to ~70% of the Metal working set
# to keep it lockable (~75 GiB on a 128 GiB box); use ROUTE=raw + env for a
# larger budget.
CACHE_GB="$((BUDGET_MIB / 1024))GB"
EXTRA_ARG="${EXTRA_ARG:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
# BASE_ARG is the streaming toggle: every route enables --ssd-streaming EXCEPT
# `full`, which serves the whole model resident (no streaming) as the A/B floor.
BASE_ARG="--ssd-streaming"
case "$ROUTE" in
    pread) PREFIX="${PREFIX:-pread-}"
           EXTRA_ARG="--ssd-streaming-cache-experts ${CACHE_GB}${EXTRA_ARG:+ $EXTRA_ARG}" ;;
    full)  PREFIX="${PREFIX:-full-}"
           BASE_ARG="" ;;   # no --ssd-streaming, no cache flag: full model residency
    raw)   PREFIX="${PREFIX:-}" ;;   # full manual: pass EXTRA_ARG / EXTRA_ENV yourself
    *) echo "[q4x] usage: q4-experiment.sh pread|full|raw   (got '$ROUTE')" >&2; exit 2 ;;
esac

# Assemble the server args: BASE_ARG (--ssd-streaming or empty) + any EXTRA_ARG.
SERVER_ARGS="$BASE_ARG"
[ -n "$EXTRA_ARG" ] && SERVER_ARGS="${SERVER_ARGS:+$SERVER_ARGS }$EXTRA_ARG"

# CLIENT selects the workload driver in run.sh (default opencode agent flow).
# Set CLIENT=verify for a deterministic fixed-seed single request (VERIFY_SEED /
# VERIFY_PROMPT / VERIFY_MAX_TOKENS / VERIFY_REF) or CLIENT=chat for one curl
# chat completion -- both pass through to run.sh via the inherited environment.
CLIENT="${CLIENT:-opencode}"
# Tag the log prefix by client so a verify A/B does not overwrite an opencode run.
CLIENT_TAG="$([ "$CLIENT" = opencode ] && echo opencode || echo "$CLIENT")"
LOG_PREFIX="${PREFIX}${TAG}-${CLIENT_TAG}-b${BUDGET_MIB}"
ENVP="${EXTRA_ENV}"
WORKDIR="$(pwd -P)"

echo "[q4x] route=$ROUTE tag=$TAG client=$CLIENT budget=${BUDGET_MIB}MiB ds4=$DS4_ABS"
echo "[q4x] args[${SERVER_ARGS:-(none)}] env[$ENVP]"
echo "[q4x] model=$MODEL_ABS  workdir=$WORKDIR  log=${LOG_PREFIX}-*"

CLIENT="$CLIENT" MODEL="$MODEL_ABS" WORKDIR="$WORKDIR" \
    bash "$RUN_SH" "$LOG_PREFIX" "$SERVER_ARGS" "$ENVP"

log="$WORKDIR/${LOG_PREFIX}-ds4-server.log"
echo "[q4x] ---- budget / enabled / finish / last decode ----"
grep -E "cache budget|streaming mode enabled|finish=" "$log" 2>/dev/null | tail -8 || true
grep -E "decoding chunk=" "$log" 2>/dev/null | tail -3 || true
echo "[q4x] done (route=$ROUTE). logs in $WORKDIR"
