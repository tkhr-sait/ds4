#!/usr/bin/env bash
# Run a single Q4 verification scenario end-to-end (no operator interaction).
#
#   1) Cache sudo credentials up-front, run `sudo purge` here (so the tmux
#      pane never has to prompt — avoids macOS `tty_tickets` cache misses).
#   2) Launch the 5-pane tmux session DETACHED (TMUX_ATTACH=0).
#   3) Wait for ds4-server's "listening on http" marker in its log.
#   4) Invoke `opencode run "<minesweeper prompt>"`, wait for completion.
#   5) Tear down the tmux session.
#
# Usage:
#   bash ./run.sh <prefix> '<server args>' ['<env vars>']
#
# Examples (paired runs for "現 prefetch 効果の上限確認"):
#   bash ./run.sh q4-baseline-rerun '--cpu-moe --prefill-metal-phases auto'
#   bash ./run.sh q4-no-prefetch    '--cpu-moe --prefill-metal-phases auto' \
#                                   'DS4_PREFETCH_ROUTED_EXPERTS=0'
#
# Env overrides:
#   MODEL                   GGUF path relative to ds4-server cwd
#   PROMPT                  Override the user prompt sent to opencode.  Default:
#                           "create minesweeper. 1 html file. file name is
#                           ${PREFIX}-minesweeper.html".  Use this to A/B a
#                           different agent task against the same engine
#                           config, e.g. PROMPT='write a fizzbuzz in python'.
#   PROMPT_FILE             Path to a UTF-8 text file whose contents become the
#                           prompt (takes precedence over PROMPT).  Handy for
#                           long multi-line prompts you do not want to inline
#                           into the shell command.
#   SESSION                 tmux session name (default: ds4mon-verify)
#   READY_TIMEOUT_S         seconds to wait for "listening on" (default: 600)
#   OPENCODE                opencode binary path / command (default: opencode)
#   OPENCODE_RUN_ARGS       extra args for `opencode run` (e.g. --model, --no-ui)
#   CLIENT                  Which client drives the workload (default: opencode):
#                             opencode -> `opencode run "<prompt>"` agent flow
#                                         (multi-turn, writes a file artifact).
#                             chat     -> single streaming POST to
#                                         /v1/chat/completions via curl, matching
#                                         the cpu-moe-memo "chat" capture (1
#                                         request, long single-turn generation).
#                                         Needs only curl + python3, no opencode.
#                             verify   -> one deterministic (fixed-seed,
#                                         non-streaming) request; writes the
#                                         decoded reasoning+answer to
#                                         /tmp/<prefix>-verify.txt.  Set
#                                         VERIFY_REF=<earlier verify .txt> to diff
#                                         and print IDENTICAL / DIFFERENT -- used
#                                         to A/B an engine toggle (e.g. pass
#                                         'DS4_DECODE_FUSE_GATEUP=0' as the 3rd
#                                         positional) for bit-equivalence.
#                                         Tunables: VERIFY_SEED (1), VERIFY_PROMPT,
#                                         VERIFY_MAX_TOKENS (200).
#   CHAT_URL                chat endpoint (default:
#                           http://127.0.0.1:8000/v1/chat/completions)
#   CHAT_MODEL              "model" field in the chat request (default: ds4;
#                           ds4-server does not validate it against the loaded
#                           GGUF, so any string works).
#   CHAT_MAX_TOKENS         max_tokens for the chat request (default: 16384).
#   POWER                   GPU duty-cycle target passed as `--power N` (1..100,
#                           default: 100 = no pacing).  NOTE: --power is NOT the
#                           throughput lever here -- in macOS "Automatic" power
#                           mode t/s stays ~8 whether --power is 90 or 100.  The
#                           real lever is the macOS system power mode: High Power
#                           mode + --power 100 sustains 10+ t/s, Automatic ~8.
#                           Kept as an overridable duty-cap knob (e.g. POWER=90
#                           to lower peak die temp / smooth variance); a --power N
#                           in the server ARGS (2nd positional) appears after and
#                           wins.

set -euo pipefail

PREFIX="${1:?need scenario prefix, e.g. q4-no-prefetch}"
ARGS="${2:?need server args, e.g. '--cpu-moe --prefill-metal-phases auto'}"
ENV_PREFIX="${3:-}"

SESSION="${SESSION:-ds4mon-verify}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
# Keep mactop / vm_stat sampling running this many seconds AFTER opencode
# returns, so the trailing post-prompt period (cache save, gen restore,
# residency teardown) lands in the plots.  Increase if mactop's --headless
# interval is large (default 5 s) so at least a few samples flush before
# tmux kills the pane.
COOLDOWN_S="${COOLDOWN_S:-30}"
OPENCODE="${OPENCODE:-opencode}"
OPENCODE_RUN_ARGS="${OPENCODE_RUN_ARGS:-}"
MODEL="${MODEL:-gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf}"

# GPU duty-cycle target (--power N), default 100 = no pacing.  --power is NOT the
# throughput lever: in macOS "Automatic" power mode t/s stays ~8 whether --power
# is 90 or 100; the real lever is the macOS system power mode (High Power mode +
# --power 100 -> 10+ t/s).  Kept as an overridable duty-cap knob (POWER=90 lowers
# peak die temp / smooths variance); a --power N in the server ARGS overrides it.
POWER="${POWER:-100}"

# Workload client: "opencode" (default agent flow) or "chat" (single curl chat
# completion against ds4-server, matching the cpu-moe-memo "chat" capture).
CLIENT="${CLIENT:-opencode}"
CHAT_URL="${CHAT_URL:-http://127.0.0.1:8000/v1/chat/completions}"
CHAT_MODEL="${CHAT_MODEL:-ds4}"
CHAT_MAX_TOKENS="${CHAT_MAX_TOKENS:-16384}"

# SCRIPTS_DIR = this script's own directory (holds launch_tmux.sh /
# mactop_filter.py); independent of where you invoke run.sh from.
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd -P)"
# WORKDIR = where logs / artifacts land.  Defaults to the invoking directory so
# `../scripts/run.sh <prefix> ...` drops logs in the current date-dir.  Override
# with WORKDIR=/abs/path to force a location.
WORKDIR="${WORKDIR:-$PWD}"
# ds4-server binary lives at the repo root (two levels above scripts/).
DS4_SERVER_BIN="${DS4_SERVER_BIN:-$(cd "$SCRIPTS_DIR/../.." && pwd -P)/ds4-server}"
DS4_LOG="$WORKDIR/${PREFIX}-ds4-server.log"
DS4_TRACE="$WORKDIR/${PREFIX}-trace.log"

# Prompt sent to `opencode run`.  Override via env to A/B different agent tasks
# against the same engine config:
#   PROMPT='draw a sine wave svg...'    bash ./run.sh foo '...'
#   PROMPT_FILE=./prompts/long.txt      bash ./run.sh foo '...'
# `${PREFIX}` is left available in the default string so the produced artifact
# file name (e.g. `<prefix>-minesweeper.html`) stays unique per scenario.
if [ -n "${PROMPT_FILE:-}" ]; then
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "[run] [$PREFIX] ERROR: PROMPT_FILE not found: $PROMPT_FILE" >&2
        exit 2
    fi
    PROMPT="$(cat "$PROMPT_FILE")"
fi
PROMPT="${PROMPT:-create minesweeper. 1 html file. file name is ${PREFIX}-minesweeper.html}"

# ---- preflight: client tools (see cpu-moe-memo/README.md "Requirements") ----
# tmux / mactop / vm_stat / python3 / ds4-server are checked by launch_tmux.sh;
# here we only verify what THIS script's selected CLIENT needs.
_missing=""
_need() { command -v "$1" >/dev/null 2>&1 || _missing="$_missing $1"; }
case "$CLIENT" in
    opencode) _need "${OPENCODE:-opencode}" ;;
    chat)     _need curl; _need python3 ;;
    verify)   _need curl; _need python3 ;;
esac
if [ -n "$_missing" ]; then
    echo "[run] [$PREFIX] ERROR: missing tool(s) for CLIENT=$CLIENT:$_missing" >&2
    echo "[run]   see cpu-moe-memo/README.md \"Requirements\" to install them." >&2
    [ "$CLIENT" = opencode ] && \
        echo "[run]   opencode setup (model/provider/endpoint): see ds4/README.md." >&2
    exit 3
fi

# ---- 1. sudo up-front -------------------------------------------------------
#echo "[run] [$PREFIX] caching sudo credentials (may prompt once)..."
#sudo -v
echo "[run] [$PREFIX] sudo purge (flushing macOS unified buffer cache)..."
sudo -n purge

# Keep sudo cache alive in the background.  macOS sudo cache decays after
# ~5min by default; long opencode sessions could outlive it, but we only need
# sudo for purge which is already done.  Skip the keepalive for simplicity.

# ---- cleanup hook -----------------------------------------------------------
cleanup() {
    echo "[run] [$PREFIX] tearing down tmux session $SESSION..."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- 2. launch tmux detached ------------------------------------------------
# Wipe stale log so the "listening on" wait sees this run's marker only.
rm -f "$DS4_LOG" "$DS4_TRACE"

echo "[run] [$PREFIX] launching tmux ($SESSION) with args: $ARGS"
[ -n "$ENV_PREFIX" ] && echo "[run] [$PREFIX] env: $ENV_PREFIX"

TMUX_ATTACH=0 \
PURGE_BEFORE=0 \
SESSION="$SESSION" \
TEST_PREFIX="$PREFIX" \
DS4_SERVER="$DS4_SERVER_BIN" \
DS4_SERVER_ENV="$ENV_PREFIX" \
DS4_SERVER_ARGS="--model ${MODEL} --power ${POWER} ${ARGS} --ctx 100000 --kv-disk-dir /tmp/${PREFIX}-ds4-kv --kv-disk-space-mb 8192 --trace ${DS4_TRACE}" \
WORKDIR="$WORKDIR" \
    bash "$SCRIPTS_DIR/launch_tmux.sh"

# ---- 3. wait for ds4-server ready ------------------------------------------
echo "[run] [$PREFIX] waiting up to ${READY_TIMEOUT_S}s for ds4-server to listen..."
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
while [ $(date +%s) -lt $deadline ]; do
    if [ -f "$DS4_LOG" ] && grep -q "listening on http" "$DS4_LOG" 2>/dev/null; then
        echo "[run] [$PREFIX] ds4-server ready."
        break
    fi
    sleep 2
done
if ! grep -q "listening on http" "$DS4_LOG" 2>/dev/null; then
    echo "[run] [$PREFIX] ERROR: timed out waiting for ds4-server"
    tail -30 "$DS4_LOG" 2>/dev/null || true
    exit 1
fi

# Small extra settle for the http server / context to fully come up.
sleep 2

# ---- 4. dispatch workload ---------------------------------------------------
set +e
if [ "$CLIENT" = "verify" ]; then
    # Deterministic bit-equivalence check.  Fixed seed + non-streaming makes the
    # response reproducible, so flipping an engine toggle (e.g. pass
    # 'DS4_DECODE_FUSE_GATEUP=0' as the 3rd positional) must yield byte-identical
    # output.  Set VERIFY_REF=<earlier verify .txt> to diff and print a verdict.
    # Fixed /tmp (NOT $TMPDIR, which on macOS is a per-user /var/folders path)
    # so the output path is predictable for VERIFY_REF and the same across runs.
    VERIFY_OUT="/tmp/${PREFIX}-verify.txt"
    VERIFY_REQ="/tmp/${PREFIX}-verify-request.json"
    echo "[run] [$PREFIX] dispatching deterministic verify request (seed=${VERIFY_SEED:-1})..."
    PROMPT="${VERIFY_PROMPT:-Count from 1 to 60 in words, one per line, then explain the Fibonacci sequence in detail.}" \
    CHAT_MODEL="$CHAT_MODEL" VERIFY_SEED="${VERIFY_SEED:-1}" VERIFY_MAX_TOKENS="${VERIFY_MAX_TOKENS:-200}" \
        python3 - "$VERIFY_REQ" <<'PY'
import json, os, sys
body = {
    "model": os.environ.get("CHAT_MODEL", "ds4"),
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ.get("VERIFY_MAX_TOKENS", "200")),
    "seed": int(os.environ.get("VERIFY_SEED", "1")),
    "stream": False,
}
with open(sys.argv[1], "w") as f:
    json.dump(body, f)
PY
    # Capture reasoning + answer (both are determined by the logit stream, so a
    # numeric divergence shows up here).  The volatile response id is excluded.
    # Extract reasoning + answer with python3 (mirrors the jq `//` null-fallback:
    # message.reasoning_content // "", then message.content // .text // "").
    curl -sS -X POST "$CHAT_URL" -H "Content-Type: application/json" \
        --data @"$VERIFY_REQ" \
        | python3 -c 'import json,sys
c=json.load(sys.stdin)["choices"][0]
m=c.get("message") or {}
r=m.get("reasoning_content")
a=m.get("content")
if a is None: a=c.get("text")
sys.stdout.write((r if r is not None else "")+"\n===ANSWER===\n"+(a if a is not None else ""))' \
        > "$VERIFY_OUT"
    oc_exit=$?
    echo "[run] [$PREFIX] verify output -> $VERIFY_OUT ($(wc -c < "$VERIFY_OUT" 2>/dev/null || echo 0) bytes)"
    if [ -n "${VERIFY_REF:-}" ]; then
        if diff -q "$VERIFY_REF" "$VERIFY_OUT" >/dev/null 2>&1; then
            echo "[run] [$PREFIX] VERIFY: IDENTICAL to $VERIFY_REF  (bit-equivalent OK)"
        else
            echo "[run] [$PREFIX] VERIFY: DIFFERENT from $VERIFY_REF  <-- outputs diverge!"
            diff "$VERIFY_REF" "$VERIFY_OUT" | head -20
        fi
    fi
elif [ "$CLIENT" = "chat" ]; then
    echo "[run] [$PREFIX] dispatching chat completion to $CHAT_URL..."
    echo "[run] [$PREFIX] prompt: $PROMPT"
    # Keep these scratch artifacts out of the (IDE/git-visible) verification dir
    # — drop them in /tmp alongside the KV cache.
    CHAT_REQ="${TMPDIR:-/tmp}/${PREFIX}-chat-request.json"
    CHAT_RESP="${TMPDIR:-/tmp}/${PREFIX}-chat-response.txt"
    # Build the request body with python3 so the prompt is safely JSON-escaped
    # (quotes / newlines / unicode), independent of shell quoting.
    PROMPT="$PROMPT" CHAT_MODEL="$CHAT_MODEL" CHAT_MAX_TOKENS="$CHAT_MAX_TOKENS" \
        python3 - "$CHAT_REQ" <<'PY'
import json, os, sys
body = {
    "model": os.environ.get("CHAT_MODEL", "ds4"),
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ.get("CHAT_MAX_TOKENS", "16384")),
    "stream": True,
}
with open(sys.argv[1], "w") as f:
    json.dump(body, f)
PY
    # --no-buffer streams the SSE deltas as they arrive (per-50-token t/s lands
    # in the ds4-server pane).  The raw event stream is saved to CHAT_RESP for
    # post-hoc inspection; stdout here stays quiet.
    curl -sS --no-buffer -X POST "$CHAT_URL" \
        -H "Content-Type: application/json" \
        --data @"$CHAT_REQ" > "$CHAT_RESP"
    oc_exit=$?
    echo "[run] [$PREFIX] chat response saved to $CHAT_RESP"
else
    echo "[run] [$PREFIX] dispatching opencode prompt..."
    echo "[run] [$PREFIX] prompt: $PROMPT"
    # `opencode run` is the headless one-shot mode.  Adjust OPENCODE /
    # OPENCODE_RUN_ARGS if your opencode invocation differs (e.g. needs --model,
    # --provider, --endpoint).
    "$OPENCODE" run $OPENCODE_RUN_ARGS "$PROMPT"
    oc_exit=$?
fi
set -e
echo "[run] [$PREFIX] $CLIENT exited with $oc_exit"

# ---- 5. cooldown: let mactop / vm_stat capture the tail ---------------------
# After opencode returns, ds4-server still does cache save, gen restore, and
# Metal residency teardown for several seconds.  Sleep here (instead of in the
# cleanup trap) so mactop / vm_stat keep emitting samples until tmux is killed.
if [ "$COOLDOWN_S" -gt 0 ]; then
    echo "[run] [$PREFIX] cooldown ${COOLDOWN_S}s (mactop / vm_stat still sampling)..."
    sleep "$COOLDOWN_S"
fi

# ---- 6. teardown via trap --------------------------------------------------
echo "[run] [$PREFIX] done."
