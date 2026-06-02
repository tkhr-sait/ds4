#!/usr/bin/env bash
# Q4 experiment driver for --routed-metal-dynamic (gen-time routed expert on
# Metal via per-expert LRU residency).  This is a shared script in
# cpu-moe-memo/scripts/; run it FROM a session dir so logs land there:
#
#   cd /workspace/ds4/cpu-moe-memo/02-routed-metal-dynamic
#   bash ../scripts/q4-experiment.sh [mode] [extra_env]
#
# [extra_env] (2nd positional arg, or EXTRA_ENV=) is a space-separated "K=V K=V"
# string injected into every dynamic run (and the ds4-bench dyn runs), appended
# after DS4_ROUTED_METAL_BUDGET_MIB.  Use it to flip the experimental residency
# toggles, e.g.:
#   bash ../scripts/q4-experiment.sh opencode 'DS4_ROUTED_METAL_CLEAR_ON_GEN=1'
#   bash ../scripts/q4-experiment.sh opencode 'DS4_PREFILL_PHASE_KEEP=0'
#   bash ../scripts/q4-experiment.sh opencode 'DS4_ROUTED_METAL_CLEAR_ON_GEN=1 DS4_PREFILL_PHASE_KEEP=0'
#
# Modes:
#   opencode (default) run.sh opencode (dyn).  Real
#                      agent workload (opencode `run` builds a file artifact) —
#                      the closest to real operational performance.  Reports gen
#                      t/s, finish=, stats (hit%/wired peak, needs SIGHUP-aware
#                      ds4-server), AND SSD reads (pageins GiB) — the key metric
#                      for "did the wired budget squeeze page cache into SSD
#                      thrash?".  Override the task via PROMPT='...'/PROMPT_FILE=.
#   oc-cpu             Same opencode workload on the pure CPU path (--cpu-moe, no
#                      dynamic).  No wired-expert competition -> max page cache ->
#                      the SSD-read FLOOR baseline to compare dyn budgets against.
#   chat               run.sh chat (dyn) + poll.  Single long streaming gen
#                      (CHAT_MAX_TOKENS), simpler than the agent flow.
#   perfab             THE honest perf A/B.  Long continuous chat (PERF_MAX_TOKENS,
#                      default 6144) for cpu vs dyn, then compares the last-third
#                      (warm STEADY-STATE) chunk t/s — excludes the cold LRU
#                      warmup that makes a short gen meaningless for this feature.
#   bench              ds4-bench cpu vs dyn, LONG single-frontier gen (GEN_TOKENS
#                      default 3072 so the LRU warms).  Use for the warmed
#                      `routed-metal-dynamic stats` line (hit%, evicts, wired peak)
#                      at engine_close.  Avg t/s ~ steady-state when gen is long.
#   sweep              ONE cpu baseline + dyn across budgets (SWEEP_BUDGETS), each
#                      a fresh long-gen ds4-bench run at a single frontier.  Prints
#                      a summary table (gen_tps / hit% / wired_peak per budget) so
#                      you pick the best budget.  RUN THIS BEFORE perfab: cpu t/s is
#                      budget-independent, dyn t/s depends on budget via hit rate.
#                      Then confirm the winner with perfab at that BUDGET_MIB.
#   all                sweep, then perfab (at the default BUDGET_MIB).
#
# Why long gen matters: --routed-metal-dynamic is gen-only, so prefill does NOT
# warm the LRU residency set.  A short gen (e.g. 256 tok) measures the cold
# all-miss regime (requestResidency every layer) and unfairly penalises dyn; the
# real benefit (warm working set, high hit rate) only appears after a few
# thousand tokens.  See the smoke runs: t/s ramped 3->11 over 4k tokens.
#
# Q4 (145 GiB > 128 GiB RAM): the always-Metal full-resident baseline
# (--n-cpu-moe 0) OOMs, so the CPU baseline is --cpu-moe.  Prefill uses
# --prefill-metal-phases auto (prefill-only) so the slow CPU prefill does not
# dominate; the gen-only dynamic path is independent of it.
#
# Output files are budget-tagged so re-runs at different budgets do NOT overwrite
# each other: dyn logs are named *-b<BUDGET_MIB>-* (e.g. q4-dyn-oc-b40960-*,
# q4-perf-dyn-b32768-*, q4-bench-dyn-b40960.log).  The CPU baseline is
# budget-independent so it keeps a single name (${TAG}-perf-cpu-*, ${TAG}-bench-cpu.log).
#
# Env overrides:
#   DS4          ds4 repo root (default: <this script>/../.. -> /workspace/ds4)
#   TAG          q4 (default, 145 GiB > RAM) or q2 (~80 GiB, fits in RAM).  Selects
#                the default model AND the log-file prefix, so q4/q2 runs don't
#                collide.  e.g.  TAG=q2 BUDGET_MIB=40960 bash ../scripts/q4-experiment.sh opencode
#                (q2 routed experts ≈ 72.6 GiB, so a budget ≥ that never evicts =
#                full residency; use BUDGET_MIB < 72.6 GiB to actually exercise the LRU).
#   MODEL        gguf path relative to $DS4 (overrides the TAG default; MODEL_Q4 is a
#                legacy alias for the same thing)
#   BUDGET_MIB   DS4_ROUTED_METAL_BUDGET_MIB for dynamic runs (default: 24576).
#                Start conservative on a 128 GiB box; raise once poll shows
#                wired headroom.  Sweep mode ignores this and loops its own set.
#   CTX_START / CTX_MAX / GEN_TOKENS   ds4-bench frontier sweep (4096/16384/256).
#   CHAT_MAX_TOKENS                    smoke chat length (default: 4096).
#   PREFILL      prefill flag for runs (default: '--prefill-metal-phases auto').
#                Set PREFILL='' to use the simpler (slow) CPU prefill if the
#                phases+dynamic combination misbehaves.

set -euo pipefail

MODE="${1:-opencode}"
# Extra env vars injected into every dynamic (and bench/sweep) run, as a single
# space-separated "K=V K=V" string.  Pass as the 2nd positional arg or via
# EXTRA_ENV=.  Use it to flip the experimental residency toggles, e.g.
#   bash ../scripts/q4-experiment.sh opencode 'DS4_ROUTED_METAL_CLEAR_ON_GEN=1'
#   bash ../scripts/q4-experiment.sh opencode 'DS4_PREFILL_PHASE_KEEP=0'
# These are appended AFTER the budget var, so they compose with it.  For the
# ds4-bench paths (bench/sweep) they are exported into the subshell `env`.
EXTRA_ENV="${2:-${EXTRA_ENV:-}}"
# join: "<budget var> [extra]" for the run.sh envp string (4th run.sh arg).
dyn_envp() { local b="$1"; printf '%s' "DS4_ROUTED_METAL_BUDGET_MIB=${b}${EXTRA_ENV:+ ${EXTRA_ENV}}"; }

# ---- resolve paths (everything anchored at the ds4 repo root) ---------------
# This script lives in cpu-moe-memo/scripts/; the repo root is two levels up.
# Resolve it from the SCRIPT location (not the cwd) so it works run as
# ../scripts/q4-experiment.sh from any session dir; logs still land in the cwd.
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd -P)"
DS4_REL="${DS4:-$SCRIPTS_DIR/../..}"
if [ ! -d "$DS4_REL" ]; then
    echo "[$TAG] ERROR: ds4 root not found at '$DS4_REL' (set DS4=/abs/path)" >&2
    exit 2
fi
DS4_ABS="$(cd "$DS4_REL" && pwd -P)"
RUN_SH="$DS4_ABS/cpu-moe-memo/scripts/run.sh"
BENCH_BIN="$DS4_ABS/ds4-bench"
PROMPT_FILE="$DS4_ABS/tests/long_context_story_prompt.txt"
TAG="${TAG:-q4}"   # selects model + log-file prefix: q4 (default, 145 GiB, >RAM) or q2 (~80 GiB, fits)
case "$TAG" in
    q4) DEFAULT_MODEL="gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf" ;;
    q2) DEFAULT_MODEL="gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" ;;
    *)  DEFAULT_MODEL="" ;;
esac
# Model path relative to $DS4_ABS.  Override with MODEL=...; MODEL_Q4 kept as a
# legacy alias.  For q2 note routed experts ≈ 72.6 GiB, so a budget ≥ that never
# evicts (degenerates to full residency) — use a smaller BUDGET_MIB to exercise the LRU.
MODEL_REL="${MODEL:-${MODEL_Q4:-$DEFAULT_MODEL}}"
MODEL_ABS="$DS4_ABS/$MODEL_REL"
PROMPT_REL="tests/long_context_story_prompt.txt"   # relative to $DS4_ABS

for f in "$RUN_SH" "$BENCH_BIN" "$MODEL_ABS" "$PROMPT_FILE"; do
    [ -e "$f" ] || { echo "[$TAG] ERROR: missing $f" >&2; exit 2; }
done

BUDGET_MIB="${BUDGET_MIB:-24576}"
PREFILL="${PREFILL:---prefill-metal-phases auto}"
# IMPORTANT: --routed-metal-dynamic is gen-only; prefill does NOT warm the LRU
# residency set (it only warms the OS page cache).  The dynamic path's benefit
# (token-to-token expert reuse keeping the working set wired) only appears after
# the LRU warms over a few thousand tokens, so a short gen measures the COLD
# all-miss regime and unfairly penalises dyn.  Hence long single-frontier gen
# below, and the `perfab` mode reports the warm STEADY-STATE (last-third) t/s.
CTX_START="${CTX_START:-8192}"
CTX_MAX="${CTX_MAX:-8192}"        # single frontier by default (sweep loops its own)
GEN_TOKENS="${GEN_TOKENS:-3072}"  # long enough to warm the LRU; avg ~ steady-state
CHAT_MAX_TOKENS="${CHAT_MAX_TOKENS:-4096}"
PERF_MAX_TOKENS="${PERF_MAX_TOKENS:-6144}"  # perfab: long continuous gen to warm + measure
PERF_PROMPT="${PERF_PROMPT:-Write an extremely detailed, exhaustive technical deep-dive (target 5000+ words) on how modern Mixture-of-Experts LLM inference works end to end: tokenization, attention and KV cache, expert routing, expert FFNs, quantization formats, and the memory-bandwidth tradeoffs of decode. Use many sections with long prose. Do not summarize, do not stop early, keep going in depth.}"
WORKDIR="$(pwd -P)"   # logs land in the cwd (run from a session dir)

# Steady-state t/s from a ds4-server log: overall avg, plus median & mean of the
# LAST THIRD of per-50-token chunks (excludes the cold LRU-warmup ramp).
steady_state_tps() {
    local log="$1"
    grep -oE "avg=[0-9.]+ t/s" "$log" 2>/dev/null | tail -1 | sed 's/^/  overall /'
    grep -oE "chunk=[0-9.]+ t/s" "$log" 2>/dev/null | grep -oE "[0-9.]+" \
      | awk '{a[NR]=$1} END{ if(NR==0){print "  steady: (no chunks)"; exit}
              s=int(NR*2/3)+1; n=0; for(i=s;i<=NR;i++){c[n++]=a[i]; sum+=a[i]}
              for(i=0;i<n;i++)for(j=i+1;j<n;j++)if(c[j]<c[i]){t=c[i];c[i]=c[j];c[j]=t}
              med=(n%2)?c[int(n/2)]:(c[n/2-1]+c[n/2])/2
              printf "  steady (last %d/%d chunks): median=%.2f t/s mean=%.2f t/s\n", n, NR, med, sum/n }'
}

echo "[$TAG] ds4=$DS4_ABS"
echo "[$TAG] model=$MODEL_ABS"
echo "[$TAG] budget=${BUDGET_MIB} MiB  prefill='${PREFILL}'  workdir=$WORKDIR"

# ---- ds4-bench A/B: cpu gen vs dyn gen --------------------------------------
# NOTE: ds4-bench (unlike ds4-server) does NOT chdir to its binary dir, and it
# loads the Metal shader sources (metal/*.metal) + gguf relative to the CURRENT
# directory.  So every ds4-bench invocation must run with cwd=$DS4_ABS (else
# "Metal source metal/flash_attn.metal not found; metal backend unavailable").
# We use a subshell `( cd "$DS4_ABS" && ./ds4-bench ... )` and tee to absolute
# log paths so the logs still land in $WORKDIR.
run_bench_pair() {
    local cpu_args="--cpu-moe ${PREFILL}"
    local dyn_args="--cpu-moe ${PREFILL} --routed-metal-dynamic"
    echo "[$TAG] bench cpu-gen (cwd=$DS4_ABS) ..."
    ( cd "$DS4_ABS" && ./ds4-bench -m "$MODEL_REL" $cpu_args \
        --chat-prompt-file "$PROMPT_REL" \
        --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --gen-tokens "$GEN_TOKENS" ) \
        2>&1 | tee "$WORKDIR/${TAG}-bench-cpu.log"
    local dyn_log="$WORKDIR/${TAG}-bench-dyn-b${BUDGET_MIB}.log"
    echo "[$TAG] bench dyn-gen (budget=${BUDGET_MIB}, extra='${EXTRA_ENV}', cwd=$DS4_ABS) ..."
    ( cd "$DS4_ABS" && env DS4_ROUTED_METAL_BUDGET_MIB="$BUDGET_MIB" ${EXTRA_ENV} ./ds4-bench -m "$MODEL_REL" $dyn_args \
        --chat-prompt-file "$PROMPT_REL" \
        --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --gen-tokens "$GEN_TOKENS" ) \
        2>&1 | tee "$dyn_log"
    echo "[$TAG] ---- stats (dyn, budget=${BUDGET_MIB}) ----"
    grep -E "routed-metal-dynamic (enabled|stats)" "$dyn_log" || \
        echo "[$TAG] WARN: no routed-metal-dynamic line in dyn bench log (fell back to CPU?)"
}

# ---- run.sh <client> (dyn) -------------------------------------------------
# client = opencode (real agent workload) or chat (single long streaming gen).
# ds4-server chdir's to its own dir, so (unlike ds4-bench) cwd does not matter
# for Metal source / gguf resolution here.
run_client() {
    local client="$1" prefix="$2" args="$3" envp="$4"
    echo "[$TAG] run.sh ${client} [${args}] env[${envp}] ..."
    CLIENT="$client" CHAT_MAX_TOKENS="$CHAT_MAX_TOKENS" MODEL="$MODEL_ABS" WORKDIR="$WORKDIR" \
        bash "$RUN_SH" "$prefix" "$args" "$envp"
    echo "[$TAG] ---- enabled / finish / stats (needs SIGHUP-aware ds4-server) ----"
    grep -E "routed-metal-dynamic (enabled|stats)|finish=" "$WORKDIR/${prefix}-ds4-server.log" | tail -8 || true
    grep -E "decoding chunk=" "$WORKDIR/${prefix}-ds4-server.log" | tail -3 || true
    echo "[$TAG] ---- memory pressure (budget=${BUDGET_MIB}) ----"
    local vmlog="$WORKDIR/${prefix}-vm_stat.log"
    echo "[$TAG]   SSD reads (pageins): $(ssd_reads_gib "$vmlog") GiB" \
         "| clean evicts (pageouts): $(ssd_pageouts_gib "$vmlog") GiB" \
         "| REAL swap-out: $(swapouts_mib "$vmlog") MiB"
    echo "[$TAG]   swap-out MUST be ~0 (>0 = past the safe cliff). High pageins is the"
    echo "[$TAG]     SSD floor (model>>RAM), roughly budget-independent on Q4."
}

# ---- perfab: long-gen chat A/B (cpu vs dyn), warm STEADY-STATE t/s ----------
# The honest perf comparison for a warm-cache-dependent feature: generate a
# long continuous response (PERF_MAX_TOKENS) so the LRU residency warms, then
# compare the last-third (steady-state) chunk t/s of cpu vs dyn.  Same prompt,
# same prefill; only the gen path differs.
run_perfab() {
    local cpu_args="--cpu-moe ${PREFILL}"
    local dyn_args="--cpu-moe ${PREFILL} --routed-metal-dynamic"
    echo "[$TAG] perfab cpu (long gen ${PERF_MAX_TOKENS} tok) ..."
    CLIENT=chat CHAT_MAX_TOKENS="$PERF_MAX_TOKENS" PROMPT="$PERF_PROMPT" \
        MODEL="$MODEL_ABS" WORKDIR="$WORKDIR" \
        bash "$RUN_SH" ${TAG}-perf-cpu "$cpu_args" ''
    local dyn_prefix="${TAG}-perf-dyn-b${BUDGET_MIB}"
    echo "[$TAG] perfab dyn (long gen ${PERF_MAX_TOKENS} tok, budget=${BUDGET_MIB}) ..."
    CLIENT=chat CHAT_MAX_TOKENS="$PERF_MAX_TOKENS" PROMPT="$PERF_PROMPT" \
        MODEL="$MODEL_ABS" WORKDIR="$WORKDIR" \
        bash "$RUN_SH" "$dyn_prefix" "$dyn_args" "$(dyn_envp "$BUDGET_MIB")"
    echo "[$TAG] ================= STEADY-STATE t/s (cold warmup excluded), budget=${BUDGET_MIB} ====="
    echo "[$TAG] CPU gen:"; steady_state_tps "$WORKDIR/${TAG}-perf-cpu-ds4-server.log"
    echo "[$TAG] DYN gen (budget ${BUDGET_MIB}):"; steady_state_tps "$WORKDIR/${dyn_prefix}-ds4-server.log"
    grep -E "routed-metal-dynamic (enabled|stats)" "$WORKDIR/${dyn_prefix}-ds4-server.log" || true
}

# ---- budget sweep: ONE cpu baseline + dyn across budgets, summary table -----
# CPU gen t/s is budget-independent, so we measure it once; dyn t/s depends on
# budget via the LRU hit rate.  Each run is a FRESH ds4-bench process (fresh
# cold LRU) at a single frontier with long gen (GEN_TOKENS, default 3072) so the
# cache warms within the run.  This is the budget-selection + rough cpu-vs-dyn
# tool; confirm the winner with `perfab` (clean steady-state) at that budget.
# Budgets (MiB) overridable via SWEEP_BUDGETS="16384 24576 ...".
bench_gen_tps()  { grep -E "^[0-9]+,[0-9]+," "$1" 2>/dev/null | tail -1 | cut -d, -f5; }
bench_hit_pct()  { grep "routed-metal-dynamic stats" "$1" 2>/dev/null | grep -oE "hit%=[0-9.]+" | head -1; }
bench_wired_pk() { grep "routed-metal-dynamic stats" "$1" 2>/dev/null | grep -oE "peak=[0-9.]+ GiB" | head -1; }
# macOS vm_stat probes (empty -> 0 on non-mac so the script still parses).
vm_swapouts()      { vm_stat 2>/dev/null | awk -F: '/Swapouts/{gsub(/[ .]/,"",$2); print $2+0; f=1} END{if(!f)print 0}'; }
vm_compressor_mb() { vm_stat 2>/dev/null | awk -F: '/compressor/{gsub(/[ .]/,"",$2); print int(($2+0)*16384/1048576); f=1} END{if(!f)print 0}'; }
vm_pageins()       { vm_stat 2>/dev/null | awk -F: '/Pageins/{gsub(/[ .]/,"",$2); print $2+0; f=1} END{if(!f)print 0}'; }
# Sum a per-interval column over a `vm_stat <interval>` log (launch_tmux runs
# `vm_stat 5` via a pty).  CRITICAL: vm_stat reprints the header AND a
# cumulative-since-boot summary row every ~20 samples; those rows carry K/M
# suffixes (e.g. faults=897824K) and must be EXCLUDED or they get mis-summed as
# huge deltas (this bug made the script report phantom SSD reads / swapouts).
# So: strip CR (pty artifact), keep numeric data rows WITHOUT any K/M suffix
# (= real per-interval deltas), sum the wanted column. 16 KiB pages.
# col 19=pageins(SSD reads), 20=pageout, 22=swapouts.
ssd_reads_gib()    { tr -d '\r' < "$1" 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]/ && $0!~/K/ && $0!~/M/ {pi+=$19} END{printf "%.2f", pi*16384/1073741824}'; }
ssd_pageouts_gib() { tr -d '\r' < "$1" 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]/ && $0!~/K/ && $0!~/M/ {po+=$20} END{printf "%.2f", po*16384/1073741824}'; }
# Real swap during the run (0 = no swap).  Same K/M-exclusion as above.
swapouts_mib()     { tr -d '\r' < "$1" 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]/ && $0!~/K/ && $0!~/M/ {s+=$22} END{printf "%.0f", s*16384/1048576}'; }
run_sweep() {
    local sctx="${CTX_START}"
    local cpu_log="$WORKDIR/${TAG}-sweep-cpu.log"
    echo "[$TAG] sweep: CPU baseline (ctx=${sctx}, gen=${GEN_TOKENS}) ..."
    ( cd "$DS4_ABS" && ./ds4-bench -m "$MODEL_REL" --cpu-moe ${PREFILL} \
        --chat-prompt-file "$PROMPT_REL" \
        --ctx-start "$sctx" --ctx-max "$sctx" --gen-tokens "$GEN_TOKENS" ) \
        2>&1 | tee "$cpu_log" >/dev/null
    local cpu_tps; cpu_tps="$(bench_gen_tps "$cpu_log")"
    local budgets="${SWEEP_BUDGETS:-16384 24576 32768 40960}"
    local sumf="$WORKDIR/.${TAG}-sweep-summary.$$"
    : > "$sumf"
    for b in $budgets; do
        echo "[$TAG] sweep: dyn budget=${b} MiB (ctx=${sctx}, gen=${GEN_TOKENS}) ..."
        local so0; so0="$(vm_swapouts)"; local pi0; pi0="$(vm_pageins)"
        ( cd "$DS4_ABS" && env DS4_ROUTED_METAL_BUDGET_MIB="$b" ${EXTRA_ENV} ./ds4-bench -m "$MODEL_REL" \
            --cpu-moe ${PREFILL} --routed-metal-dynamic \
            --chat-prompt-file "$PROMPT_REL" \
            --ctx-start "$sctx" --ctx-max "$sctx" --gen-tokens "$GEN_TOKENS" ) \
            2>&1 | tee "$WORKDIR/${TAG}-sweep-b${b}.log" >/dev/null
        local so1; so1="$(vm_swapouts)"; local comp; comp="$(vm_compressor_mb)"; local pi1; pi1="$(vm_pageins)"
        local lg="$WORKDIR/${TAG}-sweep-b${b}.log"
        local ssd_gib; ssd_gib="$(awk -v p="$(( pi1 - pi0 ))" 'BEGIN{printf "%.1f", p*16384/1073741824}')"
        printf "dyn:%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$b" \
            "$(bench_gen_tps "$lg")" \
            "$(bench_hit_pct "$lg" | sed 's/hit%=//')" \
            "$(bench_wired_pk "$lg" | sed 's/peak=//')" \
            "$(( so1 - so0 ))" "$comp" "$ssd_gib" >> "$sumf"
    done
    echo "[$TAG] ================= SWEEP SUMMARY (ctx=${sctx}, gen=${GEN_TOKENS}) ================="
    printf "[$TAG] %-12s %-9s %-7s %-13s %-11s %-9s %-10s\n" "config" "gen_tps" "hit%" "wired_peak" "swapout_d" "compr_MB" "ssd_rd_GiB"
    printf "[$TAG] %-12s %-9s %-7s %-13s %-11s %-9s %-10s\n" "cpu" "${cpu_tps:-NA}" "-" "-" "-" "-" "-"
    while IFS=$'\t' read -r cfg tps hit wired swp comp ssd; do
        printf "[$TAG] %-12s %-9s %-7s %-13s %-11s %-9s %-10s\n" "$cfg" "$tps" "$hit" "$wired" "$swp" "$comp" "$ssd"
    done < "$sumf"
    rm -f "$sumf"
    echo "[$TAG] swapout_d (delta during run) MUST be 0 — any >0 = dirty pages went to"
    echo "[$TAG]   disk = past the cliff, that budget is unsafe.  compr_MB rising across"
    echo "[$TAG]   budgets = growing pressure.  Pick the highest gen_tps with swapout_d=0,"
    echo "[$TAG]   then confirm: BUDGET_MIB=<best> bash ../scripts/q4-experiment.sh perfab"
    echo "[$TAG] NOTE: this bench (ctx=${sctx}, single gen) is LIGHT on anon RAM; real"
    echo "[$TAG]   long-ctx/multi-turn (opencode) uses much more -> validate the chosen"
    echo "[$TAG]   budget under 'opencode' with poll before trusting it for real use."
}

case "$MODE" in
    opencode) run_client opencode "${TAG}-dyn-oc-b${BUDGET_MIB}" \
                "--cpu-moe ${PREFILL} --routed-metal-dynamic" "$(dyn_envp "$BUDGET_MIB")" ;;
    oc-cpu-moe) run_client opencode "${TAG}-dyn-oc-b${BUDGET_MIB}" \
                "--cpu-moe --routed-metal-dynamic" "$(dyn_envp "$BUDGET_MIB")" ;;
    oc-mtp)  run_client opencode "${TAG}-dyn-oc-mtp-b${BUDGET_MIB}" \
                "--cpu-moe ${PREFILL} --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 --routed-metal-dynamic" "$(dyn_envp "$BUDGET_MIB")" ;;
    oc-cpu)   run_client opencode "${TAG}-cpu-oc" "--cpu-moe ${PREFILL}" "${EXTRA_ENV}" ;;
    oc-metal)   run_client opencode "${TAG}-metal" "--ctx 100000" "${EXTRA_ENV}" ;;
    chat|smoke) run_client chat "${TAG}-dyn-b${BUDGET_MIB}" \
                "--cpu-moe ${PREFILL} --routed-metal-dynamic" "$(dyn_envp "$BUDGET_MIB")" ;;
    perfab) run_perfab ;;
    bench) run_bench_pair ;;
    sweep) run_sweep ;;
    all)   run_sweep; run_perfab ;;
    *) echo "[$TAG] unknown mode: $MODE (use opencode|oc-cpu|chat|perfab|bench|sweep|all)" >&2; exit 2 ;;
esac

echo "[$TAG] done ($MODE). logs in $WORKDIR"
