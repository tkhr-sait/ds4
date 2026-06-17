# DS4 SSD-streaming benchmark results — 3 strategies

Measured: 2026-06-17 / Hardware: Apple M4 Max 128 GiB / Model: deepseek-v4-flash
Source logs: `/workspace/logs/ssd-streaming-brushup/` (plots in `plots/*.svg`)

> This document is meant to be read top to bottom. Start with **§1 Conclusion** and **§2 Overall results** to grasp "which is how fast / how memory-efficient", nail down the premises in **§3 Measurement conditions** and **§4 How memory works**, see the implementations in **§5 Strategy details**, then move on to detailed values, analysis, and selection guidance in **§6–§8**.

---

## Table of contents

- [1. Conclusion summary (TL;DR)](#1-conclusion-summary-tldr)
- [2. Overall results — full vs 3 strategies](#2-overall-results--full-vs-3-strategies)
  - [2a. q2 (full feasible)](#2a-q2-full-feasible)
  - [2b. q4 (full infeasible → streaming required)](#2b-q4-full-infeasible--streaming-required)
- [3. Measurement conditions](#3-measurement-conditions)
- [4. How memory works](#4-how-memory-works)
  - [4a. Two-tier model (fixed base + routed expert cache)](#4a-two-tier-model-fixed-base--routed-expert-cache)
  - [4b. file-backed (mmap) vs owned-buffer (nommap)](#4b-file-backed-mmap-vs-owned-buffer-nommap)
  - [4c. Machine RAM-size dependence](#4c-machine-ram-size-dependence-128-gib-assumed)
- [5. Strategy details](#5-strategy-details)
  - [5a. baseline](#5a-baseline-no-env--commit-cafc134)
  - [5b. residency-lru](#5b-residency-lru-commit-11f20b1)
  - [5c. nommap](#5c-nommap-commit-cae5e95)
  - [5d. full (non-streaming)](#5d-full-non-streaming)
- [6. Detailed value tables](#6-detailed-value-tables)
  - [6a. Prefill throughput](#6a-prefill-throughput-ts-final-avg-at-1000)
  - [6b. Generation throughput (with gen tok)](#6b-generation-throughput-with-gen-tok)
  - [6c. Memory usage (PhysMem peak)](#6c-memory-usage-physmem-peak)
- [7. Detailed memory analysis](#7-detailed-memory-analysis-vm_stat)
- [8. System metrics (mactop)](#8-system-metrics-mactop)
  - [8a. q4 b80 all-strategy charts](#8a-q4-b80-all-strategy-charts)
  - [8b. Items with large strategy differences](#8b-items-with-large-strategy-differences-all-q4-b80)

---

## 1. Conclusion summary (TL;DR)

- **For q2 (actual weights 80.76 GiB, fits in 128 GiB), `full` (non-streaming = whole model resident) is fastest.** Without the streaming overhead (per-layer cache / residency management / pread), it hits gen **22.80 t/s** and prefill 261 t/s. **If it fits, not streaming is best.**
- gen degradation of the 3 streaming strategies vs full (gen 22.80): **at b32, baseline −15% / residency-lru −27% / nommap −47%**; **at b80, baseline −15% / residency-lru −25% / nommap −13%**. nommap closes the gap sharply as budget grows, and **at q2 b80 its gen slightly overtakes baseline**.
- **For q4 (actual weights 153.33 GiB) the model does not fit in 128 GiB RAM → full is infeasible → streaming is required.** At b80 **residency-lru is best** (prefill +16, gen +0.81 vs baseline).
- **The axis is "does it fit in the box".** baseline / residency-lru are fast because they keep the model in the OS page cache, but that speed **assumes 128 GiB** (q2 fits all 80.76 GiB; q4 caps at ~116 GiB of its 153.33 GiB in RAM and the rest does evict↔re-pagein). `nommap` does not use the page cache (file-backed ≈2 GiB), is **nearly independent of RAM size**, and has the best portability to smaller machines.

**Positioning of the 3 strategies**

- **baseline (commit `cafc134`) is fast enough with no tuning** (the default baseline). There are really only 2 areas to improve — ① **prefill speed** (the region where baseline degrades at high budget), and ② **memory management when handling an oversized model (e.g. q4, larger than RAM)**.
- **residency-lru (commit `11f20b1`)** … targets ①. It lifts prefill at high budget (+15 to +31). The cost is that q2 gen is somewhat below baseline.
- **nommap (commit `cae5e95`)** … targets ②. By eliminating mmap and not using the OS page cache, it has the smallest RAM footprint and the most robust OOM avoidance, with **stable behavior independent of RAM size**. The cost is that **without the OS cache's help it is slow at low–mid budget**.

**buffer / copy cheat sheet** — how each of the 3 strategies supplies routed experts. **Only residency-lru's gen path is zero-copy (wrapping mmap pages directly in an MTLBuffer with no memcpy)**; the others copy via pread.

| Path | Buffer allocation | memcpy | backing | residency management |
|---|---|---|---|---|
| baseline | streaming expert cache: on cache miss, `newBufferWithLength` + **copy via pread** (mlock slab). The model is mmap'd as the pread source | **yes** | owned buffer (expert) + mmap (page cache = pread source) | **decaying route-hotness + clock eviction** (an evolved LRU, budget-bound) |
| residency-lru **gen** | only the selected experts, per-expert **noCopy view** (wraps mmap directly) | **no** | mmap (page cache) | **per-expert residency LRU explicitly bound** by byte budget |
| residency-lru **prefill** | allocate a pread cache slab with `newBufferWithLength` + **copy via F_NOCACHE pread** | **yes** | owned buffer (no page-cache pollution) | (1+prepare_ahead)-layer double-buffer + DONTNEED |
| nommap | owned buffer via `newBufferWithLength` + **copy via F_NOCACHE pread** | **yes** | owned buffer (**no mmap, no page cache**) | whole-tensor double-buffer (prefetch overlap) |

- **Both baseline and residency-lru share "a dedicated cache + an evolved eviction".** The difference: baseline **copies** experts via pread onto an mlock slab and evicts with **decaying-hotness/clock**, whereas residency-lru-gen wires experts as a **zero-copy mmap view** under a **per-expert LRU** (no copy, no slab).
- **Only nommap uses no mmap at all**, copying all weights into owned buffers via pread, so file-backed memory does not grow (the biggest differentiator, §7).

**Practical guidance for a 128 GiB machine, q2**

- The pure fastest is `full` (gen 22.80). But the whole model is wired, startup takes ~28s, and it competes with other processes for RAM.
- **If you want a large ctx and/or to run an IDE etc. at the same time, `baseline` at b32 is the practical sweet spot**: prefill **249** / gen **19.3** — **faster than b80 (prefill 214) with no prefill degradation**, and it fits in used 124 GiB leaving RAM headroom. Note that raising budget to b64/b80 makes the cache slab eat into the file-backed page cache (82→63 GiB) and prefill degrades from re-pagein, so **the trick is not to raise it too far**.
- **`nommap` delivers performance exactly per the budget you set** (the actual wired RAM you specify takes effect directly). It does not depend on file-backed memory, so behavior is stable and RAM-size independent. gen at low budget is weak but robust.

Implementations are in §5, the numerical evidence in §6/§8, and RAM dependence in §4c.

---

## 2. Overall results — full vs 3 strategies

Non-streaming `full` (whole model resident) is feasible **only for q2, which fits in RAM**. q4's actual weights are **153.33 GiB**, far exceeding 128 GiB RAM, so it cannot be made resident — full is N/A and streaming is required.

> Actual model weights (`ds4 --inspect` file size): **q2 = 80.76 GiB / q4 = 153.33 GiB** (matching the measured mmap span of 82697 MiB). The only difference is the quantization of the routed experts (q2: iq2_xxs 44.34 + q2_k 28.22 = 72.6 GiB → q4: q4_k 145.12 GiB; the non-routed part is f16 2.04 + q8_0 6.15 GiB, common to both). The file-backed peak "q4 ≈116 GiB" mentioned later (§7c) is **not** the model size — it is **the page-cache cap of what actually fit in RAM** (the 153.33 GiB does not all fit).

The gen token count (`gen tok`) = the **final gen token count of the longest gen sequence** for each run. walltime is wall-clock time of the agentic session; since the amount generated varies per run it is not a strict apples-to-apples comparison (the true speed signal is prefill/gen t/s).

### 2a. q2 (full feasible)

The streaming strategies list b32 / b80 side by side; full is budget-independent (one row).

| run (q2) | prefill t/s | gen t/s | gen tok | walltime | used / unused / compressor |
|---|---|---|---|---|---|
| **full (non-stream)** | **261.0** | **22.80** | 2438 | **2m43s** | 103 / 24.0 / 4.2G |
| baseline (b32) | 249.3 | 19.29 | 2112 | 3m10s | 124 / 2.9 / 0.9G |
| baseline (b80) | 214.2 | 19.37 | 2459 | 4m02s | 127 / 0.1 / 5.0G |
| residency-lru (b32) | 243.9 | 16.58 | 2150 | 3m40s | 103 / 24.0 / 0.0G |
| residency-lru (b80) | 245.0 | 17.13 | 1844 | 3m11s | 103 / 24.0 / 0.0G |
| nommap (b32) | 214.6 | 12.16 | 1728 | 3m51s | 61 / 67.0 / 0.0G |
| nommap (b80) | 215.2 | 19.92 | 2187 | 3m32s | 100 / 27.0 / 0.0G |

- **For q2, `full` is fastest on every metric.** Against gen **22.80**, the gen degradation vs full is **b32: baseline −15% / residency-lru −27% / nommap −47%**; **b80: baseline −15% / residency-lru −25% / nommap −13%**. For prefill vs full 261: b32 baseline −4.5% / residency-lru −6.6% / nommap −17.8%.
- **nommap improves dramatically at b80**: gen closes from b32's 12.16 → b80's 19.92 (vs full, −47% → −13%), overtaking baseline. More budget means more cache hits, approaching full.
- The ~28s startup residency build is a one-time cost not counted in walltime.

![q2 full throughput](plots/q2-full-opencode-tok.svg)
![q2 baseline b32 throughput](plots/q2-baseline-opencode-b32768-tok.svg)

### 2b. q4 (full infeasible → streaming required)

full does not fit, so it is N/A. b32 / b80 listed side by side.

| run (q4) | prefill t/s | gen t/s | gen tok | walltime | used / unused / compressor |
|---|---|---|---|---|---|
| full (non-stream) | — | — | — | — | **does not fit (weights 153.33 GiB > RAM 128, OOM)** |
| baseline (b32) | 120.6 | 8.98 | 2328 | 6m58s | 127 / 0.1 / 4.9G |
| baseline (b80) | 113.2 | 11.00 | 1767 | 5m25s | 127 / 0.1 / 5.3G |
| residency-lru (b32) | **136.0** | 9.62 | 2872 | 7m38s | 127 / 0.1 / 5.0G |
| residency-lru (b80) | **129.2** | **11.81** | 2644 | 6m15s | 127 / 0.1 / 5.0G |
| nommap (b32) | 117.5 | 3.60 | 2601 | 15m20s | 63 / 64.0 / 0.0G |
| nommap (b80) | 126.7 | 7.59 | 2598 | 7m22s | 106 / 21.0 / 0.0G |

- **q4 makes full infeasible → streaming required.** **On t/s, residency-lru is best** (b32 prefill +15.4, gen +0.64; b80 prefill +16.0, gen +0.81 vs baseline). nommap's prefill is close, but its gen is weak — especially b32 gen 3.60 (−5.4 vs baseline 8.98) is rough.
- **walltime caveat**: t/s and walltime do not rank the same. For example baseline b80 looks short at 5m25s, but that is **due to how much was generated** (baseline 1767 tok vs residency-lru 2644 tok). nommap b32's 15m20s is also the product of slow gen 3.60 t/s × 2601 tokens generated. The gen tok column confirms this (the true speed signal is t/s).

![q4 baseline b80 throughput](plots/q4-baseline-opencode-b81920-tok.svg)
![q4 residency-lru b80 throughput](plots/q4-residency-lru-opencode-b81920-tok.svg)

---

## 3. Measurement conditions

- **Common launch parameters**: all 3 strategies launch with `--ssd-streaming --ssd-streaming-cache-experts ${BUDGET}GB`, so SSD streaming (skipping full model residency/warmup) is common. The only thing switched by env is **the expert residency implementation**.
- **3 strategies** (switched by env in `run.sh`):
  - `baseline` … no env = **the default SSD-streaming residency path** (both mmap zero-copy and no-mmap disabled)
  - `residency-lru` … `DS4_METAL_ENABLE_STREAMING_RESIDENCY_LRU=1` (selected-expert zero-copy mmap view + residency LRU + dynamic prefill/gen budget switching)
  - `nommap` … `DS4_METAL_ENABLE_STREAMING_NO_MMAP=1` (wire from a no-cache fd without mmap)
  - `full` … the non-streaming baseline that **does not use `--ssd-streaming`**. The whole model is resident (q2 80.76 GiB mmap, ~28s residency build at startup). Measured for q2 only, budget-independent (it has no streaming cache).
- **2 quants**: q2 (q2-pro IQ2XXS+w2Q2K ≈6.75 MiB/expert) / q4 (Q4K ≈13.50 MiB/expert). attn/shared/output are Q8 in both quants.
- **5 budgets** (`--ssd-streaming-cache-experts`, GB): 8 / 16 / 32 / **64** / 80
- **Workload**: an agentic opencode session that generates minesweeper.html (same prompt).
- **Representative values**: **prefill t/s** = the final cumulative avg at the **100.0% point** of the largest prefill (TOOLS 10,656 tok); **gen t/s** = the final avg of the longest gen sequence; **gen tok** = the final token count of that longest gen sequence.
- **Memory**: PhysMem peaks (used/wired/compressor) and minimum unused from `top -l`, plus cumulative/instantaneous values from `vm_stat`.
- **Commits**: baseline `cafc134` / nommap `cae5e95` / residency-lru `11f20b1`

> Implementation background: includes nommap prefill overlap ([[project_ds4_nommap_prefill_overlap]]) + the nommap file-backed leak fix ([[project_ds4_nommap_filebacked_leak]]) + residency-lru dynamic budget switching ([[project_ds4_zerocopy_budget_switch]]).

---

## 4. How memory works

The 3 strategies differ only in **how routed experts are supplied from SSD**; the fixed base (non-routed) and the KV are common. To understand why memory behavior changes so much by strategy, first nail down two viewpoints.

### 4a. Two-tier model (fixed base + routed expert cache)

DSV4-Flash is two-tier: a **fixed base (non-routed)** + **streamed routed experts**. Only the routed experts change with quantization; the fixed parts such as attn are **exactly identical across q2/q4** (as the model name `Q8Attn-Q8Shared-Q8Out` shows, attn/shared/output are both Q8).

**Fixed allocation (common to q2 = q4, budget-independent)**

| Item | Size | Contents / quantization |
|---|---|---|
| non-routed views | **≈8.2 GiB** | attn proj(Q8) + shared expert(Q8) + output(Q8) + each norm + MLA HC/Compressor/Indexer(F16) |
| token embedding | **≈1.0 GiB** | 1 span mapped first at startup |
| context / KV buffers | **≈1.9 GiB** | ctx=100000, raw_kv 4352 + compressed_kv 25002 rows (startup log `context buffers 1933.21 MiB`) |
| **fixed total** | **≈11.1 GiB** | **q2 = q4** |

Even though SSD streaming skips full residency, this fixed part (used by every layer on every token) stays resident.

**Variable part (routed experts = what fits in the cache budget)**

DSV4-Flash has **43 layers × 256 experts/layer = 11,008 routed experts** in total, with **6 activated per token** (`ds4 --inspect`: experts count=256 used=6, layers=43). Only this expert part changes with quantization; the fixed base (f16 2.04 + q8_0 6.15 = 8.2 GiB) is common to q2/q4. Expert weights: q2 72.6 GiB (iq2_xxs 44.34 + q2_k 28.22) → q4 145.1 GiB (q4_k).

| | q2 | q4 | ratio |
|---|---|---|---|
| per-expert | 6.75 MiB (IQ2XXS+w2Q2K) | 13.50 MiB (Q4K) | ×2 |
| resident experts at 8 GiB budget | 1213 | 606 | ×2 |
| resident experts at 16 GiB budget | 2427 | 1213 | ×2 |
| prefill pread cache (512-expert reservation, residency-lru only) | 3.38 GiB | 6.75 GiB | ×2 |

(from startup log `metal SSD streaming cache budget X GiB / Y MiB per expert = N experts`)

This is the root of the gen difference in §6b: since one q2 expert is half a q4 expert, q2 **keeps twice as many experts resident at the same budget**, so gen is fast even at low–mid budget. For q4, after subtracting the fixed ≈11.1 GiB the effective cache is thin, so gen does not improve unless budget is raised (q4 8GiB=8.9 → 80GiB=11.8 t/s).

### 4b. file-backed (mmap) vs owned-buffer (nommap)

- `baseline` / `residency-lru` **mmap the model**. Re-reading an expert becomes an OS **page-cache (file-backed) hit**, which is fast — but at the cost of the model sticking in file-backed memory (q2 all 80.76 GiB; q4 caps at ~116 GiB of its 153.33 GiB that fit in RAM, the rest does evict↔re-pagein).
- `nommap` **does not mmap; it preads from an F_NOCACHE fd** into owned buffers. It does not pollute the page cache (file-backed ≈2 GiB) so the RAM footprint is smallest, but it cannot benefit from cache hits and is slow at low budget.

Below is the vm_stat trace for the same q2 b32, shown as **3 columns: baseline / residency-lru / nommap**. baseline and residency-lru keep file-backed pinned at ~82 GiB, while nommap stays tiny (≈2 GiB).

| baseline | residency-lru | nommap |
|---|---|---|
| <img src="plots/q2-baseline-opencode-b32768-vmstat-mem.svg" width="100%"/> | <img src="plots/q2-residency-lru-opencode-b32768-vmstat-mem.svg" width="100%"/> | <img src="plots/q2-nommap-opencode-b32768-vmstat-mem.svg" width="100%"/> |

### 4c. Machine RAM-size dependence (128 GiB assumed)

- **baseline / residency-lru are fast because the model sits in the OS file-backed (page cache)** (q2 all 80.76 GiB; q4 keeps ~116 GiB of its 153.33 GiB in the page cache, so re-reading experts is a cache hit). **On machines with less than 128 GiB the page cache cannot hold the model, evict↔re-pagein becomes constant, and performance drops a lot** (the numbers here will not reproduce as-is).
- **`nommap` does not use file-backed memory** (F_NOCACHE pread, file-backed ≈2 GiB), so **performance is nearly independent of machine RAM size**. Only the `--ssd-streaming-cache-experts` budget (= actual wired RAM) matters, so even on a 64 GiB machine the behavior at a given budget reproduces directly. nommap has the best portability to tight-RAM / small machines.
- → The RAM-size axis: **for "boxes that fit", use baseline/residency-lru (or full for q2); for "boxes that don't fit / are small", use nommap**.

---

## 5. Strategy details

The strategies differ only in **how routed experts are supplied**. Below: implementation points and what changed from baseline.

### 5a. baseline (no env / commit `cafc134`)

Supplies routed experts via a dedicated **streaming expert cache**. The model is mmap'd as the source of both non-routed weights and expert preads.

- **Expert supply**: on cache miss → **copy via pread** into an owned Metal buffer from `newBufferWithLength` (mlock slab). Capacity bound by budget (`--ssd-streaming-cache-experts`).
- **eviction = evolved LRU**: not a plain LRU but **decaying route-hotness + clock** (the `route_hotness >>= 1` exponential decay keeps hot experts preferentially, [[project_ds4_adaptive_cache_plan]]). Both prefill and gen use this cache.
- **Pros**: general-purpose, fast at low budget, hard to OOM.
- **Cons**: the model that serves preads lives in the page cache (file-backed q2≈82 / q4≈116 GiB; for q4 that is the portion of the 153.33 GiB model that fit in RAM). At high budget the cache slab and the page cache fight over RAM, causing a file-backed evict↔re-pagein chain → compressor pressure and prefill degradation (§6/§7).

### 5b. residency-lru (commit `11f20b1`)

Note that **the mechanism differs between prefill and gen**. The residency LRU is **gen-only** (the feature name comes from this gen mechanism).

- **gen mechanism**: `ds4_gpu_routed_moe_one_tensor_residency_lru`. Wraps **only** the router-selected experts in a zero-copy mmap view and wires them into a **per-expert residency LRU** (byte-budget bound). The gen working set stays resident across tokens while RAM stays bound. requestResidency is deferred per-layer dispatch and flushed once per token (`residency_lru_commit_pending`). Non-routed is pinned once via a static-decode map.
- **prefill mechanism**: **does not use the residency LRU**. At prefill start it drops the LRU and preads the selected experts via a dedicated **(1+prepare_ahead)-layer double-buffer on an F_NOCACHE fd** (`set_model_nocache_fd`). Each layer's already-prefilled routed pages get **MADV_DONTNEED 2 layers behind the encode front**, keeping the sweep's file-backed footprint to ~2-3 layers. On LRU miss, cold ranges are async-prefetched with F_RDADVISE (`ds4_prefetch_expert_range`). The prefill cache slab is returned to the OS on gen resume (`ds4_gpu_stream_expert_cache_release`).
- **Dynamic budget switching**: prefill budget = total − the (1+prepare_ahead)-layer pread cache reservation (≈512 experts); gen budget = full. Switched by `enter_prefill`/`enter_gen`.

> **What changed from baseline / why it improves**: as in §6a, **the reason residency-lru prefill beats baseline is not the gen residency LRU, but the prefill-side F_NOCACHE pread double-buffer + MADV_DONTNEED trailing** that avoids polluting the page cache. baseline reads experts with preads on the regular fd, so the prefill sweep pulls the model's pread-source pages into the page cache (at high budget it competes with the cache slab and thrashes).

**Implementation notes**

- **Carrying a reduced budget into gen silently slows decode to a crawl.** `enter_prefill` evicts the LRU down to the reduced bound for the pread cache and then **immediately restores it to full budget right after**. If you put the restore in `enter_gen`, then when `enter_gen` fails to be called the budget is stranded at reduced → the gen LRU is capped and decode becomes extremely slow. Keep `enter_gen` as an idempotent backstop.
- **Drop the LRU at prefill start** and let gen rebuild a demand-true set (do not carry over stale prefill warmth).
- **Put MADV_DONTNEED 2 layers behind the encode front.** Too close re-faults; too far grows the footprint.
- A noCopy MTLBuffer is counted as wired for its full length (cf. [[feedback_ds4_metal_nocopy_double_counts]]). requestResidency must be batch-flushed per token, otherwise the per-command-buffer residency verification cost piles up.

**Evaluation (vs baseline) — a speed-oriented improvement** (positioning in §1)
- Pros: prefill substantially beats baseline at high budget (q2 64/80 +30.8 to +30.9; q4 32–80 +15.4 to +16.0). For q2 it bounds used to 103 GiB, leaving ~24 GiB unused.
- Cons: q2 gen is below baseline at all budgets (−0.7 to −3.0). q4 stays pinned at used≈127 GiB (the fate of mmap-based strategies); memory savings are mainly for q2. The most complex to implement. (As an exception, only q4 gen slightly beats baseline at mid–high budget: 32 +0.6 / 64 +2.5.)

### 5c. nommap (commit `cae5e95`)

**Does not mmap the model**; preads weights from a dedicated **F_NOCACHE fd** into Metal-owned Shared buffers (`ds4_gpu_stream_nommap_*`). The aim is to avoid page-cache buildup → compressor/OOM by not growing file-backed memory (the biggest differentiator from baseline).

- **prefill mechanism**: routed prefill overlap. `ds4_gpu_stream_nommap_routed_prefetch/_ensure` **prefetches the entire gate/up/down expert tensors of layer il+1 in the background** (per-parity double-buffer). It overlaps the F_NOCACHE SSD read with layer il's GEMM so the next stage does not stall ([[project_ds4_nommap_prefill_overlap]]).
- **gen mechanism**: supplies the routed mv-addr/GEMM path via F_NOCACHE pread (whole-tensor double-buffer). Supports Flash Q4/Q2.
- Non-routed uses a persist tier (one-shot owned buffer), preaded in bulk at startup.

**Implementation notes**

- **F_NOCACHE only works with page-aligned I/O.** `ds4_gpu_model_read_into` widens the request window to page boundaries and **reads in a single pread**. Interleaving unaligned short reads makes the kernel implicitly re-enable caching and disables F_NOCACHE (via a page-aligned bounce buffer, 32 MiB ceiling).
- **F_NOCACHE alone still grows file-backed memory.** The kernel's automatic read-ahead ignores F_NOCACHE and puts prefetched pages into file-backed, so **turn off F_RDAHEAD** on the model fd (`ds4_gpu_stream_expert_readahead_enabled` is false under no-mmap). This cuts the gen-time file-backed growth to about 1/3.
- **The prefill double-buffer must be freed explicitly.** Unless you call `ds4_gpu_stream_nommap_routed_release` at end-of-prompt to free the double-buffer, it stays wired during gen too.
- KV / staged reads via stdio also land in file-backed and need to be bypassed ([[project_ds4_nommap_filebacked_leak]]). What remains is reclaimable cache from the prefill cold-load burst (inherent to macOS, no way to drop the fd, diminishing returns). This run's file-backed ≈1.3–2.8 GiB / pageins ≈1 GiB is the result of these mitigations (§6/§7).

**Evaluation (vs baseline) — a memory-oriented improvement** (positioning in §1)
- Pros: smallest RAM footprint (file-backed ≈1.3–2.8 GiB, compressor exactly 0, up to 91 GiB unused). faults/pageins are also small, swapout 0, the most robust OOM avoidance — good for tight-memory boxes / large ctx. Performance is RAM-size independent and highly portable (§4c). At high budget gen catches up to baseline (q2 80GiB +0.55).
- Cons: gen is much slower at low–mid budget (q2 8GiB −10.4 / q4 8GiB −6.7). prefill is also below baseline at low–mid budget (q2 8–32GiB −34 to −42). Achieving speed needs high budget = more RAM.

### 5d. full (non-streaming)

Uses no streaming at all; **keeps the whole model resident via mmap + a residency build** (`q4-experiment.sh full`). It has no expert cache/LRU/pread; all experts are always resident. Feasible only when the model fits in RAM (q2 80.76 GiB ≤ 128 GiB), and **fastest because there is no streaming overhead** (§2). For q4 (weights 153.33 GiB, larger than RAM) it is infeasible, so streaming is required. A one-time ~28s residency build is paid at startup.

---

## 6. Detailed value tables

### 6a. Prefill throughput (t/s, final avg at 100.0%)

Relative to baseline (parentheses = Δ vs baseline).

**q2**

| budget(GiB) | baseline | residency-lru (Δ) | nommap (Δ) |
|---|---|---|---|
| 8 | 248.9 | 243.4 (−5.5) | 206.5 (−42.4) |
| 16 | 248.7 | 245.6 (−3.1) | 210.1 (−38.6) |
| 32 | 249.3 | 243.9 (−5.4) | 214.6 (−34.7) |
| 64 | 212.5 | 243.4 (+30.9) | 215.9 (+3.4) |
| 80 | 214.2 | 245.0 (+30.8) | 215.2 (+1.0) |

**q4**

| budget(GiB) | baseline | residency-lru (Δ) | nommap (Δ) |
|---|---|---|---|
| 8 | 131.2 | 131.6 (+0.4) | 117.1 (−14.1) |
| 16 | 128.8 | 131.9 (+3.1) | 113.9 (−14.8) |
| 32 | 120.6 | 136.0 (+15.4) | 117.5 (−3.1) |
| 64 | 117.6 | 133.0 (+15.5) | 123.5 (+5.9) |
| 80 | 113.2 | 129.2 (+16.0) | 126.7 (+13.4) |

### 6b. Generation throughput (with gen tok)

t/s = avg of the longest gen sequence; gen tok = the final token count of that sequence. Parentheses = Δ t/s vs baseline.

**q2**

| budget(GiB) | baseline (tok) | residency-lru (Δ, tok) | nommap (Δ, tok) |
|---|---|---|---|
| 8 | 14.95 (2799) | 14.25 (−0.70, 2151) | 4.52 (−10.43, 1728) |
| 16 | 17.12 (2496) | 15.88 (−1.24, 1811) | 6.54 (−10.58, 2008) |
| 32 | 19.29 (2112) | 16.58 (−2.71, 2150) | 12.16 (−7.13, 1728) |
| 64 | 20.27 (2548) | 17.31 (−2.96, 2264) | 19.57 (−0.70, 2172) |
| 80 | 19.37 (2459) | 17.13 (−2.24, 1844) | 19.92 (+0.55, 2187) |

**q4**

| budget(GiB) | baseline (tok) | residency-lru (Δ, tok) | nommap (Δ, tok) |
|---|---|---|---|
| 8 | 8.86 (2831) | 7.15 (−1.71, 2784) | 2.12 (−6.74, 2708) |
| 16 | 9.31 (2908) | 8.39 (−0.92, 2760) | 2.73 (−6.58, 2084) |
| 32 | 8.98 (2328) | 9.62 (+0.64, 2872) | 3.60 (−5.38, 2601) |
| 64 | 9.00 (2846) | 11.47 (+2.47, 1877) | 7.23 (−1.77, 2051) |
| 80 | 11.00 (1767) | 11.81 (+0.81, 2644) | 7.59 (−3.41, 2598) |

> gen tok varies per run with how much the agentic session generated (it changes even with the same prompt due to agent branching). t/s is the true speed signal; gen tok is an auxiliary value for interpreting walltime.

### 6c. Memory usage (PhysMem peak)

**q2**

| strategy | budget(GiB) | used(GiB) | wired(GiB) | compressor(GiB) | min unused(GiB) | swapout |
|---|---|---|---|---|---|---|
| baseline | 8 | 100 | 24 | 1.0 | 28.0 | 0 |
| baseline | 16 | 108 | 28 | 1.0 | 19.0 | 0 |
| baseline | 32 | 124 | 48 | 0.9 | 2.9 | 0 |
| baseline | 64 | 127 | 77 | 4.7 | 0.1 | 0 |
| baseline | 80 | 127 | 84 | 5.0 | 0.1 | 40 |
| residency-lru | 8 | 103 | 28 | 0.0 | 24.0 | 0 |
| residency-lru | 16 | 103 | 35 | 0.0 | 24.0 | 0 |
| residency-lru | 32 | 103 | 51 | 0.0 | 24.0 | 0 |
| residency-lru | 64 | 103 | 83 | 0.0 | 24.0 | 0 |
| residency-lru | 80 | 103 | 85 | 0.0 | 24.0 | 0 |
| nommap | 8 | 36 | 21 | 0.0 | 91.0 | 0 |
| nommap | 16 | 44 | 31 | 0.0 | 83.0 | 0 |
| nommap | 32 | 61 | 45 | 0.0 | 67.0 | 0 |
| nommap | 64 | 92 | 77 | 0.0 | 35.0 | 0 |
| nommap | 80 | 100 | 85 | 0.0 | 27.0 | 0 |

**q4**

| strategy | budget(GiB) | used(GiB) | wired(GiB) | compressor(GiB) | min unused(GiB) | swapout |
|---|---|---|---|---|---|---|
| baseline | 8 | 127 | 21 | 5.0 | 0.1 | 96 |
| baseline | 16 | 127 | 28 | 5.0 | 0.1 | 132 |
| baseline | 32 | 127 | 44 | 4.9 | 0.1 | 0 |
| baseline | 64 | 127 | 77 | 5.0 | 0.1 | 0 |
| baseline | 80 | 127 | 88 | 5.3 | 0.1 | 0 |
| residency-lru | 8 | 127 | 36 | 5.0 | 0.1 | 0 |
| residency-lru | 16 | 127 | 43 | 4.9 | 0.1 | 36 |
| residency-lru | 32 | 127 | 59 | 5.0 | 0.1 | 0 |
| residency-lru | 64 | 127 | 90 | 4.9 | 0.1 | 184 |
| residency-lru | 80 | 127 | 101 | 5.0 | 0.1 | 0 |
| nommap | 8 | 38 | 21 | 0.0 | 89.0 | 0 |
| nommap | 16 | 47 | 29 | 0.0 | 80.0 | 0 |
| nommap | 32 | 63 | 45 | 0.0 | 64.0 | 0 |
| nommap | 64 | 94 | 77 | 0.0 | 33.0 | 0 |
| nommap | 80 | 106 | 88 | 0.0 | 21.0 | 0 |

---

## 7. Detailed memory analysis (`vm_stat`)

> Instantaneous peaks are the max/min of each sample row (high confidence). Cumulative/event values sum the per-interval rows of vm_stat (excluding the since-boot row at session start). Gaps where vm_stat paused between phases are not counted, so these lean low. Use for trend comparison. page=16 KiB.

### 7a. q2 — instantaneous peak (GiB)

| strategy | budget | wired | **file-backed** | anonymous | compressed | inactive | min free |
|---|---|---|---|---|---|---|---|
| baseline | 8 | 20.3 | 82.2 | 6.6 | 3.2 | 81.1 | 26.7 |
| baseline | 16 | 31.6 | 82.7 | 6.6 | 3.1 | 81.2 | 18.1 |
| baseline | 32 | 44.5 | 82.6 | 7.1 | 3.0 | 81.3 | 1.8 |
| baseline | 64 | 76.5 | 63.8 | 6.5 | 8.8 | 53.3 | 0.1 |
| baseline | 80 | 84.4 | 63.1 | 7.5 | 8.2 | 52.8 | 0.1 |
| residency-lru | 8 | 28.3 | 82.2 | 12.0 | 0.0 | 87.0 | 23.3 |
| residency-lru | 16 | 35.4 | 82.0 | 12.2 | 0.0 | 87.0 | 23.6 |
| residency-lru | 32 | 51.4 | 82.0 | 12.1 | 0.0 | 87.0 | 23.5 |
| residency-lru | 64 | 83.0 | 80.2 | 12.1 | 0.0 | 85.2 | 23.1 |
| residency-lru | 80 | 85.5 | 79.9 | 12.2 | 0.0 | 85.1 | 23.1 |
| nommap | 8 | 20.9 | 2.8 | 23.9 | 0.0 | 10.4 | 89.7 |
| nommap | 16 | 28.8 | 2.3 | 24.3 | 0.0 | 9.7 | 81.6 |
| nommap | 32 | 44.3 | 2.1 | 24.5 | 0.0 | 9.4 | 65.6 |
| nommap | 64 | 76.3 | 1.5 | 24.4 | 0.0 | 10.1 | 34.8 |
| nommap | 80 | 84.3 | 1.9 | 26.4 | 0.0 | 10.5 | 26.1 |

### 7b. q2 — cumulative/events (whole run)

| strategy | budget | faults | pageins(GiB) | pageout | comprs | dcomprs | swapout(pages) |
|---|---|---|---|---|---|---|---|
| baseline | 8 | 1261k | 75.8 | 0 | 0k | 12k | 0 |
| baseline | 16 | 1719k | 76.4 | 0 | 0k | 7k | 0 |
| baseline | 32 | 2806k | 76.6 | 0 | 0k | 17k | 0 |
| baseline | 64 | 4914k | 286.4 | 263 | 882k | 494k | 0 |
| baseline | 80 | 5161k | 353.6 | 288 | 1118k | 1008k | 40 |
| residency-lru | 8 | 1755k | 76.1 | 0 | 0k | 0k | 0 |
| residency-lru | 16 | 1697k | 75.7 | 0 | 0k | 0k | 0 |
| residency-lru | 32 | 1722k | 76.1 | 0 | 0k | 0k | 0 |
| residency-lru | 64 | 1760k | 76.1 | 0 | 0k | 0k | 0 |
| residency-lru | 80 | 1711k | 76.1 | 0 | 0k | 0k | 0 |
| nommap | 8 | 2506k | 1.6 | 0 | 0k | 0k | 0 |
| nommap | 16 | 2829k | 1.3 | 0 | 0k | 0k | 0 |
| nommap | 32 | 3575k | 0.8 | 0 | 0k | 0k | 0 |
| nommap | 64 | 5514k | 0.8 | 0 | 0k | 0k | 0 |
| nommap | 80 | 6016k | 1.1 | 0 | 0k | 0k | 0 |

### 7c. q4 — instantaneous peak (GiB)

| strategy | budget | wired | **file-backed** | anonymous | compressed | inactive | min free |
|---|---|---|---|---|---|---|---|
| baseline | 8 | 20.4 | 115.8 | 3.5 | 9.5 | 106.1 | 0.1 |
| baseline | 16 | 29.5 | 115.9 | 3.9 | 8.6 | 100.7 | 0.0 |
| baseline | 32 | 45.6 | 115.9 | 3.8 | 9.3 | 86.8 | 0.1 |
| baseline | 64 | 76.5 | 115.9 | 3.6 | 9.4 | 85.3 | 0.1 |
| baseline | 80 | 87.5 | 115.9 | 3.6 | 9.4 | 87.3 | 0.1 |
| residency-lru | 8 | 36.3 | 110.5 | 9.4 | 8.6 | 112.4 | 0.0 |
| residency-lru | 16 | 44.4 | 113.4 | 5.7 | 8.6 | 109.0 | 0.0 |
| residency-lru | 32 | 58.9 | 113.5 | 5.8 | 8.6 | 109.1 | 0.0 |
| residency-lru | 64 | 90.3 | 113.4 | 5.8 | 8.5 | 109.0 | 0.0 |
| residency-lru | 80 | 101.2 | 113.3 | 5.7 | 8.5 | 108.8 | 0.0 |
| nommap | 8 | 20.5 | 1.7 | 26.8 | 0.0 | 10.1 | 87.7 |
| nommap | 16 | 28.6 | 2.4 | 27.2 | 0.0 | 10.6 | 78.9 |
| nommap | 32 | 44.6 | 1.8 | 27.0 | 0.0 | 10.6 | 63.4 |
| nommap | 64 | 76.4 | 1.3 | 27.1 | 0.0 | 10.0 | 32.0 |
| nommap | 80 | 87.6 | 1.7 | 27.2 | 0.0 | 10.6 | 20.4 |

### 7d. q4 — cumulative/events (whole run)

| strategy | budget | faults | pageins(GiB) | pageout | comprs | dcomprs | swapout(pages) |
|---|---|---|---|---|---|---|---|
| baseline | 8 | 1740k | 834.2 | 685 | 1041k | 717k | 96 |
| baseline | 16 | 2139k | 916.5 | 515 | 1144k | 798k | 132 |
| baseline | 32 | 3135k | 1065.0 | 561 | 1168k | 832k | 0 |
| baseline | 64 | 5337k | 1348.2 | 574 | 1248k | 910k | 0 |
| baseline | 80 | 5925k | 922.3 | 340 | 1345k | 996k | 0 |
| residency-lru | 8 | 2389k | 841.2 | 865 | 1082k | 522k | 0 |
| residency-lru | 16 | 2122k | 816.4 | 530 | 800k | 442k | 36 |
| residency-lru | 32 | 2110k | 824.6 | 579 | 811k | 458k | 0 |
| residency-lru | 64 | 1813k | 710.6 | 348 | 575k | 232k | 184 |
| residency-lru | 80 | 2036k | 776.9 | 475 | 590k | 240k | 0 |
| nommap | 8 | 3573k | 0.9 | 0 | 0k | 0k | 0 |
| nommap | 16 | 3825k | 1.5 | 0 | 0k | 0k | 0 |
| nommap | 32 | 4547k | 1.1 | 0 | 0k | 0k | 0 |
| nommap | 64 | 6042k | 0.7 | 0 | 0k | 0k | 0 |
| nommap | 80 | 6771k | 0.8 | 0 | 0k | 0k | 0 |

The vm_stat traces for q4 b80, shown as **3 columns: baseline / residency-lru / nommap** (top row = memory breakdown vmstat-mem, bottom row = events vmstat-events). The two mmap-based strategies keep file-backed pinned at ~113–116 GiB, while nommap stays at ~2 GiB (used is only ~budget).

| baseline | residency-lru | nommap |
|---|---|---|
| <img src="plots/q4-baseline-opencode-b81920-vmstat-mem.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-vmstat-mem.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-vmstat-mem.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-vmstat-events.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-vmstat-events.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-vmstat-events.svg" width="100%"/> |

### 7e. Memory observations

**file-backed splits baseline and the 2 improvements most sharply**
- `baseline` / `residency-lru` (mmap-based) keep file-backed pinned at **q2≈80–82 GiB / q4≈110–116 GiB** → the whole model is mmap'd. For q4 the model exceeds RAM, so the page cache fills to the cap. baseline q2 drops to 63 GiB file-backed at 64/80 GiB, but that is because it gets pushed out by wired (76–84 GiB) and compressor (8 GiB).
- `nommap` keeps file-backed tiny at **≈1.3–2.8 GiB** → it does not mmap the model and reads via no-cache fd preads, so the model never lands in the OS page cache (by design = the biggest differentiator from baseline).

**faults: nommap is the same order as mmap-based**
- nommap's cumulative faults are **3.5–6.8M** for q4. With the F_NOCACHE owned-view + prefill prefetch overlap, even without mmap the minor faults stay the same order as mmap-based (1–6M).
- `residency-lru`/`baseline` are **1.2–6M faults** for both q2/q4. Faults grow gradually with budget (more resident experts to touch).

**pageins (actual disk I/O): high for mmap-based / tiny for nommap**
- `residency-lru`/`baseline` do **660–1350 GiB** of pagein for q4 → file-backed pages keep getting evicted and re-paged-in (the fate of model > RAM). q2 is small at ≈76 GiB (the ≈80.76 GiB model fits in RAM so re-pagein is mostly unnecessary).
- `nommap` is tiny at **≈0.7–1.6 GiB** pageins for both q2/q4 → preads use a separate path that does not go through the OS pagein counter, and the file-backed leak fix also suppresses reclaimable cache.
- pageout is a tiny few hundred pages (≈ a few MiB) for mmap-based, and 0 for nommap.

**compressor**
- `nommap` has compressed/comprs/dcomprs all **0** → page-cache bypass fully avoids the compressor (consistent with the memory note "compressor=0=OOM avoided").
- `residency-lru`/`baseline` keep **≈8.5–9.5 GiB** compressed resident for q4, with comprs/dcomprs in the hundreds-of-thousands to millions. The compression targets are the anonymous KV/context. `residency-lru` q2 keeps the compressor at essentially 0 (the residency LRU bounds anonymous memory).

**swapout**
- **swapout=0** for most runs. The tiny exceptions are baseline q4 at low budget (8GiB=96 / 16GiB=132 pages) and residency-lru q4 (16GiB=36 / 64GiB=184 pages), plus baseline q2 80GiB=40. All are MiB-scale; no OOM or large-scale swap.

---

## 8. System metrics (mactop)

Aggregated from `*-mactop.log` (mactop JSON samples): power, frequency, DRAM bandwidth, disk read, and temperature. Representative runs (q2 b32 / q4 b80). avg/max are over the whole run (all phases: title gen + agent→write + tool-result).

| run | total power W (avg/max) | GPU power W (avg) | DRAM BW GB/s (avg) | disk read MB/s (avg/max) | GPU temp ℃ (max) | GPU active % (avg) |
|---|---|---|---|---|---|---|
| q2 baseline b32 | 74.3 / 135.4 | 25.5 | 116.6 | 263 / 3444 | 88.1 | 86.3 |
| q2 residency-lru b32 | 63.9 / 88.8 | 21.3 | 109.3 | 223 / 3599 | 87.3 | 87.2 |
| q2 nommap b32 | 54.4 / 123.1 | 14.8 | 120.0 | **2299** / 5013 | 100.8 | 80.6 |
| q4 baseline b80 | 53.5 / 87.9 | 11.6 | **56.8** | 2943 / 6155 | 97.3 | 59.2 |
| q4 residency-lru b80 | 55.7 / 140.1 | 15.1 | 107.6 | 2147 / 6094 | 94.6 | 79.7 |
| q4 nommap b80 | 36.3 / 81.6 | 7.9 | 110.0 | 2597 / 6080 | 99.4 | 64.4 |

**Observations**
- **Disk read splits the strategies most sharply**: for q2, nommap averages **2299 MB/s** (about 9× baseline's 263). Without mmap it preads via F_NOCACHE every time, hammering the SSD constantly. Conversely baseline/residency-lru have small reads for q2 because the model sits in the page cache. For q4 all strategies read heavily (baseline/residency-lru re-pagein since model > RAM, nommap preads by design).
- **Total power is consistently lowest for nommap** (q2 74→54W, q4 54→36W). GPU power and GPU utilization are lower; while waiting on SSD I/O the GPU idles, lowering power (a tradeoff against speed).
- **DRAM BW is uniquely low for q4 baseline (56.8 GB/s)**. Against ~110 GB/s for the others, q4 baseline stalls on file-backed evict↔re-pagein and effectively cannot use the bandwidth (consistent with its excessive pagein in §7e).
- Temperature and frequency differ little by strategy (GPU temp max 88–101℃, p-cluster freq avg ~3.4–3.5 GHz, roughly flat). thermal_state was Normal for all runs.

### 8a. q4 b80 all-strategy charts

All charts for q4 b80 (model 153.33 GiB > RAM 128 GiB, streaming required), shown as **3 columns: baseline / residency-lru / nommap**. Each row is the same chart type with the same x-axis (10 min / 1 min ticks), so you can compare strategies directly. The chart type is identified by each image's title (`q4-<strategy>-…: <type>`). Order: tok / mem / events / cpu-gpu / freq / power / dram-bw / disk / temp. Discussion in §8b.

| baseline | residency-lru | nommap |
|---|---|---|
| <img src="plots/q4-baseline-opencode-b81920-tok.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-tok.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-tok.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-vmstat-mem.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-vmstat-mem.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-vmstat-mem.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-vmstat-events.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-vmstat-events.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-vmstat-events.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-mactop-cpu-gpu.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-mactop-cpu-gpu.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-mactop-cpu-gpu.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-mactop-freq.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-mactop-freq.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-mactop-freq.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-mactop-power.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-mactop-power.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-mactop-power.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-mactop-dram-bw.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-mactop-dram-bw.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-mactop-dram-bw.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-mactop-net-disk.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-mactop-net-disk.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-mactop-net-disk.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-mactop-temp.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-mactop-temp.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-mactop-temp.svg" width="100%"/> |

### 8b. Items with large strategy differences (all q4 b80)

q4 b80 (model 153.33 GiB > RAM 128 GiB) is the condition where the strategies' characters show most. Charts are in §8a; here we summarize numbers and discussion.

| Item (q4 b80) | baseline | residency-lru | nommap |
|---|---|---|---|
| prefill / gen t/s | 113.2 / 11.00 | **129.2 / 11.81** | 126.7 / **7.59** |
| disk read MB/s (avg) | **2943** | 2147 | 2597 |
| total power W (avg) | 53.5 | 55.7 | **36.3** |
| DRAM BW GB/s (avg) | **56.8** | 107.6 | 110.0 |
| GPU active % (avg) | **59.2** | **79.7** | 64.4 |
| GPU freq MHz (avg) | 1355 | 1300 | **1068** |
| CPU usage % (avg) | 4.8 | 5.6 | 3.0 |

#### ① Disk read — large for all q4 strategies, but the mechanism differs

- **nommap**: by design does not use file-backed (page cache). Experts not in the cache (the owned buffers within budget) are **preaded from SSD via F_NOCACHE every time** → constant reads (2597 MB/s).
- **baseline / residency-lru**: since model 153 GiB > RAM, the mmap pages do not all fit, causing **evict↔re-pagein** (also real disk reads). baseline thrashes the most with the highest reads (**2943**); residency-lru suppresses re-pagein with prefill MADV_DONTNEED + gen residency LRU, the lowest of the 3 (2147).
- → For q2, baseline had small reads (cache hits) and only nommap stood out (q2 nommap 2299 vs baseline 263 in the §8 top table), but **for q4 the model does not fit, so baseline also reads heavily from re-pagein**.

#### ② Total power — lowest for nommap

- **nommap 36.3W is lowest**. SSD I/O-bound, the GPU/CPU idle for long stretches; GPU active 64% and GPU power 7.9W are the lowest (many moments of underused resources, lowering average power).
- **residency-lru 55.7W is highest**. compute runs continuously with GPU active 80% and GPU power 15.1W — it works the resources hardest. baseline is in between, thrashing (53.5, GPU 59%).

#### ③ DRAM bandwidth and compute resources — high BW does not mean high gen t/s

The absolute DRAM BW is not proportional to gen t/s. **nommap has the highest BW (110) yet the lowest gen (7.59)**; conversely **baseline has the lowest BW (56.8) but still does gen 11.00**. The "content" of the BW differs by strategy.

- **nommap (high BW, low t/s)**: much of the BW is **SSD→RAM pread copies** (writing experts not in the cache into owned buffers every time), which is data movement, not MoE GEMM compute. The GPU runs intermittently waiting on preads, with the **lowest GPU freq 1068 MHz and GPU active 64%** → compute does not progress, so gen is lowest.
- **residency-lru (high BW, high t/s)**: the high BW goes to **zero-copy view weight reads**, and with no thrashing **GPU active is the highest at 80%** with freq 1300, so compute runs continuously → fastest gen.
- **baseline (low BW, mid t/s)**: BW is low because thrash page faults / SSD I/O waits mean **the time actually touching memory is short** (**GPU active the lowest at 59%**). Yet the moment a cache hit lands the GPU can read mmap pages directly, and freq is high at 1355, so during non-stalled intervals it maintains gen 11.00.
- **CPU usage is low for all strategies (3–6%)**: mostly pread issuance and GEMM dispatch; the heavy work is on the GPU side. nommap's lowest CPU at 3.0% reflects I/O waiting. p-cluster freq is flat at ~3.5 GHz (CPU is not the bottleneck).
- → What determines gen t/s is **whether the GPU can run MoE continuously without waiting on weights (GPU utilization / GPU freq)**, not the absolute DRAM BW. Even high BW does not contribute to speed if its substance is I/O copying.
