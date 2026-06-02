# 2026-05-18 ds4-server cpu-moe measurements

Logs and screenshots captured while running a 153 GiB DeepSeek-V4-Flash
model on an Apple M4 Max / 128 GiB Mac with
`--cpu-moe` + `--prefill-metal-phases auto`.
The page is ordered screenshots → charts → numeric analysis so you can
skim from top to bottom and pick up what's going on.

Japanese version: [README.ja.md](README.ja.md)

## Screenshots

### chat
<p>
  <a href="chat-01.png"><img src="chat-01.png" width="320" alt="chat-01"></a>
  <a href="chat-02.png"><img src="chat-02.png" width="320" alt="chat-02"></a>
  <a href="chat-03.png"><img src="chat-03.png" width="320" alt="chat-03"></a>
</p>

### opencode
<p>
  <a href="opencode-01.png"><img src="opencode-01.png" width="320" alt="opencode-01"></a>
  <a href="opencode-02.png"><img src="opencode-02.png" width="320" alt="opencode-02"></a>
  <a href="opencode-03.png"><img src="opencode-03.png" width="320" alt="opencode-03"></a>
</p>

## Environment

- Apple M4 Max, 128.00 GiB RAM (Metal device)
- ds4-server: `--prefill-metal-phases auto --ctx 100000 --kv-disk-space-mb 8192`
- `DS4_PREFILL_METAL_PHASES_MIN_TOKENS=0`
- `--cpu-moe`: routed MoE on the CPU for layers 0..42 (all 43 layers)
- `--prefill-metal-phases auto` resolved to **N=2**
  (total=145.12 GiB, cap=96.00 GiB, headroom=14.00 GiB, per-phase budget=82.00 GiB)
- iostat / vm_stat sampled at **30 seconds**

## Design overview

The two flags exercised in this session are independent dials on the routed-MoE
placement; they cover the prefill and gen halves of a request differently.
Authoritative description lives in `ds4-server --help`
(`ds4_server.c:7909-7927`); this is the user-facing summary.

### `--cpu-moe` / `--n-cpu-moe N`

- Routed MoE experts are computed on the CPU, and their weights are kept
  **out of the Metal wired-memory residency set**. The pages live in the OS
  file-backed cache, with the SSD as the spill tier.
- `--cpu-moe` = all 43 routed layers on CPU. `--n-cpu-moe N` = the first N
  routed layers on CPU, the rest on GPU (matches llama.cpp semantics).
- Lets a 153 GiB GGUF fit on a 128 GiB Mac because only the non-routed
  segments (attention, embeddings, output head, shared expert) need to be
  wired into Metal. That working set is the ~13 GiB `wired` floor visible in
  the gen sections of `vmstat-mem`.
- Matmul kernel selection on the CPU side (Q4_K weights × Q8_K activations):
  - **Prefill cpu-moe fallback** (`n_tok ≥ 2`): NEON **i8mm 2x2** SMMLA mini
    panel (pairs two activations × two expert rows per inner iter). Runtime
    probe via `sysctlbyname` (Darwin) / `HWCAP2_I8MM` (Linux); disable with
    `DS4_DISABLE_I8MM=1`.
  - **Gen / decode** (`n_tok = 1`): NEON **DOTPROD**. A genuine 2x2 cannot be
    formed from a single activation, so i8mm is bypassed.
  - Routed Q2_K / IQ2_XXS variants stay on NEON DOTPROD throughout.
- Metal backend only.

### `--prefill-metal-phases auto|N`

- Prefill is run on Metal in **N evenly-split phases**. Between phases, the
  routed-expert residency is swapped so that each phase's working set fits the
  Metal wired-memory cap.
- `auto` derives N from `sysctl iogpu.wired_limit_mb` (bounded by
  `hw.memsize`), subtracts the headroom
  (`DS4_PREFILL_METAL_PHASES_HEADROOM_MIB`, default **14336**), and picks the
  smallest N whose per-phase budget covers the routed-expert range. In this
  session: `total=145.12 GiB / (cap 96.00 - headroom 14.00) = per-phase budget
  82.00 GiB → N=2`.
- Per-phase entry calls `engine_activate_prefill_phase(p)`; the page-in burst
  is the 2.2–2.8 GB/s yellow-band spike on `iostat-disk`.
- When prefill exits, `engine_restore_gen_routing()` flips every routed layer
  back to CPU. Gen therefore **falls back to the `--cpu-moe` path** — the
  Metal residency shrinks to the non-routed ~13 GiB set, and gen reads
  routed-expert pages on demand through the file-backed cache. Both the
  full-prefill and the **extend-checkpoint** sync paths call `restore`; the
  per-token decode branch does not touch phase activation.
- Phase mmap is safe to repeat because `set_model_map_range` drains in-flight
  Metal command buffers and explicitly clears the previous `MTLBuffer` views
  before mapping the next phase. Without this, the old and new views both
  held mmap ranges over the same region and `cpt_mapcnt` accumulated past
  the kernel's 2048 limit.
- Mutually exclusive with `--cpu-moe` / `--n-cpu-moe`. Metal backend only.
- `DS4_PREFILL_METAL_PHASES_MIN_TOKENS` (default 0) sets a prompt-length floor
  below which prefill falls back to cpu-moe instead of building the Metal
  phase. Setting it to `0` in this capture forces the phase path even for the
  short chat warmup turn.

### Roles at a glance

| Phase | Routed-expert placement | Matmul kernel | Bound by |
|---|---|---|---|
| Prefill (yellow band) | Metal, swapped across N phases | Metal shaders | Metal GPU + sustained SSD read to fill residency |
| Prefill cpu-moe fallback (`< MIN_TOKENS`) | CPU; pages from file-backed cache | NEON i8mm 2x2 | CPU matmul + SSD random read |
| Gen (green/blue/purple bands) | CPU; pages from file-backed cache + SSD | NEON DOTPROD | SSD random read (QD=1) for cache misses |

## Capture procedure

See `../README.md`. Summary:

```sh
TEST_PREFIX=chat   # or opencode
vm_stat 30 2>&1 | tee ${TEST_PREFIX}-vm_stat.log
iostat 30 2>&1 | tee ${TEST_PREFIX}-iostat.log
DS4_PREFILL_METAL_PHASES_MIN_TOKENS=0 ./ds4-server \
  --prefill-metal-phases auto --ctx 100000 \
  --kv-disk-dir /tmp/${TEST_PREFIX}-ds4-kv --kv-disk-space-mb 8192 \
  2>&1 | tee ${TEST_PREFIX}-ds4-server.log
```

## Files

| File | Description |
|---|---|
| `chat-ds4-server.log` | ds4-server log (chat: one request, 8714 generated tokens) |
| `chat-iostat.log` | iostat at 30s interval |
| `chat-vm_stat.log` | vm_stat at 30s interval |
| `chat-01.png` / `chat-02.png` / `chat-03.png` | tmux screenshots during chat |
| `opencode-ds4-server.log` | ds4-server log (opencode integration, three turns) |
| `opencode-iostat.log` | iostat at 30s interval |
| `opencode-vm_stat.log` | vm_stat at 30s interval |
| `opencode-01.png` / `opencode-02.png` / `opencode-03.png` | tmux screenshots during opencode |
| `plots/*.svg` | auto-generated charts (see below) |

## Charts

Auto-generated from logs by `scripts/make_plots.py` (one level up):

```sh
python3 ../scripts/make_plots.py .
```

Background tints highlight ds4-server phases:

- **Yellow**: prefill (including Metal residency build / rebuild)
- **Blue**: THINKING gen
- **Green**: answer gen (normal response)
- **Purple**: TOOLS gen (opencode)
- **Orange**: thinking-checkpoint canonicalization

iostat / vm_stat run on their own clock (the two loggers start a few seconds before ds4-server), so phase positions on those plots are accurate to ±10-30s. The tok plot shares the same timestamps as the ds4-server log and is exact.

### chat

<p>
  <a href="plots/chat-tok.svg"><img src="plots/chat-tok.svg" width="480" alt="chat tokens/s"></a>
  <a href="plots/chat-iostat-disk.svg"><img src="plots/chat-iostat-disk.svg" width="480" alt="chat disk I/O"></a>
</p>
<p>
  <a href="plots/chat-iostat-cpu.svg"><img src="plots/chat-iostat-cpu.svg" width="480" alt="chat CPU/load"></a>
  <a href="plots/chat-vmstat-mem.svg"><img src="plots/chat-vmstat-mem.svg" width="480" alt="chat memory composition"></a>
</p>
<p>
  <a href="plots/chat-vmstat-events.svg"><img src="plots/chat-vmstat-events.svg" width="480" alt="chat memory events"></a>
</p>

### opencode

<p>
  <a href="plots/opencode-tok.svg"><img src="plots/opencode-tok.svg" width="480" alt="opencode tokens/s"></a>
  <a href="plots/opencode-iostat-disk.svg"><img src="plots/opencode-iostat-disk.svg" width="480" alt="opencode disk I/O"></a>
</p>
<p>
  <a href="plots/opencode-iostat-cpu.svg"><img src="plots/opencode-iostat-cpu.svg" width="480" alt="opencode CPU/load"></a>
  <a href="plots/opencode-vmstat-mem.svg"><img src="plots/opencode-vmstat-mem.svg" width="480" alt="opencode memory composition"></a>
</p>
<p>
  <a href="plots/opencode-vmstat-events.svg"><img src="plots/opencode-vmstat-events.svg" width="480" alt="opencode memory events"></a>
</p>

> `chunk t/s` (red) is the instantaneous throughput over the most recent ~50 tokens;
> `avg t/s` (blue) is the cumulative average per request.
>
> `vmstat-mem` shows the partition of 128 GiB physical RAM into `wired` (Metal residency), `file-backed` (mmap cache for experts), and `cmprssor` (compressor pool). `vmstat-events` plots `faults` / `pageins` / `pageout` on the left axis and `swap in+out` on the right — **any line lifting off zero on the right axis or in `pageout` is an anomaly / pressure signal.**
>
> Reading the opencode tok plot x-axis: there are three `prompt start` events at **t=0s** (system prompt, THINKING 533→291 tok), **t≈178s** (TOOLS, 10960→2113 tok), **t≈694s** (post-tool follow-up, 15→69 tok). `elapsed` resets to 0 each turn, so the gray dashed verticals mark session boundaries.
>
> opencode vmstat-mem wired timeline: `3→76→83 GiB` (t≈60-120s, Turn 1 phase build) → `13 GiB` (t≈150-180s, Turn 1 gen) → `62→76→85→82→40→73 GiB` (t≈210-300s, canonicalization + Turn 2 prefill's two cycles) → `13 GiB` flat (t≈390-720s, Turn 2 gen + Turn 3) → `3 GiB` (t≈780s onward, model unloaded).

## Measurement summary

Raw seconds and t/s figures live in `*-ds4-server.log`. This section instead picks out **what behaves as `--cpu-moe` + `--prefill-metal-phases` is designed to behave** and **where the design still leaves room to improve**.

### Behaviours that match the design

- **wired ~13 GiB during gen (= attention stack lives on GPU/Metal)**: attention, embeddings, output head and the other non-routed segments are Metal-resident and accounted in wired. `--cpu-moe` excludes the 145 GiB routed-expert range from the residency set by design, so wired stays flat at this value.
- **Sustained 200–520 MB/s of disk traffic during gen (= experts served by file-backed cache → SSD, in that order)**: routed experts are mmap'd on the CPU side, with `vmstat-mem`'s **file-backed (~110 GiB) acting as the primary cache**. Each token picks 6 of 256 experts, but 145 GiB > 110 GiB so the cache cannot hold the full routed set; the misses show up as the **20k–27k tps / 200–520 MB/s** band on iostat, paged in from SSD on demand.
- **2.2–2.8 GB/s bursts at phase swap**: `--prefill-metal-phases auto N=2` mmaps 70–84 GiB into Metal residency in one shot. The yellow-band peaks on iostat disk are exactly this.
- **chat prefill 53s / opencode Turn 1 prefill 55s split into ~24s + ~29s**: phase 0/2 is brought in first, then phase 1/2, by design. Phase 1/2 (layers 21..42) is the longer leg because it carries more weight pages than phase 0/2.
- **wired drops to 13 GiB the moment prefill exits**: `engine_restore_gen_routing()` flips every routed layer back to CPU and shrinks the Metal residency set to non-routed only. The drop is the boundary between yellow and green/purple bands on vm_stat.
- **pageins spikes only at phase mmap**: vm_stat's right-axis pageins (per-30s page-in count) hit the megapage range only during residency build / rebuild windows; everywhere else it sits at 0.4–1.0 M. The plot cleanly separates "CPU MoE steady page-in" from "phase mmap one-shot burst".

### Improvement candidates

- **Warmup**: on a cold start, the first prompt's prefill carries the entire first-ever Metal residency build (opencode Turn 1 prefill 54.7s, of which ~53s is phase 0/2 + phase 1/2 mmap). Triggering `engine_activate_prefill_phase(0)` → `engine_activate_prefill_phase(1)` → `engine_restore_gen_routing` in the background right after `listen` would let the first user prompt skip that build.
- **Pin hot experts into Metal residency (move routed-expert compute onto the GPU)**: today gen's routed-expert matmul runs on CPU NEON DOTPROD (i8mm 2x2 is prefill-only), and the 145 GiB routed range does not fit in the ~110 GiB file-backed cache, so every token leaks SSD page-ins (the 20-27k tps band). **Read the imatrix `ds4_imatrix_collector.observed_routes`, take the top N most-activated experts, and pin them into Metal residency.** Experts that the router hits then run on the GPU with no SSD read. The main implementation task is widening `cpu_moe_layer[]` to a per-expert decision.
- **Parallel IO prefetch for the cold path (raise queue depth)**: experts that fall outside the hot pin still go through CPU + SSD. CPU MoE picks the next expert dynamically so each page fault is serialized → SSD random reads hit the QD=1 ceiling (~25k IOPS). The M4 Max NVMe spec reaches 500k+ IOPS at QD=32, so issuing `madvise(WILLNEED)` for the router's top-K predictions lifts the cold-path queue depth.
- **MTP-batched gen**: gen advances one token at a time today, so every token's expert reads serialize. **MTP (speculative draft + verify) lets a single forward pass evaluate N draft tokens together**, batching N tokens' expert reads. Shared experts dedupe page fetches; distinct experts can be issued in parallel. **The draft head fits inside Metal residency, and the verify path reuses the hot-pinned expert pool**, so only the draft head adds capacity. ds4 already has `mtp_draft_tokens` / `metal_graph_verify_suffix_tops` infrastructure; the main task is reconciling `ds4_engine_mtp_draft_tokens()` (currently gated only on `backend != CPU`) with the cpu-moe gen path.

