# 2026-06-02 `--routed-metal-dynamic` measurements

Running the same 153 GiB DeepSeek-V4-Flash model (Q4) on an Apple M4 Max / 128 GiB
Mac, but computing the **gen-time routed experts on Metal** via the experimental
`--routed-metal-dynamic` path instead of the CPU. The key is **`F_RDADVISE` (which
issues a real read ahead) applied across all three paths — prefill, CPU, and GPU — so
the SSD page-in overlaps compute**. With it, dyn **scales with budget (LRU hit%) and
power mode**: a mid-range **32 GiB reaches 10.58 (Normal) / 10.91 (High Power) t/s — ~2×
the 01 baseline (cpu-moe gen 5.37 t/s, 2026-05-18)**, and raising the budget (8→80 GiB)
gives ~7.8→~12.6 t/s on High Power.

Japanese version: [README.ja.md](README.ja.md)

This session keeps full logs (ds4-server / vm_stat / mactop / trace) for **four
captures only** — `{cpu, dyn 32 GiB} × {High Power, Normal}` — plus charts from
them. The full wired-budget sweep (8→80 GiB, Normal / High Power) is also kept as a
numeric table.

## Performance findings (what moves throughput)

The levers that actually changed t/s across these sessions — read the rest of this
doc for the evidence behind each:

1. **`F_RDADVISE` works across all paths (prefill / CPU / GPU).** On Darwin
   `posix_madvise(WILLNEED)` is only a vm hint and issues no I/O, whereas `F_RDADVISE`
   **issues a real read ahead** so the SSD page-in overlaps compute. It paid off on all
   three: prefill (per-layer phases N=43 + next-layer prefetch) rose **1.6–2.0×**
   (111.9 → 182–222 t/s); CPU gen, prefetching the selected experts, went 5.4 → ~10 t/s;
   and the GPU dynamic path now `F_RDADVISE`-warms its per-expert make_buffer, **lifting
   the I/O cap**. (`F_RDADVISE` encodes priority by *repeat count*, so the prefill
   lookahead deliberately re-hints the current layer.) See [Evolution](#evolution-since-the-cpu-moe-baseline-01-cpumoe-baseline).
2. **Thermal headroom governs both CPU and GPU.** Both paths gain on High Power + AC and
   lose without. The GPU is no exception: at a large budget (high hit%) dyn drives the
   GPU hard, so Normal throttles its clock — the bigger the budget, the more power matters
   (+10–14% at 72–80 GiB), and cpu-moe likewise (+4.6%). See [Thermals](#thermals--cpu-and-gpu-are-both-bound-by-thermal-headroom).
3. **Budget is dyn's speed knob.** The LRU hit% sets t/s and rises with budget (8→80 GiB:
   ~7.8 → ~12.6 High Power, ~8.1 → ~11.5 Normal). The gain is from fewer re-wires
   (`requestResidency`), not SSD. See [the sweep](#wired-budget-sweep-normal--high-power).
4. **Room to improve.** Three levers remain — speeding up small/medium prefill (an SME2
   SMOPA path or moving it to the GPU; SME2 spiked **3.37× over NEON i8mm** per P-core,
   bit-exact), fine-grained optimization of the rough implementation, and efficiency to cut
   heat (perf/W). For the full list — plus what's already handled (gen-on-GPU / imatrix
   pinning via the LRU) and what was tried without effect (MTP) — see [Room to improve](#room-to-improve).

## What `--routed-metal-dynamic` does

`--cpu-moe` keeps routed-expert weights out of the Metal residency set and runs the
gen-time expert GEMV on the CPU (NEON i8mm/DOTPROD), streaming weights from the OS
page cache / SSD. `--routed-metal-dynamic` instead runs that GEMV on the **GPU**:
each gen step, only the experts the router actually selected are wired into a
dedicated Metal residency set **per-expert** (≈13.5 MiB each on this Q4); an LRU
keeps the recently-used working set resident up to a byte budget and evicts the
tail. The existing Metal `mul_mv_id` kernels are reused.

- Requires `--cpu-moe` / `--n-cpu-moe`, Metal backend only, **default off**.
- Budget via env `DS4_ROUTED_METAL_BUDGET_MIB` (MiB, default 40960).
- gen-only; prefill still follows `--prefill-metal-phases` / cpu-moe, so the LRU
  starts **cold** each generation and warms over a few thousand tokens.

```sh
# add the flag to a --cpu-moe run; wired budget via env (default 40 GiB)
DS4_ROUTED_METAL_BUDGET_MIB=81920 \
  ./ds4-server --cpu-moe --prefill-metal-phases auto --routed-metal-dynamic --ctx 100000
```

(Or drive the measurements with `../scripts/q4-experiment.sh` — see [Reproduce](#reproduce).)

## How it works (prefill & gen)

Where each stage runs, and where `--cpu-moe` / `--routed-metal-dynamic` plug in:

```text
PREFILL  — prompt length ≥ DS4_PREFILL_METAL_PHASES_MIN_TOKENS ?
        │
        ├─ YES (large) ─▶ GPU, one layer per phase (--prefill-metal-phases, N=43)
        │                   compute layer L while F_RDADVISE prefetches L+1
        │                   (prefetch overlaps the next layer's page-in)
        │
        └─ NO  (small) ─▶ CPU (cpu-moe fallback), NEON i8mm 2×2
                            experts streamed from page cache / SSD

GEN  — decode, one token at a time
  embed → attention → norm → … → output head     [GPU, Metal-resident, ~13 GiB wired]
            │
            └─▶ routed MoE — router picks 6 of 256 experts (prefetched ahead):
                  --cpu-moe               → CPU: NEON DOTPROD; experts via page cache / SSD
                  --routed-metal-dynamic  → GPU: only the selected experts wired (dynamic LRU)
```

Both stages prefetch the weights they are about to need (prefill prefetches the next
layer; gen prefetches the router's selected experts), so the SSD read overlaps compute
instead of stalling it.

## Evolution since the cpu-moe baseline (`01-cpumoe-baseline/`)

Two things improved between the 01 baseline (2026-05-18) and this session, on top
of which `--routed-metal-dynamic` is layered:

1. **Prefill: per-layer phases + lookahead prefetch.** `--prefill-metal-phases auto`
   used to resolve to **N=2** coarse phases (layers 0–20 / 21–42, an 82 GiB residency
   swap each, **no prefetch**). It now resolves to **N=43 — one layer per phase**
   (3.38 GiB swap each) and `F_RDADVISE`-prefetches the *next* layer's experts while
   the current layer computes (`lookahead prefetch … next layers N+1` in the log).
   The small per-layer swaps (~80 ms map+residency) overlap with compute instead of
   stalling the prompt on two large 82 GiB swaps. On the ~10–11 k-token prefill this
   lifts throughput 1.6–2.0× (up to ~2× on High Power) — **111.9 t/s (01) → 182–222 t/s (02)** (prefill speed
   is the same for the cpu and dyn captures; `--routed-metal-dynamic` is gen-only):

   | prefill, ~10–11 k-token prompt | t/s |
   |--------------------------------|-----|
   | 01 baseline (2026-05-18)       | 111.9 |
   | 02 — High Power (cpu / dyn)     | 200 / 222 |
   | 02 — Normal (cpu / dyn)         | 182 / 195 |
2. **Gen: routed-expert prefetch, then the Metal path.** Gen-time prefetch of the
   selected experts lifted the **CPU** path itself (opencode CPU gen 5.37 → ~10 t/s vs
   the 01 baseline); `--routed-metal-dynamic` then moves that GEMV onto the GPU + LRU
   residency and scales it further with budget (hit%) and power.

| opencode gen t/s              | value |
|-------------------------------|-------|
| 01 baseline (CPU, 2026-05-18) | 5.37  |
| 02 CPU — Normal               | 9.62  |
| 02 CPU — High Power           | 10.06 |
| 02 dyn 32 GiB — Normal        | 10.58 |
| 02 dyn 32 GiB — High Power    | 10.91 |

So: prefill got much faster from per-layer phasing + prefetch; gen got much faster from
prefetch (CPU, 5.4 → ~10), and `--routed-metal-dynamic` moves that onto the GPU + LRU
residency and scales it further with budget and power (a mid-range 32 GiB gives 10.58 /
10.91 = ~2× the 01 baseline; 80 GiB reaches 11.46 / 12.60 — see the sweep).

## Environment

- Apple M4 Max, 128.00 GiB RAM (Metal device)
- Q4 GGUF (routed experts 145.12 GiB, whole model 153 GiB), `--cpu-moe` (all 43 routed layers), `--prefill-metal-phases auto`
- Workload: opencode "create minesweeper" agent flow (3 turns, ctx grows to ~13 k).
  Gen t/s = the server's running `avg=` (cumulative tokens ÷ time) over the long
  ~2.5 k-token turn. opencode is non-deterministic (gen-token counts vary 2.3 k–3.2 k),
  so treat these as single representative runs.
- Captures: `cpu-highpower`, `dyn-highpower`, `cpu-normal`, `dyn-normal`
  (`dyn` = `--routed-metal-dynamic` at `DS4_ROUTED_METAL_BUDGET_MIB=32768` = 32 GiB, a mid-range budget).
  High Power = System Settings → Battery "High Power" + AC-cooled room; Normal =
  Automatic mode, no AC.

## Dynamic Metal gen t/s (the four captures, vs the 01 baseline)

Sustained gen t/s (the long TOOLS turn's avg, warmup included) — `--routed-metal-dynamic` at 32 GiB, High Power vs Normal, **relative to the 01 baseline**:

| condition        | dyn 32 GiB | vs 01 baseline |
|------------------|------------|----------------|
| High Power + AC  | 10.91      | 2.03×          |
| Automatic/Normal | 10.58      | 1.97×          |

The 01 baseline is cpu-moe gen **5.37 t/s** (2026-05-18, [`01-cpumoe-baseline/`](../01-cpumoe-baseline/README.md)).
With `F_RDADVISE` across all paths, dyn is **~2× the baseline even at a mid-range 32 GiB
budget**. It gains from power too (Normal 10.58 → High Power 10.91) and, like cpu-moe, is
subject to thermal headroom (the GPU throttles too — see Thermals). Raising the budget
reaches 11.46 / 12.60 at 80 GiB (see the sweep). swap-free.

### Throughput

<p>
  <a href="plots/cpu-highpower-tok.svg"><img src="plots/cpu-highpower-tok.svg" width="380" alt="cpu high power t/s"></a>
  <a href="plots/dyn-highpower-tok.svg"><img src="plots/dyn-highpower-tok.svg" width="380" alt="dyn high power t/s"></a>
</p>
<p>
  <a href="plots/cpu-normal-tok.svg"><img src="plots/cpu-normal-tok.svg" width="380" alt="cpu normal t/s"></a>
  <a href="plots/dyn-normal-tok.svg"><img src="plots/dyn-normal-tok.svg" width="380" alt="dyn normal t/s"></a>
</p>

### Thermals — CPU and GPU are both bound by thermal headroom

Die temperature, P/E/GPU clocks, and package power from `mactop`. The **CPU path across
power modes** (Normal vs High Power) isolates the throttling most clearly. But **the GPU
path is the same**: at a large budget (high hit%) dyn drives the GPU hard enough to heat
up, and on Normal the GPU clock drops and t/s falls (High Power recovers it, +10%).
Losing speed off power is not a CPU-only phenomenon.

**CPU path — Normal (left) vs High Power (right):**

<p>
  <a href="plots/cpu-normal-mactop-temp.svg"><img src="plots/cpu-normal-mactop-temp.svg" width="380" alt="cpu normal temp"></a>
  <a href="plots/cpu-highpower-mactop-temp.svg"><img src="plots/cpu-highpower-mactop-temp.svg" width="380" alt="cpu high power temp"></a>
</p>
<p>
  <a href="plots/cpu-normal-mactop-freq.svg"><img src="plots/cpu-normal-mactop-freq.svg" width="380" alt="cpu normal freq"></a>
  <a href="plots/cpu-highpower-mactop-freq.svg"><img src="plots/cpu-highpower-mactop-freq.svg" width="380" alt="cpu high power freq"></a>
</p>
<p>
  <a href="plots/cpu-normal-mactop-power.svg"><img src="plots/cpu-normal-mactop-power.svg" width="380" alt="cpu normal power"></a>
  <a href="plots/cpu-highpower-mactop-power.svg"><img src="plots/cpu-highpower-mactop-power.svg" width="380" alt="cpu high power power"></a>
</p>

**`--routed-metal-dynamic`, Normal (temp / freq / power):**

<p>
  <a href="plots/dyn-normal-mactop-temp.svg"><img src="plots/dyn-normal-mactop-temp.svg" width="250" alt="dyn normal temp"></a>
  <a href="plots/dyn-normal-mactop-freq.svg"><img src="plots/dyn-normal-mactop-freq.svg" width="250" alt="dyn normal freq"></a>
  <a href="plots/dyn-normal-mactop-power.svg"><img src="plots/dyn-normal-mactop-power.svg" width="250" alt="dyn normal power"></a>
</p>

#### The three levers that move thermal headroom

gen t/s is set by **thermal headroom (cooling capacity − accumulated heat)**. CPU GEMV and
GPU GEMV share the same SoC power/thermal budget, so both move with three levers:

1. **Ambient temp**: a cool room (AC) lowers the starting die temp. Measured on
   continuous chat in Automatic mode: starting at 23 °C → ~10 t/s, at 56 °C → ~8.6
   (P-cluster clamped ~65% of the run).
2. **macOS power mode**: High Power (System Settings → Battery) raises the sustained
   power budget and fans → cpu-moe ~10.6 t/s on opencode (vs ~8 on Automatic), and GPU
   dynamic gains from power too (the bigger the budget, the more so — 32 GiB 10.58 → 10.91,
   +10–14% at 72–80 GiB).
3. **Workload shape**: opencode splits gen into short turns with cooling gaps; chat runs
   gen continuously so the die soaks to ~96 °C and clamps more — chat is thermally worse
   even in the same power mode.

So **to go faster, add thermal headroom** (High Power mode / cool the room); on CPU or
GPU, a warmer start just throttles sooner. When comparing runs, equalise the starting
die temp or most of the difference is thermal-history noise.

Per-capture chart sets: `plots/<capture>-{tok,vmstat-mem,vmstat-events,mactop-cpu-gpu,mactop-freq,mactop-power,mactop-dram-bw,mactop-net-disk,mactop-temp}.svg`.

## CPU path vs GPU path: how each uses RAM

The two paths make opposite RAM trade-offs — **portability vs a tunable speed knob**:

- **`--cpu-moe` (CPU): tiny wired, rides the page cache.** It pins only the ~13 GiB
  non-routed set, so it needs no raised `iogpu.wired_limit_mb`, never risks a
  wired-budget swap, and will run on machines with much less RAM. Its gen speed instead
  rides the **file-backed cache size**: the more RAM left to cache the mmap'd model, the
  fewer experts miss to SSD — so more free RAM ≈ faster, a RAM-starved box slower. (And
  that speed is heat-sensitive — see the thermals above.)
- **`--routed-metal-dynamic` (GPU): the wired budget *is* the speed knob.** t/s scales
  with the budget (via hit% — quantified in the sweep below) and, now that `F_RDADVISE`
  has lifted the I/O cap, also gains from High Power (≈+10%); you pay in pinned RAM: a
  bigger budget is faster yet needs RAM headroom and a raised `iogpu.wired_limit_mb`, and
  overshooting swaps (see the swap note under the sweep).

In short: **CPU for portability / a small wired footprint; GPU when you have RAM to pin
and want predictable, tunable t/s.**

The `vmstat-mem` charts show the split directly: `--cpu-moe` keeps wired tiny (~13 GiB)
and leaves the rest of RAM as reclaimable page cache, while `--routed-metal-dynamic`
pins the budget (e.g. ~45 GiB at a 32 GiB budget = 32 GiB experts + the ~13 GiB non-routed
set), and the page cache shrinks to match.

<p>
  <a href="plots/cpu-normal-vmstat-mem.svg"><img src="plots/cpu-normal-vmstat-mem.svg" width="380" alt="cpu normal memory"></a>
  <a href="plots/dyn-normal-vmstat-mem.svg"><img src="plots/dyn-normal-vmstat-mem.svg" width="380" alt="dyn normal memory"></a>
</p>

## Wired-budget sweep (Normal / High Power)

The opencode long-TOOLS-turn avg (warmup included) per budget, at both power modes. The
CPU row is the cpu-moe reference on the same workload.

| budget   | Normal | High Power | hit% | wired peak | SSD read N/HP (GiB) |
|----------|--------|------------|------|------------|---------------------|
| CPU      | 9.62   | 10.06      | –    | ~13 GiB    | 461 / 409           |
| dyn 8 G  | 8.14   | 7.81       | ~41  | 8.0 GiB    | 489 / 579           |
| dyn 16 G | 9.33   | 9.22       | ~59  | 16.0 GiB   | 496 / 524           |
| dyn 24 G | 10.18  | 10.02      | ~69  | 24.0 GiB   | 425 / 502           |
| dyn 32 G | 10.58  | 10.91      | ~76  | 32.0 GiB   | 482 / 447           |
| dyn 40 G | 10.55  | 11.27      | ~80  | 40.0 GiB   | 480 / 445           |
| dyn 48 G | 10.88  | 11.78      | ~85  | 48.0 GiB   | 416 / 450           |
| dyn 56 G | 11.08  | 12.05      | ~88  | 56.0 GiB   | 459 / 438           |
| dyn 64 G | 11.27  | 12.42      | ~91  | 64.0 GiB   | 426 / 440           |
| dyn 72 G | 11.36  | 12.93      | ~93  | 72.0 GiB   | 422 / 421           |
| dyn 80 G | 11.46  | 12.60      | ~94  | 80.0 GiB   | 422 / 453           |

- **Budget sets the LRU hit rate, which sets t/s** — rising with budget. The gain is from
  fewer re-wires (`requestResidency`), not SSD.
- **Power mode matters more at larger budgets.** Small budgets (≤24 GiB) miss a lot and
  stay I/O-leaning, so Normal ≈ High Power (sometimes Normal is even a touch higher). The
  bigger the budget, the higher the hit% and the harder the GPU runs, so High Power pulls
  ahead (+10–14% at 72–80 GiB).
- **Going larger swaps.** Wired (resident) memory cannot be reclaimed. The higher the
  budget, the more RAM is pinned, squeezing the rest (OS + the KV cache, which grows with
  context). Once `wired + OS + KV` exceeds physical RAM, `swap-out` appears (possible past
  ~96 GiB on this box), so **validate `swap-out = 0` at your longest context** (not a short
  prompt). 80 GiB wires at the default `iogpu.wired_limit_mb`; beyond that, raise it (e.g.
  `sudo sysctl iogpu.wired_limit_mb=102400`). Note `free ≈ 0` is normal — spare RAM is
  filled with reclaimable page cache; the signal to watch is `swap-out`. (Reading vm_stat:
  `vm_stat <interval>` reprints the column header every few dozen samples, and the row right
  after it is a cumulative-since-boot snapshot — exclude that post-header row *by position*
  when summing `Pageins` / `Swapouts`; don't key on the `K`/`M` suffix, which is absent at
  low uptime when the cumulative count still fits as a plain integer.)
- **SSD reads stay ≈ 410–580 GiB whatever the budget** (the SSD column above, summed
  vm_stat pageins). The routed experts (145.12 GiB) far exceed RAM, so they are constantly
  streamed from SSD; that total tracks the workload, not the gen budget (the spread within
  the column is opencode's run-to-run noise, with no trend against budget). Since gen serves
  its experts from the wired (in-RAM) set, this SSD traffic is off gen's critical path — it
  does not cap gen t/s.

## Q2 (a model that fits in RAM)

Gen t/s for the IQ2_XXS/Q2_K build (routed experts ≈ 72.6 GiB total, whole model ≈ 80 GiB)
on the same opencode workload (long TOOLS-turn avg, warmup included; 2026-06-03). **One
routed expert is ≈6.8 MiB here — about half Q4's
≈13.5 MiB** (so a given budget holds ~2× as many resident). `--routed-metal-dynamic` lands
**between full Metal and cpu-moe — slower than full Metal, faster than cpu-moe**:

| Q2 gen t/s                     | t/s   | vs full Metal |
|--------------------------------|-------|---------------|
| full Metal (all experts wired) | 22.13 | 100%          |
| dyn 24 GiB wired (hit 85.6%)   | 17.19 | 78%           |
| dyn 40 GiB wired (hit 94.3%)   | 17.56 | 79%           |
| dyn 80 GiB wired (hit 98.3%)¹  | 14.89 | 67%           |
| `--cpu-moe` (CPU gen)          | 7.96  | 36%           |

All swap-free.

- **dyn sits between full Metal and cpu-moe.** Running the expert GEMV on the GPU beats
  cpu-moe (7.96), but cycling experts through the LRU keeps it short of full Metal (22.13),
  which keeps every expert resident.
- **Budget sensitivity is flatter than on Q4.** 24 GiB (85.6%) and 40 GiB (94.3%) are nearly
  tied (the gap is run-to-run noise). Q2 experts are small, so an LRU miss re-wires cheaply —
  raising the hit% saves little wall-time, so t/s barely tracks budget (on Q4 the sweep rose
  cleanly with hit%).
- **The Q2/Q4 speed ratio tracks the expert-size ratio (~2×) only partly, and only on the GPU
  path.** At a matched budget, dyn makes Q2 ~1.6–1.7× Q4 (24–40 GiB) — the expert GEMV halves
  with the bytes, but the fixed costs (attention, non-routed, dispatch) are shared, so it falls
  short of 2×. cpu-moe is the opposite: Q2 is on par with or a touch slower than Q4 (~0.8×),
  confirming CPU gen is compute/thermal-bound, not weight-bandwidth-bound.
- **SSD reads happen until the cache fully warms.** The LRU residency starts cold, so early
  in a turn there are many misses (= SSD page-ins); as the turn proceeds the working set
  becomes wired and the hit% climbs (72 → 94% at 40 GiB), so the warmup-included avg reads a
  bit below steady state. But because the whole 80 GiB Q2 model fits in RAM, SSD reads total
  only **~80 GiB across every config (the model read roughly once, budget-independent;
  measured vm_stat pageins)** and stop once warm — unlike Q4, which streams its 145 GiB routed
  set at ~450 GiB throughout.

¹ At 80 GiB (hit 98.3%, evict 0) almost all experts are resident, but wiring ~68 GiB drives
the GPU hardest, so it throttles thermally late in the long turn (early steady ~17, on par
with 40 GiB; the avg sinks to 14.89). Even when the model fits, dyn is still a useful **cap
on wired memory** — most of full-Metal speed at a smaller wired footprint.

## Room to improve

Tracked against the candidate improvements from the cpu-moe baseline
([`01-cpumoe-baseline/`](../01-cpumoe-baseline/README.md)). Two of those are already
handled. **gen-on-GPU / hot-expert pinning** is realised here via a dynamic LRU of the
*actually-selected* experts — which subsumes the baseline's **imatrix** idea too: a
static top-N pin from `observed_routes` misses because the hot set is prompt-dependent,
whereas the LRU wires exactly what the router selects, per turn. And the cold-start
**warmup** idea is moot — per-layer phases (N=43) + `F_RDADVISE` prefetch shrank the
residency build from ~53 s to small, compute-overlapped per-layer swaps, so the first
prompt no longer pays a big build.

**Still on the table**

- **Speed up small/medium prefill (SME2 or move it to the GPU).** Large prefills run on
  Metal phases; prefills below `DS4_PREFILL_METAL_PHASES_MIN_TOKENS` fall back to CPU
  NEON i8mm 2×2. Two ways to close it — (a) an **SME2 SMOPA path** (spiked **3.37× over
  NEON i8mm** per P-core, bit-exact), or (b) widen the Metal phases to cover small
  prefills too and **run them on the GPU**.
- **Fine-grained optimization (the implementation is rough).** The current
  `--routed-metal-dynamic` is a coarse Claude-Code implementation aimed at proving the
  mechanism works. The hot-path details — LRU/wire constants, how prefetch is batched,
  the granularity of residency updates — are not yet tuned, so there is still headroom in
  fine-grained optimization.
- **Efficiency to cut heat (better perf/W).** gen t/s is bound by thermal headroom (see
  [Thermals](#thermals--cpu-and-gpu-are-both-bound-by-thermal-headroom)) — it's why the Q2
  80 GiB run, which drives the GPU hardest, throttles late in a long turn. Trimming wasted
  wires/re-residency, copies, and dispatches to do **the same work at lower power** lowers
  the heat, delays the clamp, and lifts sustained t/s. This overlaps with the fine-grained
  work above (more perf/W converts directly into thermal headroom).

**Tried, no effect (here)**

- **MTP (speculative draft + verify).** Meant to batch the per-token expert reads; did
  not speed gen in this setup (the MTP run landed at/below the non-MTP baseline).

## Takeaway

- **`F_RDADVISE` makes every path faster.** prefill (~2×), CPU gen (5.4 → ~10), and the
  GPU dynamic path (I/O cap lifted). dyn scales with budget (hit%) and power: a mid-range
  32 GiB gives 10.58 (Normal) / 10.91 (High Power) t/s — **~2× the 01 baseline (5.37)**
  (80 GiB reaches 11.46 / 12.60).
- **Thermals govern both CPU and GPU.** Both paths gain on High Power + AC and lose without.
- **A wired-memory knob** (clearest on Q2): cap wired memory, keep most of the GPU benefit.
- Default off / experimental; going too large on the budget swaps — validate `swap-out = 0`
  at your longest context (80 GiB wires at the default `iogpu.wired_limit_mb`).

## Reproduce

Shared driver script `../scripts/q4-experiment.sh [mode]` (see its header for env
overrides like `BUDGET_MIB` / `TAG`). Run it **from this session directory** (logs
land in the cwd). Modes: `sweep` (budget→t/s table), `opencode` (real agent run),
`oc-cpu` (CPU baseline), `perfab` (warm steady-state A/B), `bench`, `chat`.

```sh
bash ../scripts/q4-experiment.sh sweep                      # cpu baseline + dyn budget sweep
BUDGET_MIB=81920 bash ../scripts/q4-experiment.sh opencode  # dyn at 80 GiB, real workload
bash ../scripts/q4-experiment.sh oc-cpu                     # pure-CPU baseline
# Q2 (fits in RAM): TAG=q2 selects the IQ2_XXS model + q2-* log prefixes.
# Use a budget < 72.6 GiB or the LRU never evicts (= full residency):
TAG=q2 BUDGET_MIB=40960 bash ../scripts/q4-experiment.sh opencode
```

Charts: `python3 ../scripts/make_plots.py 02-routed-metal-dynamic` (auto-detects the
four `<capture>-ds4-server.log` prefixes; writes `plots/`).
