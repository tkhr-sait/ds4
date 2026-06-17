# DS4 SSD-streaming ベンチ結果 — 3方式

計測日: 2026-06-17 / 機材: Apple M4 Max 128 GiB / モデル: deepseek-v4-flash
対象ログ: `/workspace/logs/ssd-streaming-brushup/`（plot は `plots/*.svg`）

> このドキュメントは上から順に読めば理解できる構成です。まず **§1 結論** と **§2 全体結果** で「どれがどれくらい速い／省メモリか」を掴み、**§3 計測条件**・**§4 メモリの考え方** で前提を押さえ、**§5 方式の解説** で実装、**§6〜§8** で詳細値・解析・使い分けに進みます。

---

## 目次

- [1. 結論サマリ (TL;DR)](#1-結論サマリ-tldr)
- [2. 全体結果 — full vs 3方式](#2-全体結果--full-vs-3方式)
  - [2a. q2（full 成立）](#2a-q2full-成立)
  - [2b. q4（full 不成立 → streaming 必須）](#2b-q4full-不成立--streaming-必須)
- [3. 計測条件](#3-計測条件)
- [4. メモリの考え方](#4-メモリの考え方)
  - [4a. 二層モデル（固定土台 + routed expert cache）](#4a-二層モデル固定土台--routed-expert-cache)
  - [4b. file-backed（mmap）と owned-buffer（nommap）](#4b-file-backedmmapと-owned-buffernommap)
  - [4c. マシン RAM サイズ依存性](#4c-マシン-ram-サイズ依存性本計測は-128-gib-前提)
- [5. 方式の解説](#5-方式の解説)
  - [5a. baseline](#5a-baseline-env-無--commit-cafc134)
  - [5b. residency-lru](#5b-residency-lru-commit-11f20b1)
  - [5c. nommap](#5c-nommap-commit-cae5e95)
  - [5d. full（非ストリーミング）](#5d-full非ストリーミング)
- [6. 詳細値リスト](#6-詳細値リスト)
  - [6a. Prefill スループット](#6a-prefill-スループット-ts-1000-最終-avg)
  - [6b. Generation スループット（gen tok 付き）](#6b-generation-スループットgen-tok-付き)
  - [6c. メモリ使用量（PhysMem ピーク）](#6c-メモリ使用量physmem-ピーク)
- [7. メモリ詳細解析](#7-メモリ詳細解析-vm_stat)
- [8. システムメトリクス (mactop)](#8-システムメトリクス-mactop)
  - [8a. baseline 基準チャート](#8a-baseline-基準チャート-q4-b80)
  - [8b. 方式間で差が大きい項目](#8b-方式間で差が大きい項目すべて-q4-b80)

---

## 1. 結論サマリ (TL;DR)

- **q2（weight 実体 80.76 GiB、128 GiB に収まる）は `full`（非ストリーミング＝全モデル常駐）が最速**。streaming の per-layer cache / residency 管理 / pread のオーバーヘッドが無いぶん、gen **22.80 t/s**・prefill 261 t/s。**収まるなら streaming しないのが最良**。
- streaming 3 方式の full(gen 22.80) 比 gen 劣化率: **b32 で baseline −15% / residency-lru −27% / nommap −47%**、**b80 で baseline −15% / residency-lru −25% / nommap −13%**。nommap は budget を上げると差が大きく詰まり、**q2 b80 では gen が baseline をわずかに逆転**する。
- **q4（weight 実体 153.33 GiB）は RAM 128 GiB に載らず full 不成立 → streaming 必須**。b80 では **residency-lru が最良**（prefill +16・gen +0.81 vs baseline）。
- **軸は「箱に収まるか」**: baseline / residency-lru はモデルを OS page cache に載せて速いが、これは **128 GiB を前提**にした速さ（q2 は 80.76 GiB 全体が載る／q4 は 153.33 GiB のうち RAM に載りきる ~116 GiB で頭打ちし、残りは evict↔re-pagein）。`nommap` は page cache を使わず（file-backed ≈2 GiB）、**RAM サイズにほぼ非依存**で小容量機への移植性が最も高い。

**3 方式の位置づけ**

- **baseline（commit `cafc134`）は調整不要で十分速い**（デフォルト基準線）。改善の余地は実質 2 つだけ —— ① **prefill 速度**（高 budget で baseline が劣化する領域）、② **オーバーサイズモデル（q4 など RAM 超）処理時のメモリ管理**。
- **residency-lru（commit `11f20b1`）** … ① を狙う改善。prefill を高 budget で底上げ（+15〜+31）。代償は q2 gen がやや baseline に劣る。
- **nommap（commit `cae5e95`）** … ② を狙う改善。mmap を排除して OS page cache を使わないため RAM フットプリント最小・OOM 回避が堅牢で、**RAM サイズに非依存で安定動作**。代償は **OS キャッシュの支援を受けられず低〜中 budget で遅い**。

**buffer / copy 早見表** — 3方式が routed expert をどう供給するか。**zero-copy（mmap ページを直接 MTLBuffer で包み memcpy しない）なのは residency-lru の gen 経路だけ**で、他は pread でコピーする。

| 経路 | バッファ確保 | memcpy | backing | residency 管理 |
|---|---|---|---|---|
| baseline | streaming expert cache: cache miss を `newBufferWithLength` + **pread でコピー** (mlock slab)。model は pread 供給元として mmap | **する** | owned buffer (expert) + mmap (page cache = pread 源) | **decaying route-hotness + clock eviction**（進化版 LRU, budget bound） |
| residency-lru **gen** | selected expert **のみ** per-expert **noCopy view** (mmap 直包み) | **しない** | mmap (page cache) | byte budget の **per-expert residency LRU で明示 bound** |
| residency-lru **prefill** | pread cache slab を `newBufferWithLength` 確保 + **F_NOCACHE pread でコピー** | **する** | owned buffer (page cache 非汚染) | (1+prepare_ahead) 層 double-buffer + DONTNEED |
| nommap | owned buffer を `newBufferWithLength` + **F_NOCACHE pread でコピー** | **する** | owned buffer (**mmap も page cache も無し**) | whole-tensor double-buffer (prefetch overlap) |

- **baseline も residency-lru も「専用 cache + 進化した eviction」を持つ点は共通**。違いは: baseline は expert を **pread でコピー**して mlock slab に載せ **decaying-hotness/clock** で evict、residency-lru-gen は expert を **zero-copy mmap view** のまま **per-expert LRU** で wire（コピーも slab も持たない）。
- **nommap だけが mmap を一切使わず**、全 weights を owned buffer へ pread コピーするため file-backed が増えない（§7 の最大の差別化点）。

**128 GiB マシン・q2 での実践指針**

- 純粋な最速は `full`（gen 22.80）。ただし全モデル wired で起動 ~28s、他プロセスと RAM を奪い合う。
- **大きな ctx を取りたい／IDE 等を同時起動したいなら `baseline` の b32 が実用的なスイートスポット**: prefill **249** / gen **19.3** で、**b80（prefill 214）より速く prefill 劣化が無い**うえ used 124 GiB に収まり RAM に余裕が残る。ただし budget を b64/b80 に上げると cache slab が file-backed page cache を削り（82→63 GiB）re-pagein で prefill が劣化するため、**上げ過ぎないのがコツ**。
- **`nommap` は budget 指定どおりの性能**（指定した実 wired RAM がそのまま効く）。file-backed に依存しないので動作が安定し、RAM サイズにも非依存。低 budget の gen は弱いが堅牢。

各方式の実装は §5、数値の根拠は §6/§8、RAM 依存性は §4c。

---

## 2. 全体結果 — full vs 3方式

非ストリーミング `full`（全モデル resident）が成立するのは **RAM に収まる q2 のみ**。q4 は weight 実体 **153.33 GiB** で RAM 128 GiB を大きく超えるため resident に載らず full は N/A、streaming が必須。

> モデル weight 実体（`ds4 --inspect` の file size）: **q2 = 80.76 GiB / q4 = 153.33 GiB**（mmap span 実測 82697 MiB とも一致）。両者の差は routed expert の量子化だけ（q2: iq2_xxs 44.34 + q2_k 28.22 = 72.6 GiB → q4: q4_k 145.12 GiB、非 routed は f16 2.04 + q8_0 6.15 GiB で共通）。後述の file-backed ピーク「q4 ≈116 GiB」（§7c）はモデルサイズではなく、**RAM に載りきった page cache の頭打ち値**（153.33 GiB は全部は載らない）。

gen token 数（`gen tok`）= 各 run の **最長 gen 系列の最終 gen token 数**。walltime は agentic セッションの実時間で、生成量が run ごとに変動するため厳密な等価比較ではない（速度の真値は prefill/gen t/s 側）。

### 2a. q2（full 成立）

streaming 系は b32 / b80 を並記、full は budget 非依存（1 行）。

| run (q2) | prefill t/s | gen t/s | gen tok | walltime | used / unused / compressor |
|---|---|---|---|---|---|
| **full (非stream)** | **261.0** | **22.80** | 2438 | **2m43s** | 103 / 24.0 / 4.2G |
| baseline (b32) | 249.3 | 19.29 | 2112 | 3m10s | 124 / 2.9 / 0.9G |
| baseline (b80) | 214.2 | 19.37 | 2459 | 4m02s | 127 / 0.1 / 5.0G |
| residency-lru (b32) | 243.9 | 16.58 | 2150 | 3m40s | 103 / 24.0 / 0.0G |
| residency-lru (b80) | 245.0 | 17.13 | 1844 | 3m11s | 103 / 24.0 / 0.0G |
| nommap (b32) | 214.6 | 12.16 | 1728 | 3m51s | 61 / 67.0 / 0.0G |
| nommap (b80) | 215.2 | 19.92 | 2187 | 3m32s | 100 / 27.0 / 0.0G |

- **q2 は `full` が全項目で最速**。gen **22.80** に対する full 比 gen 劣化は **b32: baseline −15% / residency-lru −27% / nommap −47%**、**b80: baseline −15% / residency-lru −25% / nommap −13%**。prefill は full 261 比で b32 baseline −4.5% / residency-lru −6.6% / nommap −17.8%。
- **nommap は b80 で大きく改善**: gen が b32 の 12.16 → b80 19.92（full 比 −47% → −13%）まで詰まり、baseline を逆転。budget を積めば cache ヒットが増え full に近づく。
- 起動時 residency 確保 ~28s は walltime 未計上の一回コスト。

![q2 full throughput](plots/q2-full-opencode-tok.svg)
![q2 baseline b32 throughput](plots/q2-baseline-opencode-b32768-tok.svg)

### 2b. q4（full 不成立 → streaming 必須）

full は載らないため N/A。b32 / b80 を並記。

| run (q4) | prefill t/s | gen t/s | gen tok | walltime | used / unused / compressor |
|---|---|---|---|---|---|
| full (非stream) | — | — | — | — | **載らない (weight 153.33 GiB > RAM 128, OOM)** |
| baseline (b32) | 120.6 | 8.98 | 2328 | 6m58s | 127 / 0.1 / 4.9G |
| baseline (b80) | 113.2 | 11.00 | 1767 | 5m25s | 127 / 0.1 / 5.3G |
| residency-lru (b32) | **136.0** | 9.62 | 2872 | 7m38s | 127 / 0.1 / 5.0G |
| residency-lru (b80) | **129.2** | **11.81** | 2644 | 6m15s | 127 / 0.1 / 5.0G |
| nommap (b32) | 117.5 | 3.60 | 2601 | 15m20s | 63 / 64.0 / 0.0G |
| nommap (b80) | 126.7 | 7.59 | 2598 | 7m22s | 106 / 21.0 / 0.0G |

- **q4 は full 不成立 → streaming 必須**。**t/s は residency-lru が最良**（b32 prefill +15.4・gen +0.64、b80 prefill +16.0・gen +0.81 vs baseline）。nommap は prefill こそ近いが gen が弱く、特に b32 gen 3.60（baseline 8.98 比 −5.4）と苦しい。
- **walltime caveat**: t/s と walltime の順位は一致しない。例えば baseline b80 が 5m25s と短く見えるのは **生成量の差**（baseline 1767 tok vs residency-lru 2644 tok）。nommap b32 の 15m20s も gen 3.60 t/s の遅さ × 2601 tok 生成の積。gen tok 列がこれを裏付ける（速度の真値は t/s 側）。

![q4 baseline b80 throughput](plots/q4-baseline-opencode-b81920-tok.svg)
![q4 residency-lru b80 throughput](plots/q4-residency-lru-opencode-b81920-tok.svg)

---

## 3. 計測条件

- **共通起動パラメータ**: 3方式とも `--ssd-streaming --ssd-streaming-cache-experts ${BUDGET}GB` で起動し、SSD streaming (full model residency/warmup skip) は共通。env で切替えているのは **expert の residency 実装だけ**。
- **3 strategy** (`run.sh` の env で切替):
  - `baseline` … env 無し = **SSD streaming のデフォルト residency 経路**（mmap zero-copy/no-mmap いずれも無効）
  - `residency-lru` … `DS4_METAL_ENABLE_STREAMING_RESIDENCY_LRU=1`（selected-expert zero-copy mmap view + residency LRU + prefill/gen budget 動的切替）
  - `nommap` … `DS4_METAL_ENABLE_STREAMING_NO_MMAP=1`（mmap せず no-cache fd で wire）
  - `full` … **`--ssd-streaming` を使わない**非ストリーミング基準。全モデルを resident（q2 80.76 GiB mmap、起動時に residency 確保 ~28s）。q2 のみ計測・budget 非依存（streaming cache を持たない）
- **2 quant**: q2 (q2-pro IQ2XXS+w2Q2K ≈6.75 MiB/expert) / q4 (Q4K ≈13.50 MiB/expert)。attn/shared/output は両 quant とも Q8。
- **5 budget** (`--ssd-streaming-cache-experts`, GB): 8 / 16 / 32 / **64** / 80
- **ワークロード**: opencode で minesweeper.html を生成する agentic セッション（同一プロンプト）
- **代表値**: **prefill t/s** = 最大 prefill (TOOLS 10,656 tok) の **100.0% 時点の最終 cumulative avg**、**gen t/s** = 最長 gen 系列の最終 avg、**gen tok** = その最長 gen 系列の最終 token 数
- **メモリ**: `top -l` の PhysMem ピーク (used/wired/compressor) と最小 unused、`vm_stat` の累積/瞬間値
- **コミット**: baseline `cafc134` / nommap `cae5e95` / residency-lru `11f20b1`

> 実装の背景: nommap prefill overlap ([[project_ds4_nommap_prefill_overlap]]) + nommap file-backed leak 対策 ([[project_ds4_nommap_filebacked_leak]]) + residency-lru budget 動的切替 ([[project_ds4_zerocopy_budget_switch]]) を含む。

---

## 4. メモリの考え方

3 方式は **routed expert を SSD からどう供給するか**だけが違い、固定土台（非 routed）と KV は共通。なぜ方式によってメモリ挙動が大きく変わるのか、まず二つの観点を押さえる。

### 4a. 二層モデル（固定土台 + routed expert cache）

DSV4-Flash は「**固定土台（non-routed）**」+「**ストリーミングされる routed expert**」の二層。量子化で変わるのは routed expert だけで、attn 等の固定分は **q2/q4 で完全に同一**（モデル名 `Q8Attn-Q8Shared-Q8Out` の通り attn/shared/output は両方 Q8）。

**固定確保分（q2 = q4 共通、budget 非依存）**

| 項目 | サイズ | 中身・量子化 |
|---|---|---|
| non-routed views | **≈8.2 GiB** | attn proj(Q8) + shared expert(Q8) + output(Q8) + 各 norm + MLA HC/Compressor/Indexer(F16) |
| token embedding | **≈1.0 GiB** | 起動時に最初にマップする 1 span |
| context / KV buffers | **≈1.9 GiB** | ctx=100000, raw_kv 4352 + compressed_kv 25002 rows（起動ログ `context buffers 1933.21 MiB`） |
| **固定合計** | **≈11.1 GiB** | **q2 = q4** |

SSD streaming は full residency を skip しても、この固定分（毎トークン全層で必ず使う）は常駐させる。

**可変分（routed expert = cache budget で載る分）**

DSV4-Flash の routed expert は **43 層 × 256 個/層 = 計 11,008 個**、token あたり **6 個を活性化**（`ds4 --inspect`: experts count=256 used=6, layers=43）。量子化で変わるのはこの expert 分だけで、固定土台（f16 2.04 + q8_0 6.15 = 8.2 GiB）は q2/q4 共通。expert weight は q2 72.6 GiB（iq2_xxs 44.34 + q2_k 28.22）→ q4 145.1 GiB（q4_k）。

| | q2 | q4 | 比 |
|---|---|---|---|
| per-expert | 6.75 MiB (IQ2XXS+w2Q2K) | 13.50 MiB (Q4K) | ×2 |
| 8 GiB budget の常駐 expert 数 | 1213 | 606 | ×2 |
| 16 GiB budget の常駐 expert 数 | 2427 | 1213 | ×2 |
| prefill pread cache (512 exp 予約, residency-lru のみ) | 3.38 GiB | 6.75 GiB | ×2 |

（起動ログ `metal SSD streaming cache budget X GiB / Y MiB per expert = N experts` より）

これが §6b の gen 差の根本: q2 は 1 expert が q4 の半分なので**同じ budget で倍の expert を常駐**でき、低〜中 budget でも gen が速い。q4 は固定 ≈11.1 GiB を引くと実効 cache が薄く、budget を増やさないと gen が伸びない（q4 8GiB=8.9 → 80GiB=11.8 t/s）。

### 4b. file-backed（mmap）と owned-buffer（nommap）

- `baseline` / `residency-lru` は **モデルを mmap** する。expert 再読込が OS の **page cache（file-backed）ヒット**になり速いが、その代償でモデルが file-backed に張り付く（q2 は 80.76 GiB 全体／q4 は 153.33 GiB のうち RAM に載りきる ~116 GiB で頭打ち、残りは evict↔re-pagein）。
- `nommap` は **mmap せず F_NOCACHE fd で pread** し owned buffer に積む。page cache を汚さない（file-backed ≈2 GiB）ため RAM フットプリント最小だが、cache ヒットの恩恵を受けられず低 budget では遅い。

下は同じ q2 b32 の vm_stat 推移を **baseline / residency-lru / nommap の 3 列横並び**で示す。baseline・residency-lru は file-backed が ~82 GiB に張り付く一方、nommap は極小（≈2 GiB）に保たれる。

| baseline | residency-lru | nommap |
|---|---|---|
| <img src="plots/q2-baseline-opencode-b32768-vmstat-mem.svg" width="100%"/> | <img src="plots/q2-residency-lru-opencode-b32768-vmstat-mem.svg" width="100%"/> | <img src="plots/q2-nommap-opencode-b32768-vmstat-mem.svg" width="100%"/> |

### 4c. マシン RAM サイズ依存性（本計測は 128 GiB 前提）

- **baseline / residency-lru が速いのは、モデルが OS の file-backed (page cache) に載っているから**（q2 は 80.76 GiB 全体、q4 は 153.33 GiB のうち ~116 GiB を page cache が保持し、expert 再読込が cache hit になる）。**128 GiB 未満のマシンでは page cache がモデルを保持しきれず、evict↔re-pagein が常態化して性能が大きく落ちる**（この計測の数値はそのまま出ない）。
- **`nommap` は file-backed を使わない**（F_NOCACHE pread、file-backed ≈2 GiB）ので、**性能はマシン RAM サイズにほぼ非依存**。効くのは `--ssd-streaming-cache-experts` の budget（= 実 wired RAM）だけで、64 GiB 機などでも budget 相当の挙動がそのまま再現する。tight-RAM／小容量機での移植性は nommap が最も高い。
- → **「収まる箱」では baseline/residency-lru（や q2 の full）、「収まらない／小さい箱」では nommap**、という軸が RAM サイズで効いてくる。

---

## 5. 方式の解説

各方式は **routed expert の供給方法**だけが異なる。以下、実装ポイントと baseline からの変更点を整理する。

### 5a. baseline (env 無し / commit `cafc134`)

routed expert を専用 **streaming expert cache** で供給する。model は非 routed 重み + expert pread の供給元として mmap。

- **expert 供給**: cache miss → `newBufferWithLength` の owned Metal buffer に **pread でコピー**（mlock slab）。budget (`--ssd-streaming-cache-experts`) で容量 bound。
- **eviction = 進化版 LRU**: 単純 LRU ではなく **decaying route-hotness + clock**（`route_hotness >>= 1` の指数減衰でホットな expert を優先保持、[[project_ds4_adaptive_cache_plan]]）。prefill/gen 共通でこの cache を引く。
- **利点**: 汎用・低 budget で速い・OOM し難い。
- **欠点**: pread 供給元の model が page cache に載る (file-backed q2≈82 / q4≈116 GiB; q4 はモデル 153.33 GiB のうち RAM に載りきった分)。高 budget では cache slab と page cache が RAM を奪い合い、file-backed の evict↔re-pagein 連鎖 → compressor 圧と prefill 劣化（§6/§7）。

### 5b. residency-lru (commit `11f20b1`)

**prefill と gen で機構が異なる**点が要注意。residency LRU は **gen 専用**（機能名はこの gen 機構に由来）。

- **gen 方式**: `ds4_gpu_routed_moe_one_tensor_residency_lru`。router が選んだ expert **だけ**を zero-copy mmap view で包み、**per-expert residency LRU**（byte budget で bound）に wire。gen の working set がトークン跨ぎで resident に留まりつつ RAM は bound。requestResidency は per-layer dispatch で defer し token ごとに 1 回 flush (`residency_lru_commit_pending`)。非 routed は static-decode map で一度だけ pin。
- **prefill 方式**: **residency LRU を使わない**。prefill 開始時に LRU を drop し、selected expert を専用 **F_NOCACHE fd の (1+prepare_ahead) 層 double-buffer**で pread (`set_model_nocache_fd`)。各層の prefill 済 routed page は **encode front の 2 層後ろで MADV_DONTNEED** し、sweep の file-backed footprint を ~2-3 層に抑える。LRU miss 時は F_RDADVISE で cold range を async 先読み (`ds4_prefetch_expert_range`)。prefill cache slab は gen 再開時に OS へ返却 (`ds4_gpu_stream_expert_cache_release`)。
- **budget 動的切替**: prefill budget = total − (1+prepare_ahead) 層分の pread cache 予約（≈512 expert）、gen budget = full。`enter_prefill`/`enter_gen` で切替。

> **baseline からの変更点 / 改善要因**: §6a の **residency-lru prefill が baseline を上回る要因は gen の residency LRU ではなく、prefill 側の F_NOCACHE pread double-buffer + MADV_DONTNEED trailing** による page-cache 非汚染である。baseline は通常 fd の pread で expert を読むため、prefill sweep が pread 供給元の model ページを page cache に呼び込み（高 budget では cache slab と競合して thrash）。

**実装上の注意**

- **reduced budget を gen に持ち越すと decode が静かに激遅化**する。`enter_prefill` は pread cache 用に LRU を reduced bound まで evict した**直後に full budget へ即復元**する。復元を `enter_gen` 側に置くと、`enter_gen` が呼ばれ損ねた場合に budget が reduced のまま stranded → gen LRU が cap され decode が激遅に。`enter_gen` は冪等な backstop に留める。
- **prefill 開始時に LRU を drop** し、gen が demand-true な set を再構築する（古い prefill warm を引きずらない）。
- **MADV_DONTNEED は encode front の 2 層後ろ**へ。近すぎると再 fault、遠すぎると footprint 増。
- noCopy MTLBuffer は length 分が wired 計上される（[[feedback_ds4_metal_nocopy_double_counts]] 系の注意）。requestResidency は token 単位で batch flush しないと per-command-buffer の residency 検証コストが嵩む。

**評価（vs baseline）— 速度寄りの改善**（位置づけは §1）
- メリット: prefill が高 budget で baseline を大きく上回る（q2 64/80 で +30.8〜+30.9、q4 32–80 で +15.4〜+16.0）。q2 で used を 103 GiB に bound し unused ~24 GiB を残す。
- デメリット: q2 gen は全 budget で下回る（−0.7〜−3.0）。q4 は used≈127 GiB に張り付き（mmap 系の宿命）、メモリ削減は q2 中心。実装が最も複雑。（例外的に q4 gen のみ中〜高 budget で baseline を僅かに上回る: 32 +0.6 / 64 +2.5）

### 5c. nommap (commit `cae5e95`)

モデルを **mmap せず**、専用の **F_NOCACHE fd** から weights を pread して Metal-owned Shared buffer に積む (`ds4_gpu_stream_nommap_*`)。file-backed メモリを増やさず page-cache 蓄積→compressor/OOM を回避するのが狙い（baseline からの最大の差別化点）。

- **prefill 方式**: routed prefill overlap。`ds4_gpu_stream_nommap_routed_prefetch/_ensure` が **層 il+1 の gate/up/down expert テンソル全体を background で先読み**（per-parity double-buffer）。F_NOCACHE SSD read を層 il の GEMM と overlap させ、前段で stall しないようにする ([[project_ds4_nommap_prefill_overlap]])。
- **gen 方式**: routed mv-addr/GEMM 経路に F_NOCACHE pread で供給（whole-tensor double-buffer）。Flash Q4/Q2 対応。
- 非 routed は persist tier (one-shot owned buffer) で起動時に一括 pread。

**実装上の注意**

- **F_NOCACHE は page-aligned I/O でしか効かない**。`ds4_gpu_model_read_into` は要求窓を page 境界に広げ **1 回の pread で読む**。unaligned な短絡 read 継続を挟むと kernel が暗黙にキャッシュを再有効化し F_NOCACHE が無効化される（page-aligned bounce buffer 経由、32 MiB ceiling）。
- **F_NOCACHE だけでは file-backed が増える**。kernel の自動 read-ahead は F_NOCACHE を無視して先読みページを file-backed に載せるため、model fd の **F_RDAHEAD を off** にする (`ds4_gpu_stream_expert_readahead_enabled` は no-mmap 時 false)。これで gen 時の file-backed 増加が約 1/3。
- **prefill double-buffer は明示解放が必須**。`ds4_gpu_stream_nommap_routed_release` を end-of-prompt で呼び double-buffer を解放しないと、gen 中も wired のまま残る。
- KV / staged 読み込みも stdio 経由だと file-backed に乗るため bypass が要る ([[project_ds4_nommap_filebacked_leak]])。残留は prefill cold-load バーストの reclaimable cache（macOS inherent、fd drop 手段がなく diminishing returns）。本 run の file-backed ≈1.3–2.8 GiB / pageins ≈1 GiB はこの対策が効いた結果（§6/§7）。

**評価（vs baseline）— メモリ寄りの改善**（位置づけは §1）
- メリット: RAM フットプリント最小（file-backed ≈1.3–2.8 GiB、compressor 完全 0、unused 最大 91 GiB）。faults/pageins も小さく swapout 0・OOM 回避が最も堅牢で tight-memory box / 高 ctx 向き。性能がマシン RAM サイズに非依存で移植性が高い（§4c）。高 budget では gen が baseline に追いつく（q2 80GiB +0.55）。
- デメリット: gen が低〜中 budget で大幅に遅い（q2 8GiB −10.4 / q4 8GiB −6.7）。prefill も低〜中 budget で baseline 以下（q2 8–32GiB −34〜−42）。速度を出すのに高 budget = より多い RAM が要る。

### 5d. full（非ストリーミング）

streaming を一切使わず**全モデルを mmap + residency 確保で常駐**させる経路（`q4-experiment.sh full`）。expert の cache/LRU/pread を持たず、全 expert が常時 resident。モデルが RAM に収まる場合（q2 80.76 GiB ≤ 128 GiB）のみ成立し、**streaming オーバーヘッドが無いぶん最速**（§2）。q4（weight 153.33 GiB, RAM 超）では成立しないため streaming が要る。起動時に residency 確保コスト（~28s）が一度かかる。

---

## 6. 詳細値リスト

### 6a. Prefill スループット (t/s, 100.0% 最終 avg)

baseline 基準（括弧 = Δ vs baseline）。

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

### 6b. Generation スループット（gen tok 付き）

t/s = 最長 gen 系列 avg、gen tok = その系列の最終 token 数。括弧 = Δ t/s vs baseline。

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

> gen tok は agentic セッションの生成量で run ごとに変動する（同一プロンプトでもエージェントの分岐で変わる）。t/s が速度の真値、gen tok は walltime の解釈に使う補助値。

### 6c. メモリ使用量（PhysMem ピーク）

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

## 7. メモリ詳細解析 (`vm_stat`)

> 瞬間ピークは各サンプル行の最大/最小（信頼度高）。累積/イベントは vm_stat の各区間行を合計した値（セッション先頭の since-boot 行は除外）。フェーズ間の vm_stat 停止区間（gap）は未計上のため下限寄り。傾向比較用とする。page=16 KiB。

### 7a. q2 — 瞬間ピーク (GiB)

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

### 7b. q2 — 累積/イベント (run 全体)

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

### 7c. q4 — 瞬間ピーク (GiB)

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

### 7d. q4 — 累積/イベント (run 全体)

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

q4 b80 の vm_stat 推移を **baseline / residency-lru / nommap の 3 列横並び**で示す（上段=メモリ内訳 vmstat-mem、下段=イベント vmstat-events）。mmap 系 2 方式は file-backed が ~113–116 GiB で張り付くのに対し、nommap は ~2 GiB に保たれる（used は budget 相当のみ）。

| baseline | residency-lru | nommap |
|---|---|---|
| <img src="plots/q4-baseline-opencode-b81920-vmstat-mem.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-vmstat-mem.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-vmstat-mem.svg" width="100%"/> |
| <img src="plots/q4-baseline-opencode-b81920-vmstat-events.svg" width="100%"/> | <img src="plots/q4-residency-lru-opencode-b81920-vmstat-events.svg" width="100%"/> | <img src="plots/q4-nommap-opencode-b81920-vmstat-events.svg" width="100%"/> |

### 7e. メモリ所見

**file-backed が baseline と 2 改善案を最も鋭く分ける**
- `baseline` / `residency-lru` (mmap 系) は file-backed が **q2≈80–82 GiB / q4≈110–116 GiB** に張り付く → モデル全体を mmap している。q4 はモデルが RAM 超なので page cache が上限まで埋まる。baseline q2 は 64/80 GiB で file-backed が 63 GiB に落ちるが、これは wired (76–84 GiB) と compressor (8 GiB) に押し出された結果。
- `nommap` は file-backed が **≈1.3–2.8 GiB** と極小 → モデルを mmap せず no-cache fd の pread で読むため、OS page cache にモデルが乗らない（設計通り＝baseline からの最大の差別化点）。

**faults: nommap も mmap 系と同オーダー**
- nommap の累積 faults は q4 で **3.5–6.8M**。F_NOCACHE owned-view + prefill prefetch overlap により、no-mmap でも minor fault が mmap 系 (1〜6M) と同オーダーに収まる。
- `residency-lru`/`baseline` は q2/q4 とも **1.2–6M faults**。budget が大きいほど faults も漸増（常駐 expert の touch 増）。

**pageins (実ディスク I/O): mmap 系が多い / nommap は極小**
- `residency-lru`/`baseline` は q4 で **660–1350 GiB** の pagein → file-backed ページが evict と re-pagein を繰り返す（モデル>RAM の宿命）。q2 は ≈76 GiB と少ない（モデル 80.76 GiB が RAM に収まり再 pagein がほぼ不要）。
- `nommap` は q2/q4 とも pageins **≈0.7–1.6 GiB** と極小 → pread が OS の pagein カウンタを経由しない独自経路で、かつ file-backed leak 対策で reclaimable cache も抑えたため。
- pageout は mmap 系で数百 pages（≈ 数 MiB）の微小発生、nommap は 0。

**compressor**
- `nommap` は compressed/comprs/dcomprs すべて **0** → page-cache bypass で compressor を完全回避（memory の "compressor=0=OOM回避" と整合）。
- `residency-lru`/`baseline` は q4 で compressed **≈8.5–9.5 GiB** 常駐、comprs/dcomprs が数十万〜百万回オーダー。anonymous な KV/context が圧縮対象。`residency-lru` q2 は compressor を実質 0 に抑える（residency LRU で anonymous を bound）。

**swapout**
- 大半の run で **swapout=0**。微小発生は baseline q4 低 budget (8GiB=96 / 16GiB=132 pages) と residency-lru q4 (16GiB=36 / 64GiB=184 pages)、baseline q2 80GiB=40 のみ。いずれも MiB 規模で OOM・大規模 swap は無し。

---

## 8. システムメトリクス (mactop)

`*-mactop.log`（mactop の JSON サンプル）から電力・周波数・DRAM 帯域・ディスク read・温度を集計。代表 run（q2 b32 / q4 b80）。avg/max は run 全体（title gen + agent→write + tool-result の全フェーズ）。

| run | 総電力 W (avg/max) | GPU電力 W (avg) | DRAM BW GB/s (avg) | disk read MB/s (avg/max) | GPU温度 ℃ (max) | GPU稼働 % (avg) |
|---|---|---|---|---|---|---|
| q2 baseline b32 | 74.3 / 135.4 | 25.5 | 116.6 | 263 / 3444 | 88.1 | 86.3 |
| q2 residency-lru b32 | 63.9 / 88.8 | 21.3 | 109.3 | 223 / 3599 | 87.3 | 87.2 |
| q2 nommap b32 | 54.4 / 123.1 | 14.8 | 120.0 | **2299** / 5013 | 100.8 | 80.6 |
| q4 baseline b80 | 53.5 / 87.9 | 11.6 | **56.8** | 2943 / 6155 | 97.3 | 59.2 |
| q4 residency-lru b80 | 55.7 / 140.1 | 15.1 | 107.6 | 2147 / 6094 | 94.6 | 79.7 |
| q4 nommap b80 | 36.3 / 81.6 | 7.9 | 110.0 | 2597 / 6080 | 99.4 | 64.4 |

**所見**
- **ディスク read が方式を最も鋭く分ける**: q2 で nommap は avg **2299 MB/s**（baseline 263 の約 9 倍）。mmap せず毎回 F_NOCACHE pread するため常時 SSD を叩く。逆に baseline/residency-lru は q2 ではモデルが page cache に載るため read が小さい。q4 は全方式とも read 大（baseline/residency-lru もモデル >RAM で re-pagein、nommap は設計上 pread）。
- **総電力は nommap が一貫して低い**（q2 74→54W、q4 54→36W）。GPU 電力・GPU 稼働率が低めで、SSD I/O 待ちで GPU が遊ぶぶん消費電力が下がる（速度とのトレードオフ）。
- **DRAM BW は q4 baseline だけ低い（56.8 GB/s）**。他方式が ~110 GB/s なのに対し、baseline q4 は file-backed の evict↔re-pagein で stall し、実効的に帯域を使い切れていない（§7e の pagein 過多と整合）。
- 温度・周波数は方式差が小さい（GPU温度 max 88–101℃、p-cluster freq avg ~3.4–3.5 GHz でほぼ横並び）。thermal_state は全 run Normal。

### 8a. q4 b80 全方式チャート（横並び比較）

q4 b80（モデル 153.33 GiB > RAM 128 GiB、streaming 必須）の全チャートを **baseline / residency-lru / nommap の 3 列横並び**で示す。各行が同一チャート種・同一 x 軸（10分 / 1分刻み）なので、方式差を直接見比べられる。チャート種は各画像のタイトル（`q4-<方式>-…: <種別>`）で判別。並びは tok / mem / events / cpu-gpu / freq / power / dram-bw / disk / temp の順。考察は §8b。

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

### 8b. 方式間で差が大きい項目（すべて q4 b80）

q4 b80 はモデル 153.33 GiB > RAM 128 GiB で、3方式の性格差が最も出る条件。チャートは §8a の横並びを参照し、ここでは数値と考察をまとめる。

| 項目 (q4 b80) | baseline | residency-lru | nommap |
|---|---|---|---|
| prefill / gen t/s | 113.2 / 11.00 | **129.2 / 11.81** | 126.7 / **7.59** |
| disk read MB/s (avg) | **2943** | 2147 | 2597 |
| 総電力 W (avg) | 53.5 | 55.7 | **36.3** |
| DRAM BW GB/s (avg) | **56.8** | 107.6 | 110.0 |
| GPU稼働 % (avg) | **59.2** | **79.7** | 64.4 |
| GPU freq MHz (avg) | 1355 | 1300 | **1068** |
| CPU usage % (avg) | 4.8 | 5.6 | 3.0 |

#### ① ディスク read — q4 は全方式とも大きいが機序が違う

- **nommap**: file-backed（page cache）を使わない設計。cache（budget 内の owned buffer）に無い expert は **毎回 F_NOCACHE で SSD から pread** → 恒常的に read（2597 MB/s）。
- **baseline / residency-lru**: モデル 153 GiB > RAM なので mmap ページが全部は載らず、**evict↔re-pagein**（これも実 disk read）が発生。baseline は thrash が最激で read 最大（**2943**）、residency-lru は prefill の MADV_DONTNEED + gen の residency LRU で再 pagein を抑え 3方式中最小（2147）。
- → q2 では baseline が cache hit で read 小・nommap だけ突出だった（§8 冒頭表で q2 nommap 2299 vs baseline 263）が、**q4 ではモデルが載りきらないため baseline も re-pagein で read 大**になる。

#### ② 総電力 — nommap が最も低い

- **nommap 36.3W が最低**。SSD I/O bound で GPU/CPU のアイドルが長く、GPU稼働 64%・GPU電力 7.9W と最も低い（リソースを使い切れていない瞬間が多く平均電力が下がる）。
- **residency-lru 55.7W が最高**。compute が連続し GPU稼働 80%・GPU電力 15.1W で最もリソースを回す。baseline は thrash で中間（53.5、GPU 59%）。

#### ③ DRAM 帯域と演算リソース — 高 BW でも gen t/s が高いとは限らない

DRAM BW の絶対値は gen t/s と比例しない。**nommap は BW 最高（110）なのに gen 最低（7.59）**、逆に **baseline は BW 最低（56.8）でも gen 11.00** を出す。BW の「中身」が方式で違うため。

- **nommap（高 BW・低 t/s）**: BW の多くが **SSD→RAM の pread コピー**（cache に無い expert を毎回 owned buffer へ書き込む）で、これはデータ移動であって MoE GEMM の compute ではない。GPU は pread 待ちで断続稼働し、**GPU freq 1068 MHz が最低・GPU稼働 64%** → 演算が前に進まず gen 最低。
- **residency-lru（高 BW・高 t/s）**: 高 BW が **zero-copy view の weight 読み**に使われ、thrash も無いので **GPU稼働 80% が最高**・freq 1300 で compute が連続 → gen 最速。
- **baseline（低 BW・中 t/s）**: BW が低いのは、thrash の page fault / SSD I/O 待ちで **メモリを触る時間そのものが短い**ため（**GPU稼働 59% が最低**）。一方 cache hit した瞬間は GPU が mmap ページを直読でき、freq は 1355 と高いので、stall していない区間の効率で gen 11.00 を保つ。
- **CPU usage はどの方式も低い（3–6%）**: pread 発行と GEMM ディスパッチが主で、重い処理は GPU 側。nommap が CPU 3.0% と最低なのも I/O 待ちを反映。p-cluster freq は ~3.5 GHz で横並び（CPU は律速でない）。
- → gen t/s を決めるのは **GPU が weight を待たずに連続して MoE を回せるか（GPU稼働率・GPU freq）** であって、DRAM BW の絶対値ではない。BW が高くてもその実体が I/O コピーなら速度に寄与しない。
