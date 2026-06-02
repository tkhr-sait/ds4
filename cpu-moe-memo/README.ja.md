# cpu-moe-memo

Apple M4 Max / 128 GiB Mac で `--cpu-moe` + `--prefill-metal-phases` を使い
153 GiB の DeepSeek-V4-Flash モデルを動かした際の実測ログ、スクリーンショット、解析。
`<date>/` 形式の各ディレクトリが 1 回のキャプチャセッションで、それぞれ
スクショ・プロット・所見をまとめた独自の README を持つ。

English version: [README.md](README.md)

## レイアウト

- `01-cpumoe-baseline/` — 最初のキャプチャセッション: chat (1 リクエストで
  長文生成) と opencode (複数ターンの agent フロー)。同ディレクトリの
  `README.md` / `README.ja.md` を参照。
- `02-routed-metal-dynamic/` — `--routed-metal-dynamic` セッション: gen の routed
  expert を Metal で計算。wired budget スイープと高出力 vs 通常モードの比較。
  **性能に関する所見もこちらにまとめてある** — prefetch (`F_RDADVISE`)、
  CPU vs GPU の熱特性、改善の余地。同ディレクトリの `README.md` / `README.ja.md` を参照。
- `scripts/`
  - `launch_tmux.sh` — 各セッションのスクショと同じ 5 ペイン tmux レイアウトを起動
  - `make_plots.py` — セッションログから SVG プロット (`*-tok.svg` /
    `*-iostat-*.svg` / `*-vmstat-mem.svg` / `*-vmstat-events.svg`) を生成

## 前提ツール

Apple Silicon の macOS 前提。スクリプトが使うツールを入れる:

```sh
brew install tmux mactop          # tmux レイアウト + mactop テレメトリ
```

- **macOS に最初から入っている**(不要): `vm_stat` / `curl` / `python3`(Xcode
  Command Line Tools = `xcode-select --install`)。プロット系と chat / verify
  クライアントは Python 標準ライブラリのみなので **`pip` パッケージは不要**。
- **ワークロードクライアント**: 既定の `opencode` エージェントフローは、このサーバを
  指すよう設定した `opencode` CLI が必要 — opencode の model / provider / endpoint 設定は
  [`../README.md`](../README.md)(ds4 リポジトリ root)を参照。または `CLIENT=chat`
  (curl + python3 のみ、opencode 不要)。
- **ビルド済みバイナリ**: リポジトリ root の `ds4-server` / `ds4-bench`。
- `run.sh` と `launch_tmux.sh` は起動時チェックを行い、必須ツールが無ければ早期に中断する。

## セッションを回す

### Option 1: 5 ペイン tmux レイアウト (vm_stat / ds4-server / mactop / 空 shell / iostat)

```sh
cd <session-dir>          # 例: cpu-moe-memo/<date>。vm_stat / iostat ログがここに落ちる
TEST_PREFIX=opencode ../scripts/launch_tmux.sh
```

左下ペインはコマンド送信なしの空 shell のままなので、好きなクライアント
(opencode, curl, ds4 cli, …) を起動して使う。他の 4 ペインは自動的に
それぞれのコマンドで起動する。

終了は `tmux kill-session -t ds4mon`。

### Option 2: コマンドを個別に起動

```sh
TEST_PREFIX=chat

vm_stat 30 2>&1 | tee ${TEST_PREFIX}-vm_stat.log
iostat 30 2>&1 | tee ${TEST_PREFIX}-iostat.log
DS4_PREFILL_METAL_PHASES_MIN_TOKENS=0 ./ds4-server \
  --prefill-metal-phases auto --ctx 100000 \
  --kv-disk-dir /tmp/${TEST_PREFIX}-ds4-kv --kv-disk-space-mb 8192 \
  2>&1 | tee ${TEST_PREFIX}-ds4-server.log
```

シングルターンのチャットリクエスト例 (streaming、簡易疎通確認用に残してある):

```sh
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ds4","stream":true,"messages":[{"role":"user","content":"create minesweeper. 1html file."}]}' \
  | tee /tmp/ds4_resp1.sse \
  | sed -n 's/^data: //p' \
  | grep -v '^\[DONE\]' \
  | jq -rj '.choices[0].delta.content // empty' \
  > /tmp/ds4_assistant1.txt
```

直前 assistant の返答を会話履歴に積んで follow-up ターン:

```sh
jq -n \
  --arg q1 "create minesweeper. 1html file." \
  --rawfile a1 /tmp/ds4_assistant1.txt \
  --arg q2 "A modern design and the ability to choose the difficulty level." \
  '{model:"ds4",stream:true,messages:[
    {role:"user",content:$q1},
    {role:"assistant",content:$a1},
    {role:"user",content:$q2}
  ]}' \
  | curl -sN http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' -d @-
```

## プロット生成

セッション終了後 (logger を Ctrl-C、ds4-server を停止):

```sh
python3 scripts/make_plots.py <session-dir>
```

`<session-dir>/plots/*.svg` が出力される。各セッションの README に埋め込まれている。
