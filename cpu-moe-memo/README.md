# cpu-moe-memo

Measurement logs, screenshots, and analysis from running a 153 GiB DeepSeek-V4-Flash
model on an Apple M4 Max / 128 GiB Mac with `--cpu-moe` + `--prefill-metal-phases`.
Each `<date>/` directory is one capture session and contains its own README that
walks through the screenshots, plots, and findings for that run.

Japanese version: [README.ja.md](README.ja.md)

## Layout

- `01-cpumoe-baseline/` — first capture session: chat (long single-request gen) and
  opencode (multi-turn agent flow). See its `README.md` / `README.ja.md`.
- `02-routed-metal-dynamic/` — `--routed-metal-dynamic` session: gen routed experts
  on Metal, with a wired-budget sweep and a High-Power-vs-Normal power-mode
  comparison. **Also collects the performance findings** — prefetch (`F_RDADVISE`),
  CPU-vs-GPU thermals, and where the room to improve is. See its `README.md` /
  `README.ja.md`.
- `scripts/`
  - `launch_tmux.sh` — start the standard 5-pane tmux layout that the
    screenshots in each session were taken from.
  - `make_plots.py` — render SVG plots (`*-tok.svg`, `*-iostat-*.svg`,
    `*-vmstat-mem.svg`, `*-vmstat-events.svg`) from the session's logs.

## Requirements

macOS on Apple Silicon. Install the tools the scripts use:

```sh
brew install tmux mactop          # tmux layout + mactop telemetry
```

- **Pre-installed on macOS** (no action): `vm_stat`, `curl`, `python3` (via Xcode
  Command Line Tools — `xcode-select --install`). The plot scripts and the chat /
  verify clients use the Python standard library only, so there are **no `pip`
  packages** to install.
- **Workload client**: the default `opencode` agent flow needs the `opencode` CLI
  configured to talk to this server — see [`../README.md`](../README.md) (ds4 repo
  root) for the opencode model / provider / endpoint setup. Or use `CLIENT=chat`
  (curl + python3 only, no opencode).
- **Built binaries**: `ds4-server` / `ds4-bench` at the repo root (build the ds4 repo).
- `run.sh` and `launch_tmux.sh` run a startup check and abort early if a required
  tool is missing.

## Run a session

### Option 1: 5-pane tmux layout (vm_stat / ds4-server / mactop / free shell / iostat)

```sh
cd <session-dir>          # e.g. cpu-moe-memo/<date>; vm_stat / iostat logs land here
TEST_PREFIX=opencode ../scripts/launch_tmux.sh
```

The bottom-left pane is left empty so you can launch whatever client you want
there (opencode, curl, ds4 cli, …). The other four panes auto-start with the
right commands.

Tear down with `tmux kill-session -t ds4mon`.

### Option 2: launch each command manually

```sh
TEST_PREFIX=chat

vm_stat 30 2>&1 | tee ${TEST_PREFIX}-vm_stat.log
iostat 30 2>&1 | tee ${TEST_PREFIX}-iostat.log
DS4_PREFILL_METAL_PHASES_MIN_TOKENS=0 ./ds4-server \
  --prefill-metal-phases auto --ctx 100000 \
  --kv-disk-dir /tmp/${TEST_PREFIX}-ds4-kv --kv-disk-space-mb 8192 \
  2>&1 | tee ${TEST_PREFIX}-ds4-server.log
```

Example single-turn chat request (streamed, kept around as a quick smoke test):

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

Follow-up turn that includes the assistant's first reply as conversation history:

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

## Render plots

After the session finishes (Ctrl-C the loggers, stop ds4-server):

```sh
python3 scripts/make_plots.py <session-dir>
```

This populates `<session-dir>/plots/*.svg`. Each session README embeds them.
