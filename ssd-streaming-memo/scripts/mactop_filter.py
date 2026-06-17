#!/usr/bin/env python3
"""Trim mactop --headless output to compact NDJSON, dropping bulky fields that
make_plots.py never reads (temperatures / core_usages / etc.).

``processes`` is NOT dropped: mactop reports per-process ``rss_kb`` /
``memory_percent``, which is what lets us see WHICH process (krunkit = the
Podman VM hosting opencode, vs ds4-server) holds the growing anonymous memory
that the macOS compressor feeds on.  The full list would bloat a 2h capture, so
only the top ``PROC_TOP_N`` by rss are kept, with the few fields we need.  (rss
excludes compressed pages — pair this with the top_cmprs_sampler.sh ``cmprs``
log for the compressed split.)

Usage:
    mactop --headless --count 0 -i 5000 | mactop_filter.py > mactop.log

Robust to either ``--pretty`` (whitespace+newlines) or compact mactop output:
brackets and commas around the JSON-array elements are stripped between
samples, and each sample is decoded via raw_decode so the script can run as a
streaming filter without waiting for the closing ``]``.
"""

import json
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

KEEP = {
    "timestamp",
    "soc_metrics",
    "memory",
    "net_disk",
    "cpu_usage",
    "gpu_usage",
    "gpu_metrics",
    "thermal_state",
}

# Per-process memory: keep only the biggest residents, trimmed to the fields we
# actually plot/attribute, so the NDJSON stays compact over a long session.
PROC_TOP_N = 15
PROC_FIELDS = ("pid", "command", "rss_kb", "memory_percent", "cpu_percent")


def main():
    decoder = json.JSONDecoder()
    buf = ""
    while True:
        chunk = sys.stdin.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            i = 0
            while i < len(buf) and buf[i] in " \t\r\n[],":
                i += 1
            if i:
                buf = buf[i:]
            if not buf:
                break
            try:
                obj, idx = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break  # incomplete; wait for more input
            buf = buf[idx:]
            if isinstance(obj, dict):
                filtered = {k: obj[k] for k in KEEP if k in obj}
                procs = obj.get("processes")
                if isinstance(procs, list):
                    top = sorted(
                        procs, key=lambda p: p.get("rss_kb", 0), reverse=True
                    )[:PROC_TOP_N]
                    filtered["processes"] = [
                        {k: p[k] for k in PROC_FIELDS if k in p} for p in top
                    ]
                json.dump(filtered, sys.stdout, separators=(",", ":"))
                sys.stdout.write("\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
