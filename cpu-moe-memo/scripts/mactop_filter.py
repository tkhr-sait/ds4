#!/usr/bin/env python3
"""Trim mactop --headless output to compact NDJSON, dropping bulky fields that
make_plots.py never reads (processes / temperatures / core_usages / etc.).

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
                json.dump(filtered, sys.stdout, separators=(",", ":"))
                sys.stdout.write("\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
