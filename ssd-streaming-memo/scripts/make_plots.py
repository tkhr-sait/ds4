#!/usr/bin/env python3
"""Generate SVG plots from ds4-server / iostat / vm_stat logs.

Usage:
    python3 scripts/make_plots.py <session-dir>
    python3 scripts/make_plots.py <session-dir> --prefix opencode-K8 --prefix opencode-K16

Without --prefix, the session-dir is scanned for ``<prefix>-ds4-server.log``
files and each matching prefix is processed.  This auto-detect is backwards
compatible with the original ``chat`` / ``opencode`` capture layout: those
prefixes are still picked up when present.

Outputs SVGs under <session-dir>/plots/. The helpers (nice_ceil, nice_step,
fmt_tick, frange, points_to_polyline) and SVG styling follow
/workspace/ds4/speed-bench/plot_speed.py.
"""

import argparse
import datetime as _dt
import html
import json
import math
import re
import sys
from pathlib import Path

COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#f59e0b",
    "#0891b2",
]
GRID_COLOR = "#e2e8f0"
AXIS_COLOR = "#334155"

# Default sample intervals.  06-q2/ and later use 5s for vm_stat; earlier
# captures (01-04) used 30s, so pass --vmstat-interval 30 when re-plotting
# those.  iostat is no longer captured for new runs but the parser stays for
# backward compatibility with 01-cpumoe-baseline / 05-mtp captures.
IOSTAT_INTERVAL_DEFAULT = 30.0
VMSTAT_INTERVAL_DEFAULT = 5.0

# When set (seconds), forces the x-axis tick spacing on time-formatted plots
# instead of the auto ~6-tick "nice" step.  Set from --x-tick-minutes in main().
X_TICK_SECONDS = None

# When set (seconds), a target/reference x-axis span: the axis is pinned to at
# least this width (so short runs render with right-side whitespace), but it
# auto-extends to the real data span when a run overruns the target.  Set from
# --x-target-minutes in main().
X_TARGET_SECONDS = None

PAGE_BYTES = 16384
GIB = 1024**3
MIB = 1024**2


def nice_ceil(value):
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    # Finer steps in [1, 2] and [5, 10] so e.g. a 113 GiB peak rounds to 120
    # rather than 200, which leaves vmstat-mem with most of the plot empty.
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def nice_step(span, target_ticks):
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    for step in (1, 2, 2.5, 5, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def fmt_tick(value):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    if abs(value) < 1 and value != 0:
        return f"{value:.2f}"
    return f"{value:g}"


def fmt_duration(secs):
    """Compact wall-clock duration. Switches to minute form at 90s so the
    x-axis and footer stay short on long agent sessions."""
    if secs is None:
        return ""
    if secs == 0:
        return "0s"
    m, s = divmod(int(round(secs)), 60)
    if s == 0 and m >= 1:
        return f"{m}m"          # exact minutes: 1m, 2m, ... (consistent ticks)
    if secs < 10:
        return f"{secs:.1f}s"
    if secs < 90:
        return f"{secs:.0f}s"
    return f"{m}m{s:02d}s"


def frange(start, stop, step):
    value = start
    while value <= stop + step * 0.001:
        yield round(value, 10)
        value += step


def project(point, x_min, x_max, y_min, y_max, plot):
    left, top, width, height = plot
    x, y = point
    if x_max == x_min:
        px = left
    else:
        px = left + (x - x_min) / (x_max - x_min) * width
    if y_max == y_min:
        py = top + height
    else:
        py = top + height - (y - y_min) / (y_max - y_min) * height
    return px, py


def points_to_polyline(points, x_min, x_max, y_min, y_max, plot):
    return " ".join(
        f"{px:.2f},{py:.2f}"
        for (px, py) in (project(p, x_min, x_max, y_min, y_max, plot) for p in points)
    )


def render_svg(
    title,
    x_label,
    series_left,
    y_left_label,
    series_right=None,
    y_right_label=None,
    width=960,
    height=540,
    annotations=None,
    phase_bands=None,
    y_left_max_override=None,
    y_right_max_override=None,
    x_max_override=None,
    session_footer=None,
    session_header=None,
    x_axis_format=None,
):
    """series_left / series_right: list of (label, color, [(x, y), ...]).

    A point of `None` inside the point list breaks the polyline so multiple
    segments (e.g., per-session) can share a single series entry.

    `annotations`: optional list of (x_value, label_text) drawn as vertical
    dashed markers with a label at the top.

    `phase_bands`: optional list of (t_start, t_end, kind) where `kind` is a
    key into PHASE_COLORS; drawn as a coloured background rect.

    `y_left_max_override` / `y_right_max_override`: when set, replace the
    data-derived Y-axis upper bound (used to anchor a comparison plot to a
    baseline scenario's scale).
    """
    margin_left = 86
    margin_right = 86 if series_right else 30
    # Annotation rows are computed below from x-collision detection so that
    # arbitrarily long detail labels stack into as many rows as needed and
    # never overlap.  Reserve a sentinel here; the real value is patched in
    # after row assignment.  Each row is 16px tall.
    annotation_rows_estimate = len(annotations) if annotations else 0
    margin_top = 66 + (annotation_rows_estimate * 16 + 8 if annotation_rows_estimate else 0)
    # Extra bottom room for the per-session prompt-excerpt legend, one line
    # per session.
    footer_lines = len(session_footer) if session_footer else 0
    margin_bottom = 72 + (16 * footer_lines + 10 if footer_lines else 0)
    # Grow the canvas when header + footer would squeeze the plot itself
    # below ~MIN_PLOT_HEIGHT.  This keeps the curves readable even when many
    # sessions push the annotation rows up and the footer rows down.
    MIN_PLOT_HEIGHT = 320
    required_height = margin_top + MIN_PLOT_HEIGHT + margin_bottom
    if required_height > height:
        height = required_height
    plot = (
        margin_left,
        margin_top,
        width - margin_left - margin_right,
        height - margin_top - margin_bottom,
    )
    left, top, plot_width, plot_height = plot
    right = left + plot_width
    bottom = top + plot_height

    def real_points(pts):
        return [pt for pt in pts if pt is not None]

    all_x = [pt[0] for _, _, pts in series_left for pt in real_points(pts)]
    if series_right:
        all_x += [pt[0] for _, _, pts in series_right for pt in real_points(pts)]
    if not all_x:
        return None
    x_min = 0
    x_max = max(all_x) or 1
    if x_max_override is not None and x_max_override > 0:
        x_max = x_max_override

    left_values = [pt[1] for _, _, pts in series_left for pt in real_points(pts)] or [0]
    if y_left_max_override is not None:
        y_left_max = y_left_max_override
    else:
        y_left_max = nice_ceil(max(left_values) * 1.05) if max(left_values) > 0 else 1.0
    y_left_min = 0
    y_right_max = None
    if series_right:
        right_values = [pt[1] for _, _, pts in series_right for pt in real_points(pts)] or [0]
        if y_right_max_override is not None:
            y_right_max = y_right_max_override
        else:
            y_right_max = nice_ceil(max(right_values) * 1.05) if max(right_values) > 0 else 1.0
        y_right_min = 0

    # Time-formatted axes can be pinned to a fixed tick interval (e.g. one tick
    # per minute) via --x-tick-minutes; otherwise auto-pick ~6 nice ticks.
    if x_axis_format == "time" and X_TICK_SECONDS and X_TICK_SECONDS > 0:
        x_step = X_TICK_SECONDS
    else:
        x_step = nice_step(x_max - x_min, 6)
    x_ticks = []
    tick = math.ceil(x_min / x_step) * x_step
    while tick <= x_max + x_step * 0.001:
        x_ticks.append(tick)
        tick += x_step

    y_left_step = nice_step(y_left_max - y_left_min, 5)
    y_left_ticks = list(frange(y_left_min, y_left_max, y_left_step))
    y_right_ticks = []
    if y_right_max is not None:
        y_right_step = nice_step(y_right_max - y_right_min, 5)
        y_right_ticks = list(frange(y_right_min, y_right_max, y_right_step))

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".title { font-size: 22px; font-weight: 700; fill: #1f2933; }",
        ".axis-label { font-size: 13px; font-weight: 600; fill: #334155; }",
        ".tick { font-size: 11px; fill: #64748b; }",
        ".legend { font-size: 12px; font-weight: 600; fill: #1f2933; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{html.escape(title)}</text>',
    ]

    if phase_bands:
        for t_start, t_end, kind in phase_bands:
            fill = PHASE_COLORS.get(kind)
            if fill is None:
                continue
            x0 = max(t_start, x_min)
            x1 = min(t_end, x_max)
            if x1 <= x0:
                continue
            px0 = left + (x0 - x_min) / (x_max - x_min) * plot_width
            px1 = left + (x1 - x_min) / (x_max - x_min) * plot_width
            parts.append(
                f'<rect x="{px0:.2f}" y="{top}" width="{px1 - px0:.2f}" height="{plot_height}" fill="{fill}" fill-opacity="0.55"/>'
            )

    for tk in y_left_ticks:
        y = bottom - (tk - y_left_min) / (y_left_max - y_left_min) * plot_height
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{fmt_tick(tk)}</text>'
        )
    if y_right_max is not None:
        for tk in y_right_ticks:
            y = bottom - (tk - y_right_min) / (y_right_max - y_right_min) * plot_height
            parts.append(
                f'<text class="tick" x="{right + 10}" y="{y + 4:.2f}" text-anchor="start">{fmt_tick(tk)}</text>'
            )

    for tk in x_ticks:
        x = left + (tk - x_min) / (x_max - x_min) * plot_width
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        tick_text = fmt_duration(tk) if x_axis_format == "time" else fmt_tick(tk)
        parts.append(
            f'<text class="tick" x="{x:.2f}" y="{bottom + 20}" text-anchor="middle">{tick_text}</text>'
        )

    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.2"/>',
        ]
    )
    if y_right_max is not None:
        parts.append(
            f'<line x1="{right}" y1="{top}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.2"/>'
        )

    parts.append(
        f'<text class="axis-label" x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle">{html.escape(x_label)}</text>'
    )

    if session_footer:
        # Per-session prompt-excerpt legend just below the x-axis.  Each entry
        # is either a string or a (label, excerpt) tuple; render as
        # "<label>  ▶  <excerpt>" when both are present.
        fy0 = bottom + 30
        for i, entry in enumerate(session_footer):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                label, excerpt = entry[0], entry[1]
                text = f"{label}  ▶  {excerpt}" if excerpt else label
            else:
                text = str(entry)
            parts.append(
                f'<text class="legend" x="{left:.2f}" y="{fy0 + i * 16:.2f}">{html.escape(text)}</text>'
            )
    parts.append(
        f'<text class="axis-label" x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 22 {top + plot_height / 2:.1f})">{html.escape(y_left_label)}</text>'
    )
    if y_right_max is not None:
        parts.append(
            f'<text class="axis-label" x="{width - 22}" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(90 {width - 22} {top + plot_height / 2:.1f})">{html.escape(y_right_label or "")}</text>'
        )

    def split_segments(pts):
        segs = []
        current = []
        for pt in pts:
            if pt is None:
                if current:
                    segs.append(current)
                    current = []
                continue
            current.append(pt)
        if current:
            segs.append(current)
        return segs

    for _, color, pts in series_left:
        for seg in split_segments(pts):
            if not seg:
                continue
            poly = points_to_polyline(seg, x_min, x_max, y_left_min, y_left_max, plot)
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="{poly}"/>'
            )
    if series_right:
        for _, color, pts in series_right:
            for seg in split_segments(pts):
                if not seg:
                    continue
                poly = points_to_polyline(seg, x_min, x_max, y_right_min, y_right_max, plot)
                parts.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="6 3" points="{poly}"/>'
                )

    if annotations:
        # Vertical session-start markers + detail labels staggered above the
        # plot.  Rows are assigned greedily by x-collision: a label is
        # placed on the first row whose previous label has already finished
        # to its left.  N rows are added on demand so 3+ sessions never
        # overlap regardless of how long each label string is.
        char_px = 6.6  # rough average proportional-font width at 12px
        min_gap_px = 8
        rows_end_x = []  # the right-edge x (in pixels) of the last label per row
        row_assignment = []  # row index parallel to annotations
        # Sort by ax_val so the greedy assignment is left-to-right.
        anno_indexed = list(enumerate(annotations))
        anno_indexed.sort(key=lambda t: t[1][0])
        for orig_i, (ax_val, label) in anno_indexed:
            if ax_val < x_min or ax_val > x_max:
                row_assignment.append((orig_i, ax_val, label, -1))
                continue
            x = left + (ax_val - x_min) / (x_max - x_min) * plot_width
            label_right = x + len(label) * char_px + min_gap_px
            row = None
            for r, end_x in enumerate(rows_end_x):
                if x > end_x:
                    row = r
                    break
            if row is None:
                rows_end_x.append(label_right)
                row = len(rows_end_x) - 1
            else:
                rows_end_x[row] = label_right
            row_assignment.append((orig_i, ax_val, label, row))
        n_rows = max(1, len(rows_end_x))
        # Each row is 16px tall; the lowest row sits at top - 8 and stacks
        # upward.
        for _orig_i, ax_val, label, row in row_assignment:
            if row < 0:
                continue
            x = left + (ax_val - x_min) / (x_max - x_min) * plot_width
            parts.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3"/>'
            )
            # row 0 = highest (furthest from plot top), row n_rows-1 = closest.
            ly = top - 8 - (n_rows - 1 - row) * 16
            parts.append(
                f'<line x1="{x:.2f}" y1="{ly + 3:.2f}" x2="{x:.2f}" y2="{top:.2f}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="2 2"/>'
            )
            parts.append(
                f'<text class="tick" x="{x + 4:.2f}" y="{ly:.2f}" text-anchor="start">{html.escape(label)}</text>'
            )

    legend_items = [(lbl, col, False) for lbl, col, _ in series_left]
    if series_right:
        legend_items += [(lbl, col, True) for lbl, col, _ in series_right]
    legend_y = top + 6
    box_w = 170
    box_h = 18 * len(legend_items) + 10
    legend_x = right - box_w - 10 if y_right_max is None else right - box_w - 60
    parts.append(
        f'<rect x="{legend_x - 8}" y="{legend_y - 4}" width="{box_w}" height="{box_h}" rx="6" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>'
    )
    for i, (lbl, col, dashed) in enumerate(legend_items):
        ly = legend_y + 14 + i * 18
        if dashed:
            parts.append(
                f'<line x1="{legend_x}" y1="{ly - 4}" x2="{legend_x + 18}" y2="{ly - 4}" stroke="{col}" stroke-width="2" stroke-dasharray="6 3"/>'
            )
        else:
            parts.append(
                f'<rect x="{legend_x}" y="{ly - 10}" width="12" height="12" fill="{col}"/>'
            )
        parts.append(f'<text class="legend" x="{legend_x + 24}" y="{ly}">{html.escape(lbl)}</text>')

    if phase_bands:
        used_kinds = []
        seen = set()
        for _, _, kind in phase_bands:
            if kind in seen or kind not in PHASE_COLORS:
                continue
            seen.add(kind)
            used_kinds.append(kind)
        swatch_w = 14
        gap = 4
        labels = {
            "prefill": "prefill (Metal residency build)",
            "thinking": "THINKING gen",
            "answer": "answer gen",
            "tools": "TOOLS gen",
            "canonicalization": "thinking-checkpoint canonicalization",
        }
        text_w = {kind: len(labels.get(kind, kind)) * 6.5 + swatch_w + gap + 12 for kind in used_kinds}
        total = sum(text_w.values())
        cx = (width - total) / 2
        py = height - 4
        for kind in used_kinds:
            color = PHASE_COLORS[kind]
            parts.append(
                f'<rect x="{cx:.1f}" y="{py - 11}" width="{swatch_w}" height="11" fill="{color}" fill-opacity="0.85" stroke="#cbd5e1"/>'
            )
            parts.append(
                f'<text class="tick" x="{cx + swatch_w + gap:.1f}" y="{py - 1}" text-anchor="start">{html.escape(labels.get(kind, kind))}</text>'
            )
            cx += text_w[kind]

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"^\s*\d{4}\s+(\d{2}):(\d{2}):(\d{2})\s")
GEN_RE = re.compile(
    r"gen=(?P<gen>\d+).+?chunk=(?P<chunk>[\d.]+) t/s avg=(?P<avg>[\d.]+) t/s\s+(?P<elapsed>[\d.]+)s"
)
PREFILL_RE = re.compile(
    r"prefill chunk (?P<current>\d+)/(?P<total>\d+)\s*\([\d.]+%\)\s+chunk=(?P<chunk>[\d.]+) t/s avg=(?P<avg>[\d.]+) t/s\s+(?P<elapsed>[\d.]+)s"
)

# Background tints used to highlight ds4-server phases.
PHASE_COLORS = {
    "prefill": "#fef3c7",        # light yellow
    "thinking": "#dbeafe",       # light blue
    "answer": "#dcfce7",         # light green
    "tools": "#f3e8ff",          # light purple
    "canonicalization": "#fed7aa",  # light orange
}


def _to_seconds(match):
    h, m, s = (int(x) for x in match.groups())
    return h * 3600 + m * 60 + s


def parse_ds4_log(path):
    """Walk the ds4-server log once, emitting samples, session markers, and
    phase bands. Wall-clock seconds since the first timestamped line is used
    as the x axis so multi-turn `elapsed` resets do not overlap."""
    samples = []  # (t_wall, chunk_tps, avg_tps, gen_tokens, session_idx)
    prefill_samples = []  # (t_wall, chunk_tps, avg_tps, current_tokens, session_idx)
    sessions = []  # list of (t_start, label)
    phases = []  # list of (t_start, t_end, kind)
    gen_origins = {}  # session_idx -> t_wall at gen=0 (first sample t - elapsed)
    session_idx = -1
    epoch = None

    current_phase = None  # (kind, t_start)

    def close_phase(t_end):
        nonlocal current_phase
        if current_phase is None:
            return
        kind, t_start = current_phase
        if t_end > t_start:
            phases.append((t_start, t_end, kind))
        current_phase = None

    def open_phase(kind, t_start):
        nonlocal current_phase
        if current_phase is not None and current_phase[0] == kind:
            return
        close_phase(t_start)
        current_phase = (kind, t_start)

    with path.open() as fp:
        for line in fp:
            ts = TIMESTAMP_RE.match(line)
            if not ts:
                continue
            t_abs = _to_seconds(ts)
            if epoch is None:
                epoch = t_abs
            t_wall = t_abs - epoch
            if t_wall < 0:
                t_wall += 24 * 3600  # wrap past midnight (defensive)

            if "prompt start" in line:
                session_idx += 1
                # ctx=A..B:N  →  N = prompt token count of THIS request
                # (B-A is the new-token delta; N is the full prompt length)
                ctx_m = re.search(r"ctx=\d+\.\.\d+:(\d+)", line)
                prompt_tokens = int(ctx_m.group(1)) if ctx_m else 0
                kind = "TOOLS" if "TOOLS" in line else "system"
                label = f"#{session_idx + 1} {kind} {prompt_tokens} tok"
                sessions.append((t_wall, label))
                open_phase("prefill", t_wall)
                continue

            if "prompt done" in line:
                close_phase(t_wall)
                continue

            if "canonicalization needs rebuild" in line:
                open_phase("canonicalization", t_wall)
                continue

            if "canonicalized" in line and "needs rebuild" not in line:
                close_phase(t_wall)
                continue

            if "finish=stop" in line:
                close_phase(t_wall)
                continue

            mp = PREFILL_RE.search(line)
            if mp:
                if session_idx < 0:
                    session_idx = 0
                    sessions.append((t_wall, "prompt 1"))
                open_phase("prefill", t_wall)
                prefill_samples.append(
                    (
                        t_wall,
                        float(mp.group("chunk")),
                        float(mp.group("avg")),
                        int(mp.group("current")),
                        session_idx,
                    )
                )
                continue

            m = GEN_RE.search(line)
            if not m:
                continue
            if session_idx < 0:
                session_idx = 0
                sessions.append((t_wall, "prompt 1"))
            if "THINKING" in line:
                kind = "thinking"
            elif "TOOLS" in line:
                kind = "tools"
            else:
                kind = "answer"
            if session_idx not in gen_origins:
                # Anchor the gen line at the instant generation began (gen=0),
                # reconstructed as this first sample's wall-clock minus its
                # cumulative elapsed. Mirrors prefill, whose line starts at
                # prompt-start (current_tokens=0, elapsed=0).
                gen_origins[session_idx] = t_wall - float(m.group("elapsed"))
                # Open the first gen phase at that origin (≈ prompt done) so the
                # phase-band tint also covers the lead-in, matching the line.
                open_phase(kind, gen_origins[session_idx])
            else:
                open_phase(kind, t_wall)
            samples.append(
                (
                    t_wall,
                    float(m.group("chunk")),
                    float(m.group("avg")),
                    int(m.group("gen")),
                    session_idx,
                )
            )

    # close any dangling phase at the last seen timestamp
    last_t = None
    if samples:
        last_t = samples[-1][0]
    if prefill_samples and (last_t is None or prefill_samples[-1][0] > last_t):
        last_t = prefill_samples[-1][0]
    if current_phase is not None and last_t is not None:
        close_phase(last_t)

    return {
        "samples": samples,
        "prefill_samples": prefill_samples,
        "sessions": sessions,
        "phases": phases,
        "gen_origins": gen_origins,
    }


def parse_ds4_session_metrics(path):
    """Per-session timing / tps / finish-reason / executed tool list / error
    detail scraped from the ds4-server log.  Sessions are delimited by
    "... prompt start" markers; per-session state is finalized when the
    next "prompt start" arrives (or at EOF)."""
    if not path.exists():
        return []
    out = []
    cur = None
    with path.open(errors="replace") as fp:
        for line in fp:
            if "prompt start" in line:
                if cur is not None:
                    out.append(cur)
                cur = {"prefill_s": None, "gen_tokens": None,
                       "gen_avg": None, "finish": None, "total_s": None,
                       "error_msg": None, "tool_names": []}
                continue
            if cur is None:
                continue
            m = re.search(r"prompt done\s+([\d.]+)s", line)
            if m:
                cur["prefill_s"] = float(m.group(1))
            m = re.search(r"gen=(\d+).*decoding chunk=[\d.]+ t/s avg=([\d.]+) t/s",
                          line)
            if m:
                cur["gen_tokens"] = int(m.group(1))
                cur["gen_avg"] = float(m.group(2))
            # "finish=<reason> [error="..."] <total>s"  — the trailing seconds
            # is the wall-clock from "prompt start" to "finish", i.e. the
            # full request total (prefill + gen + overhead).
            m = re.search(r"finish=(\w+)(?:.*?error=\"([^\"]+)\")?.*?([\d.]+)s\s*$",
                          line.rstrip())
            if m:
                cur["finish"] = m.group(1)
                if m.group(2):
                    cur["error_msg"] = m.group(2)
                try:
                    cur["total_s"] = float(m.group(3))
                except (TypeError, ValueError):
                    pass
            m = re.search(r"tool calls .* names=\[([^\]]+)\]", line)
            if m:
                cur["tool_names"] = [n.strip() for n in m.group(1).split(",")]
    if cur is not None:
        out.append(cur)
    return out


def parse_trace_request_meta(path, max_excerpt_len=40):
    """Per-request meta extracted from the `--trace` log: detected intent
    (title gen / agent / tool-result / chat), tool count, cache decision
    fields, and a short excerpt of the last user message."""
    if not path.exists():
        return []
    metas = []
    in_request = False
    cache_section = False
    json_next = False
    cur = None
    with path.open(errors="replace") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if line.startswith("===== request "):
                in_request = True
                cache_section = False
                json_next = False
                cur = {"intent": "?", "n_tools": 0, "excerpt": "",
                       "cache_source": "?", "cached_pct": 0.0,
                       "cached_tokens": 0, "prompt_tokens": 0,
                       "miss_reason": None}
                continue
            if line.startswith("===== end request"):
                if cur is not None:
                    metas.append(cur)
                cur = None
                in_request = False
                continue
            if not in_request or cur is None:
                continue
            if line.startswith("--- cache decision"):
                cache_section = True
                continue
            if line.startswith("--- raw request json"):
                json_next = True
                cache_section = False
                continue
            if line.startswith("---"):
                cache_section = False
                continue
            if cache_section:
                m = re.match(r"cache_source:\s*(\S+)", line)
                if m:
                    cur["cache_source"] = m.group(1)
                m = re.match(r"cached_tokens:\s*(\d+)", line)
                if m:
                    cur["cached_tokens"] = int(m.group(1))
                m = re.match(r"prompt_tokens:\s*(\d+)", line)
                if m:
                    cur["prompt_tokens"] = int(m.group(1))
                m = re.match(r"memory_miss_reason:\s*(\S+)", line)
                if m and m.group(1) != "":
                    cur["miss_reason"] = m.group(1)
            if json_next and line.startswith("{"):
                json_next = False
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                msgs = obj.get("messages") or []
                tools = obj.get("tools") or []
                cur["n_tools"] = len(tools)
                sys_content = ""
                for m in msgs:
                    if m.get("role") == "system":
                        c = m.get("content")
                        if isinstance(c, str):
                            sys_content = c
                        break
                last_role = msgs[-1].get("role") if msgs else ""
                last_is_tool_result = False
                if msgs and last_role == "user":
                    c = msgs[-1].get("content")
                    if isinstance(c, list):
                        for blk in c:
                            if isinstance(blk, dict) and blk.get("type") == "tool_result":
                                last_is_tool_result = True
                                break
                if "title generator" in sys_content.lower():
                    cur["intent"] = "title gen"
                elif last_role == "tool" or last_is_tool_result:
                    cur["intent"] = "tool-result"
                elif len(tools) > 0:
                    # Just the category — the available-tool count (len(tools))
                    # is roughly constant per client and adds no performance
                    # signal beyond prompt_tokens; read the trace log if the
                    # exact toolset size is ever needed.  n_tools is still
                    # parsed above for that purpose.
                    cur["intent"] = "agent"
                else:
                    cur["intent"] = "chat"
                # excerpt selection — show the most informative "what is
                # happening right now" snippet:
                #   - title gen:                first user msg (= instruction)
                #   - last role=tool:           tool result content head
                #   - last role=assistant +tc:  "<tool>(<arg summary>)"
                #   - everything else:          last user message
                def msg_text(c):
                    if isinstance(c, str):
                        return c
                    if isinstance(c, list):
                        out = []
                        for blk in c:
                            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                                out.append(blk["text"])
                            elif isinstance(blk, dict) and isinstance(blk.get("content"), str):
                                out.append(blk["content"])
                        return " ".join(out)
                    return ""

                user_msgs = [msg_text(m.get("content"))
                             for m in msgs if m.get("role") == "user"]
                pick = ""
                last_msg = msgs[-1] if msgs else {}
                last_role = last_msg.get("role")
                if cur["intent"] == "title gen" and user_msgs:
                    pick = user_msgs[0]
                elif last_role == "tool":
                    pick = msg_text(last_msg.get("content"))
                elif last_role == "assistant":
                    tcs = last_msg.get("tool_calls") or []
                    if tcs:
                        first_tc = tcs[0]
                        tname = first_tc.get("function", {}).get("name", "?")
                        targs = first_tc.get("function", {}).get("arguments", "")
                        args_summary = ""
                        try:
                            args_obj = json.loads(targs) if isinstance(targs, str) else targs
                            if isinstance(args_obj, dict):
                                first_keys = list(args_obj.items())[:2]
                                args_summary = ", ".join(
                                    f"{k}={str(v)[:30]}" for k, v in first_keys)
                        except Exception:
                            args_summary = str(targs)[:30]
                        pick = f"{tname}({args_summary})"
                    else:
                        pick = msg_text(last_msg.get("content"))
                elif user_msgs:
                    pick = user_msgs[-1]
                pick = re.sub(r"\s+", " ", pick).strip().strip('"')
                if len(pick) > max_excerpt_len:
                    pick = pick[: max_excerpt_len - 1] + "…"
                cur["excerpt"] = pick
                if cur["prompt_tokens"] > 0:
                    cur["cached_pct"] = 100.0 * cur["cached_tokens"] / cur["prompt_tokens"]
    return metas


def _intent_with_tool(meta, metric):
    """Format the "[intent→tool,…]" cell used in both header and footer."""
    intent = meta.get("intent", "?") if meta else "?"
    tool_names = metric.get("tool_names") if metric else None
    if tool_names:
        shown = ",".join(tool_names[:3])
        if len(tool_names) > 3:
            shown += f"+{len(tool_names) - 3}"
        return f"[{intent}→{shown}]"
    return f"[{intent}]"


def _format_cache_cell(meta):
    """cache cell is a single percent now — type (mem-txt / disk-tok / etc.)
    and miss reason are dropped to keep the footer to one line per session;
    look at the trace log directly for the full cache decision."""
    if meta is None:
        return "cache=-"
    cs = meta.get("cache_source") or "?"
    if cs in ("?", None):
        return "cache=-"
    pct = meta.get("cached_pct", 0.0)
    return f"cache={pct:.0f}%"


def build_session_short_lines(sessions, metas, metrics):
    """Short anchor label per session for the in-plot staggered markers
    above the plot.  Form: "#N <total> [<intent>→<tool>]" — time goes first
    so even when the right-edge labels get clipped, the wall-clock cost of
    each request stays readable."""
    lines = []
    for i, (_t, lbl) in enumerate(sessions):
        meta = metas[i] if i < len(metas) else None
        metric = metrics[i] if i < len(metrics) else None
        idx_m = re.match(r"(#\d+)", lbl)
        prefix = idx_m.group(1) if idx_m else lbl
        parts = [prefix]
        if metric and metric.get("total_s") is not None:
            parts.append(fmt_duration(metric["total_s"]))
        if meta:
            parts.append(_intent_with_tool(meta, metric))
        lines.append(" ".join(parts))
    return lines


def build_session_footer_lines(sessions, metas, metrics):
    """Footer line per session drawn BELOW the plot — one row per session:
      "#N <total> [<intent>→<tool>] prefill=<ptok>[@<rate>t/s] gen=<gtok>@<rate>t/s cache=<pct>% <finish> | <excerpt>"
    Notes:
      - time goes right after "#N" so it lines up with the in-plot anchor
      - prefill rate is dropped on cache hit (cached_pct > 0) because the
        figure becomes meaninglessly large (cache replay, not real prefill)
      - cache cell is just the percent — type and miss reason are off-table
      - excerpt is appended after " | " on the same line (no second row).
    """
    lines = []
    for i, (_t, lbl) in enumerate(sessions):
        meta = metas[i] if i < len(metas) else None
        metric = metrics[i] if i < len(metrics) else None
        idx_m = re.match(r"(#\d+)", lbl)
        prefix = idx_m.group(1) if idx_m else lbl
        parts = [prefix]
        if metric and metric.get("total_s") is not None:
            parts.append(fmt_duration(metric["total_s"]))
        if meta:
            parts.append(_intent_with_tool(meta, metric))
        cached_pct = meta.get("cached_pct", 0.0) if meta else 0.0
        if metric:
            prefill_s = metric.get("prefill_s")
            ptok = meta.get("prompt_tokens", 0) if meta else 0
            if ptok > 0 and prefill_s and prefill_s > 0:
                if cached_pct > 0:
                    parts.append(f"prefill={ptok}")
                else:
                    parts.append(f"prefill={ptok}@{ptok / prefill_s:.1f}t/s")
            elif prefill_s is not None:
                parts.append(f"prefill={fmt_duration(prefill_s)}")
            if metric.get("gen_tokens") and metric.get("gen_avg"):
                parts.append(
                    f"gen={metric['gen_tokens']}@{metric['gen_avg']:.1f}t/s")
        if meta:
            parts.append(_format_cache_cell(meta))
        if metric:
            finish = metric.get("finish")
            if finish == "error" and metric.get("error_msg"):
                parts.append(f'error("{metric["error_msg"]}")')
            elif finish:
                parts.append(finish)
        line = " ".join(parts)
        excerpt = meta.get("excerpt") if meta else ""
        if excerpt:
            line = f"{line} | {excerpt}"
        lines.append(line)
    # Summary row: total wall-clock across all sessions (sum of per-session
    # serving time) so the whole run's elapsed time is visible at the bottom.
    totals = [m["total_s"] for m in metrics
              if m and m.get("total_s") is not None]
    if totals:
        n = len(totals)
        lines.append(
            f"Σ total {fmt_duration(sum(totals))} "
            f"({n} session{'s' if n != 1 else ''})"
        )
    return lines


def parse_trace_user_excerpts(path, max_len=80):
    """Return one short excerpt of the *last user message* per request from a
    ds4-server `--trace` log, in request order. Used to attach the agent's
    actual prompt to each session annotation in the plot legend."""
    if not path.exists():
        return []
    excerpts = []
    in_request = False
    json_next = False
    with path.open(errors="replace") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if line.startswith("===== request "):
                in_request = True
                json_next = False
                continue
            if line.startswith("===== end request"):
                in_request = False
                continue
            if not in_request:
                continue
            if line.startswith("--- raw request json"):
                json_next = True
                continue
            if json_next and line.startswith("{"):
                json_next = False
                try:
                    obj = json.loads(line)
                except Exception:
                    excerpts.append("")
                    continue
                msgs = obj.get("messages") or []
                last_user = ""
                for m in msgs:
                    if m.get("role") != "user":
                        continue
                    c = m.get("content")
                    if isinstance(c, str):
                        last_user = c
                    elif isinstance(c, list):
                        parts = []
                        for blk in c:
                            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                                parts.append(blk["text"])
                            elif isinstance(blk, dict) and isinstance(blk.get("content"), str):
                                parts.append(blk["content"])
                        last_user = " ".join(parts)
                last_user = re.sub(r"\s+", " ", last_user).strip().strip('"')
                if len(last_user) > max_len:
                    last_user = last_user[: max_len - 1] + "…"
                excerpts.append(last_user)
    return excerpts


IOSTAT_HEADER_RE = re.compile(r"^\s*KB/t\b")
IOSTAT_DISK_HEADER_RE = re.compile(r"disk\d")


def parse_iostat(path, interval=IOSTAT_INTERVAL_DEFAULT):
    rows = []
    t = 0.0
    with path.open() as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped:
                continue
            if IOSTAT_DISK_HEADER_RE.search(stripped) and "cpu" in stripped:
                continue
            if IOSTAT_HEADER_RE.match(line):
                continue
            tokens = stripped.split()
            if len(tokens) < 9:
                continue
            try:
                rows.append(
                    dict(
                        t=t,
                        kbt=float(tokens[0]),
                        tps=float(tokens[1]),
                        mbs=float(tokens[2]),
                        us=float(tokens[3]),
                        sy=float(tokens[4]),
                        idle=float(tokens[5]),
                        load1=float(tokens[6]),
                        load5=float(tokens[7]),
                        load15=float(tokens[8]),
                    )
                )
            except ValueError:
                continue
            t += interval
    return rows


def parse_kvalue(tok):
    if tok.endswith("K"):
        return int(tok[:-1]) * 1000
    return int(tok)


def parse_vmstat(path, interval=VMSTAT_INTERVAL_DEFAULT):
    """Parse `vm_stat <interval>` output.

    The data row that follows each header re-print contains cumulative-since-boot
    values for delta columns (pageins, faults, copy, 0fill, comprs, dcomprs).
    We skip those rows so they don't show up as artificial spikes in the plot.

    Column layout (0-indexed) in `vm_stat 30`:
        0:free 1:active 2:specul 3:inactive 4:throttle 5:wired 6:prgable
        7:faults 8:copy 9:0fill 10:reactive 11:purged 12:file-backed
        13:anonymous 14:cmprssed 15:cmprssor 16:dcomprs 17:comprs
        18:pageins 19:pageout 20:swapins 21:swapouts
    """
    rows = []
    t = 0.0
    just_saw_header = True
    with path.open() as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped:
                continue
            if "Mach Virtual Memory Statistics" in stripped:
                just_saw_header = True
                continue
            if stripped.startswith("free"):
                continue
            tokens = stripped.split()
            if len(tokens) < 22:
                continue
            if just_saw_header:
                just_saw_header = False
                continue
            try:
                rows.append(
                    dict(
                        t=t,
                        wired=parse_kvalue(tokens[5]),
                        faults=parse_kvalue(tokens[7]),
                        file_backed=parse_kvalue(tokens[12]),
                        cmprssor=parse_kvalue(tokens[15]),
                        pageins=parse_kvalue(tokens[18]),
                        pageout=parse_kvalue(tokens[19]),
                        swapins=parse_kvalue(tokens[20]),
                        swapouts=parse_kvalue(tokens[21]),
                    )
                )
            except ValueError:
                continue
            t += interval
    return rows


def parse_mactop(path):
    """Tolerant parser for ``mactop --headless`` output filtered through
    scripts/mactop_filter.py.

    The filter emits one compact JSON object per line (NDJSON).  The raw
    ``mactop --headless --pretty`` stream is a JSON array with whitespace and
    may be truncated at Ctrl+C, so we skip ``[`` / ``]`` / ``,`` separators
    and raw_decode one object at a time.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("[")
    if start >= 0 and start < 1024:
        text = text[start + 1 :]
    samples = []
    decoder = json.JSONDecoder()
    while True:
        text = text.lstrip()
        if text.startswith(","):
            text = text[1:].lstrip()
        if not text or text.startswith("]"):
            break
        try:
            obj, idx = decoder.raw_decode(text)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            samples.append(obj)
        text = text[idx:]
    return samples


def _parse_iso(ts):
    return _dt.datetime.fromisoformat(ts)


def mactop_timeline(samples):
    """Return list of dicts keyed by t_wall (seconds since first sample)."""
    out = []
    epoch = None
    for s in samples:
        ts = s.get("timestamp")
        if not ts:
            continue
        try:
            t = _parse_iso(ts)
        except ValueError:
            continue
        if epoch is None:
            epoch = t
        soc = s.get("soc_metrics") or {}
        mem = s.get("memory") or {}
        net = s.get("net_disk") or {}
        gpu_metrics = s.get("gpu_metrics") or {}
        out.append(
            dict(
                t=(t - epoch).total_seconds(),
                wall=t,
                cpu_usage=s.get("cpu_usage") or 0.0,
                gpu_usage=s.get("gpu_usage") or 0.0,
                e_cluster_active=soc.get("e_cluster_active") or 0.0,
                p_cluster_active=soc.get("p_cluster_active") or 0.0,
                e_cluster_freq=soc.get("e_cluster_freq_mhz") or 0.0,
                p_cluster_freq=soc.get("p_cluster_freq_mhz") or 0.0,
                gpu_freq=gpu_metrics.get("freq_mhz") or soc.get("gpu_freq_mhz") or 0.0,
                cpu_power=soc.get("cpu_power") or 0.0,
                gpu_power=soc.get("gpu_power") or 0.0,
                ane_power=soc.get("ane_power") or 0.0,
                dram_power=soc.get("dram_power") or 0.0,
                system_power=soc.get("system_power") or 0.0,
                total_power=soc.get("total_power") or 0.0,
                dram_read_gbs=soc.get("dram_read_bw_gbs") or 0.0,
                dram_write_gbs=soc.get("dram_write_bw_gbs") or 0.0,
                dram_combined_gbs=soc.get("dram_bw_combined_gbs") or 0.0,
                mem_used=mem.get("used") or 0,
                swap_used=mem.get("swap_used") or 0,
                disk_read_kbps=net.get("read_kbytes_per_sec") or 0.0,
                disk_write_kbps=net.get("write_kbytes_per_sec") or 0.0,
                net_in_bps=net.get("in_bytes_per_sec") or 0.0,
                net_out_bps=net.get("out_bytes_per_sec") or 0.0,
                soc_temp=soc.get("soc_temp") or 0.0,
                cpu_temp=soc.get("cpu_temp") or 0.0,
                gpu_temp=soc.get("gpu_temp") or 0.0,
            )
        )
    return out


# ----------------------------------------------------------------------------
# Plot builders
# ----------------------------------------------------------------------------


def _series_with_session_breaks(samples, t_idx, val_idx, sess_idx):
    out = []
    last_session = None
    for row in samples:
        t_wall = row[t_idx]
        v = row[val_idx]
        sess = row[sess_idx]
        if last_session is not None and sess != last_session:
            out.append(None)
        out.append((t_wall, v))
        last_session = sess
    return out


def _prefill_forward_fill(samples, t_idx, val_idx, sess_idx):
    """Forward-fill a prefill rate series so zero / missing samples inherit
    the previous non-zero value, while real measurements pass through
    unchanged. Leading zero(s) before the first non-zero sample of a session
    are back-filled with that session's first non-zero value (so the line
    does not dip to 0 on prefill kickoff)."""
    if not samples:
        return []
    by_session = {}
    order = []
    for row in samples:
        sess = row[sess_idx]
        if sess not in by_session:
            by_session[sess] = []
            order.append(sess)
        by_session[sess].append((row[t_idx], row[val_idx]))
    out = []
    for i, sess in enumerate(order):
        rows = by_session[sess]
        first_nonzero = 0.0
        for (_, v) in rows:
            if v > 0.0:
                first_nonzero = v
                break
        if i > 0:
            out.append(None)
        last_val = first_nonzero
        for (t, v) in rows:
            if v > 0.0:
                last_val = v
            out.append((t, last_val))
    return out


def _gen_series_with_origin(samples, val_idx, gen_origins):
    """Like _series_with_session_breaks, but prepend a per-session origin
    anchor at the instant generation began (gen=0), carrying that session's
    first measured value. Without it the gen line floats, starting at the
    first 50-token sample ~several seconds into the phase; with it the line is
    anchored at the gen-phase start, mirroring the prefill line (which
    _prefill_forward_fill anchors at prompt-start)."""
    out = []
    last_session = None
    for row in samples:
        t_wall = row[0]
        v = row[val_idx]
        sess = row[4]
        if sess != last_session:
            if last_session is not None:
                out.append(None)
            origin_t = gen_origins.get(sess)
            # First row of the session: anchor at gen=0 carrying this first
            # value, drawing a flat lead-in from the gen-phase start.
            if origin_t is not None and origin_t < t_wall:
                out.append((origin_t, v))
        out.append((t_wall, v))
        last_session = sess
    return out


def build_tok_plot(parsed, title, y_prefill_max=None, y_gen_max=None, x_max_override=None,
                   session_footer=None, x_axis_format=None):
    """Combined prefill + gen throughput, X-axis shifted so the first request's
    prompt start sits at t=0 (so scenarios with different cold-load times
    overlay fairly).  Prefill on the left axis (solid), gen on the right
    (dashed).  Phase bands and session markers are shifted with the data.
    ``y_prefill_max`` / ``y_gen_max`` override the data-derived upper bounds
    so multiple scenarios can share a comparison scale.
    """
    if not parsed:
        return None
    samples = parsed.get("samples") or []
    prefill_samples = parsed.get("prefill_samples") or []
    if not samples and not prefill_samples:
        return None
    sessions = parsed.get("sessions") or []
    phases = parsed.get("phases", [])
    gen_origins = parsed.get("gen_origins") or {}

    # Shift origin to the first request's prompt start so prefill/gen overlays
    # across scenarios start from the same X=0.  Falls back to 0 (no shift) if
    # no session markers were captured.
    origin = sessions[0][0] if sessions else 0.0

    def shift_rows(rows, t_idx):
        return [tuple(r[i] - origin if i == t_idx else r[i] for i in range(len(r)))
                for r in rows]

    prefill_samples = shift_rows(prefill_samples, 0)
    samples = shift_rows(samples, 0)
    sessions = [(t - origin, lbl) for (t, lbl) in sessions]
    phases = [(s - origin, e - origin, k) for (s, e, k) in phases]
    gen_origins = {s: t - origin for s, t in gen_origins.items()}

    series_left = []
    if prefill_samples:
        # tuple: (t_wall, chunk_tps, avg_tps, current_tokens, session_idx)
        prefill_chunk = _prefill_forward_fill(prefill_samples, 0, 1, 4)
        prefill_avg = _prefill_forward_fill(prefill_samples, 0, 2, 4)
        series_left = [
            ("prefill chunk t/s", COLORS[1], prefill_chunk),
            ("prefill avg t/s", COLORS[0], prefill_avg),
        ]

    series_right = None
    y_right_label = None
    if samples:
        # tuple: (t_wall, chunk_tps, avg_tps, gen_tokens, session_idx)
        gen_chunk = _gen_series_with_origin(samples, 1, gen_origins)
        gen_avg = _gen_series_with_origin(samples, 2, gen_origins)
        series_right = [
            ("gen chunk t/s", COLORS[4], gen_chunk),
            ("gen avg t/s", COLORS[2], gen_avg),
        ]
        y_right_label = "gen tokens / second"

    return render_svg(
        title=title,
        x_label="seconds since first request",
        series_left=series_left or [("(no prefill)", COLORS[0], [])],
        y_left_label="prefill tokens / second",
        series_right=series_right,
        y_right_label=y_right_label,
        annotations=[(t, label) for t, label in sessions],
        phase_bands=phases,
        y_left_max_override=y_prefill_max,
        y_right_max_override=y_gen_max,
        x_max_override=x_max_override,
        session_footer=session_footer,
        x_axis_format=x_axis_format,
    )


def build_iostat_disk_plot(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    if not rows:
        return None
    return render_svg(
        title=title,
        x_label="elapsed seconds (approx. aligned to ds4-server epoch)",
        series_left=[("disk0 MB/s", COLORS[0], [(r["t"], r["mbs"]) for r in rows])],
        y_left_label="disk MB/s",
        series_right=[("disk0 tps", COLORS[1], [(r["t"], r["tps"]) for r in rows])],
        y_right_label="disk tps",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def build_iostat_cpu_plot(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    if not rows:
        return None
    return render_svg(
        title=title,
        x_label="elapsed seconds (approx. aligned to ds4-server epoch)",
        series_left=[
            ("user %", COLORS[0], [(r["t"], r["us"]) for r in rows]),
            ("sys %", COLORS[1], [(r["t"], r["sy"]) for r in rows]),
            ("idle %", COLORS[2], [(r["t"], r["idle"]) for r in rows]),
        ],
        y_left_label="CPU %",
        series_right=[("load 1m", COLORS[3], [(r["t"], r["load1"]) for r in rows])],
        y_right_label="load avg",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def _mactop_serieslist(rows, *fields_labels_colors):
    out = []
    for field, label, color, scale in fields_labels_colors:
        pts = [(r["t"], r[field] * scale) for r in rows]
        out.append((label, color, pts))
    return out


def build_mactop_cpu_gpu(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    if not rows:
        return None
    return render_svg(
        title=title,
        x_label="elapsed seconds (mactop epoch)",
        series_left=_mactop_serieslist(
            rows,
            ("cpu_usage", "cpu %", COLORS[0], 1),
            ("e_cluster_active", "E-cluster active %", COLORS[2], 1),
            ("p_cluster_active", "P-cluster active %", COLORS[1], 1),
            ("gpu_usage", "gpu %", COLORS[3], 1),
        ),
        y_left_label="utilization %",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def build_mactop_freq(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    if not rows:
        return None
    return render_svg(
        title=title,
        x_label="elapsed seconds (mactop epoch)",
        series_left=_mactop_serieslist(
            rows,
            ("e_cluster_freq", "E-cluster MHz", COLORS[2], 1),
            ("p_cluster_freq", "P-cluster MHz", COLORS[1], 1),
            ("gpu_freq", "GPU MHz", COLORS[3], 1),
        ),
        y_left_label="frequency MHz",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def build_mactop_power(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None,
                       y_power_max=None, y_total_power_max=None):
    if not rows:
        return None
    return render_svg(
        title=title,
        x_label="elapsed seconds (mactop epoch)",
        series_left=_mactop_serieslist(
            rows,
            ("cpu_power", "cpu W", COLORS[0], 1),
            ("gpu_power", "gpu W", COLORS[3], 1),
            ("ane_power", "ANE W", COLORS[5], 1),
            ("dram_power", "dram W", COLORS[2], 1),
            ("system_power", "system W", COLORS[4], 1),
        ),
        y_left_label="power (W)",
        series_right=[
            ("total W", COLORS[1], [(r["t"], r["total_power"]) for r in rows]),
        ],
        y_right_label="total power (W)",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
        y_left_max_override=y_power_max,
        y_right_max_override=y_total_power_max,
    )


def build_mactop_dram_bw(rows, title, phase_bands=None, annotations=None, y_bw_max=None,
                         x_max_override=None, session_footer=None, x_axis_format=None):
    if not rows:
        return None
    return render_svg(
        title=title,
        x_label="elapsed seconds (mactop epoch)",
        series_left=_mactop_serieslist(
            rows,
            ("dram_read_gbs", "DRAM read GB/s", COLORS[0], 1),
            ("dram_write_gbs", "DRAM write GB/s", COLORS[1], 1),
            ("dram_combined_gbs", "DRAM combined GB/s", COLORS[2], 1),
        ),
        y_left_label="DRAM bandwidth (GB/s)",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
        y_left_max_override=y_bw_max,
    )


def build_mactop_memory(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    if not rows:
        return None
    used_gib = [(r["t"], r["mem_used"] / GIB) for r in rows]
    swap_gib = [(r["t"], r["swap_used"] / GIB) for r in rows]
    return render_svg(
        title=title,
        x_label="elapsed seconds (mactop epoch)",
        series_left=[
            ("memory used (GiB)", COLORS[0], used_gib),
            ("swap used (GiB)", COLORS[1], swap_gib),
        ],
        y_left_label="memory (GiB)",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def build_mactop_net_disk(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    if not rows:
        return None
    disk_read_mbs = [(r["t"], r["disk_read_kbps"] / 1024.0) for r in rows]
    disk_write_mbs = [(r["t"], r["disk_write_kbps"] / 1024.0) for r in rows]
    return render_svg(
        title=title,
        x_label="elapsed seconds (mactop epoch)",
        series_left=[
            ("disk read MB/s", COLORS[0], disk_read_mbs),
            ("disk write MB/s", COLORS[1], disk_write_mbs),
        ],
        y_left_label="disk MB/s",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def build_mactop_temp(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    if not rows:
        return None
    return render_svg(
        title=title,
        x_label="elapsed seconds (mactop epoch)",
        series_left=_mactop_serieslist(
            rows,
            ("soc_temp", "SoC °C", COLORS[0], 1),
            ("cpu_temp", "CPU °C", COLORS[1], 1),
            ("gpu_temp", "GPU °C", COLORS[3], 1),
        ),
        y_left_label="temperature (°C)",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def build_vmstat_mem_plot(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    """Memory composition: how the 128 GiB physical RAM is partitioned.

    wired + file-backed + cmprssor are the three biggest consumers in the
    cpu-moe + prefill-metal-phases workload. The trade-off between wired
    (Metal residency) and file-backed (OS page cache for mmap'd experts)
    is the central story. cmprssor stays small in a healthy run; growth
    signals memory pressure (compressor pool expanding before swap).
    """
    if not rows:
        return None
    wired_gib = [(r["t"], r["wired"] * PAGE_BYTES / GIB) for r in rows]
    file_gib = [(r["t"], r["file_backed"] * PAGE_BYTES / GIB) for r in rows]
    compress_gib = [(r["t"], r["cmprssor"] * PAGE_BYTES / GIB) for r in rows]
    return render_svg(
        title=title,
        x_label="elapsed seconds (approx. aligned to ds4-server epoch)",
        series_left=[
            ("wired (GiB)", COLORS[0], wired_gib),
            ("file-backed (GiB)", COLORS[2], file_gib),
            ("cmprssor (GiB)", COLORS[3], compress_gib),
        ],
        y_left_label="memory (GiB)",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def build_vmstat_events_plot(rows, title, phase_bands=None, annotations=None, x_max_override=None, session_footer=None, x_axis_format=None):
    """Per-30s event rates: anomaly + pressure detection.

    faults / pageins / pageout share the left axis; swap-in+swap-out lives
    on the right axis (separate scale because both are expected to be
    flat-zero in a healthy run). Any line lifting off zero on the right
    axis is a red flag (swap activity = memory exhausted). pageout > 0
    means unexpected dirty pages going to disk on this read-only model.
    """
    if not rows:
        return None
    faults = [(r["t"], r["faults"]) for r in rows]
    pageins = [(r["t"], r["pageins"]) for r in rows]
    pageout = [(r["t"], r["pageout"]) for r in rows]
    swap = [(r["t"], r["swapins"] + r["swapouts"]) for r in rows]
    return render_svg(
        title=title,
        x_label="elapsed seconds (approx. aligned to ds4-server epoch)",
        series_left=[
            ("faults / interval", COLORS[0], faults),
            ("pageins / interval", COLORS[1], pageins),
            ("pageout / interval", COLORS[2], pageout),
        ],
        y_left_label="events / interval",
        series_right=[("swap in+out / interval", COLORS[3], swap)],
        y_right_label="swap events / interval",
        phase_bands=phase_bands, annotations=annotations, x_max_override=x_max_override, session_footer=session_footer, x_axis_format=x_axis_format,
    )


def write_svg(plot_dir, name, svg):
    if svg is None:
        print(f"skip {name} (no data)")
        return
    out = plot_dir / name
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}")


def discover_prefixes(session_dir):
    """Return sorted list of TEST_PREFIX values present in session_dir.
    A prefix is anything before '-ds4-server.log' OR '-mactop.log'."""
    prefixes = set()
    for p in session_dir.glob("*-ds4-server.log"):
        prefixes.add(p.name[: -len("-ds4-server.log")])
    for p in session_dir.glob("*-mactop.log"):
        prefixes.add(p.name[: -len("-mactop.log")])
    return sorted(prefixes)


def _compute_baseline_caps(session_dir, baseline):
    """Read the baseline scenario's logs and return Y-axis upper bounds to
    anchor comparison plots.  Returns a dict with optional keys:
        prefill_tps, gen_tps, dram_bw_gbs, power_w, total_power_w
    """
    caps = {}
    if not baseline:
        return caps
    ds4_log = session_dir / f"{baseline}-ds4-server.log"
    if ds4_log.exists():
        parsed = parse_ds4_log(ds4_log)
        prefill_vals = [c for (_, c, _, _, _) in parsed.get("prefill_samples", [])] + \
                       [a for (_, _, a, _, _) in parsed.get("prefill_samples", [])]
        gen_vals = [c for (_, c, _, _, _) in parsed.get("samples", [])] + \
                   [a for (_, _, a, _, _) in parsed.get("samples", [])]
        if prefill_vals:
            caps["prefill_tps"] = nice_ceil(max(prefill_vals) * 1.05)
        if gen_vals:
            caps["gen_tps"] = nice_ceil(max(gen_vals) * 1.05)
    mactop_log = session_dir / f"{baseline}-mactop.log"
    if mactop_log.exists():
        rows = mactop_timeline(parse_mactop(mactop_log))
        if rows:
            bw_max = max(max(r["dram_read_gbs"], r["dram_write_gbs"],
                             r["dram_combined_gbs"]) for r in rows)
            if bw_max > 0:
                caps["dram_bw_gbs"] = nice_ceil(bw_max * 1.05)
            pwr_max = max(max(r["cpu_power"], r["gpu_power"], r["ane_power"],
                              r["dram_power"], r["system_power"]) for r in rows)
            if pwr_max > 0:
                caps["power_w"] = nice_ceil(pwr_max * 1.05)
            tot_max = max(r["total_power"] for r in rows)
            if tot_max > 0:
                caps["total_power_w"] = nice_ceil(tot_max * 1.05)
    return caps


def process_session(
    session_dir,
    scenarios=None,
    iostat_interval=IOSTAT_INTERVAL_DEFAULT,
    vmstat_interval=VMSTAT_INTERVAL_DEFAULT,
    mactop_offset=0.0,
    baseline=None,
):
    plot_dir = session_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    all_prefixes = discover_prefixes(session_dir)
    if scenarios is None or len(scenarios) == 0:
        scenarios = all_prefixes
        if not scenarios:
            print(f"no *-ds4-server.log or *-mactop.log found under {session_dir}")
            return
    else:
        # Each --prefix value is first tried as an exact prefix (a real
        # <NAME>-ds4-server.log / -mactop.log).  If none exists, it is treated as
        # a startswith filter over the auto-detected prefixes, so e.g.
        # ``--prefix q2-`` selects every ``q2-*`` scenario.
        resolved = []
        seen = set()
        for req in scenarios:
            if ((session_dir / f"{req}-ds4-server.log").exists() or
                    (session_dir / f"{req}-mactop.log").exists()):
                cands = [req]
            else:
                cands = [p for p in all_prefixes if p.startswith(req)]
            if not cands:
                print(f"--prefix {req!r}: no matching scenario "
                      f"(exact <prefix>-ds4-server.log/-mactop.log nor startswith match)")
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    resolved.append(c)
        scenarios = resolved
        if not scenarios:
            print(f"no scenarios matched the given --prefix filter(s) under {session_dir}")
            return
    caps = _compute_baseline_caps(session_dir, baseline)
    if baseline and caps:
        print(f"baseline {baseline} caps: " + ", ".join(f"{k}={v:g}" for k, v in caps.items()))

    # Pre-scan to compute the longest wall-clock span across selected scenarios,
    # so every plot uses the same X-axis upper bound and short runs render with
    # right-side whitespace (= "this run finished faster") for direct side-by-
    # side comparison.
    tok_x_max = 0.0
    mactop_x_max = 0.0
    vmstat_x_max = 0.0
    iostat_x_max = 0.0
    # Per-scenario spans, so --x-target-minutes can extend ONLY the runs that
    # overrun the target (instead of one slow run stretching the shared axis).
    tok_span = {}
    mactop_span = {}
    vmstat_span = {}
    iostat_span = {}
    for scenario in scenarios:
        ds4 = session_dir / f"{scenario}-ds4-server.log"
        if ds4.exists():
            p = parse_ds4_log(ds4)
            for arr in (p.get("samples") or [], p.get("prefill_samples") or []):
                if arr:
                    tok_x_max = max(tok_x_max, arr[-1][0])
                    tok_span[scenario] = max(tok_span.get(scenario, 0.0), arr[-1][0])
        for vmcand in (session_dir / f"{scenario}-vmstat.log",
                       session_dir / f"{scenario}-vm_stat.log"):
            if vmcand.exists():
                rows = parse_vmstat(vmcand, interval=vmstat_interval)
                if rows:
                    vmstat_x_max = max(vmstat_x_max, rows[-1]["t"])
                    vmstat_span[scenario] = rows[-1]["t"]
                break
        io_log = session_dir / f"{scenario}-iostat.log"
        if io_log.exists():
            rows = parse_iostat(io_log, interval=iostat_interval)
            if rows:
                iostat_x_max = max(iostat_x_max, rows[-1]["t"])
                iostat_span[scenario] = rows[-1]["t"]
        mac_log = session_dir / f"{scenario}-mactop.log"
        if mac_log.exists():
            rows = mactop_timeline(parse_mactop(mac_log))
            if rows:
                mactop_x_max = max(mactop_x_max, rows[-1]["t"])
                mactop_span[scenario] = rows[-1]["t"]
    # Target span: each plot is pinned to the target width, but a run that
    # overruns the target keeps its own (larger) span so nothing is clipped.
    # Applied PER SCENARIO in the render loop below; here we just report which
    # runs overran.  (A single slow run no longer stretches the others.)
    if X_TARGET_SECONDS and X_TARGET_SECONDS > 0:
        over = sorted({s for d in (tok_span, mactop_span, vmstat_span, iostat_span)
                       for s, v in d.items() if v > X_TARGET_SECONDS})
        if over:
            print(f"x-target: {fmt_duration(X_TARGET_SECONDS)} pinned; "
                  f"{len(over)} run(s) overran -> full span: {', '.join(over)}")
        else:
            print(f"x-target: {fmt_duration(X_TARGET_SECONDS)} (all runs pinned)")
    if tok_x_max or mactop_x_max or vmstat_x_max:
        print(f"x-max: tok={tok_x_max:.1f}s mactop={mactop_x_max:.1f}s "
              f"vmstat={vmstat_x_max:.1f}s iostat={iostat_x_max:.1f}s")

    for scenario in scenarios:
        # Effective per-plot x-axis upper bound.  With --x-target-minutes, pin to
        # the target but extend to THIS scenario's own span when it overruns
        # (so one slow run doesn't stretch the others).  Without a target, use
        # the shared cross-scenario max so axes stay directly comparable.
        if X_TARGET_SECONDS and X_TARGET_SECONDS > 0:
            eff_tok = max(X_TARGET_SECONDS, tok_span.get(scenario, 0.0))
            eff_mactop = max(X_TARGET_SECONDS, mactop_span.get(scenario, 0.0))
            eff_vmstat = max(X_TARGET_SECONDS, vmstat_span.get(scenario, 0.0))
            eff_iostat = max(X_TARGET_SECONDS, iostat_span.get(scenario, 0.0))
        else:
            eff_tok, eff_mactop, eff_vmstat, eff_iostat = (
                tok_x_max, mactop_x_max, vmstat_x_max, iostat_x_max)
        ds4_log = session_dir / f"{scenario}-ds4-server.log"
        io_log = session_dir / f"{scenario}-iostat.log"
        vm_candidates = [
            session_dir / f"{scenario}-vmstat.log",
            session_dir / f"{scenario}-vm_stat.log",
        ]
        vm_log = next((p for p in vm_candidates if p.exists()), None)
        mactop_log = session_dir / f"{scenario}-mactop.log"

        phases = []
        sessions = []
        session_footer = None
        if ds4_log.exists():
            parsed = parse_ds4_log(ds4_log)
            phases = parsed.get("phases", [])
            sessions = parsed.get("sessions", [])
            # Build a rich header + footer per session: intent (title gen /
            # agent / tool-result / chat) from the trace log, cache
            # decision from the trace log, and prefill / gen / finish
            # timing from the server log.  Header carries the detail
            # table; footer carries the intent + excerpt summary.
            trace_log = session_dir / f"{scenario}-trace.log"
            trace_metas = parse_trace_request_meta(trace_log) if trace_log.exists() else []
            ds4_metrics = parse_ds4_session_metrics(ds4_log)
            if sessions:
                # In-plot annotation gets the short anchor ("#N [intent] <total>")
                # so the staggered markers stay readable even at high session
                # counts.  Detail metrics + excerpt live in the footer.
                short_lines = build_session_short_lines(
                    sessions, trace_metas, ds4_metrics)
                sessions = [(t, short_lines[i] if i < len(short_lines) else lbl)
                            for i, (t, lbl) in enumerate(sessions)]
                parsed["sessions"] = sessions
                session_footer = build_session_footer_lines(
                    sessions, trace_metas, ds4_metrics)
            write_svg(
                plot_dir,
                f"{scenario}-tok.svg",
                build_tok_plot(
                    parsed,
                    f"{scenario}: prefill + gen throughput",
                    y_prefill_max=caps.get("prefill_tps"),
                    y_gen_max=caps.get("gen_tps"),
                    x_max_override=eff_tok if eff_tok > 0 else None,
                    session_footer=session_footer, x_axis_format="time",
                ),
            )
        io_xmax = eff_iostat if eff_iostat > 0 else None
        vm_xmax = eff_vmstat if eff_vmstat > 0 else None
        if io_log.exists():
            rows = parse_iostat(io_log, interval=iostat_interval)
            write_svg(
                plot_dir,
                f"{scenario}-iostat-disk.svg",
                build_iostat_disk_plot(
                    rows, f"{scenario}: disk I/O (iostat {iostat_interval:g}s)", phases,
                    annotations=sessions, x_max_override=io_xmax,
                    session_footer=session_footer, x_axis_format="time"),
            )
            write_svg(
                plot_dir,
                f"{scenario}-iostat-cpu.svg",
                build_iostat_cpu_plot(
                    rows, f"{scenario}: CPU and load (iostat {iostat_interval:g}s)", phases,
                    annotations=sessions, x_max_override=io_xmax,
                    session_footer=session_footer, x_axis_format="time"),
            )
        if vm_log is not None:
            rows = parse_vmstat(vm_log, interval=vmstat_interval)
            write_svg(
                plot_dir,
                f"{scenario}-vmstat-mem.svg",
                build_vmstat_mem_plot(
                    rows, f"{scenario}: memory composition (vm_stat {vmstat_interval:g}s)", phases,
                    annotations=sessions, x_max_override=vm_xmax,
                    session_footer=session_footer, x_axis_format="time"),
            )
            write_svg(
                plot_dir,
                f"{scenario}-vmstat-events.svg",
                build_vmstat_events_plot(
                    rows, f"{scenario}: memory events (vm_stat {vmstat_interval:g}s)", phases,
                    annotations=sessions, x_max_override=vm_xmax,
                    session_footer=session_footer, x_axis_format="time"),
            )
        if mactop_log.exists():
            samples = parse_mactop(mactop_log)
            rows = mactop_timeline(samples)
            if not rows:
                print(f"skip {scenario}-mactop (no samples parsed)")
            else:
                print(f"{scenario}-mactop: {len(rows)} samples, "
                      f"span {rows[-1]['t']:.1f}s (offset={mactop_offset:+.1f}s)")
                mactop_bands = None
                if phases:
                    mactop_bands = [
                        (s - mactop_offset, e - mactop_offset, k)
                        for (s, e, k) in phases
                    ]
                mactop_annotations = [(t - mactop_offset, lbl) for (t, lbl) in sessions]
                mac_xmax = eff_mactop if eff_mactop > 0 else None
                write_svg(plot_dir, f"{scenario}-mactop-cpu-gpu.svg",
                          build_mactop_cpu_gpu(rows, f"{scenario}: CPU / GPU utilization", mactop_bands,
                                               annotations=mactop_annotations, x_max_override=mac_xmax,
                                               session_footer=session_footer, x_axis_format="time"))
                write_svg(plot_dir, f"{scenario}-mactop-freq.svg",
                          build_mactop_freq(rows, f"{scenario}: cluster frequencies", mactop_bands,
                                            annotations=mactop_annotations, x_max_override=mac_xmax,
                                            session_footer=session_footer, x_axis_format="time"))
                write_svg(plot_dir, f"{scenario}-mactop-power.svg",
                          build_mactop_power(
                              rows, f"{scenario}: power consumption", mactop_bands,
                              annotations=mactop_annotations,
                              x_max_override=mac_xmax,
                              session_footer=session_footer, x_axis_format="time",
                              y_power_max=caps.get("power_w"),
                              y_total_power_max=caps.get("total_power_w")))
                write_svg(plot_dir, f"{scenario}-mactop-dram-bw.svg",
                          build_mactop_dram_bw(
                              rows, f"{scenario}: DRAM bandwidth", mactop_bands,
                              annotations=mactop_annotations,
                              x_max_override=mac_xmax,
                              session_footer=session_footer, x_axis_format="time",
                              y_bw_max=caps.get("dram_bw_gbs")))
                write_svg(plot_dir, f"{scenario}-mactop-net-disk.svg",
                          build_mactop_net_disk(rows, f"{scenario}: disk I/O", mactop_bands,
                                                annotations=mactop_annotations, x_max_override=mac_xmax,
                                                session_footer=session_footer, x_axis_format="time"))
                write_svg(plot_dir, f"{scenario}-mactop-temp.svg",
                          build_mactop_temp(rows, f"{scenario}: temperatures", mactop_bands,
                                            annotations=mactop_annotations, x_max_override=mac_xmax,
                                            session_footer=session_footer, x_axis_format="time"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session_dir",
        type=Path,
        help="session directory containing *-ds4-server.log / *-iostat.log / *-vm_stat.log",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "TEST_PREFIX to process (matches <NAME>-ds4-server.log etc.).  If no "
            "exact match exists, NAME is used as a startswith filter over the "
            "auto-detected prefixes (e.g. '--prefix q2-' selects every q2-* "
            "scenario).  May be repeated.  Without --prefix, all prefixes present "
            "in session_dir are auto-detected."
        ),
    )
    parser.add_argument(
        "--iostat-interval",
        type=float,
        default=IOSTAT_INTERVAL_DEFAULT,
        metavar="SECONDS",
        help="iostat sample interval (default 30; older captures only).",
    )
    parser.add_argument(
        "--vmstat-interval",
        type=float,
        default=VMSTAT_INTERVAL_DEFAULT,
        metavar="SECONDS",
        help="vm_stat sample interval (default 5; pass 30 for 01-04 captures).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Scenario name (e.g. q2-metal) whose prefill/gen t/s and mactop "
            "DRAM bw / power maxima are used as fixed Y-axis upper bounds on "
            "all comparison plots in this session.  Without --baseline each "
            "plot auto-scales independently."
        ),
    )
    parser.add_argument(
        "--x-tick-minutes",
        type=float,
        default=None,
        metavar="MIN",
        help=(
            "Force the x-axis tick spacing on time plots to MIN minutes (e.g. "
            "1 = a tick every minute, 0.5 = every 30s).  Accepts fractions.  "
            "Without it, ~6 'nice' ticks are auto-chosen."
        ),
    )
    parser.add_argument(
        "--x-target-minutes",
        type=float,
        default=None,
        metavar="MIN",
        help=(
            "Target x-axis span in minutes.  Every time plot is pinned to at "
            "least this width (short runs show right-side whitespace = finished "
            "early); if a run overruns the target the axis auto-extends to the "
            "full data span so nothing is clipped.  Accepts fractions."
        ),
    )
    parser.add_argument(
        "--mactop-offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Offset to add to ds4-server-elapsed seconds when overlaying "
            "prefill/gen phase bands onto mactop plots.  Positive means mactop "
            "started later than ds4-server."
        ),
    )
    args = parser.parse_args()
    if not args.session_dir.is_dir():
        sys.exit(f"not a directory: {args.session_dir}")
    if args.x_tick_minutes is not None:
        if args.x_tick_minutes <= 0:
            sys.exit("--x-tick-minutes must be > 0")
        global X_TICK_SECONDS
        X_TICK_SECONDS = args.x_tick_minutes * 60.0
    if args.x_target_minutes is not None:
        if args.x_target_minutes <= 0:
            sys.exit("--x-target-minutes must be > 0")
        global X_TARGET_SECONDS
        X_TARGET_SECONDS = args.x_target_minutes * 60.0
    process_session(
        args.session_dir.resolve(),
        scenarios=args.prefix,
        iostat_interval=args.iostat_interval,
        vmstat_interval=args.vmstat_interval,
        mactop_offset=args.mactop_offset,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()
