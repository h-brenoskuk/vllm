# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Plot pXX metric vs max concurrency from saved benchmark JSON results.

Usage examples:
  python3 plot_max_concurrency.py \
    --inputs "e2el_percentiles/*.json" \
    --metric-key p95_e2el_ms \
    --output max_concurrency_p95.png


Notes:
  - Accepts a glob or multiple file paths via repeated --inputs.
  - Supports both single-JSON files and newline-delimited JSON (append mode).
  - Default metric-key is p95_e2el_ms.
"""

import argparse
import glob
import json
import os
from contextlib import suppress
from typing import Any

import matplotlib.pyplot as plt  # type: ignore


def _read_json_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return records
        # Try NDJSON first
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if len(lines) > 1:
            for ln in lines:
                with suppress(Exception):
                    records.append(json.loads(ln))
            if records:
                return records
        # Fallback: single JSON
        with suppress(Exception):
            obj = json.loads(content)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _coerce_value(raw: str) -> Any:
    s = raw.strip()
    # Strip surrounding quotes if present
    if (s.startswith("\"") and s.endswith("\"")) or (
        s.startswith("'") and s.endswith("'")
    ):
        s = s[1:-1]
    # Special tokens
    low = s.lower()
    if low == "null":
        return None
    if low in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if low in {"-inf", "-infinity"}:
        return float("-inf")
    if low in {"nan"}:
        return float("nan")
    # Try int, then float
    with suppress(Exception):
        return int(s)
    with suppress(Exception):
        return float(s)
    return s


def _read_kv_blocks(path: str) -> list[dict[str, Any]]:
    """Parse simple key-per-line blocks.

    Expected layout example:
      root
      date
      20250826-104313
      max_concurrency
      8
      __

    Multiple blocks may be separated by lines equal to "root" or "__".
    """
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    records: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line in {"root", "__"}:
            if cur:
                records.append(cur)
            cur = {}
            i += 1
            continue

        # Treat current line as key; next line as value if available
        key = line.strip('"')
        value: Any = None
        if i + 1 < n and lines[i + 1] not in {"root", "__"}:
            value = _coerce_value(lines[i + 1])
            i += 2
        else:
            # If value is glued to key (e.g., max_concurrency8),
            # best-effort split by trailing digits/float.
            import regex as _re

            m = _re.match(r"^([A-Za-z0-9_]+)(.*)$", key)
            if m:
                key = m.group(1)
                tail = m.group(2)
                value = _coerce_value(tail) if tail else None
            i += 1

        if cur is None:
            cur = {}
        if key:
            cur[key] = value

    if cur:
        records.append(cur)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSON files or glob patterns.",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default="p95_ttft_ms",
        help=(
            "Metric key to plot as Y (e.g., p95_ttft_ms, p95_e2el_ms, "
            "p95_tpot_ms)."
        ),
    )
    parser.add_argument(
        "--x-field",
        type=str,
        default="max_concurrency",
        help="Field name for X axis (default: max_concurrency).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path. If omitted, show interactively.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="pXX vs Max Concurrency",
        help="Figure title.",
    )
    args = parser.parse_args()

    # Expand globs
    file_paths: list[str] = []
    for pat in args.inputs:
        expanded = glob.glob(pat, recursive=True)
        if expanded:
            file_paths.extend(expanded)
        elif os.path.isfile(pat):
            file_paths.append(pat)

    xs: list[float] = []
    ys: list[float] = []

    all_keys: set[str] = set()
    for path in sorted(set(file_paths)):
        # Try JSON/NDJSON first
        recs = _read_json_records(path)
        if not recs:
            # Fallback to key-per-line blocks
            recs = _read_kv_blocks(path)
        for rec in recs:
            all_keys.update(rec.keys())
            if args.metric_key not in rec:
                continue
            x_val = rec.get(args.x_field)
            y_val = rec.get(args.metric_key)
            if x_val is None or y_val is None:
                continue
            with suppress(Exception):
                xs.append(float(x_val))
                ys.append(float(y_val))

    # Sort by X for nicer plotting
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    if not pairs:
        # Provide a helpful diagnostic if wrong keys were used
        print(
            "No data points collected. Check --metric-key and --x-field.",
        )
        # Show a short list of available keys
        sample_keys = sorted(all_keys)
        if sample_keys:
            # Prefer to surface common metric-looking keys
            metric_like = [
                k for k in sample_keys if k.endswith("_ms") or k.startswith("p")
            ]
            shown = metric_like[:20] or sample_keys[:20]
            print("Available keys (sample):")
            for k in shown:
                print(" -", k)
            if args.metric_key not in all_keys:
                print(
                    "Hint: Your files contain the keys above; "
                    "you may want --metric-key p95_e2el_ms and "
                    "--x-field max_concurrency."
                )
        return

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(args.x_field.replace("_", " "))
    plt.ylabel(args.metric_key)
    plt.title(args.title)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        plt.savefig(args.output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()


