# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Plot TTFT/TPOT vs input length across multiple experiments.

Intended for experiments produced by text_ttft_tpot_input_size.yaml.
We group points by an experiment label (e.g., "label") and plot bars
for TTFT and TPOT per input length, labeling bars by experiment.

Usage example:
  python3 plot_ttft_tpot_vs_input.py \
    --inputs "results/**/detailed_metrics/*.json" \
    --group-field label \
    --output ttft_tpot_vs_input.png

Assumptions:
  - Results JSON contains fields: median_ttft_ms, median_tpot_ms.
  - Input length can be inferred by best effort from metadata or filename.
    If not present, we try to read "random_input_len" from metadata; else
    parse from filename pattern "*_<inputlen>_*.json".
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from contextlib import suppress
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import regex as re


def _read_json_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return records
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if len(lines) > 1:
            for ln in lines:
                with suppress(Exception):
                    records.append(json.loads(ln))
            if records:
                return records
        with suppress(Exception):
            obj = json.loads(content)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _infer_input_len(rec: dict[str, Any], path: str) -> int | None:
    # Prefer explicit metadata if present
    for k in ("random_input_len", "input_len", "input_length"):
        if k in rec:
            with suppress(Exception):
                return int(rec[k])

    # Try filename pattern ..._<num>_... .json
    m = re.search(r"_(\d+)_.*\.json$", os.path.basename(path))
    if m:
        with suppress(Exception):
            return int(m.group(1))
    return None


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
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path. If omitted, show interactively.",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="median",
        choices=["mean", "median"],
        help="Use mean or median for TTFT/TPOT bars (default: median).",
    )
    parser.add_argument(
        "--group-field",
        type=str,
        default="label",
        help=(
            "Record field to group experiments by (e.g., label, model_id). "
            "Bars at each input length are separated per unique group value."
        ),
    )
    args = parser.parse_args()

    file_paths: list[str] = []
    for pat in args.inputs:
        expanded = glob.glob(pat, recursive=True)
        if expanded:
            file_paths.extend(expanded)
        elif os.path.isfile(pat):
            file_paths.append(pat)

    # Map (input_len) -> list of (group_value, ttft_ms, tpot_ms)
    data: dict[int, list[tuple[str, float, float]]] = defaultdict(list)

    for path in sorted(set(file_paths)):
        for rec in _read_json_records(path):
            input_len = _infer_input_len(rec, path)
            if input_len is None:
                continue
            group_value = rec.get(args.group_field)
            if group_value is None:
                group_value = "(unknown)"
            # pick stat
            if args.stat == "mean":
                ttft_key, tpot_key = "mean_ttft_ms", "mean_tpot_ms"
            else:
                ttft_key, tpot_key = "median_ttft_ms", "median_tpot_ms"
            ttft = rec.get(ttft_key)
            tpot = rec.get(tpot_key)
            if ttft is None or tpot is None:
                continue
            with suppress(Exception):
                data[int(input_len)].append(
                    (str(group_value), float(ttft), float(tpot)))

    # Prepare plotting: for each input_len, multiple bars labeled by concurrency
    input_lens = sorted(data.keys())
    group_labels = sorted({g for rows in data.values() for (g, _, _) in rows})

    # Build a matrix [len(group_labels) x len(input_lens)] for TTFT/TPOT
    import numpy as np
    ttft_matrix = np.full((len(group_labels), len(input_lens)), np.nan)
    tpot_matrix = np.full((len(group_labels), len(input_lens)), np.nan)

    group_to_idx = {g: i for i, g in enumerate(group_labels)}
    len_to_idx = {input_length: j for j, input_length in enumerate(input_lens)}

    for input_length in input_lens:
        for g, ttft, tpot in data[input_length]:
            ttft_matrix[group_to_idx[g], len_to_idx[input_length]] = ttft
            tpot_matrix[group_to_idx[g], len_to_idx[input_length]] = tpot

    # Plot grouped bars per input length; each group has bars for each lenght
    width = 0.35
    x = np.arange(len(input_lens))

    plt.figure(figsize=(10, 5))
    for i, g in enumerate(group_labels):
        # offset bars for TTFT
        plt.bar(
            x + (i - len(group_labels) / 2)
            * (width / max(1, len(group_labels))),
            ttft_matrix[i, :],
            width / max(1, len(group_labels)),
            label=f"TTFT ({args.group_field}={g})",
            alpha=0.8,
        )

    for i, g in enumerate(group_labels):
        # offset bars for TPOT (stacked group to the right)
        plt.bar(
            x
            + (i - len(group_labels) / 2)
            * (width / max(1, len(group_labels)))
            + width,
            tpot_matrix[i, :],
            width / max(1, len(group_labels)),
            label=f"TPOT ({args.group_field}={g})",
            alpha=0.6,
        )

    plt.xticks(
        x + width / 2,
        [str(input_length) for input_length in input_lens],
    )
    plt.xlabel("Input length")
    plt.ylabel(f"{args.stat.title()} latency (ms)")
    plt.title(
        f"TTFT/TPOT vs Input Length by "
        f"{args.group_field.replace('_', ' ').title()}"
    )
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        plt.savefig(args.output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()


