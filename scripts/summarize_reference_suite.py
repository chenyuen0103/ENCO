#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit //= 10


DEFAULT_BENCHMARK_ROOT = Path("/Users/yuenc2/Desktop/ENCO/benchmark_data/reference_suite")
DEFAULT_BIF_ROOT = Path("/Users/yuenc2/Desktop/ENCO/causal_graphs/real_data/small_graphs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MICAD reference-suite benchmark CSV artifacts."
    )
    parser.add_argument(
        "--benchmark-root",
        default=str(DEFAULT_BENCHMARK_ROOT),
        help="Directory containing one CSV per dataset built by scripts/build_benchmark_data.py.",
    )
    parser.add_argument(
        "--bif-root",
        default=str(DEFAULT_BIF_ROOT),
        help="Directory containing <dataset>.bif files used to recover graph sizes.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path for per-dataset summary CSV.",
    )
    parser.add_argument(
        "--output-tex",
        default=None,
        help="Optional path for LaTeX table summarizing per-dataset statistics.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for full suite summary JSON.",
    )
    return parser.parse_args()


def parse_bif_stats(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    variable_blocks = re.findall(r"variable\s+([^{\s]+)\s*\{(.*?)\}", text, flags=re.S)
    probability_headers = re.findall(r"probability\s*\(\s*([^)]+)\)", text)

    categories: dict[str, int] = {}
    for name, body in variable_blocks:
        inner = body.split("{")[-1].split("}")[0]
        values = [v.strip() for v in inner.split(",") if v.strip()]
        categories[name.strip()] = len(values)

    edge_count = 0
    for header in probability_headers:
        if "|" not in header:
            continue
        parents = [p.strip() for p in header.split("|", 1)[1].split(",") if p.strip()]
        edge_count += len(parents)

    node_count = len(categories)
    density = (edge_count / (node_count * (node_count - 1))) if node_count > 1 else 0.0
    cardinalities = list(categories.values())
    return {
        "nodes": node_count,
        "edges": edge_count,
        "density": density,
        "cardinality_min": min(cardinalities) if cardinalities else None,
        "cardinality_median": median(cardinalities) if cardinalities else None,
        "cardinality_max": max(cardinalities) if cardinalities else None,
    }


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    xs = sorted(values)
    idx = round((len(xs) - 1) * q)
    idx = max(0, min(len(xs) - 1, idx))
    return xs[idx]


def maybe_count_tokens(prompt_text: str) -> int | None:
    if tiktoken is None:
        return None
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt_text))


def token_count_for_row(row: dict[str, str]) -> int | None:
    raw = (row.get("prompt_tokens_ref") or "").strip()
    if raw and raw != "-1":
        return int(raw)
    return maybe_count_tokens(row["prompt_text"])


def format_counter(counter: Counter[Any]) -> str:
    return ", ".join(f"{k}:{counter[k]}" for k in sorted(counter))


def format_budget_counter(counter: Counter[tuple[int, int]]) -> str:
    items = sorted(counter.items())
    return ", ".join(f"({obs},{intr}):{count}" for (obs, intr), count in items)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        "Dataset & Nodes & Edges & Density & Cells & Rows & Tok. min & Tok. mean & Tok. max \\\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(str(row["dataset"])),
                    str(row["nodes"]),
                    str(row["edges"]),
                    f'{float(row["density"]):.3f}',
                    str(row["benchmark_cells"]),
                    str(row["rows"]),
                    str(row["token_min"] if row["token_min"] is not None else "--"),
                    str(row["token_mean"] if row["token_mean"] is not None else "--"),
                    str(row["token_max"] if row["token_max"] is not None else "--"),
                ]
            )
            + r" \\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).resolve()
    bif_root = Path(args.bif_root).resolve()

    csv_paths = sorted(
        p for p in benchmark_root.glob("*.csv") if not p.name.endswith(".manifest.csv")
    )
    if not csv_paths:
        raise SystemExit(f"No benchmark CSVs found in {benchmark_root}")

    dataset_rows: list[dict[str, Any]] = []
    overall_cells = 0
    overall_rows = 0
    overall_tokens: list[int] = []
    cells_by_format: Counter[str] = Counter()
    cells_by_view: Counter[str] = Counter()
    cells_by_name: Counter[str] = Counter()
    cells_by_budget: Counter[tuple[int, int]] = Counter()

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue

        dataset = rows[0].get("dataset") or csv_path.stem
        bif_path = bif_root / f"{dataset}.bif"
        if not bif_path.exists():
            raise SystemExit(f"Missing BIF for dataset '{dataset}': {bif_path}")
        graph_stats = parse_bif_stats(bif_path)

        config_representatives: dict[str, dict[str, str]] = {}
        token_values: list[int] = []
        for row in rows:
            config_representatives.setdefault(row["config_name"], row)
            token_value = token_count_for_row(row)
            if token_value is not None:
                token_values.append(token_value)
                overall_tokens.append(token_value)

        configs = list(config_representatives.values())
        dataset_format_counts: Counter[str] = Counter()
        naming_counter: Counter[str] = Counter()
        view_counter: Counter[str] = Counter()
        budget_counter: Counter[tuple[int, int]] = Counter()

        for row in configs:
            prompt_format = (
                "names_only" if row["benchmark_view"] == "names_only" else row["prompt_style"]
            )
            dataset_format_counts[prompt_format] += 1
            view_counter[row["benchmark_view"]] += 1
            naming_counter["anonymized" if row["anonymize"] == "1" else "real"] += 1
            budget_counter[(int(row["obs_per_prompt"]), int(row["int_per_combo"]))] += 1

        cells_by_format.update(dataset_format_counts)
        cells_by_view.update(view_counter)
        cells_by_name.update(naming_counter)
        cells_by_budget.update(budget_counter)

        overall_rows += len(rows)
        overall_cells += len(configs)

        dataset_rows.append(
            {
                "dataset": dataset,
                "nodes": graph_stats["nodes"],
                "edges": graph_stats["edges"],
                "density": round(graph_stats["density"], 4),
                "cardinality_min": graph_stats["cardinality_min"],
                "cardinality_median": graph_stats["cardinality_median"],
                "cardinality_max": graph_stats["cardinality_max"],
                "rows": len(rows),
                "benchmark_cells": len(configs),
                "prompts_per_cell": round(len(rows) / len(configs), 2),
                "views": format_counter(view_counter),
                "naming": format_counter(naming_counter),
                "obs_grid": ",".join(str(v) for v in sorted({int(r["obs_per_prompt"]) for r in configs})),
                "int_grid": ",".join(str(v) for v in sorted({int(r["int_per_combo"]) for r in configs})),
                "budget_counts": format_budget_counter(budget_counter),
                "token_min": min(token_values) if token_values else None,
                "token_mean": round(mean(token_values), 1) if token_values else None,
                "token_max": max(token_values) if token_values else None,
            }
        )

    dataset_rows.sort(key=lambda row: (row["nodes"], row["dataset"]))

    suite_summary = {
        "datasets": len(dataset_rows),
        "total_rows": overall_rows,
        "total_benchmark_cells": overall_cells,
        "cells_by_format": dict(sorted(cells_by_format.items())),
        "cells_by_view": dict(sorted(cells_by_view.items())),
        "cells_by_name": dict(sorted(cells_by_name.items())),
        "cells_by_budget": [
            {"obs_per_prompt": obs, "int_per_combo": intr, "count": count}
            for (obs, intr), count in sorted(cells_by_budget.items())
        ],
        "tokenizer": "cl100k_base" if tiktoken is not None else "unavailable",
        "token_min": min(overall_tokens) if overall_tokens else None,
        "token_mean": round(mean(overall_tokens), 1) if overall_tokens else None,
        "token_max": max(overall_tokens) if overall_tokens else None,
    }

    print("Per-dataset benchmark summary")
    for row in dataset_rows:
        print(
            f"{row['dataset']:10s} "
            f"nodes={row['nodes']:>3} edges={row['edges']:>3} density={float(row['density']):.4f} "
            f"rows={row['rows']:>4} cells={row['benchmark_cells']:>3} "
            f"naming=[{row['naming']}] obs=[{row['obs_grid']}] int=[{row['int_grid']}] "
            f"tok(min/mean/max)={row['token_min']}/{row['token_mean']}/{row['token_max']}"
        )

    print("\nOverall suite summary")
    for key, value in suite_summary.items():
        print(f"{key}={value}")

    if args.output_csv:
        write_csv(Path(args.output_csv).resolve(), dataset_rows)
    if args.output_tex:
        write_tex(Path(args.output_tex).resolve(), dataset_rows)
    if args.output_json:
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"per_dataset": dataset_rows, "suite": suite_summary}, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
