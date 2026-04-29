#!/usr/bin/env python3
"""Collect the Sachs M-sweep results needed for the Figure 1 falsifiability plot."""

from __future__ import annotations

import ast
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_builder.graph_io import load_causal_graph
from experiments.evaluate import eval_pair


OUT_DIR = REPO_ROOT / "benchmark_runs" / "sachs_figure1"
SACHS_BIF = REPO_ROOT / "causal_graphs" / "real_data" / "small_graphs" / "sachs.bif"


def _parse_matrix(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue

        if isinstance(parsed, dict) and isinstance(parsed.get("adjacency_matrix"), list):
            arr = np.asarray(parsed["adjacency_matrix"]).astype(int)
        elif isinstance(parsed, list):
            arr = np.asarray(parsed).astype(int)
        else:
            continue
        np.fill_diagonal(arr, 0)
        return arr
    return None


def _preferred_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _required_cells() -> list[tuple[str, str, str, str, str, int, int, Path]]:
    cells: list[tuple[str, str, str, str, str, int, int, Path]] = []
    for m in [50, 100, 200]:
        cells.append(
            (
                "enco_ceiling",
                "ENCO",
                "data_only",
                "baseline",
                "anonymized",
                1000,
                m,
                _preferred_existing(
                    [
                        REPO_ROOT / f"experiments/responses/sachs/predictions_obs1000_int{m}_ENCO_seed42.csv",
                        REPO_ROOT / f"experiments/responses/sachs/predictions_obs1000_int{m}_ENCO.csv",
                    ]
                ),
            )
        )
    for m in [0, 50, 100, 200]:
        cells.append(
            (
                "llm_real",
                "gpt-5-mini",
                "mixed_information",
                "summary",
                "real",
                1000,
                m,
                _preferred_existing(
                    [
                        REPO_ROOT
                        / f"experiments/responses/sachs/responses_obs1000_int{m}_shuf1_p3_summary_joint_gpt-5-mini.csv",
                        REPO_ROOT
                        / f"experiments/responses/sachs/responses_obs1000_int{m}_shuf1_p5_summary_joint_gpt-5-mini.csv",
                    ]
                ),
            )
        )
        cells.append(
            (
                "llm_anonymized",
                "gpt-5-mini",
                "mixed_information",
                "summary",
                "anonymized",
                1000,
                m,
                _preferred_existing(
                    [
                        REPO_ROOT
                        / f"experiments/responses/sachs/responses_obs1000_int{m}_shuf1_p3_anon_summary_joint_gpt-5-mini.csv",
                        REPO_ROOT
                        / f"experiments/responses/sachs/responses_obs1000_int{m}_shuf1_p5_anon_summary_joint_gpt-5-mini.csv",
                    ]
                ),
            )
        )
    cells.extend(
        [
            (
                "semantic_floor",
                "gpt-5-mini",
                "semantic_only",
                "names_only",
                "names_only",
                0,
                0,
                _preferred_existing(
                    [
                        REPO_ROOT / "experiments/responses/sachs/responses_names_only_p5_gpt-5-mini.csv",
                        REPO_ROOT / "experiments/responses/sachs/responses_names_only_p3_gpt-5-mini.csv",
                    ]
                ),
            ),
            (
                "pc_anchor",
                "PC",
                "classical_observational",
                "baseline",
                "anonymized",
                1000,
                0,
                _preferred_existing(
                    [
                        REPO_ROOT / "experiments/responses/sachs/predictions_obs1000_int0_PC_seed42.csv",
                        REPO_ROOT / "experiments/responses/sachs/predictions_obs1000_int0_PC.csv",
                    ]
                ),
            ),
            (
                "ges_anchor",
                "GES",
                "classical_observational",
                "baseline",
                "anonymized",
                1000,
                0,
                _preferred_existing(
                    [
                        REPO_ROOT / "experiments/responses/sachs/predictions_obs1000_int0_GES_seed42.csv",
                        REPO_ROOT / "experiments/responses/sachs/predictions_obs1000_int0_GES.csv",
                    ]
                ),
            ),
        ]
    )
    return cells


def _stats(values: list[float]) -> tuple[int, float | str, float | str, float | str, float | str]:
    n = len(values)
    if not n:
        return 0, "", "", "", ""
    mean = sum(values) / n
    sd = (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / math.sqrt(n)
    ci95 = 1.96 * se if n > 1 else 0.0
    return n, mean, sd, se, ci95


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    graph = load_causal_graph(SACHS_BIF)
    answer = np.asarray(graph.adj_matrix).astype(int)
    np.fill_diagonal(answer, 0)

    per_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for line_id, system, info_class, prompt_style, naming, obs_n, int_n, path in _required_cells():
        rows: list[dict[str, Any]] = []
        valid = 0
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for idx, row in enumerate(reader):
                    rows.append(row)
                    pred = _parse_matrix(row.get("prediction"))
                    if pred is None:
                        pred = _parse_matrix(row.get("raw_response"))
                    if pred is None or pred.shape != answer.shape:
                        continue

                    metrics = eval_pair(answer, pred)
                    valid += 1
                    per_rows.append(
                        {
                            "line_id": line_id,
                            "system": system,
                            "information_class": info_class,
                            "prompt_style": prompt_style,
                            "naming_regime": naming,
                            "obs_n": obs_n,
                            "int_n": int_n,
                            "source_csv": str(path.relative_to(REPO_ROOT)),
                            "replicate_index": row.get("replicate_index") or row.get("data_idx") or idx,
                            "replicate_seed": row.get("replicate_seed", ""),
                            "shuffle_idx": row.get("shuffle_idx", ""),
                            "valid": 1,
                            **{
                                key: metrics.get(key)
                                for key in [
                                    "TP",
                                    "TN",
                                    "FP",
                                    "FN",
                                    "accuracy",
                                    "precision",
                                    "recall",
                                    "f1",
                                    "shd",
                                    "orientation_eval_pairs",
                                    "orientation_TP",
                                    "orientation_FN",
                                    "orientation_accuracy",
                                ]
                            },
                        }
                    )

        minimum = 1 if line_id in {"enco_ceiling", "pc_anchor", "ges_anchor"} else 3
        audit_rows.append(
            {
                "line_id": line_id,
                "system": system,
                "prompt_style": prompt_style,
                "naming_regime": naming,
                "obs_n": obs_n,
                "int_n": int_n,
                "source_csv": str(path.relative_to(REPO_ROOT)),
                "exists": int(path.exists()),
                "rows": len(rows),
                "valid_prediction_rows": valid,
                "minimum_required": minimum,
                "meets_minimum": int(valid >= minimum),
            }
        )

    per_fields = [
        "line_id",
        "system",
        "information_class",
        "prompt_style",
        "naming_regime",
        "obs_n",
        "int_n",
        "source_csv",
        "replicate_index",
        "replicate_seed",
        "shuffle_idx",
        "valid",
        "TP",
        "TN",
        "FP",
        "FN",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "shd",
        "orientation_eval_pairs",
        "orientation_TP",
        "orientation_FN",
        "orientation_accuracy",
    ]
    with (OUT_DIR / "figure1_per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fields)
        writer.writeheader()
        writer.writerows(per_rows)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in per_rows:
        key = (
            row["line_id"],
            row["system"],
            row["information_class"],
            row["prompt_style"],
            row["naming_regime"],
            row["obs_n"],
            row["int_n"],
            row["source_csv"],
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][6]), item[0][4])):
        f1_values = [float(row["f1"]) for row in rows if row.get("f1") not in (None, "")]
        shd_values = [float(row["shd"]) for row in rows if row.get("shd") not in (None, "")]
        n, f1_mean, f1_sd, f1_se, f1_ci95 = _stats(f1_values)
        _, shd_mean, shd_sd, _, _ = _stats(shd_values)
        line_id, system, info_class, prompt_style, naming, obs_n, int_n, source_csv = key
        summary_rows.append(
            {
                "line_id": line_id,
                "system": system,
                "information_class": info_class,
                "prompt_style": prompt_style,
                "naming_regime": naming,
                "obs_n": obs_n,
                "int_n": int_n,
                "source_csv": source_csv,
                "n_valid": n,
                "f1_mean": f1_mean,
                "f1_sd": f1_sd,
                "f1_se": f1_se,
                "f1_ci95_halfwidth": f1_ci95,
                "shd_mean": shd_mean,
                "shd_sd": shd_sd,
            }
        )

    summary_fields = [
        "line_id",
        "system",
        "information_class",
        "prompt_style",
        "naming_regime",
        "obs_n",
        "int_n",
        "source_csv",
        "n_valid",
        "f1_mean",
        "f1_sd",
        "f1_se",
        "f1_ci95_halfwidth",
        "shd_mean",
        "shd_sd",
    ]
    with (OUT_DIR / "figure1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    audit_fields = [
        "line_id",
        "system",
        "prompt_style",
        "naming_regime",
        "obs_n",
        "int_n",
        "source_csv",
        "exists",
        "rows",
        "valid_prediction_rows",
        "minimum_required",
        "meets_minimum",
    ]
    with (OUT_DIR / "figure1_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    missing = [row for row in audit_rows if not row["meets_minimum"]]
    readme = [
        "# Sachs Figure 1 Result Collection",
        "",
        "Collected from saved response/prediction CSVs and evaluated against `causal_graphs/real_data/small_graphs/sachs.bif`.",
        "",
        "## Audit",
        "",
    ]
    if missing:
        readme.append("Missing or under-replicated cells:")
        for row in missing:
            readme.append(
                f"- {row['line_id']} {row['system']} {row['prompt_style']} {row['naming_regime']} "
                f"N={row['obs_n']} M={row['int_n']}: valid={row['valid_prediction_rows']} "
                f"rows={row['rows']} exists={row['exists']}"
            )
    else:
        readme.append("All required Figure 1 cells meet the minimum replicate count.")
    readme.extend(["", "## Summary F1", ""])
    for row in summary_rows:
        readme.append(
            f"- {row['line_id']} {row['naming_regime']} M={row['int_n']}: "
            f"F1={float(row['f1_mean']):.3f} (n={row['n_valid']}, sd={float(row['f1_sd']):.3f})"
        )
    (OUT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    for path in ["figure1_per_run.csv", "figure1_summary.csv", "figure1_audit.csv", "README.md"]:
        print(OUT_DIR / path)
    print("ALL_REQUIRED_CELLS_PRESENT" if not missing else "UNDER_REPLICATED")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
