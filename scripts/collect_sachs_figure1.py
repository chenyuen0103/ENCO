#!/usr/bin/env python3
"""Collect the Sachs M-sweep results needed for the Figure 1 falsifiability plot."""

from __future__ import annotations

import ast
import argparse
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
SACHS_SUMMARY_CSV = REPO_ROOT / "scripts" / "responses" / "sachs" / "sachs_summary.csv"
DEFAULT_LLM_MODEL = "gpt-5-pro"


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


def _required_cells(llm_model: str) -> list[tuple[str, str, str, str, str, int, int, Path]]:
    cells: list[tuple[str, str, str, str, str, int, int, Path]] = []
    for n in [1000, 5000]:
        cells.append(
            (
                "enco_observational",
                "ENCO",
                "data_only",
                "baseline",
                "anonymized",
                n,
                0,
                REPO_ROOT / f"experiments/responses/sachs/predictions_obs{n}_int0_ENCO.csv",
            )
        )
    for m in [50, 100, 200, 500]:
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
                llm_model,
                "mixed_information",
                "summary",
                "real",
                1000,
                m,
                SACHS_SUMMARY_CSV,
            )
        )
        cells.append(
            (
                "llm_anonymized",
                llm_model,
                "mixed_information",
                "summary",
                "anonymized",
                1000,
                m,
                SACHS_SUMMARY_CSV,
            )
        )
    cells.extend(
        [
            (
                "semantic_floor",
                llm_model,
                "semantic_only",
                "names_only",
                "names_only",
                0,
                0,
                SACHS_SUMMARY_CSV,
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


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _int_field(row: dict[str, Any], key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    if raw in (None, ""):
        return default
    return int(float(raw))


def _float_or_blank(raw: Any) -> float | str:
    if raw in (None, ""):
        return ""
    return float(raw)


def _has_metric(row: dict[str, Any]) -> bool:
    return (row.get("avg_f1") or row.get("avg_F1")) not in (None, "")


def _candidate_rank(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    response_csv = row.get("response_csv", "")
    return (
        int(_has_metric(row)),
        _int_field(row, "valid_rows", 0),
        _int_field(row, "num_rows", 0),
        int("_p5_" in response_csv or "_p5" in response_csv),
        response_csv,
    )


def _source_csv(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _find_sachs_summary_row(
    rows: list[dict[str, Any]],
    *,
    system: str,
    prompt_style: str,
    naming: str,
    obs_n: int,
    int_n: int,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("model") != system or row.get("dataset") != "sachs":
            continue
        if row.get("prompt_style") != prompt_style:
            continue

        if prompt_style == "names_only":
            if _int_field(row, "is_names_only", 0) == 1:
                candidates.append(row)
            continue

        if _int_field(row, "obs_n") != obs_n or _int_field(row, "int_n") != int_n:
            continue
        expected_anon = 1 if naming == "anonymized" else 0
        if _int_field(row, "anonymize") != expected_anon:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=_candidate_rank)


def _summary_output_row(
    row: dict[str, Any],
    *,
    line_id: str,
    system: str,
    info_class: str,
    prompt_style: str,
    naming: str,
    obs_n: int,
    int_n: int,
    source_csv: str,
) -> dict[str, Any]:
    n = _int_field(row, "valid_rows", 0)
    f1_se = _float_or_blank(row.get("avg_f1_se"))
    f1_ci95 = 1.96 * f1_se if isinstance(f1_se, float) and n > 1 else 0.0
    return {
        "line_id": line_id,
        "system": system,
        "information_class": info_class,
        "prompt_style": prompt_style,
        "naming_regime": naming,
        "obs_n": obs_n,
        "int_n": int_n,
        "source_csv": source_csv,
        "n_valid": n,
        "f1_mean": _float_or_blank(row.get("avg_f1") or row.get("avg_F1")),
        "f1_sd": _float_or_blank(row.get("avg_f1_sd")),
        "f1_se": f1_se,
        "f1_ci95_halfwidth": f1_ci95,
        "shd_mean": _float_or_blank(row.get("avg_shd")),
        "shd_sd": _float_or_blank(row.get("avg_shd_sd")),
    }


def _stats(values: list[float]) -> tuple[int, float | str, float | str, float | str, float | str]:
    n = len(values)
    if not n:
        return 0, "", "", "", ""
    mean = sum(values) / n
    sd = (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / math.sqrt(n)
    ci95 = 1.96 * se if n > 1 else 0.0
    return n, mean, sd, se, ci95


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    aliases = {
        "TP": "tp",
        "TN": "tn",
        "FP": "fp",
        "FN": "fn",
        "orientation_eval_pairs": "orient_eval_pairs",
        "orientation_TP": "orient_tp",
        "orientation_FN": "orient_fn",
        "orientation_accuracy": "orient_acc",
    }
    value = metrics.get(key)
    if value is None:
        value = metrics.get(aliases.get(key, key))
    if value is None and key in {"precision", "f1"} and metrics.get("tp") == 0 and metrics.get("fp") == 0:
        return 0.0
    return value


def _available_models(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({row["model"] for row in rows if row.get("dataset") == "sachs" and row.get("model")})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"Language model to import from sachs_summary.csv (default: {DEFAULT_LLM_MODEL}).",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=SACHS_SUMMARY_CSV)
    parser.add_argument("--list-models", action="store_true", help="List Sachs models in --summary-csv and exit.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any required Figure 1 cell is missing or under-replicated.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    sachs_summary_csv = args.summary_csv
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = load_causal_graph(SACHS_BIF)
    answer = np.asarray(graph.adj_matrix).astype(int)
    np.fill_diagonal(answer, 0)

    per_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    imported_summary_rows: list[dict[str, Any]] = []
    sachs_summary_rows = _read_csv_rows(sachs_summary_csv)
    if args.list_models:
        for model in _available_models(sachs_summary_rows):
            print(model)
        return 0
    for line_id, system, info_class, prompt_style, naming, obs_n, int_n, path in _required_cells(args.model):
        if path == SACHS_SUMMARY_CSV:
            path = sachs_summary_csv
        rows: list[dict[str, Any]] = []
        valid = 0
        minimum = 1 if line_id in {"enco_ceiling", "enco_observational", "pc_anchor", "ges_anchor"} else 3
        if system == args.model and path == sachs_summary_csv:
            summary_row = _find_sachs_summary_row(
                sachs_summary_rows,
                system=system,
                prompt_style=prompt_style,
                naming=naming,
                obs_n=obs_n,
                int_n=int_n,
            )
            valid = _int_field(summary_row, "valid_rows", 0) if summary_row else 0
            if summary_row:
                imported_summary_rows.append(
                    _summary_output_row(
                        summary_row,
                        line_id=line_id,
                        system=system,
                        info_class=info_class,
                        prompt_style=prompt_style,
                        naming=naming,
                        obs_n=obs_n,
                        int_n=int_n,
                        source_csv=_source_csv(path),
                    )
                )
            audit_rows.append(
                {
                    "line_id": line_id,
                    "system": system,
                    "prompt_style": prompt_style,
                    "naming_regime": naming,
                    "obs_n": obs_n,
                    "int_n": int_n,
                    "source_csv": _source_csv(path),
                    "exists": int(path.exists()),
                    "rows": _int_field(summary_row, "num_rows", 0) if summary_row else 0,
                    "valid_prediction_rows": valid,
                    "minimum_required": minimum,
                    "meets_minimum": int(valid >= minimum),
                }
            )
            continue

        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for idx, row in enumerate(reader):
                    rows.append(row)
                    row_answer = _parse_matrix(row.get("answer"))
                    if row_answer is None:
                        row_answer = answer
                    pred = _parse_matrix(row.get("prediction"))
                    if pred is None:
                        pred = _parse_matrix(row.get("raw_response"))
                    if pred is None or pred.shape != row_answer.shape:
                        continue

                    metrics = eval_pair(row_answer, pred)
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
                            "source_csv": _source_csv(path),
                            "replicate_index": row.get("replicate_index") or row.get("data_idx") or idx,
                            "replicate_seed": row.get("replicate_seed", ""),
                            "shuffle_idx": row.get("shuffle_idx", ""),
                            "valid": 1,
                            **{
                                key: _metric_value(metrics, key)
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

        audit_rows.append(
            {
                "line_id": line_id,
                "system": system,
                "prompt_style": prompt_style,
                "naming_regime": naming,
                "obs_n": obs_n,
                "int_n": int_n,
                "source_csv": _source_csv(path),
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
    with (out_dir / "figure1_per_run.csv").open("w", newline="", encoding="utf-8") as handle:
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

    summary_rows: list[dict[str, Any]] = list(imported_summary_rows)
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
    summary_rows.sort(key=lambda row: (row["line_id"], int(row["int_n"]), row["naming_regime"]))

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
    with (out_dir / "figure1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
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
    with (out_dir / "figure1_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    missing = [row for row in audit_rows if not row["meets_minimum"]]
    readme = [
        "# Sachs Figure 1 Result Collection",
        "",
        f"Collected from saved response/prediction CSVs and `{_source_csv(sachs_summary_csv)}` for `{args.model}`; raw predictions are evaluated against `causal_graphs/real_data/small_graphs/sachs.bif` when needed.",
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
        f1_text = f"{float(row['f1_mean']):.3f}" if row["f1_mean"] not in (None, "") else "NA"
        sd_text = f"{float(row['f1_sd']):.3f}" if row["f1_sd"] not in (None, "") else "NA"
        readme.append(
            f"- {row['line_id']} {row['naming_regime']} M={row['int_n']}: "
            f"F1={f1_text} (n={row['n_valid']}, sd={sd_text})"
        )
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    for path in ["figure1_per_run.csv", "figure1_summary.csv", "figure1_audit.csv", "README.md"]:
        print(out_dir / path)
    print("ALL_REQUIRED_CELLS_PRESENT" if not missing else "UNDER_REPLICATED")
    return 1 if missing and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
