from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interfaces import CommandPlan, Evaluator
from .schema import EvaluatorSpec

PYTHON_EXE = sys.executable or "python3"

_CONTRASTIVE_METRIC_COLS = (
    "avg_f1",
    "avg_skeleton_f1",
    "avg_ancestor_f1",
    "acyclic_rate",
    "consensus_f1",
    "consensus_skeleton_f1",
    "consensus_ancestor_f1",
    "consensus_acyclic",
)


@dataclass
class EvalScriptEvaluator(Evaluator):
    repo_root: Path

    def build_eval_command(self, *, response_csv: Path, evaluator: EvaluatorSpec) -> CommandPlan:
        cmd = [
            PYTHON_EXE,
            "evaluate.py",
            "--csv",
            str(response_csv),
            "--tau",
            str(evaluator.tau),
        ]
        if evaluator.answer_col:
            cmd.extend(["--answer-col", evaluator.answer_col])
        if evaluator.pred_col:
            cmd.extend(["--pred-col", evaluator.pred_col])
        return CommandPlan(label=f"evaluate:{response_csv.name}", cmd=cmd, cwd=self.repo_root / "experiments")


def direct_csv_summary(*, response_csv: Path, evaluator: EvaluatorSpec) -> dict[str, Any]:
    from experiments.evaluate import evaluate_response_csv

    result = evaluate_response_csv(
        response_csv,
        answer_col=evaluator.answer_col,
        pred_col=evaluator.pred_col,
        tau=evaluator.tau,
        write_artifacts=False,
        verbose=False,
    )
    return dict(result["summary"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def attach_contrastive_metrics(rows: list[dict[str, Any]]) -> None:
    metric_cols = _available_metric_cols(rows)
    if not metric_cols:
        return

    names_only_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    names_only_by_relaxed_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    naming_groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    representation_groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    obs_budget_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    int_budget_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for row in rows:
        naming = _naming_regime(row)
        row["naming_regime"] = naming

        if naming == "names_only":
            names_only_by_key.setdefault(_semantic_floor_key(row), row)
            names_only_by_relaxed_key.setdefault(_semantic_floor_relaxed_key(row), row)

        naming_key = _naming_pair_key(row)
        if naming_key is not None:
            naming_groups.setdefault(naming_key, {})[naming] = row

        repr_key = _representation_key(row)
        if repr_key is not None:
            representation_groups.setdefault(repr_key, {})[_prompt_style(row)] = row

        obs_key = _obs_budget_key(row)
        if obs_key is not None:
            obs_budget_groups.setdefault(obs_key, []).append(row)

        int_key = _int_budget_key(row)
        if int_key is not None:
            int_budget_groups.setdefault(int_key, []).append(row)

    prev_obs_by_row: dict[int, dict[str, Any]] = {}
    for group_rows in obs_budget_groups.values():
        prev: dict[str, Any] | None = None
        for row in sorted(group_rows, key=lambda r: (_coerce_int(r.get("obs_n")) or 0, str(r.get("response_csv") or ""))):
            obs_n = _coerce_int(row.get("obs_n"))
            if prev is not None:
                prev_n = _coerce_int(prev.get("obs_n"))
                if obs_n is not None and prev_n is not None and obs_n > prev_n:
                    prev_obs_by_row[id(row)] = prev
            prev = row

    prev_int_by_row: dict[int, dict[str, Any]] = {}
    for group_rows in int_budget_groups.values():
        prev = None
        for row in sorted(group_rows, key=lambda r: (_coerce_int(r.get("int_n")) or 0, str(r.get("response_csv") or ""))):
            int_n = _coerce_int(row.get("int_n"))
            if prev is not None:
                prev_n = _coerce_int(prev.get("int_n"))
                if int_n is not None and prev_n is not None and int_n > prev_n:
                    prev_int_by_row[id(row)] = prev
            prev = row

    for row in rows:
        naming = _naming_regime(row)
        names_only_row = names_only_by_key.get(_semantic_floor_key(row))
        if names_only_row is None:
            names_only_row = names_only_by_relaxed_key.get(_semantic_floor_relaxed_key(row))
        if names_only_row is not None:
            for metric in metric_cols:
                row[f"names_only_{metric}"] = names_only_row.get(metric)
                row[f"{metric}_minus_names_only"] = (
                    None
                    if naming == "names_only"
                    else _diff(
                        _coerce_float(row.get(metric)),
                        _coerce_float(names_only_row.get(metric)),
                    )
                )

        naming_key = _naming_pair_key(row)
        if naming_key is not None:
            naming_rows = naming_groups.get(naming_key, {})
            real_row = naming_rows.get("real")
            anonymized_row = naming_rows.get("anonymized")
            if real_row is not None or anonymized_row is not None:
                for metric in metric_cols:
                    real_val = _coerce_float(real_row.get(metric)) if real_row is not None else None
                    anon_val = _coerce_float(anonymized_row.get(metric)) if anonymized_row is not None else None
                    row[f"real_{metric}"] = real_val
                    row[f"anonymized_{metric}"] = anon_val
                    row[f"real_minus_anonymized_{metric}"] = _diff(real_val, anon_val)

        repr_key = _representation_key(row)
        if repr_key is not None:
            repr_rows = representation_groups.get(repr_key, {})
            summary_row = repr_rows.get("summary")
            matrix_row = repr_rows.get("matrix")
            if summary_row is not None or matrix_row is not None:
                for metric in metric_cols:
                    summary_val = _coerce_float(summary_row.get(metric)) if summary_row is not None else None
                    matrix_val = _coerce_float(matrix_row.get(metric)) if matrix_row is not None else None
                    row[f"summary_{metric}"] = summary_val
                    row[f"matrix_{metric}"] = matrix_val
                    row[f"summary_minus_matrix_{metric}"] = _diff(summary_val, matrix_val)

        prev_obs = prev_obs_by_row.get(id(row))
        if prev_obs is not None:
            row["prev_obs_n"] = _coerce_int(prev_obs.get("obs_n"))
            for metric in metric_cols:
                prev_val = _coerce_float(prev_obs.get(metric))
                row[f"prev_obs_{metric}"] = prev_val
                row[f"obs_budget_gain_{metric}"] = _diff(_coerce_float(row.get(metric)), prev_val)

        prev_int = prev_int_by_row.get(id(row))
        if prev_int is not None:
            row["prev_int_n"] = _coerce_int(prev_int.get("int_n"))
            for metric in metric_cols:
                prev_val = _coerce_float(prev_int.get(metric))
                row[f"prev_int_{metric}"] = prev_val
                row[f"int_budget_gain_{metric}"] = _diff(_coerce_float(row.get(metric)), prev_val)


def contamination_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_cols = _available_metric_cols(rows)
    grouped: dict[tuple[str, str, str, int | None, int | None], dict[str, float | None]] = {}
    grouped_metrics: dict[tuple[str, str, str, int | None, int | None], dict[str, dict[str, float | None]]] = {}
    for row in rows:
        method = row.get("system")
        if not method or method in {"ENCO", "PC", "GES", "GIES"}:
            continue
        key = (
            str(row.get("dataset")),
            str(row.get("prompt_style")),
            str(row.get("model")),
            _coerce_int(row.get("obs_n")),
            _coerce_int(row.get("int_n")),
        )
        grouped.setdefault(key, {})
        grouped_metrics.setdefault(key, {})
        naming = str(row.get("naming_regime"))
        grouped[key][naming] = _primary_metric_value(row)
        for metric in metric_cols:
            grouped_metrics[key].setdefault(metric, {})
            grouped_metrics[key][metric][naming] = _coerce_float(row.get(metric))

    audit_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        dataset, prompt_style, model, obs_n, int_n = key
        real = values.get("real")
        anonymized = values.get("anonymized")
        names_only = values.get("names_only")
        audit_row: dict[str, Any] = {
            "dataset": dataset,
            "prompt_style": prompt_style,
            "model": model,
            "obs_n": obs_n,
            "int_n": int_n,
            "real_f1": real,
            "anonymized_f1": anonymized,
            "names_only_f1": names_only,
            "real_minus_anonymized": _diff(real, anonymized),
            "real_minus_names_only": _diff(real, names_only),
            "anonymized_minus_names_only": _diff(anonymized, names_only),
        }
        for metric in metric_cols:
            metric_values = grouped_metrics[key].get(metric, {})
            real_val = metric_values.get("real")
            anon_val = metric_values.get("anonymized")
            names_only_val = metric_values.get("names_only")
            audit_row[f"real_{metric}"] = real_val
            audit_row[f"anonymized_{metric}"] = anon_val
            audit_row[f"names_only_{metric}"] = names_only_val
            audit_row[f"real_minus_anonymized_{metric}"] = _diff(real_val, anon_val)
            audit_row[f"real_minus_names_only_{metric}"] = _diff(real_val, names_only_val)
            audit_row[f"anonymized_minus_names_only_{metric}"] = _diff(anon_val, names_only_val)
        audit_rows.append(audit_row)
    return audit_rows


def _available_metric_cols(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    cols = {key for row in rows for key in row.keys()}
    return [metric for metric in _CONTRASTIVE_METRIC_COLS if metric in cols]


def _prompt_style(row: dict[str, Any]) -> str:
    return str(row.get("prompt_style") or "").strip().lower()


def _system_key(row: dict[str, Any]) -> str:
    for key in ("system", "parsed_model", "model"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _base_control_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("dataset") or ""),
        _system_key(row),
        str(row.get("reasoning_guidance") or ""),
        str(row.get("row_order") or ""),
        str(row.get("col_order") or ""),
        str(row.get("wrapper_mode") or ""),
        _coerce_int(row.get("append_format_hint")),
        _coerce_int(row.get("causal_rules")),
        _coerce_int(row.get("give_steps")),
    )


def _semantic_floor_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return _base_control_key(row)


def _semantic_floor_relaxed_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("dataset") or ""),
        _system_key(row),
    )


def _naming_regime(row: dict[str, Any]) -> str:
    raw = str(row.get("naming_regime") or "").strip().lower()
    if raw in {"real", "anonymized", "names_only"}:
        return raw
    if _coerce_int(row.get("is_names_only")) == 1 or _prompt_style(row) == "names_only":
        return "names_only"
    if _coerce_int(row.get("anonymize")) == 1:
        return "anonymized"
    return "real"


def _naming_pair_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    naming = _naming_regime(row)
    if naming == "names_only":
        return None
    obs_n = _coerce_int(row.get("obs_n"))
    int_n = _coerce_int(row.get("int_n"))
    if obs_n is None and int_n is None:
        return None
    return _base_control_key(row) + (_prompt_style(row), obs_n, int_n)


def _representation_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    prompt_style = _prompt_style(row)
    if prompt_style not in {"summary", "matrix"}:
        return None
    obs_n = _coerce_int(row.get("obs_n"))
    int_n = _coerce_int(row.get("int_n"))
    return _base_control_key(row) + (_naming_regime(row), obs_n, int_n)


def _obs_budget_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if _naming_regime(row) == "names_only":
        return None
    obs_n = _coerce_int(row.get("obs_n"))
    int_n = _coerce_int(row.get("int_n"))
    if obs_n is None:
        return None
    return _base_control_key(row) + (_naming_regime(row), _prompt_style(row), int_n)


def _int_budget_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if _naming_regime(row) == "names_only":
        return None
    obs_n = _coerce_int(row.get("obs_n"))
    int_n = _coerce_int(row.get("int_n"))
    if int_n is None:
        return None
    return _base_control_key(row) + (_naming_regime(row), _prompt_style(row), obs_n)


def _primary_metric_value(row: dict[str, Any]) -> float | None:
    return _coerce_float(row.get("consensus_f1") or row.get("avg_f1"))


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _diff(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)
