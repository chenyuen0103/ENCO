from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interfaces import CommandPlan, Evaluator
from .schema import EvaluatorSpec

PYTHON_EXE = sys.executable or "python3"


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


def contamination_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int | None, int | None], dict[str, float | None]] = {}
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
        naming = str(row.get("naming_regime"))
        grouped[key][naming] = _coerce_float(row.get("consensus_f1") or row.get("avg_f1"))

    audit_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        dataset, prompt_style, model, obs_n, int_n = key
        real = values.get("real")
        anonymized = values.get("anonymized")
        names_only = values.get("names_only")
        audit_rows.append(
            {
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
        )
    return audit_rows


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
