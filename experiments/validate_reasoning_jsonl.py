#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VAR_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$", re.M)


def _iter_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for p in inputs:
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        else:
            paths.append(p)
    return paths


def _extract_variables(prompt: str) -> list[str]:
    m = re.search(r"--- VARIABLE ORDER \(ORDER MATTERS\) ---\n(.*?)(?:\n\n|\n---)", prompt, re.S)
    if not m:
        m = re.search(r"--- VARIABLES ---\n(.*?)(?:\n\n|\n---)", prompt, re.S)
    block = m.group(1) if m else prompt
    pairs: list[tuple[int, str]] = []
    for mm in VAR_LINE_RE.finditer(block):
        pairs.append((int(mm.group(1)), mm.group(2).strip()))
    pairs.sort()
    return [v for _, v in pairs]


def _parse_answer_adj(answer: str) -> list[list[int]] | None:
    if "</think><answer>" not in answer or not answer.endswith("</answer>"):
        return None
    raw = answer.split("<answer>", 1)[1].rsplit("</answer>", 1)[0]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    mat = obj.get("adjacency_matrix")
    return mat if isinstance(mat, list) else None


def _gt_edges(adj: list[list[int]], variables: list[str]) -> set[tuple[str, str]]:
    return {
        (variables[i], variables[j])
        for i, row in enumerate(adj)
        for j, v in enumerate(row)
        if v == 1
    }


def _mentioned_edges(text: str, variables: list[str]) -> set[tuple[str, str]]:
    if not variables:
        return set()
    alt = "|".join(re.escape(v) for v in sorted(variables, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w.-])({alt})\s*->\s*({alt})(?![\w.-])")
    return set(pattern.findall(text or ""))


def _validate_record(rec: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    answer = str(rec.get("answer") or "")
    gold = str(rec.get("gold_think") or "")
    adj = _parse_answer_adj(answer)

    if adj is None:
        return ["answer JSON/tag parse failed"]

    expected_answer = (
        f"{gold}</think>"
        f"<answer>{json.dumps({'adjacency_matrix': adj}, ensure_ascii=False)}</answer>"
    )
    if answer != expected_answer:
        issues.append("answer is not exactly gold_think + </think><answer>{matrix}</answer>")

    n = len(adj)
    if any(not isinstance(row, list) or len(row) != n for row in adj):
        issues.append("adjacency matrix is not square")
    if any(v not in (0, 1) for row in adj for v in row):
        issues.append("adjacency matrix has non-binary entries")
    if "[ERROR]" in gold or "[ERROR]" in answer:
        issues.append("contains [ERROR]")

    if rec.get("reasoning_target") == "teacher_evidence":
        variables = _extract_variables(str(rec.get("prompt") or ""))
        if len(variables) != n:
            issues.append(f"variable count {len(variables)} != adjacency size {n}")
        else:
            extra = _mentioned_edges(gold, variables) - _gt_edges(adj, variables)
            if extra:
                issues.append(
                    "teacher mentioned non-GT edges: "
                    + ", ".join(f"{a}->{b}" for a, b in sorted(extra))
                )

    return issues


def _validate_path(path: Path) -> tuple[dict[str, Any], list[tuple[int, list[str], dict[str, Any]]]]:
    bad: list[tuple[int, list[str], dict[str, Any]]] = []
    summary: dict[str, Any] = {
        "rows": 0,
        "targets": {},
        "graphs": {},
        "duplicate_source_keys": 0,
    }
    seen_keys: set[tuple[str, str, int]] = set()
    duplicate_keys: set[tuple[str, str, int]] = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            summary["rows"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                bad.append((lineno, [f"json parse failed: {e}"], {}))
                continue
            if not isinstance(rec, dict):
                bad.append((lineno, ["record is not a JSON object"], {}))
                continue

            target = str(rec.get("reasoning_target") or "")
            graph = str(rec.get("graph") or "")
            summary["targets"][target] = summary["targets"].get(target, 0) + 1
            summary["graphs"][graph] = summary["graphs"].get(graph, 0) + 1

            row_idx = rec.get("source_row_idx")
            if row_idx is not None:
                key = (str(rec.get("source") or ""), target, int(row_idx))
                if key in seen_keys:
                    duplicate_keys.add(key)
                seen_keys.add(key)

            issues = _validate_record(rec)
            if issues:
                bad.append((lineno, issues, rec))

    summary["duplicate_source_keys"] = len(duplicate_keys)
    summary["bad_rows"] = len(bad)
    return summary, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path, help="JSONL files or directories to validate.")
    ap.add_argument("--max-errors", type=int, default=30)
    ap.add_argument(
        "--write-valid-only-dir",
        type=Path,
        default=None,
        help="Optional directory where cleaned JSONLs containing only valid rows are written.",
    )
    args = ap.parse_args()

    paths = _iter_paths(args.paths)
    total_bad = 0
    report: dict[str, Any] = {}

    if args.write_valid_only_dir is not None:
        args.write_valid_only_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        summary, bad = _validate_path(path)
        total_bad += len(bad)
        report[path.name] = summary

        if args.write_valid_only_dir is not None:
            bad_lines = {lineno for lineno, _issues, _rec in bad}
            out_path = args.write_valid_only_dir / path.name
            with path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
                for lineno, line in enumerate(src, 1):
                    if lineno not in bad_lines:
                        dst.write(line)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"errors: {total_bad}")

    shown = 0
    for path in paths:
        _summary, bad = _validate_path(path)
        for lineno, issues, _rec in bad:
            if shown >= int(args.max_errors):
                raise SystemExit(1 if total_bad else 0)
            print(f"{path.name}:{lineno}: " + "; ".join(issues))
            shown += 1

    if total_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
