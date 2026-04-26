#!/usr/bin/env python3
"""
Generate a LaTeX table for an LLM baseline sweep (table-range grid), comparable to ENCO.

Expected inputs:
  - consolidated summary CSV: experiments/responses/<dataset>/<dataset>_summary.csv

The table cells report F1 with SHD in parentheses, using either:
  - consensus_{f1,shd} (recommended when there are multiple rows / shuffles), or
  - avg_{f1,shd}

Example:
  python experiments/make_llm_baseline_table.py \
    --dataset sachs \
    --model gpt-5-mini \
    --prompt-style summary \
    --metric consensus
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Optional


_RESP_RE = re.compile(
    r"^responses_obs(?P<obs>\d+)_int(?P<int>\d+)_shuf(?P<shuf>\d+)(?P<tags>.*?)(?:_(?P<model>[^_]+))?$",
    flags=re.IGNORECASE,
)
_ENCO_PRED_RE = re.compile(r"^predictions_obs(?P<obs>\d+)_int(?P<int>\d+)_ENCO\.csv$", flags=re.IGNORECASE)


def _has_style_tag(tags: str, style: str) -> bool:
    tag_tokens = {tok for tok in tags.strip("_").split("_") if tok}
    if style == "cases":
        # "cases" is untagged in this repo: treat rows without explicit style tags as cases.
        explicit = {"matrix", "summary", "payload", "payload_topk"}
        return len(tag_tokens.intersection(explicit)) == 0
    # Multi-token styles are encoded as substrings in filenames/tags.
    if style in {"payload_topk"}:
        return style in tags
    return style in tag_tokens


def _parse_int_list(values: list[str]) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for part in values:
        for tok in part.replace(",", " ").split():
            if tok:
                out.append(int(tok))
    return out


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _fmt_cell(f1: float, shd: int) -> str:
    return f"{f1:.2f} ({shd})"


def _render_table(
    *,
    dataset: str,
    model: str,
    prompt_style: str,
    metric: str,
    anonymize: str,
    obs_sizes: Iterable[int],
    int_sizes: Iterable[int],
    values: dict[tuple[int, int], tuple[float, int]],
    label: str,
) -> str:
    obs_sizes = list(obs_sizes)
    int_sizes = list(int_sizes)

    col_spec = "l" + ("c" * len(int_sizes))
    header_cells = [r"Obs $N$ / Int $M$"] + [str(m) for m in int_sizes]

    lines: list[str] = []
    lines.append(r"% Requires \usepackage{booktabs}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for n in obs_sizes:
        row = [str(n)]
        for m in int_sizes:
            v = values.get((n, m))
            row.append("--" if v is None else _fmt_cell(*v))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    pretty_style = {
        "payload_topk": "payload (top-K)",
    }.get(prompt_style, prompt_style)

    caption = (
        f"LLM baseline ({_latex_escape(pretty_style)} prompts) on {_latex_escape(dataset)}, "
        f"model={_latex_escape(model)}, anonymize={_latex_escape(anonymize)}, "
        f"metric={_latex_escape(metric)}. "
        r"Cells report F1 (higher is better) with SHD in parentheses (lower is better)."
    )
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows[0]


def _to_int(x: object) -> int | None:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _to_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _render_enco_table(
    *,
    dataset: str,
    obs_sizes: list[int],
    int_sizes: list[int],
    values: dict[tuple[int, int], tuple[float, int]],
    label: str,
) -> str:
    col_spec = "l" + ("c" * len(int_sizes))
    header_cells = [r"Obs $N$ / Int $M$"] + [str(m) for m in int_sizes]

    lines: list[str] = []
    lines.append(r"% Requires \usepackage{booktabs}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for n in obs_sizes:
        row = [str(n)]
        for m in int_sizes:
            v = values.get((n, m))
            row.append("--" if v is None else _fmt_cell(*v))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{ENCO baseline on {_latex_escape(dataset)}. Cells report F1 (higher is better) with SHD in parentheses (lower is better).}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _build_enco_tex_from_responses(
    *,
    dataset: str,
    responses_dir: Path,
    obs_sizes: list[int],
    int_sizes: list[int],
    label: str,
) -> str:
    values: dict[tuple[int, int], tuple[float, int]] = {}
    for csv_path in sorted(responses_dir.glob("predictions_obs*_int*_ENCO.csv")):
        m = _ENCO_PRED_RE.match(csv_path.name)
        if not m:
            continue
        obs_n = int(m.group("obs"))
        int_m = int(m.group("int"))
        try:
            row = _read_single_row_csv(csv_path)
        except Exception:
            continue
        f1 = _to_float(row.get("f1"))
        shd = _to_int(row.get("SHD"))
        if f1 is None or shd is None:
            continue
        values[(obs_n, int_m)] = (float(f1), int(shd))

    if not values:
        raise FileNotFoundError(
            f"No ENCO prediction CSVs found under {responses_dir} (expected predictions_obs*_int*_ENCO.csv)."
        )
    return _render_enco_table(
        dataset=dataset,
        obs_sizes=obs_sizes,
        int_sizes=int_sizes,
        values=values,
        label=label,
    )


def _render_comparison_table(
    *,
    dataset: str,
    model: str,
    metric: str,
    anonymize: str,
    obs_sizes: list[int],
    int_sizes: list[int],
    # values[(method, obs_n, int_n)] = (f1, shd)
    values: dict[tuple[str, int, int], tuple[float, int]],
    methods: list[tuple[str, str]],
    excluded: set[tuple[str, int, int]] | None = None,
    label: str,
) -> str:
    """
    Render a comparison grid with (obs_n as rows) x (int_n as column groups) and
    one subcolumn per method.
    """
    col_spec = "l" + ("c" * (len(int_sizes) * len(methods)))

    dataset_title = dataset[:1].upper() + dataset[1:]
    anon_text = r"\textbf{variable names hidden}" if anonymize == "anon" else r"\textbf{variable names given}"
    caption = (
        f"Prompt comparison on {_latex_escape(dataset_title)} "
        f"(Model={_latex_escape(model)}, {anon_text})."
    )
    if excluded:
        caption += r" \texttt{CTX} indicates the prompt exceeded the model context window for that configuration."

    lines: list[str] = []
    lines.append(r"% Requires \usepackage{booktabs}")
    lines.append(r"% Requires \usepackage{graphicx}  % for \resizebox")
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header1 = [r"Obs $N$ / Int $M$"]
    for m in int_sizes:
        header1.append(rf"\multicolumn{{{len(methods)}}}{{c}}{{{m}}}")
    lines.append(" & ".join(header1) + r" \\")

    # Horizontal rules above each method group (Summary/Matrix/ENCO) per Int M.
    if int_sizes and methods:
        cmids: list[str] = []
        group_w = len(methods)
        for gi in range(len(int_sizes)):
            start = 2 + gi * group_w
            end = start + group_w - 1
            cmids.append(rf"\cmidrule(lr){{{start}-{end}}}")
        lines.append("".join(cmids))

    header2 = [""]
    for _m in int_sizes:
        for _tag, disp in methods:
            header2.append(_latex_escape(disp))
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for n in obs_sizes:
        row_cells: list[str] = [str(n)]
        for m in int_sizes:
            for method_tag, _disp in methods:
                key = (method_tag, n, m)
                v = values.get(key)
                if v is not None:
                    row_cells.append(_fmt_cell(*v))
                elif excluded is not None and key in excluded:
                    row_cells.append(r"\texttt{CTX}")
                else:
                    row_cells.append("--")
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sachs", help="Dataset name under experiments/responses/<dataset>/")
    p.add_argument("--responses-dir", type=Path, default=None)
    p.add_argument("--model", default=None, help="Model tag in filenames (e.g., gpt-5-mini).")
    p.add_argument(
        "--prompt-style",
        choices=["cases", "matrix", "summary", "payload", "payload_topk"],
        default="summary",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Build comparison tables across methods (summary/matrix/ENCO) using "
            "experiments/responses/<dataset>/<dataset>_summary.csv."
        ),
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Path to <dataset>_summary.csv (default: experiments/responses/<dataset>/<dataset>_summary.csv).",
    )
    p.add_argument(
        "--compare-styles",
        nargs="*",
        default=["summary", "matrix", "enco"],
        choices=["summary", "matrix", "enco"],
        help='Which methods to include in --compare tables (default: "summary matrix enco").',
    )
    p.add_argument(
        "--exclude-context-exceeded",
        action="store_true",
        help="In --compare mode, drop matrix cells where context_exceeded_any==1.",
    )
    p.add_argument(
        "--append-styles",
        nargs="*",
        default=[],
        choices=["cases", "matrix", "summary", "payload", "payload_topk"],
        help=(
            "Additional prompt styles to append in the combined --all-four output. "
            "Example: --append-styles matrix"
        ),
    )
    p.add_argument(
        "--only-styles",
        nargs="*",
        default=[],
        choices=["cases", "matrix", "summary", "payload", "payload_topk"],
        help=(
            "If set, restrict the combined --all-four output to exactly these prompt styles "
            "(disables auto-appending of payload/payload_topk). Example: --only-styles summary matrix"
        ),
    )
    p.add_argument(
        "--no-payload",
        action="store_true",
        help="If set, do not append payload prompt-style tables to the combined output (default: append if available).",
    )
    p.add_argument(
        "--no-payload-topk",
        action="store_true",
        help="If set, do not append payload_topk prompt-style tables to the combined output (default: append if available).",
    )
    p.add_argument("--metric", choices=["consensus", "avg"], default="consensus")
    p.add_argument(
        "--anonymize",
        choices=["any", "anon", "non"],
        default="any",
        help="Filter to anonymized runs ('anon'), non-anonymized runs ('non'), or include both ('any').",
    )
    p.add_argument(
        "--all-four",
        action="store_true",
        help="Write 4 tables: (avg vs consensus) x (anon vs non). Ignores --metric/--anonymize output-wise.",
    )
    p.add_argument(
        "--all-four-out",
        type=Path,
        default=None,
        help="If set with --all-four, write all 4 tables into this single .tex file (instead of 4 separate files).",
    )
    p.add_argument(
        "--all-models-avg",
        action="store_true",
        help="In --compare mode, write avg-only tables for all discovered LLM models (non+anon) into one .tex file.",
    )
    p.add_argument(
        "--all-models-avg-out",
        type=Path,
        default=None,
        help="Output path for --all-models-avg (default: experiments/out/baselines/<dataset>_compare_all_models_avg.tex).",
    )
    p.add_argument(
        "--include-enco",
        action="store_true",
        help="Deprecated (now the default with --all-four). Append the ENCO baseline table to the same output .tex file.",
    )
    p.add_argument(
        "--no-enco",
        action="store_true",
        help="If set with --all-four, do not append the ENCO baseline table to the combined output .tex file.",
    )
    p.add_argument(
        "--enco-responses-dir",
        type=Path,
        default=None,
        help="Optional ENCO responses directory override (default: experiments/responses/<dataset>, then responses/<dataset>).",
    )
    p.add_argument("--obs-sizes", nargs="*", default=[], help="Optional explicit list (space/comma separated).")
    p.add_argument("--int-sizes", nargs="*", default=[], help="Optional explicit list (space/comma separated).")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory (default: experiments/out/baselines). Ignored if --out is set.",
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    if args.compare:
        if not args.all_models_avg and not args.model:
            p.error("--model is required for --compare unless --all-models-avg is set.")
    else:
        if not args.model:
            p.error("--model is required unless --compare --all-models-avg is set.")

    def _resolve_summary_csv_path() -> Path:
        if args.summary_csv is not None:
            return Path(args.summary_csv)

        candidates: list[Path] = []
        if args.responses_dir is not None:
            candidates.append(Path(args.responses_dir) / f"{args.dataset}_summary.csv")
        candidates.extend([
            Path("experiments") / "responses" / args.dataset / f"{args.dataset}_summary.csv",
            Path("responses") / args.dataset / f"{args.dataset}_summary.csv",
        ])
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    summary_csv = _resolve_summary_csv_path()
    rows = _read_summary_csv(summary_csv)

    if args.compare:
        obs_sizes = _parse_int_list(args.obs_sizes) or [0, 100, 1000, 5000, 8000]
        int_sizes = _parse_int_list(args.int_sizes) or [0, 50, 100, 200, 500]

        methods: list[tuple[str, str]] = []
        for tag in args.compare_styles:
            if tag == "summary":
                methods.append(("summary", "Summary"))
            elif tag == "matrix":
                methods.append(("matrix", "Tabular"))
            elif tag == "enco":
                methods.append(("enco", "ENCO"))

        def _select_values(
            *,
            anonymize: str,
            metric: str,
            model: str,
        ) -> tuple[dict[tuple[str, int, int], tuple[float, int]], set[tuple[str, int, int]]]:
            metric_prefix = "consensus" if metric == "consensus" else "avg"
            f1_key = f"{metric_prefix}_f1"
            shd_key = f"{metric_prefix}_shd"
            allowed = {m for m, _ in methods}
            out: dict[tuple[str, int, int], tuple[float, int]] = {}
            excluded: set[tuple[str, int, int]] = set()
            for r in rows:
                ps = (r.get("prompt_style") or "").strip().lower()
                if ps not in allowed:
                    continue

                is_anon = _to_int(r.get("anonymize")) == 1
                # ENCO is reported for its default setting (anonymized by default),
                # even when the comparison table is for non-anonymized LLM prompts.
                if ps != "enco":
                    if anonymize == "anon" and not is_anon:
                        continue
                    if anonymize == "non" and is_anon:
                        continue

                model_tag = (r.get("model") or "").strip()
                if ps == "enco":
                    if model_tag != "ENCO":
                        continue
                else:
                    if model_tag != model:
                        continue

                obs_n = _to_int(r.get("obs_n"))
                int_n = _to_int(r.get("int_n"))
                if obs_n is None or int_n is None:
                    continue

                if args.exclude_context_exceeded and ps == "matrix":
                    if _to_int(r.get("context_exceeded_any")) == 1:
                        excluded.add((ps, int(obs_n), int(int_n)))
                        continue

                f1 = _to_float(r.get(f1_key))
                shd = _to_int(r.get(shd_key))
                if f1 is None or shd is None:
                    continue
                out[(ps, int(obs_n), int(int_n))] = (float(f1), int(shd))

            # Special-case: (obs=0,int=0) corresponds to the "names-only" prompts.
            # In the summary CSV those rows use prompt_style=names_only and have obs_n/int_n blank.
            # For non-anonymized comparison tables, we report that baseline in the (0,0) cell
            # for non-ENCO methods (Summary/Matrix).
            if anonymize == "non":
                best_names_only: tuple[float, int] | None = None
                best_score = -1.0
                for r in rows:
                    if (r.get("prompt_style") or "").strip().lower() != "names_only":
                        continue
                    if _to_int(r.get("anonymize")) == 1:
                        continue
                    if (r.get("model") or "").strip() != model:
                        continue
                    f1 = _to_float(r.get(f1_key))
                    shd = _to_int(r.get(shd_key))
                    if f1 is None or shd is None:
                        continue
                    # Prefer the run with more evaluated rows (or num_rows as a fallback).
                    score = float(_to_float(r.get("valid_rows")) or _to_float(r.get("num_rows")) or 0.0)
                    if score > best_score:
                        best_score = score
                        best_names_only = (float(f1), int(shd))
                if best_names_only is not None:
                    for ps in allowed:
                        if ps == "enco":
                            continue
                        out.setdefault((ps, 0, 0), best_names_only)
            return out, excluded

        out_dir = args.out_dir or (Path("experiments") / "out" / "baselines")
        if args.all_models_avg:
            out_path = (
                Path(args.all_models_avg_out)
                if args.all_models_avg_out is not None
                else (out_dir / f"{args.dataset}_compare_all_models_avg.tex")
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            discovered_models = sorted(
                {
                    (r.get("model") or "").strip()
                    for r in rows
                    if (r.get("model") or "").strip() and (r.get("model") or "").strip() != "ENCO"
                }
            )
            if not discovered_models:
                raise FileNotFoundError(f"No LLM model rows found in summary CSV: {summary_csv}")

            chunks: list[str] = []
            for model_name in discovered_models:
                for anonymize in ("non", "anon"):
                    values, excluded = _select_values(anonymize=anonymize, metric="avg", model=model_name)
                    label = f"tab:{args.dataset}-{model_name}-compare-{anonymize}-avg"
                    chunks.append(f"% ===== model={model_name} anonymize={anonymize} metric=avg =====\n")
                    chunks.append(
                        _render_comparison_table(
                            dataset=args.dataset,
                            model=model_name,
                            metric="avg",
                            anonymize=anonymize,
                            obs_sizes=obs_sizes,
                            int_sizes=int_sizes,
                            values=values,
                            methods=methods,
                            excluded=excluded,
                            label=label,
                        )
                    )
            out_path.write_text("\n".join(chunks), encoding="utf-8")
            print(f"[done] Wrote {out_path}")
            return 0

        if args.all_four:
            out_path = Path(args.all_four_out) if args.all_four_out is not None else (out_dir / f"{args.dataset}_{args.model}_compare_all_four_tables.tex")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            chunks: list[str] = []
            for anonymize in ("non", "anon"):
                for metric in ("avg", "consensus"):
                    values, excluded = _select_values(anonymize=anonymize, metric=metric, model=args.model)
                    label = f"tab:{args.dataset}-{args.model}-compare-{anonymize}-{metric}"
                    chunks.append(f"% ===== compare anonymize={anonymize} metric={metric} =====\n")
                    chunks.append(
                        _render_comparison_table(
                            dataset=args.dataset,
                            model=args.model,
                            metric=metric,
                            anonymize=anonymize,
                            obs_sizes=obs_sizes,
                            int_sizes=int_sizes,
                            values=values,
                            methods=methods,
                            excluded=excluded,
                            label=label,
                        )
                    )
            out_path.write_text("\n".join(chunks), encoding="utf-8")
            print(f"[done] Wrote {out_path}")
            return 0

        values, excluded = _select_values(
            anonymize=("anon" if args.anonymize == "anon" else "non" if args.anonymize == "non" else "non"),
            metric=args.metric,
            model=args.model,
        )
        out_path = args.out or (out_dir / f"{args.dataset}_{args.model}_compare_{args.anonymize}_{args.metric}.tex")
        label = args.label or f"tab:{args.dataset}-{args.model}-compare-{args.anonymize}-{args.metric}"
        tex = _render_comparison_table(
            dataset=args.dataset,
            model=args.model,
            metric=args.metric,
            anonymize=args.anonymize,
            obs_sizes=obs_sizes,
            int_sizes=int_sizes,
            values=values,
            methods=methods,
            excluded=excluded,
            label=label,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex, encoding="utf-8")
        print(f"[done] Wrote {out_path}")
        return 0

    def _build_one_tex(
        *,
        prompt_style: str,
        metric: str,
        anonymize: str,
        label: str,
        responses_dir: Path | None = None,
        allow_empty: bool = False,
    ) -> str:
        metric_prefix = "consensus" if metric == "consensus" else "avg"
        f1_key = f"{metric_prefix}_f1"
        shd_key = f"{metric_prefix}_shd"

        values: dict[tuple[int, int], tuple[float, int]] = {}
        discovered_obs: set[int] = set()
        discovered_int: set[int] = set()
        matched_rows = 0

        if anonymize == "non":
            best_names_only: tuple[float, int] | None = None
            best_score = -1.0
            for r in rows:
                if (r.get("prompt_style") or "").strip().lower() != "names_only":
                    continue
                if _to_int(r.get("anonymize")) == 1:
                    continue
                if (r.get("model") or "").strip() != args.model:
                    continue
                f1 = _to_float(r.get(f1_key))
                shd = _to_int(r.get(shd_key))
                if f1 is None or shd is None:
                    continue
                score = float(_to_float(r.get("valid_rows")) or _to_float(r.get("num_rows")) or 0.0)
                if score > best_score:
                    best_score = score
                    best_names_only = (float(f1), int(shd))
            if best_names_only is not None:
                values[(0, 0)] = best_names_only
                discovered_obs.add(0)
                discovered_int.add(0)

        for r in rows:
            if (r.get("prompt_style") or "").strip().lower() != prompt_style:
                continue
            if (r.get("model") or "").strip() != args.model:
                continue
            is_anon = _to_int(r.get("anonymize")) == 1
            if anonymize == "anon" and not is_anon:
                continue
            if anonymize == "non" and is_anon:
                continue

            obs_n = _to_int(r.get("obs_n"))
            int_n = _to_int(r.get("int_n"))
            if obs_n is None or int_n is None:
                continue

            matched_rows += 1
            f1 = _to_float(r.get(f1_key))
            shd = _to_int(r.get(shd_key))
            if f1 is None or shd is None:
                continue
            values[(int(obs_n), int(int_n))] = (float(f1), int(shd))
            discovered_obs.add(int(obs_n))
            discovered_int.add(int(int_n))

        if not values:
            if allow_empty:
                obs_sizes = _parse_int_list(args.obs_sizes) or [0, 100, 1000, 5000, 8000]
                int_sizes = _parse_int_list(args.int_sizes) or [0, 50, 100, 200, 500]
                return _render_table(
                    dataset=args.dataset,
                    model=args.model,
                    prompt_style=prompt_style,
                    metric=metric,
                    anonymize=anonymize,
                    obs_sizes=obs_sizes,
                    int_sizes=int_sizes,
                    values={},
                    label=label,
                )
            extra = ""
            if matched_rows > 0:
                extra = (
                    " Found matching summary rows, but none have non-null evaluation metrics.\n"
                    "Rebuild the summary table first, e.g.:\n"
                    "  python experiments/pipelines/run_cd_eval_pipeline.py --dataset sachs --steps analyze"
                )
            raise FileNotFoundError(
                f"No evaluated rows found in {summary_csv} "
                f"for model={args.model}, prompt_style={prompt_style}, anonymize={anonymize}, metric={metric}.{extra}"
            )

        obs_sizes = _parse_int_list(args.obs_sizes) or sorted(discovered_obs)
        int_sizes = _parse_int_list(args.int_sizes) or sorted(discovered_int)

        return _render_table(
            dataset=args.dataset,
            model=args.model,
            prompt_style=prompt_style,
            metric=metric,
            anonymize=anonymize,
            obs_sizes=obs_sizes,
            int_sizes=int_sizes,
            values=values,
            label=label,
        )

    def _style_available(style_tag: str) -> Path | None:
        for r in rows:
            if (r.get("model") or "").strip() != args.model:
                continue
            if (r.get("prompt_style") or "").strip().lower() != style_tag.lower():
                continue
            return summary_csv.parent
        return None

    out_dir = args.out_dir or (Path("experiments") / "out" / "baselines")

    if args.all_four:
        if args.all_four_out is not None:
            out_path = Path(args.all_four_out)
        else:
            out_path = out_dir / f"{args.dataset}_{args.model}_{args.prompt_style}_all_four_llm_tables.tex"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        chunks: list[str] = []
        prompt_styles: list[str] = []
        if args.only_styles:
            for ps in list(args.only_styles or []):
                if ps not in prompt_styles:
                    prompt_styles.append(ps)
        else:
            # Preserve order while de-duplicating.
            for ps in [args.prompt_style, *list(args.append_styles or [])]:
                if ps not in prompt_styles:
                    prompt_styles.append(ps)
        payload_dir: Path | None = None
        payload_topk_dir: Path | None = None

        # Auto-append optional styles unless the user explicitly restricted styles via --only-styles.
        if not args.only_styles:
            include_payload = (not args.no_payload) and args.prompt_style != "payload"
            if include_payload:
                payload_dir = _style_available("payload")
                if "payload" not in prompt_styles:
                    prompt_styles.append("payload")

            include_payload_topk = (not args.no_payload_topk) and args.prompt_style != "payload_topk"
            if include_payload_topk:
                payload_topk_dir = _style_available("payload_topk")
                if "payload_topk" not in prompt_styles:
                    prompt_styles.append("payload_topk")

        for ps in prompt_styles:
            chunks.append(f"% ===== prompt_style={ps} =====\n")
            for anonymize in ("non", "anon"):
                for metric in ("avg", "consensus"):
                    label = f"tab:{args.dataset}-{args.model}-{ps}-{anonymize}-{metric}-llm-table-range"
                    if ps == "payload":
                        resp_dir_override = payload_dir
                    elif ps == "payload_topk":
                        resp_dir_override = payload_topk_dir
                    else:
                        resp_dir_override = None
                    chunks.append(
                        _build_one_tex(
                            prompt_style=ps,
                            metric=metric,
                            anonymize=anonymize,
                            label=label,
                            responses_dir=resp_dir_override,
                            # In --all-four, prefer producing placeholder tables over failing due to
                            # missing anon/non (or missing optional styles).
                            allow_empty=True,
                        )
                    )

        include_enco = (not args.no_enco) or args.include_enco
        if include_enco:
            obs_sizes = _parse_int_list(args.obs_sizes) or [0, 100, 1000, 5000, 8000]
            int_sizes = _parse_int_list(args.int_sizes) or [0, 50, 100, 200, 500]
            if args.enco_responses_dir is not None:
                enco_resp_dirs = [Path(args.enco_responses_dir)]
            else:
                enco_resp_dirs = [
                    Path("experiments") / "responses" / args.dataset,
                    Path("responses") / args.dataset,
                ]
            enco_resp_dirs = [d for d in enco_resp_dirs if d.exists()]
            if not enco_resp_dirs:
                raise FileNotFoundError(
                    f"--include-enco set but ENCO responses dir not found (checked: {', '.join(str(d) for d in enco_resp_dirs) or 'none'})."
                )
            enco_label = f"tab:{args.dataset}-enco-table-range"
            last_error: Exception | None = None
            enco_tex: str | None = None
            for enco_dir in enco_resp_dirs:
                try:
                    enco_tex = _build_enco_tex_from_responses(
                        dataset=args.dataset,
                        responses_dir=enco_dir,
                        obs_sizes=obs_sizes,
                        int_sizes=int_sizes,
                        label=enco_label,
                    )
                    break
                except FileNotFoundError as exc:
                    last_error = exc
            if enco_tex is None:
                if last_error is not None:
                    raise last_error
                raise FileNotFoundError("--include-enco set but no ENCO prediction CSVs were found.")
            chunks.append(enco_tex)
        out_path.write_text("\n".join(chunks), encoding="utf-8")
        print(f"[done] Wrote {out_path}")
        return 0

    dataset = args.dataset
    out_path = args.out or (out_dir / f"{dataset}_{args.model}_{args.prompt_style}_{args.anonymize}_{args.metric}_llm_table.tex")
    label = args.label or f"tab:{dataset}-{args.model}-{args.prompt_style}-{args.anonymize}-{args.metric}-llm-table-range"
    tex = _build_one_tex(prompt_style=args.prompt_style, metric=args.metric, anonymize=args.anonymize, label=label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")
    print(f"[done] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
