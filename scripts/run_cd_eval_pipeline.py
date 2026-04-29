#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


_DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    # Keep a small mapping; adjust/extend as needed.
    "gpt-5-mini": 400_000,
    "gpt-5": 400_000,
}


def _context_window_for(model: str) -> int:
    key = (model or "").split("/")[-1]
    return int(_DEFAULT_CONTEXT_WINDOWS.get(key, 128_000))


def _safe_model_tag(model: str) -> str:
    model_path = Path(str(model))
    tag = model_path.name or str(model)
    if tag.startswith("checkpoint-") and model_path.parent.name:
        tag = f"{model_path.parent.name}_{tag}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "model"


_RESP_RE = re.compile(
    r"^responses_obs(?P<obs>\d+)_int(?P<int>\d+)_shuf(?P<shuf>\d+)(?P<suffix>.*)$",
    flags=re.IGNORECASE,
)

_ENCO_RE = re.compile(
    r"^predictions_obs(?P<obs>\d+)_int(?P<int>\d+)_ENCO(?:_seed(?P<seed>\d+))?$",
    flags=re.IGNORECASE,
)

_PRED_RE = re.compile(
    r"^predictions_obs(?P<obs>\d+)_int(?P<int>\d+)_(?P<method>.+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ResponseMeta:
    dataset: str
    csv_path: Path
    model: str
    reasoning_guidance: str
    is_names_only: bool
    obs_n: Optional[int]
    int_n: Optional[int]
    shuf_n: Optional[int]
    anonymize: bool
    prompt_style: str  # "cases" or "matrix" or "names_only"
    row_order: str
    col_order: str
    wrapper_mode: str
    append_format_hint: bool
    causal_rules: bool
    give_steps: bool


def _repo_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    experiments_dir = repo_root / "experiments"
    return repo_root, experiments_dir


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)

def _resolve_file_arg(path_str: str, *, repo_root: Path, invocation_cwd: Path) -> str:
    """
    Resolve a user-provided path string to an absolute path when possible.

    We try common bases in a user-friendly order:
    1) as-is (absolute or relative to invocation cwd)
    2) relative to repo root (matches README examples run from repo root)
    3) relative to experiments/ (useful when running from that directory)
    """
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)

    cand1 = (invocation_cwd / p).resolve()
    if cand1.exists():
        return str(cand1)

    cand2 = (repo_root / p).resolve()
    if cand2.exists():
        return str(cand2)

    experiments_dir = repo_root / "experiments"
    cand3 = (experiments_dir / p).resolve()
    if cand3.exists():
        return str(cand3)

    # best effort fallback; let downstream scripts raise a clear error
    return str((invocation_cwd / p).resolve())


def _infer_model_from_stem(stem: str) -> str:
    stem = _normalize_stem_for_parse(stem)
    # Preserve full model suffix (including underscores), e.g. grpo_sft_8192_from4096.
    m_names = re.match(
        r"^responses_names_only(?:_p\d+)?"
        r"(?P<tags>(?:_(?:anon|rules|steps|wrapchat|fmthint|reason(?:concise|none)|shuf\d+|"
        r"row[A-Za-z0-9]+|col[A-Za-z0-9]+))*)_(?P<model>.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if m_names:
        return m_names.group("model")
    m_resp = re.match(
        r"^responses_obs\d+_int\d+_shuf\d+_p\d+_"
        r"(?:(?:[A-Za-z0-9]+_)*)"
        r"(?:payload_topk|summary|matrix|cases|payload)"
        r"_(?P<model>.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if m_resp:
        return m_resp.group("model")
    return "unknown"


def _normalize_stem_for_parse(stem: str) -> str:
    out = stem
    out = out.replace("summary_joint", "summary")
    out = out.replace("summary_join", "summary")
    out = out.replace("thinktags_cothint", "wrapchat_fmthint")
    out = out.replace("cothint_thinktags", "wrapchat_fmthint")
    out = out.replace("thinktags", "wrapchat")
    out = out.replace("cothint", "fmthint")
    out = out.replace("respthink_answer", "")
    out = out.replace("wrapplain", "")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _parse_prompt_suffix(suffix: str) -> tuple[str, str]:
    """
    Parse the filename suffix after `_shuf{n}` into:
      1) the tag blob (may be empty, usually starts with `_p...`)
      2) the model name

    Example suffixes:
      `_p5_anon_wrapchat_summary_gpt-5-mini`
      `_p3_matrix_gpt-5.2-pro`
      `_p100_wrapchat_fmthint_summary_sft`
    """
    suffix = _normalize_stem_for_parse(suffix)
    m = re.match(
        r"^(?P<tags>_p\d+_(?:(?:[A-Za-z0-9]+_)*)"
        r"(?:summary_probs|payload_topk|payload|summary|matrix|cases))"
        r"_(?P<model>.+)$",
        suffix,
        flags=re.IGNORECASE,
    )
    if not m:
        return suffix, "unknown"
    return m.group("tags"), m.group("model").strip()


def _infer_reasoning_guidance(text: str) -> str:
    lowered = _normalize_stem_for_parse(text).lower()
    if "reasonconcise" in lowered:
        return "concise"
    if "reasonnone" in lowered:
        return "none"
    return "staged"


def _parse_response_meta(dataset: str, csv_path: Path) -> ResponseMeta:
    raw_stem = csv_path.stem
    stem = _normalize_stem_for_parse(raw_stem)

    # ENCO baseline: "predictions_obs{N}_int{M}_ENCO.csv"
    m_enco = _ENCO_RE.match(stem)
    if m_enco:
        return ResponseMeta(
            dataset=dataset,
            csv_path=csv_path,
            model="ENCO",
            reasoning_guidance="baseline",
            is_names_only=False,
            obs_n=int(m_enco.group("obs")),
            int_n=int(m_enco.group("int")),
            shuf_n=None,
            anonymize=False,
            prompt_style="baseline",
            row_order="random",
            col_order="original",
            wrapper_mode="plain",
            append_format_hint=False,
            causal_rules=False,
            give_steps=False,
        )

    # Generic prediction baseline:
    #   predictions_obs{N}_int{M}_{Method}.csv
    #   predictions_obs{N}_int{M}_{Method}_anon.csv
    #   predictions_obs0_int0_{Method}_names_only.csv
    m_pred = _PRED_RE.match(stem)
    if m_pred:
        method = m_pred.group("method")
        anonymize = False
        is_names_only = False
        prompt_style = "baseline"
        if method.endswith("_anon"):
            anonymize = True
            method = method[: -len("_anon")]
        if method.endswith("_names_only"):
            is_names_only = True
            method = method[: -len("_names_only")]
            prompt_style = "names_only"
        method = re.sub(r"_seed\d+$", "", method)
        if method.startswith("TakayamaSCP") or method in {"JiralerspongBFS", "CausalLLMData", "JiralerspongPairwise"}:
            prompt_style = "summary"
        elif method == "ENCO":
            prompt_style = "enco"
        return ResponseMeta(
            dataset=dataset,
            csv_path=csv_path,
            model=method,
            reasoning_guidance="baseline",
            is_names_only=is_names_only,
            obs_n=int(m_pred.group("obs")),
            int_n=int(m_pred.group("int")),
            shuf_n=None,
            anonymize=anonymize,
            prompt_style=prompt_style,
            row_order="random",
            col_order="original",
            wrapper_mode="plain",
            append_format_hint=False,
            causal_rules=False,
            give_steps=False,
        )

    # Names-only: "responses_names_only..." (no obs/int/shuf encoded)
    if "responses_names_only" in stem:
        model = _infer_model_from_stem(stem)
        col_order = "original"
        m = re.search(r"_col([A-Za-z0-9]+)", stem)
        if m:
            col_order = m.group(1).lower()
        return ResponseMeta(
            dataset=dataset,
            csv_path=csv_path,
            model=model,
            reasoning_guidance=_infer_reasoning_guidance(stem),
            is_names_only=True,
            obs_n=None,
            int_n=None,
            shuf_n=None,
            anonymize=False,
            prompt_style="names_only",
            row_order="random",
            col_order=col_order,
            wrapper_mode=("chat" if "wrapchat" in stem.lower() else "plain"),
            append_format_hint=("fmthint" in stem.lower()),
            causal_rules=("rules" in stem.lower()),
            give_steps=("steps" in stem.lower()),
        )

    m = _RESP_RE.match(stem)
    if not m:
        # best-effort fallback
        prompt_style = "cases"
        stem_l = stem.lower()
        if "payload_topk" in stem_l:
            prompt_style = "payload_topk"
        elif "payload" in stem_l:
            prompt_style = "payload"
        elif "summary" in stem_l:
            prompt_style = "summary"
        elif "matrix" in stem_l:
            prompt_style = "matrix"

        return ResponseMeta(
            dataset=dataset,
            csv_path=csv_path,
            model=_infer_model_from_stem(stem),
            reasoning_guidance=_infer_reasoning_guidance(stem),
            is_names_only=False,
            obs_n=None,
            int_n=None,
            shuf_n=None,
            anonymize=("anon" in stem.lower()),
            prompt_style=prompt_style,
            row_order="random",
            col_order=("topo" if "coltopo" in stem.lower() else "original"),
            wrapper_mode=("chat" if "wrapchat" in stem.lower() else "plain"),
            append_format_hint=("fmthint" in stem.lower()),
            causal_rules=("rules" in stem.lower()),
            give_steps=("steps" in stem.lower()),
        )

    obs_n = int(m.group("obs"))
    int_n = int(m.group("int"))
    shuf_n = int(m.group("shuf"))
    suffix = m.group("suffix") or ""
    parsed_tags, parsed_model = _parse_prompt_suffix(suffix)
    tags = parsed_tags.lower()
    model = parsed_model if parsed_model != "unknown" else _infer_model_from_stem(stem)

    row_order = "random"
    col_order = "original"
    for token in tags.split("_"):
        token = token.strip()
        if token.startswith("row"):
            row_order = token.removeprefix("row")
        if token.startswith("col"):
            col_order = token.removeprefix("col")

    prompt_style = "cases"
    if "payload_topk" in tags:
        prompt_style = "payload_topk"
    elif "payload" in tags:
        prompt_style = "payload"
    elif "summary" in tags:
        prompt_style = "summary"
    elif "matrix" in tags:
        prompt_style = "matrix"
    return ResponseMeta(
        dataset=dataset,
        csv_path=csv_path,
        model=model,
        reasoning_guidance=_infer_reasoning_guidance(tags),
        is_names_only=False,
        obs_n=obs_n,
        int_n=int_n,
        shuf_n=shuf_n,
        anonymize=("anon" in tags),
        prompt_style=prompt_style,
        row_order=row_order,
        col_order=col_order,
        wrapper_mode=("chat" if "wrapchat" in tags else "plain"),
        append_format_hint=("fmthint" in tags),
        causal_rules=("rules" in tags),
        give_steps=("steps" in tags),
    )


def _default_response_dirs(experiments_dir: Path, dataset: str) -> list[Path]:
    # Responses may live either under experiments/responses/<dataset> (when run from experiments/)
    # or under repo_root/responses/<dataset> (when run from repo root).
    repo_root = experiments_dir.parent
    return [
        experiments_dir / "responses" / dataset,
        repo_root / "responses" / dataset,
    ]


def _resolve_response_dirs(args: argparse.Namespace, experiments_dir: Path) -> list[Path]:
    if not getattr(args, "responses_dir", None):
        return _default_response_dirs(experiments_dir, args.dataset)

    repo_root = experiments_dir.parent
    invocation_cwd = getattr(args, "invocation_cwd", Path.cwd().resolve())
    return [
        Path(_resolve_file_arg(str(path), repo_root=repo_root, invocation_cwd=invocation_cwd))
        for path in args.responses_dir
    ]


def _format_searched_response_dirs(resp_dirs: list[Path]) -> str:
    return ", ".join(str(d) for d in resp_dirs) or "(none)"


def _analysis_summary_dir(args: argparse.Namespace, experiments_dir: Path, resp_dirs: list[Path]) -> Path:
    if getattr(args, "responses_dir", None) and resp_dirs:
        return Path(resp_dirs[0])
    return experiments_dir / "responses" / args.dataset


def _find_response_csvs(resp_dirs: list[Path]) -> list[Path]:
    resp_dirs = [Path(d) for d in resp_dirs]
    resp_dirs = [d for d in resp_dirs if d.exists()]
    if not resp_dirs:
        return []
    # Include both raw model responses and prediction CSVs from classical /
    # semantic / mixed-method baselines. Exclude derived artifacts like
    # *.per_row.csv so we do not mis-evaluate them as primary outputs.
    out: list[Path] = []
    for resp_dir in resp_dirs:
        for p in resp_dir.glob("responses_*.csv"):
            if p.name.endswith(".per_row.csv"):
                continue
            out.append(p)
        for p in resp_dir.glob("predictions_*.csv"):
            if p.name.endswith(".per_row.csv"):
                continue
            out.append(p)
    return sorted({p.resolve(): p for p in out}.values())


def _display_model_for_meta(meta: ResponseMeta, args: argparse.Namespace) -> str:
    if getattr(args, "model_label", None):
        return str(args.model_label)
    return meta.model


def _prompt_token_stats(csv_path: Path, *, context_window: int) -> dict[str, Any]:
    """
    Best-effort scan of a response CSV to quantify prompt lengths and context errors.

    This is used in the analyze step so you can see which configurations exceeded
    the model context window (often producing all-[ERROR] rows / valid=0).
    """
    stats: dict[str, Any] = {
        "context_window": int(context_window),
        "prompt_tokens_max": None,
        "prompt_tokens_mean": None,
        "prompt_tokens_rows": 0,
        "prompt_tokens_missing_rows": 0,
        "context_exceeded_by_tokens_rows": 0,
        "context_exceeded_by_error_rows": 0,
        "context_exceeded_any": 0,
    }

    toks: list[int] = []
    exceed_tokens = 0
    exceed_err = 0
    missing = 0

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = (row.get("raw_response") or "").lower()
                if "context window" in raw or "context_length_exceeded" in raw:
                    exceed_err += 1

                pt = row.get("prompt_tokens")
                if pt is None or pt == "":
                    missing += 1
                    continue
                try:
                    v = int(float(pt))
                except Exception:
                    missing += 1
                    continue
                if v >= 0:
                    toks.append(v)
                    if v > int(context_window):
                        exceed_tokens += 1
    except Exception:
        return stats

    if toks:
        stats["prompt_tokens_max"] = int(max(toks))
        stats["prompt_tokens_mean"] = float(sum(toks) / float(len(toks)))
        stats["prompt_tokens_rows"] = int(len(toks))
    stats["prompt_tokens_missing_rows"] = int(missing)
    stats["context_exceeded_by_tokens_rows"] = int(exceed_tokens)
    stats["context_exceeded_by_error_rows"] = int(exceed_err)
    stats["context_exceeded_any"] = int((exceed_tokens > 0) or (exceed_err > 0))
    return stats


def step_generate_and_run_in_memory(args: argparse.Namespace, *, experiments_dir: Path, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        str(experiments_dir.parent / "scripts" / "eval_cd_configs.py"),
        "--bif-file",
        args.bif_file,
        "--num-prompts",
        str(args.num_prompts),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
    ]
    if getattr(args, "styles", None):
        cmd.append("--styles")
        cmd.extend([str(s) for s in args.styles])
    if getattr(args, "wrapper_mode", None):
        cmd.extend(["--wrapper-mode", str(args.wrapper_mode)])
    if getattr(args, "append_format_hint", False):
        cmd.append("--append-format-hint")
    if getattr(args, "cot_hint", False):
        cmd.append("--cot-hint")
    for s in (args.shuffles_per_graph or []):
        cmd.extend(["--shuffles-per-graph", str(int(s))])
    for m in args.model:
        cmd.extend(["--model", m])
    if args.overwrite:
        cmd.append("--overwrite")
    if getattr(args, "only_names_only", False):
        cmd.append("--only-names-only")
    if getattr(args, "save_example_prompt", False):
        cmd.append("--save-example-prompt")
    if getattr(args, "example_prompt_dir", None):
        cmd.extend(["--example-prompt-dir", str(args.example_prompt_dir)])
    if getattr(args, "overwrite_example_prompt", False):
        cmd.append("--overwrite-example-prompt")
    if args.dry_run:
        cmd.append("--dry-run")
    _run(cmd, cwd=experiments_dir, dry_run=dry_run)


def _load_evaluated_response_csvs(summary_csv: Path) -> set[Path]:
    if not summary_csv.exists():
        return set()
    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return set()
    if "response_csv" not in df.columns:
        return set()
    if "evaluated" in df.columns:
        df = df[df["evaluated"] == 1].copy()
    out: set[Path] = set()
    for value in df["response_csv"].dropna():
        try:
            out.add(Path(str(value)).resolve())
        except Exception:
            continue
    return out


def step_evaluate(args: argparse.Namespace, *, experiments_dir: Path, dry_run: bool) -> None:
    resp_dirs = _resolve_response_dirs(args, experiments_dir)
    resp_csvs = _find_response_csvs(resp_dirs)
    if not resp_csvs:
        raise SystemExit(
            f"No response CSVs found under: {_format_searched_response_dirs(resp_dirs)}. "
            "Run the model step first."
        )

    evaluated_cache: dict[Path, set[Path]] = {}
    for csv_path in resp_csvs:
        summary_csv = csv_path.parent / "eval_summary.csv"
        if summary_csv not in evaluated_cache:
            evaluated_cache[summary_csv] = _load_evaluated_response_csvs(summary_csv)

        per_row_path = csv_path.with_suffix(csv_path.suffix + ".per_row.csv")
        if (
            per_row_path.exists()
            and per_row_path.stat().st_mtime >= csv_path.stat().st_mtime
            and not args.overwrite_eval
        ):
            continue
        if not args.overwrite_eval and csv_path.resolve() in evaluated_cache[summary_csv]:
            continue

        cmd = [
            sys.executable,
            "evaluate.py",
            "--csv",
            str(csv_path),
            "--tau",
            str(args.tau),
            "--summary-csv",
            str(summary_csv),
        ]
        _run(cmd, cwd=experiments_dir, dry_run=dry_run)
        if not dry_run:
            evaluated_cache[summary_csv].add(csv_path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_parent(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(float(v))
    except Exception:
        return None


def _add_metric_spread_columns(row: dict[str, Any]) -> None:
    """
    Add explicit per-metric SD/SE columns for downstream plotting.

    Inputs are expected from evaluate.py summary keys:
      - averages: avg_*
      - variability SDs: var_*_sd
      - row counts: valid_rows / num_rows
    """
    n = _safe_int(row.get("valid_rows"))
    if not n or n <= 0:
        n = _safe_int(row.get("num_rows"))
    if not n or n <= 0:
        n = 1

    metric_to_var_sd = {
        "avg_shd": "var_shd_sd",
        "avg_accuracy": "var_accuracy_sd",
        "avg_precision": "var_precision_sd",
        "avg_recall": "var_recall_sd",
        "avg_f1": "var_f1_sd",
        "num_pred_edges": "var_num_pred_edges_sd",
    }

    # Promote evaluate.py's var_*_sd into explicit <metric>_sd and <metric>_se.
    for metric_key, var_sd_key in metric_to_var_sd.items():
        metric_val = _safe_float(row.get(metric_key))
        sd_val = _safe_float(row.get(var_sd_key))
        if metric_val is None or sd_val is None:
            row[f"{metric_key}_sd"] = None
            row[f"{metric_key}_se"] = None
            continue
        row[f"{metric_key}_sd"] = float(sd_val)
        row[f"{metric_key}_se"] = float(sd_val / (n ** 0.5))

    # NHD fields already use *_sd names; add matching *_se.
    for metric_key, sd_key in (("nhd_mean", "nhd_sd"), ("nhd_ratio_mean", "nhd_ratio_sd")):
        metric_val = _safe_float(row.get(metric_key))
        sd_val = _safe_float(row.get(sd_key))
        if metric_val is None or sd_val is None:
            row[f"{metric_key}_se"] = None
            continue
        row[f"{metric_key}_se"] = float(sd_val / (n ** 0.5))

    row["spread_n"] = int(n)


def _safe_rate(numer: Any, denom: Any) -> Optional[float]:
    n = _safe_float(numer)
    d = _safe_float(denom)
    if n is None or d is None or d <= 0:
        return None
    return float(n / d)


def _build_config_summary(
    *,
    prompt_style: str,
    anonymize: Any,
    obs_n: Any,
    int_n: Any,
    shuffles_per_graph: Any,
    reasoning_guidance: str,
    row_order: str,
    col_order: str,
    wrapper_mode: str,
    append_format_hint: Any,
    causal_rules: Any,
    give_steps: Any,
    is_names_only: Any,
) -> str:
    parts: list[str] = []
    if _safe_int(is_names_only) == 1:
        parts.append("style=names_only")
    else:
        parts.append(f"style={prompt_style}")
        if obs_n is not None:
            parts.append(f"obs={obs_n}")
        if int_n is not None:
            parts.append(f"int={int_n}")
        if shuffles_per_graph is not None:
            parts.append(f"shuf={shuffles_per_graph}")

    parts.append(f"anon={int(bool(_safe_int(anonymize) == 1))}")
    parts.append(f"reason={reasoning_guidance}")

    if row_order:
        parts.append(f"row={row_order}")
    if col_order:
        parts.append(f"col={col_order}")
    if wrapper_mode:
        parts.append(f"wrap={wrapper_mode}")
    if _safe_int(append_format_hint) == 1:
        parts.append("fmthint=1")
    if _safe_int(causal_rules) == 1:
        parts.append("rules=1")
    if _safe_int(give_steps) == 1:
        parts.append("steps=1")
    return ",".join(parts)


def _enrich_summary_row(row: dict[str, Any]) -> None:
    row["config"] = _build_config_summary(
        prompt_style=str(row.get("prompt_style") or ""),
        anonymize=row.get("anonymize"),
        obs_n=row.get("obs_n"),
        int_n=row.get("int_n"),
        shuffles_per_graph=row.get("shuffles_per_graph"),
        reasoning_guidance=str(row.get("reasoning_guidance") or ""),
        row_order=str(row.get("row_order") or ""),
        col_order=str(row.get("col_order") or ""),
        wrapper_mode=str(row.get("wrapper_mode") or ""),
        append_format_hint=row.get("append_format_hint"),
        causal_rules=row.get("causal_rules"),
        give_steps=row.get("give_steps"),
        is_names_only=row.get("is_names_only"),
    )
    row["valid_rate"] = _safe_rate(row.get("valid_rows"), row.get("num_rows"))
    row["avg_F1"] = row.get("avg_f1")


def _ordered_fieldnames(rows: list[dict[str, Any]], priority: list[str]) -> list[str]:
    all_fields = {k for r in rows for k in r.keys()}
    ordered = [k for k in priority if k in all_fields]
    ordered.extend(sorted(all_fields - set(ordered)))
    return ordered


def _ordering_bias_from_csv(csv_path: Path) -> dict[str, Any]:
    """
    Estimate ordering sensitivity by grouping rows by data_idx and measuring the spread
    across shuffle_idx for each data_idx, then aggregating across data_idx.
    Uses the per-row metrics CSV produced by evaluate.py (<csv>.per_row.csv) when available.
    """
    import pandas as pd  # local import to keep script import-light

    per_row_path = csv_path.with_suffix(csv_path.suffix + ".per_row.csv")
    df_path = per_row_path if per_row_path.exists() else csv_path
    df = pd.read_csv(df_path)
    for col in ("data_idx", "shuffle_idx", "shd"):
        if col not in df.columns:
            return {
                "ok": False,
                "reason": f"missing required column: {col}",
            }

    # Ensure numeric
    df["data_idx"] = pd.to_numeric(df["data_idx"], errors="coerce")
    df["shuffle_idx"] = pd.to_numeric(df["shuffle_idx"], errors="coerce")
    df["shd"] = pd.to_numeric(df["shd"], errors="coerce")
    df = df.dropna(subset=["data_idx", "shuffle_idx", "shd"])
    if df.empty:
        return {"ok": False, "reason": "no valid rows after cleaning"}

    # per data_idx, standard deviation across shuffles
    per_data = df.groupby("data_idx")["shd"].agg(["count", "mean", "std"])
    per_data = per_data.rename(columns={"count": "k", "mean": "shd_mean", "std": "shd_sd"})

    # only data_idx with at least 2 shuffles contribute to an SD
    per_data_sd = per_data[per_data["k"] >= 2]["shd_sd"].dropna()

    shuf_n = int(df["shuffle_idx"].max()) + 1
    return {
        "ok": True,
        "rows": int(len(df)),
        "unique_data_idx": int(df["data_idx"].nunique()),
        "shuffles_detected": int(shuf_n),
        "per_data_shd_sd_mean": float(per_data_sd.mean()) if len(per_data_sd) else None,
        "per_data_shd_sd_median": float(per_data_sd.median()) if len(per_data_sd) else None,
    }


def step_analyze(args: argparse.Namespace, *, experiments_dir: Path, dry_run: bool) -> None:
    resp_dirs = _resolve_response_dirs(args, experiments_dir)
    resp_csvs = _find_response_csvs(resp_dirs)
    if not resp_csvs:
        raise SystemExit(
            f"No response CSVs found under: {_format_searched_response_dirs(resp_dirs)}. "
            "Run the model step first."
        )
    try:
        from evaluate import evaluate_response_csv
    except ModuleNotFoundError:
        from experiments.evaluate import evaluate_response_csv

    out_dir = experiments_dir / "out" / "experiment1"
    summary_dir = _analysis_summary_dir(args, experiments_dir, resp_dirs)
    _ensure_parent(out_dir / "placeholder.txt", dry_run=dry_run)
    _ensure_parent(summary_dir / "placeholder.txt", dry_run=dry_run)

    # 1) Collect per-condition summaries into one CSV
    summary_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []

    for csv_path in resp_csvs:
        meta = _parse_response_meta(args.dataset, csv_path)
        display_model = _display_model_for_meta(meta, args)
        ctx = _context_window_for(display_model)
        pt_stats = _prompt_token_stats(csv_path, context_window=ctx)
        summary = evaluate_response_csv(
            csv_path,
            tau=args.tau,
            write_artifacts=False,
            verbose=False,
        )["summary"]
        row: dict[str, Any] = {
            "dataset": meta.dataset,
            "model": display_model,
            "parsed_model": meta.model,
            "reasoning_guidance": meta.reasoning_guidance,
            "is_names_only": int(meta.is_names_only),
            "obs_n": meta.obs_n,
            "int_n": meta.int_n,
            "shuffles_per_graph": meta.shuf_n,
            "anonymize": int(meta.anonymize),
            "prompt_style": meta.prompt_style,
            "row_order": meta.row_order,
            "col_order": meta.col_order,
            "wrapper_mode": meta.wrapper_mode,
            "append_format_hint": int(meta.append_format_hint),
            "causal_rules": int(meta.causal_rules),
            "give_steps": int(meta.give_steps),
            "response_csv": str(csv_path),
            "summary_json": "",
            "evaluated": 1,
        }
        row.update(summary)
        row.update(pt_stats)
        _add_metric_spread_columns(row)
        _enrich_summary_row(row)
        summary_rows.append(row)

        # 2) Ordering bias analysis (only makes sense if shuffle_idx varies)
        if meta.shuf_n is not None and meta.shuf_n >= 2:
            ob = _ordering_bias_from_csv(csv_path)
            ordering_rows.append(
                {
                    "dataset": meta.dataset,
                    "model": display_model,
                    "parsed_model": meta.model,
                    "reasoning_guidance": meta.reasoning_guidance,
                    "obs_n": meta.obs_n,
                    "int_n": meta.int_n,
                    "shuffles_per_graph": meta.shuf_n,
                    "anonymize": int(meta.anonymize),
                    "prompt_style": meta.prompt_style,
                    "row_order": meta.row_order,
                    "col_order": meta.col_order,
                    "wrapper_mode": meta.wrapper_mode,
                    "append_format_hint": int(meta.append_format_hint),
                    "response_csv": str(csv_path),
                    **ob,
                }
            )

    if not summary_rows:
        raise SystemExit(
            "No response CSV summaries could be computed. Check the response CSV files."
        )

    summary_csv = summary_dir / f"{args.dataset}_summary.csv"
    if not dry_run:
        summary_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = _ordered_fieldnames(
            summary_rows,
            ["model", "config", "valid_rate", "avg_F1", "avg_shd"],
        )
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
    print(f"[info] Wrote summary table: {summary_csv}")

    ordering_csv = out_dir / f"{args.dataset}_ordering_bias.csv"
    if ordering_rows and not dry_run:
        with ordering_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in ordering_rows for k in r.keys()}))
            writer.writeheader()
            for r in ordering_rows:
                writer.writerow(r)
        print(f"[info] Wrote ordering-bias table: {ordering_csv}")
    elif ordering_rows:
        print(f"[info] Would write ordering-bias table: {ordering_csv}")


def main() -> None:
    repo_root, experiments_dir = _repo_paths()
    invocation_cwd = Path.cwd().resolve()
    for path in (repo_root, experiments_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    ap = argparse.ArgumentParser(
        description="Run Experiment 1 end-to-end: generate prompts, query models, evaluate, and analyze."
    )
    ap.add_argument("--dataset", default="cancer", help="Dataset name (defaults to bif basename).")
    ap.add_argument(
        "--bif-file",
        default=str(repo_root / "causal_graphs" / "real_data" / "small_graphs" / "sachs.bif"),
        help="Path to the BIF file.",
    )
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # NOTE: default=[] is important; we forward these flags into sub-scripts that
    # also have defaults. If we used default=[1] here, we'd accidentally pass "1"
    # and the callee would get [1,1] (duplicated grid).
    ap.add_argument("--shuffles-per-graph", type=int, action="append", default=[])
    ap.add_argument(
        "--styles",
        nargs="*",
        default=None,
        help=(
            'Optional subset of prompt styles to generate (any of: "cases", "matrix", "summary", '
            '"payload", "payload_topk").'
        ),
    )
    ap.add_argument(
        "--wrapper-mode",
        choices=["plain", "chat"],
        default=None,
        help="Prompt transport for the in-memory run step.",
    )
    ap.add_argument(
        "--append-format-hint",
        action="store_true",
        help=(
            "Append the canonical Formatting requirement line in the in-memory run step. "
            "For causal discovery this enables the optional stage-by-stage reasoning instructions."
        ),
    )
    ap.add_argument(
        "--cot-hint",
        action="store_true",
        help="Legacy alias for chat-style prompt wrapping in the in-memory run step.",
    )
    ap.add_argument(
        "--include-enco-in-summary",
        action="store_true",
        default=True,
        help="Deprecated; prediction CSVs are now summarized automatically for all methods.",
    )
    ap.add_argument(
        "--enco-responses-dir",
        default="experiments/responses/sachs",
        help=(
            "Deprecated legacy option; prediction CSVs are discovered automatically. "
            "Default retained for backward compatibility."
        ),
    )
    ap.add_argument(
        "--responses-dir",
        action="append",
        default=[],
        help=(
            "Directory to search for response CSVs during evaluate/analyze. Repeatable. "
            "When omitted, searches experiments/responses/<dataset> and responses/<dataset>."
        ),
    )

    ap.add_argument("--model", action="append", default=[], help="Repeatable.")
    ap.add_argument(
        "--model-label",
        default=None,
        help=(
            "Override the model label written by the analyze step. Intended only for one-off "
            "single-model legacy repairs; multi-model summaries should use unique model tags in filenames."
        ),
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.7, help="Consensus threshold for evaluate.py.")

    ap.add_argument("--overwrite", action="store_true", help="Re-query model responses.")
    ap.add_argument("--overwrite-eval", action="store_true", help="Re-run evaluation even if summary exists.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument(
        "--only-names-only",
        action="store_true",
        help="In in-memory mode, run only the names-only configuration.",
    )
    ap.add_argument(
        "--save-example-prompt",
        action="store_true",
        help="(In in-memory mode) Save one example prompt per configuration for debugging.",
    )
    ap.add_argument(
        "--example-prompt-dir",
        default=None,
        help="(In in-memory mode) Directory to write example prompts (passed to scripts/eval_cd_configs.py).",
    )
    ap.add_argument(
        "--overwrite-example-prompt",
        action="store_true",
        help="(In in-memory mode) Overwrite existing example prompt files.",
    )

    ap.add_argument(
        "--steps",
        default="run,evaluate,analyze",
        help=(
            "Comma-separated subset of steps to execute: run,evaluate,analyze. "
            "The 'run' step generates prompts and queries models in-memory via scripts/eval_cd_configs.py. "
            "The deprecated alias 'generate' is treated as 'run' for backward compatibility."
        ),
    )

    args = ap.parse_args()
    args.invocation_cwd = invocation_cwd
    args.bif_file = _resolve_file_arg(args.bif_file, repo_root=repo_root, invocation_cwd=invocation_cwd)

    # If dataset wasn’t explicitly set, default to bif basename
    if not args.dataset or args.dataset == "cancer" and Path(args.bif_file).stem != "cancer":
        args.dataset = Path(args.bif_file).stem

    steps = [s.strip().lower() for s in str(args.steps).split(",") if s.strip()]
    steps = ["run" if s == "generate" else s for s in steps]  # backward-compat alias
    allowed = {"run", "evaluate", "analyze"}
    if any(s not in allowed for s in steps):
        bad = [s for s in steps if s not in allowed]
        raise SystemExit(f"Unknown step(s): {bad}. Allowed: {sorted(allowed)}")

    if not args.model:
        args.model = ["gpt-5-mini"]

    # Small guardrail: OpenAI models need OPENAI_API_KEY available at runtime.
    if any(("gpt" in m.lower() or m.lower().startswith("o")) for m in args.model):
        if not os.getenv("OPENAI_API_KEY") and not args.dry_run:
            print(
                "[warn] OPENAI_API_KEY is not set in this environment. "
                "If you rely on ~/.bashrc, run this script from an interactive shell that sourced it.",
                file=sys.stderr,
            )

    if "run" in steps:
        step_generate_and_run_in_memory(args, experiments_dir=experiments_dir, dry_run=args.dry_run)
    if "evaluate" in steps:
        step_evaluate(args, experiments_dir=experiments_dir, dry_run=args.dry_run)
    if "analyze" in steps:
        step_analyze(args, experiments_dir=experiments_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
