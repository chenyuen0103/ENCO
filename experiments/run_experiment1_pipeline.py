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


_RESP_RE = re.compile(
    r"^responses_obs(?P<obs>\d+)_int(?P<int>\d+)_shuf(?P<shuf>\d+)(?P<tags>.*?)(?:_(?P<model>[^_]+))?$",
    flags=re.IGNORECASE,
)

_ENCO_RE = re.compile(
    r"^predictions_obs(?P<obs>\d+)_int(?P<int>\d+)_ENCO$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ResponseMeta:
    dataset: str
    csv_path: Path
    model: str
    is_names_only: bool
    obs_n: Optional[int]
    int_n: Optional[int]
    shuf_n: Optional[int]
    anonymize: bool
    prompt_style: str  # "cases" or "matrix" or "names_only"
    row_order: str
    col_order: str
    causal_rules: bool
    give_steps: bool


def _repo_paths() -> tuple[Path, Path]:
    experiments_dir = Path(__file__).resolve().parent
    repo_root = experiments_dir.parent
    return repo_root, experiments_dir


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _infer_model_from_stem(stem: str) -> str:
    # query_gemini.py appends args.model.split("/")[-1] to the stem
    # if not already present, so the model is typically the last underscore token.
    parts = stem.split("_")
    if parts:
        return parts[-1]
    return "unknown"


def _parse_response_meta(dataset: str, csv_path: Path) -> ResponseMeta:
    stem = csv_path.stem

    # ENCO baseline: "predictions_obs{N}_int{M}_ENCO.csv"
    m_enco = _ENCO_RE.match(stem)
    if m_enco:
        return ResponseMeta(
            dataset=dataset,
            csv_path=csv_path,
            model="ENCO",
            is_names_only=False,
            obs_n=int(m_enco.group("obs")),
            int_n=int(m_enco.group("int")),
            shuf_n=None,
            anonymize=False,
            prompt_style="baseline",
            row_order="random",
            col_order="original",
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
            is_names_only=True,
            obs_n=None,
            int_n=None,
            shuf_n=None,
            anonymize=False,
            prompt_style="names_only",
            row_order="random",
            col_order=col_order,
            causal_rules=("rules" in stem.lower()),
            give_steps=("steps" in stem.lower()),
        )

    m = _RESP_RE.match(stem)
    if not m:
        # best-effort fallback
        return ResponseMeta(
            dataset=dataset,
            csv_path=csv_path,
            model=_infer_model_from_stem(stem),
            is_names_only=False,
            obs_n=None,
            int_n=None,
            shuf_n=None,
            anonymize=("anon" in stem.lower()),
            prompt_style=("matrix" if "matrix" in stem.lower() else "cases"),
            row_order="random",
            col_order=("topo" if "coltopo" in stem.lower() else "original"),
            causal_rules=("rules" in stem.lower()),
            give_steps=("steps" in stem.lower()),
        )

    obs_n = int(m.group("obs"))
    int_n = int(m.group("int"))
    shuf_n = int(m.group("shuf"))
    tags = (m.group("tags") or "").lower()
    model = (m.group("model") or _infer_model_from_stem(stem)).strip()

    row_order = "random"
    col_order = "original"
    for token in tags.split("_"):
        token = token.strip()
        if token.startswith("row"):
            row_order = token.removeprefix("row")
        if token.startswith("col"):
            col_order = token.removeprefix("col")

    prompt_style = "matrix" if "matrix" in tags else "cases"
    return ResponseMeta(
        dataset=dataset,
        csv_path=csv_path,
        model=model,
        is_names_only=False,
        obs_n=obs_n,
        int_n=int_n,
        shuf_n=shuf_n,
        anonymize=("anon" in tags),
        prompt_style=prompt_style,
        row_order=row_order,
        col_order=col_order,
        causal_rules=("rules" in tags),
        give_steps=("steps" in tags),
    )


def _find_prompt_csvs(experiments_dir: Path, dataset: str) -> tuple[list[Path], list[Path]]:
    # Core prompts: prompts_obs*_int*_shuf*.csv
    core = sorted((experiments_dir / "prompts" / "experiment1" / dataset).rglob("prompts_obs*_int*_shuf*.csv"))
    # Names-only prompts (produced by generate_prompts_names_only.py inside experiment1 dirs)
    names_only = sorted((experiments_dir / "prompts" / "experiment1" / dataset).rglob("prompts_names_only*.csv"))
    return core, names_only


def _find_response_csvs(experiments_dir: Path, dataset: str) -> list[Path]:
    resp_dir = experiments_dir / "responses" / dataset
    if not resp_dir.exists():
        return []
    return sorted(resp_dir.glob("*.csv"))


def step_generate(args: argparse.Namespace, *, experiments_dir: Path, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "generate_prompt_files.py",
        "--bif-file",
        args.bif_file,
        "--num-prompts",
        str(args.num_prompts),
        "--seed",
        str(args.seed),
    ]
    for s in args.shuffles_per_graph:
        cmd.extend(["--shuffles-per-graph", str(int(s))])
    _run(cmd, cwd=experiments_dir, dry_run=dry_run)


def step_generate_and_run_in_memory(args: argparse.Namespace, *, experiments_dir: Path, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "run_experiment1_in_memory.py",
        "--bif-file",
        args.bif_file,
        "--num-prompts",
        str(args.num_prompts),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
    ]
    for s in args.shuffles_per_graph:
        cmd.extend(["--shuffles-per-graph", str(int(s))])
    for m in args.model:
        cmd.extend(["--model", m])
    if args.overwrite:
        cmd.append("--overwrite")
    if getattr(args, "only_names_only", False):
        cmd.append("--only-names-only")
    if args.dry_run:
        cmd.append("--dry-run")
    _run(cmd, cwd=experiments_dir, dry_run=dry_run)


def step_run_models(args: argparse.Namespace, *, experiments_dir: Path, dry_run: bool) -> None:
    core_csvs, names_only_csvs = _find_prompt_csvs(experiments_dir, args.dataset)
    if not core_csvs and not names_only_csvs:
        raise SystemExit(
            f"No prompt CSVs found under {experiments_dir/'prompts'/'experiment1'/args.dataset}. "
            "Run the generate step first."
        )

    # 1) Run core prompts via the wrapper
    if core_csvs:
        cmd = [
            sys.executable,
            "run_api_models.py",
            "--base-root",
            "prompts/experiment1",
            "--dataset",
            args.dataset,
            "--pattern",
            "prompts_obs*_int*_shuf*.csv",
            "--temperature",
            str(args.temperature),
        ]
        for m in args.model:
            cmd.extend(["--model", m])
        if args.overwrite:
            cmd.append("--overwrite")
        _run(cmd, cwd=experiments_dir, dry_run=dry_run)

    # 2) Run names-only CSVs directly (not discovered by run_api_models.py’s default pattern)
    for csv_path in names_only_csvs:
        for model in args.model:
            cmd = [
                sys.executable,
                "query_gemini.py",
                "--csv",
                str(csv_path),
                "--model",
                model,
                "--temperature",
                str(args.temperature),
                "--provider",
                "auto",
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            _run(cmd, cwd=experiments_dir, dry_run=dry_run)


def step_evaluate(args: argparse.Namespace, *, experiments_dir: Path, dry_run: bool) -> None:
    resp_csvs = _find_response_csvs(experiments_dir, args.dataset)
    if not resp_csvs:
        raise SystemExit(
            f"No response CSVs found under {experiments_dir/'responses'/args.dataset}. "
            "Run the model step first."
        )

    for csv_path in resp_csvs:
        summary_path = csv_path.with_suffix(csv_path.suffix + ".summary.json")
        if summary_path.exists() and not args.overwrite_eval:
            continue
        cmd = [
            sys.executable,
            "evaluate.py",
            "--csv",
            str(csv_path),
            "--tau",
            str(args.tau),
        ]
        _run(cmd, cwd=experiments_dir, dry_run=dry_run)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_parent(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def _ordering_bias_from_csv(csv_path: Path) -> dict[str, Any]:
    """
    Estimate ordering sensitivity by grouping rows by data_idx and measuring the spread
    across shuffle_idx for each data_idx, then aggregating across data_idx.
    Requires that evaluate.py has already appended per-row metrics (e.g. shd).
    """
    import pandas as pd  # local import to keep script import-light

    df = pd.read_csv(csv_path)
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
    resp_csvs = _find_response_csvs(experiments_dir, args.dataset)
    if not resp_csvs:
        raise SystemExit(
            f"No response CSVs found under {experiments_dir/'responses'/args.dataset}. "
            "Run the model step first."
        )

    out_dir = experiments_dir / "out" / "experiment1"
    _ensure_parent(out_dir / "placeholder.txt", dry_run=dry_run)

    # 1) Collect per-condition summaries into one CSV
    summary_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []

    for csv_path in resp_csvs:
        meta = _parse_response_meta(args.dataset, csv_path)
        summary_path = csv_path.with_suffix(csv_path.suffix + ".summary.json")
        if not summary_path.exists():
            # Require evaluation first; keeps pipeline explicit
            continue

        summary = _read_json(summary_path)
        row: dict[str, Any] = {
            "dataset": meta.dataset,
            "model": meta.model,
            "is_names_only": int(meta.is_names_only),
            "obs_n": meta.obs_n,
            "int_n": meta.int_n,
            "shuffles_per_graph": meta.shuf_n,
            "anonymize": int(meta.anonymize),
            "prompt_style": meta.prompt_style,
            "row_order": meta.row_order,
            "col_order": meta.col_order,
            "causal_rules": int(meta.causal_rules),
            "give_steps": int(meta.give_steps),
            "response_csv": str(csv_path),
        }
        row.update(summary)
        summary_rows.append(row)

        # 2) Ordering bias analysis (only makes sense if shuffle_idx varies)
        if meta.shuf_n is not None and meta.shuf_n >= 2:
            ob = _ordering_bias_from_csv(csv_path)
            ordering_rows.append(
                {
                    "dataset": meta.dataset,
                    "model": meta.model,
                    "obs_n": meta.obs_n,
                    "int_n": meta.int_n,
                    "shuffles_per_graph": meta.shuf_n,
                    "anonymize": int(meta.anonymize),
                    "prompt_style": meta.prompt_style,
                    "row_order": meta.row_order,
                    "col_order": meta.col_order,
                    "response_csv": str(csv_path),
                    **ob,
                }
            )

    if not summary_rows:
        raise SystemExit(
            "No *.summary.json found next to response CSVs. Run the evaluate step first."
        )

    summary_csv = out_dir / f"{args.dataset}_summary.csv"
    if not dry_run:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in summary_rows for k in r.keys()}))
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

    ap = argparse.ArgumentParser(
        description="Run Experiment 1 end-to-end: generate prompts, query models, evaluate, and analyze."
    )
    ap.add_argument("--dataset", default="cancer", help="Dataset name (defaults to bif basename).")
    ap.add_argument(
        "--bif-file",
        default=str(repo_root / "causal_graphs" / "real_data" / "small_graphs" / "cancer.bif"),
        help="Path to the BIF file.",
    )
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffles-per-graph", type=int, action="append", default=[1])

    ap.add_argument("--model", action="append", default=["gpt-5-mini"], help="Repeatable.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.7, help="Consensus threshold for evaluate.py.")

    ap.add_argument("--overwrite", action="store_true", help="Re-query model responses.")
    ap.add_argument("--overwrite-eval", action="store_true", help="Re-run evaluation even if summary exists.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument(
        "--in-memory",
        action="store_true",
        help="Generate prompts in-memory and query models without writing prompt files.",
    )
    ap.add_argument(
        "--only-names-only",
        action="store_true",
        help="In in-memory mode, run only the names-only configuration.",
    )

    ap.add_argument(
        "--steps",
        default="generate,run,evaluate,analyze",
        help="Comma-separated subset of: generate,run,evaluate,analyze",
    )

    args = ap.parse_args()

    # If dataset wasn’t explicitly set, default to bif basename
    if not args.dataset or args.dataset == "cancer" and Path(args.bif_file).stem != "cancer":
        args.dataset = Path(args.bif_file).stem

    steps = [s.strip().lower() for s in str(args.steps).split(",") if s.strip()]
    allowed = {"generate", "run", "evaluate", "analyze"}
    if any(s not in allowed for s in steps):
        bad = [s for s in steps if s not in allowed]
        raise SystemExit(f"Unknown step(s): {bad}. Allowed: {sorted(allowed)}")

    # Small guardrail: OpenAI models need OPENAI_API_KEY available at runtime.
    if any(("gpt" in m.lower() or m.lower().startswith("o")) for m in args.model):
        if not os.getenv("OPENAI_API_KEY") and not args.dry_run:
            print(
                "[warn] OPENAI_API_KEY is not set in this environment. "
                "If you rely on ~/.bashrc, run this script from an interactive shell that sourced it.",
                file=sys.stderr,
            )

    if args.in_memory and ("generate" in steps or "run" in steps):
        step_generate_and_run_in_memory(args, experiments_dir=experiments_dir, dry_run=args.dry_run)
    else:
        if "generate" in steps:
            step_generate(args, experiments_dir=experiments_dir, dry_run=args.dry_run)
        if "run" in steps:
            step_run_models(args, experiments_dir=experiments_dir, dry_run=args.dry_run)
    if "evaluate" in steps:
        step_evaluate(args, experiments_dir=experiments_dir, dry_run=args.dry_run)
    if "analyze" in steps:
        step_analyze(args, experiments_dir=experiments_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
