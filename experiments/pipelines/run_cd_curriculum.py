#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from cd_generation.format import (
        DEFAULT_FORMAT_HINT_TEXT,
        build_payload_completion,
        canonicalize_cd_prompt,
        default_format_hint_text,
        default_short_think_text,
        validate_sft_example,
    )
except ModuleNotFoundError:
    from experiments.cd_generation.format import (
        DEFAULT_FORMAT_HINT_TEXT,
        build_payload_completion,
        canonicalize_cd_prompt,
        default_format_hint_text,
        default_short_think_text,
        validate_sft_example,
    )

try:
    from train_sft import run_sft as _train_sft_run_sft
except ModuleNotFoundError:
    from experiments.train_sft import run_sft as _train_sft_run_sft


def run_sft(
    *,
    base_model: str,
    sft_jsonl: Path,
    sft_output_dir: Path,
    sft_epochs: float,
    sft_lr: float,
    sft_batch_size: int,
    sft_grad_accum: int,
    sft_max_seq_length: int,
    sft_save_steps: int,
    sft_logging_steps: int,
    **_unused: Any,
) -> None:
    """Compatibility adapter for the older curriculum SFT call shape."""
    _train_sft_run_sft(
        base_model=base_model,
        distributed_base_model=None,
        sft_jsonl=sft_jsonl,
        sft_output_dir=sft_output_dir,
        sft_epochs=sft_epochs,
        sft_lr=sft_lr,
        sft_batch_size=sft_batch_size,
        sft_eval_batch_size=1,
        sft_grad_accum=sft_grad_accum,
        sft_max_seq_length=sft_max_seq_length,
        sft_save_steps=sft_save_steps,
        sft_logging_steps=sft_logging_steps,
        sft_save_total_limit=None,
        train_log_jsonl=None,
        eval_jsonl=None,
        eval_log_jsonl=None,
        validation_split_ratio=0.0,
        eval_every=None,
        use_unsloth=True,
    )


SUPPORTED_TASKS = {"causal_discovery", "cd_descendants"}


def _set_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


def _resolve_path(path_str: str | None, *, base_dir: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    candidate = (base_dir / p).resolve()
    if candidate.exists():
        return candidate
    return p.resolve()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    _set_csv_field_limit()
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _parse_extra_args(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value).strip()
    if not s:
        return []
    return shlex.split(s)


def _append_flag_once(args: List[str], flag: str, enabled: bool) -> List[str]:
    if not enabled:
        return list(args)
    if flag in args:
        return list(args)
    return [*args, flag]


def _extract_payload_text(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("empty answer payload")
    if "<answer>" in s and "</answer>" in s:
        start = s.find("<answer>")
        end = s.rfind("</answer>")
        if start >= 0 and end > start:
            return s[start + len("<answer>") : end].strip()
        return s
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return s


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_generic_sft_jsonl(
    *,
    in_csv: Path,
    out_jsonl: Path,
    task: str,
    think_text: str,
    prompt_text_col: str,
    prompt_path_col: str,
    answer_col: str,
    answer_path_col: str,
    answer_mode: str,
    wrap_system_prompt: bool,
    append_format_hint: bool,
    format_hint_text: str,
) -> tuple[int, int]:
    _set_csv_field_limit()
    rows = _read_csv_rows(in_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows):
            try:
                prompt = str(row.get(prompt_text_col) or "").strip()
                if not prompt:
                    prompt_path = str(row.get(prompt_path_col) or "").strip()
                    if prompt_path:
                        prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
                if not prompt:
                    raise ValueError("empty prompt")

                answer_raw = str(row.get(answer_col) or "").strip()
                if not answer_raw:
                    answer_path = str(row.get(answer_path_col) or "").strip()
                    if answer_path:
                        answer_raw = Path(answer_path).read_text(encoding="utf-8").strip()
                if not answer_raw:
                    raise ValueError("empty answer")

                prompt = canonicalize_cd_prompt(
                    prompt,
                    task=task,
                    wrap_system_prompt=wrap_system_prompt,
                    append_format_hint=append_format_hint,
                    format_hint_text=format_hint_text,
                    prefill_think=(answer_mode == "payload"),
                )

                if answer_mode == "completion":
                    completion = answer_raw.strip()
                elif answer_mode == "payload":
                    payload = _extract_payload_text(answer_raw)
                    completion = build_payload_completion(payload, think_text=think_text, task=task)
                else:
                    raise ValueError(f"unsupported answer_mode={answer_mode!r}")

                rec = {
                    "prompt": prompt,
                    "answer": completion,
                    "text": prompt + "\n\n" + completion,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
            except Exception as exc:
                skipped += 1
                print(f"[warn] skip row {idx} from {in_csv}: {exc}", file=sys.stderr)
    return wrote, skipped


def _validate_sft_jsonl(path: Path, *, sample_limit: int = 8) -> Dict[str, Any]:
    total = 0
    issues: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            if len(issues) >= sample_limit:
                continue
            row = json.loads(s)
            row_issues = validate_sft_example(str(row.get("prompt") or ""), str(row.get("answer") or ""))
            if row_issues:
                issues.append(
                    {
                        "row_idx": total - 1,
                        "issues": row_issues,
                    }
                )
    return {
        "rows": total,
        "sample_issues": issues,
        "ok": not issues,
    }


def _sample_rows(
    rows: List[Dict[str, str]],
    n: int,
    *,
    rng: random.Random,
) -> List[Dict[str, str]]:
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        return [dict(r) for r in rows]
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    return [dict(rows[i]) for i in idxs[:n]]


def build_replay_mixed_csv(
    *,
    current_csv: Path,
    previous_stage_csvs: List[Path],
    out_csv: Path,
    replay_ratio: float,
    replay_seed: int,
) -> Dict[str, Any]:
    current_rows = _read_csv_rows(current_csv)
    mixed_rows: List[Dict[str, Any]] = []
    for row in current_rows:
        row2 = dict(row)
        row2["_source_stage"] = "current"
        row2["_is_replay"] = "0"
        mixed_rows.append(row2)

    replay_added = 0
    source_counts: Dict[str, int] = {}
    if replay_ratio > 0 and previous_stage_csvs:
        rng = random.Random(replay_seed)
        replay_target = max(0, int(round(len(current_rows) * float(replay_ratio))))
        prev_stage_rows: List[tuple[str, List[Dict[str, str]]]] = []
        for path in previous_stage_csvs:
            rows = _read_csv_rows(path)
            prev_stage_rows.append((path.stem, rows))
        if replay_target > 0 and prev_stage_rows:
            per_stage = max(1, replay_target // len(prev_stage_rows))
            sampled: List[Dict[str, Any]] = []
            sampled_keys: set[tuple[str, str]] = set()
            for stage_name, rows in prev_stage_rows:
                take_n = min(len(rows), per_stage)
                for row in _sample_rows(rows, take_n, rng=rng):
                    row_key = json.dumps(row, sort_keys=True, ensure_ascii=False)
                    sampled_keys.add((stage_name, row_key))
                    row["_source_stage"] = stage_name
                    row["_is_replay"] = "1"
                    sampled.append(row)
                    source_counts[stage_name] = source_counts.get(stage_name, 0) + 1
            if len(sampled) < replay_target:
                leftovers: List[tuple[str, Dict[str, str]]] = []
                for stage_name, rows in prev_stage_rows:
                    for row in rows:
                        row_key = json.dumps(row, sort_keys=True, ensure_ascii=False)
                        if (stage_name, row_key) in sampled_keys:
                            continue
                        leftovers.append((stage_name, row))
                rng.shuffle(leftovers)
                need = replay_target - len(sampled)
                for stage_name, row in leftovers[:need]:
                    row2 = dict(row)
                    row2["_source_stage"] = stage_name
                    row2["_is_replay"] = "1"
                    sampled.append(row2)
                    source_counts[stage_name] = source_counts.get(stage_name, 0) + 1
            mixed_rows.extend(sampled[:replay_target])
            replay_added = min(len(sampled), replay_target)

    if replay_added > 0:
        rng = random.Random(replay_seed)
        rng.shuffle(mixed_rows)

    _write_csv_rows(out_csv, mixed_rows)
    return {
        "current_rows": len(current_rows),
        "replay_rows": replay_added,
        "total_rows": len(mixed_rows),
        "source_counts": source_counts,
        "mixed_csv": str(out_csv),
    }


def _latest_checkpoint_or_dir(path: Path) -> Path:
    checkpoints = sorted(path.glob("checkpoint-*"), key=lambda p: p.name)
    return checkpoints[-1] if checkpoints else path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_last_jsonl(path: Path) -> Dict[str, Any]:
    last: Dict[str, Any] | None = None
    last_metrics: Dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            last = row
            if any(
                isinstance(k, str) and (
                    k.startswith("rewards/")
                    or k.startswith("completions/")
                    or k in {"reward", "reward_std", "entropy", "loss", "grad_norm"}
                )
                for k in row.keys()
            ):
                last_metrics = row
    if last_metrics is not None:
        return last_metrics
    if last is None:
        raise ValueError(f"no JSONL rows in {path}")
    return last


def _metric_ok(value: float, spec: Dict[str, Any]) -> bool:
    if "min" in spec and value < float(spec["min"]):
        return False
    if "max" in spec and value > float(spec["max"]):
        return False
    if "equals" in spec and value != float(spec["equals"]):
        return False
    return True


def _check_gate(metrics: Dict[str, Any], gate: Dict[str, Any]) -> tuple[bool, List[str]]:
    messages: List[str] = []
    ok = True
    for key, spec in gate.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Gate spec for {key!r} must be an object")
        raw = metrics.get(key)
        if raw is None:
            ok = False
            messages.append(f"{key}=missing")
            continue
        try:
            value = float(raw)
        except Exception:
            ok = False
            messages.append(f"{key}={raw!r} (non-numeric)")
            continue
        passed = _metric_ok(value, spec)
        ok = ok and passed
        parts = [f"{key}={value:.4f}"]
        if "min" in spec:
            parts.append(f"min={float(spec['min']):.4f}")
        if "max" in spec:
            parts.append(f"max={float(spec['max']):.4f}")
        if "equals" in spec:
            parts.append(f"equals={float(spec['equals']):.4f}")
        parts.append("PASS" if passed else "FAIL")
        messages.append(" ".join(parts))
    return ok, messages


def _run_cmd(cmd: List[str], *, env: Dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("[run]", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_grpo_train(
    *,
    python_exe: str,
    nproc_per_node: int,
    grpo_script: Path,
    task: str,
    model_id: str,
    train_csv: Path,
    eval_csv: Path | None,
    output_dir: Path,
    extra_args: List[str],
    env: Dict[str, str] | None,
    dry_run: bool,
) -> None:
    if int(nproc_per_node) > 1:
        cmd = [
            python_exe,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(int(nproc_per_node)),
            str(grpo_script),
        ]
    else:
        cmd = [python_exe, str(grpo_script)]
    cmd.extend(
        [
            "--mode",
            "train",
            "--task",
            task,
            "--model_id",
            str(model_id),
            "--cd-train-csv",
            str(train_csv),
            "--output_dir",
            str(output_dir),
        ]
    )
    if eval_csv is not None:
        cmd.extend(["--cd-test-csv", str(eval_csv)])
    cmd.extend(extra_args)
    _run_cmd(cmd, env=env, dry_run=dry_run)


def _run_grpo_eval(
    *,
    python_exe: str,
    grpo_script: Path,
    task: str,
    eval_model: Path,
    eval_csv: Path,
    output_json: Path,
    extra_args: List[str],
    env: Dict[str, str] | None,
    dry_run: bool,
) -> None:
    cmd = [
        python_exe,
        str(grpo_script),
        "--mode",
        "eval",
        "--task",
        task,
        "--eval_model",
        str(eval_model),
        "--cd-test-csv",
        str(eval_csv),
        "--eval_output_json",
        str(output_json),
    ]
    cmd.extend(extra_args)
    _run_cmd(cmd, env=env, dry_run=dry_run)


def _normalize_stage_from_manifest(
    stage: Dict[str, Any],
    *,
    default_task: str,
    base_dir: Path,
) -> Dict[str, Any]:
    normalized = dict(stage)
    if "train_csv" not in normalized and "csv_path" in normalized:
        normalized["train_csv"] = normalized["csv_path"]
    normalized.setdefault("task", default_task)
    normalized.setdefault("name", Path(str(normalized["train_csv"])).stem)
    if normalized["task"] not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported stage task={normalized['task']!r}. "
            f"Supported tasks: {sorted(SUPPORTED_TASKS)}"
        )
    normalized["train_csv"] = str(_resolve_path(str(normalized["train_csv"]), base_dir=base_dir))
    if normalized.get("eval_csv"):
        normalized["eval_csv"] = str(_resolve_path(str(normalized["eval_csv"]), base_dir=base_dir))
    return normalized


def load_curriculum(curriculum_path: Path) -> Dict[str, Any]:
    payload = _load_json(curriculum_path)
    if not isinstance(payload, dict):
        raise ValueError("Curriculum file must be a JSON object.")
    base_dir = curriculum_path.parent
    default_task = str(payload.get("task") or "causal_discovery")
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Curriculum file must contain a non-empty 'stages' list.")
    payload["stages"] = [
        _normalize_stage_from_manifest(dict(stage), default_task=default_task, base_dir=base_dir)
        for stage in stages
    ]
    return payload


def write_example_curriculum(path: Path) -> None:
    example = {
        "base_model": "unsloth/qwen3-4b-thinking-2507-unsloth-bnb-4bit",
        "python_exe": sys.executable,
        "grpo_script": "experiments/grpo.py",
        "nproc_per_node": 1,
        "cuda_visible_devices": "0",
        "task": "cd_descendants",
        "default_eval_args": [
            "--eval_n",
            "64",
            "--eval_max_new_tokens",
            "256",
            "--eval_batch_size",
            "4",
        ],
        "default_grpo_args": [
            "--no-use-vllm",
            "--max_completion_length",
            "512",
            "--num_generations",
            "2",
        ],
        "stages": [
            {
                "name": "stage_1_descendants",
                "train_csv": "experiments/prompts/cd_descendants/sachs/stage_1_named_obs50_int10.csv",
                "eval_csv": "experiments/prompts/cd_descendants/sachs/stage_1_named_obs50_int10.csv",
                "task": "cd_descendants",
                "enable_sft": True,
                "enable_grpo": True,
                "replay_ratio": 0.0,
                "sft": {
                    "epochs": 1.0,
                    "lr": 2e-5,
                    "batch_size": 1,
                    "grad_accum": 4,
                    "max_seq_length": 2048,
                    "load_in_4bit": True,
                },
                "gate": {
                    "after_sft_eval": {
                        "format_rate": {"min": 0.95},
                        "accuracy": {"min": 0.50},
                    },
                    "after_grpo_eval": {
                        "format_rate": {"min": 0.95},
                        "accuracy": {"min": 0.70},
                    },
                    "after_grpo_train": {
                        "rewards/cd_format_reward/mean": {"min": 0.80},
                        "completions/clipped_ratio": {"max": 0.20},
                    },
                },
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a staged causal-discovery curriculum with SFT warmup, GRPO handoff, replay mixing, and eval gating."
    )
    p.add_argument("--curriculum-file", type=Path, default=None, help="JSON curriculum manifest.")
    p.add_argument("--output-root", type=Path, required=False, default=Path("experiments/checkpoints/cd_curriculum"))
    p.add_argument("--base-model", type=str, default=None, help="Override base model in the curriculum file.")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--grpo-script", type=Path, default=None, help="Override GRPO script path.")
    p.add_argument("--nproc-per-node", type=int, default=None, help="Override distributed process count.")
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES for GPU stages. Defaults to CLI value, else shell env, else curriculum value, else 0.",
    )
    p.add_argument("--start-stage", type=int, default=1, help="1-based stage index to start from.")
    p.add_argument("--stop-stage", type=int, default=0, help="1-based inclusive stop stage (0 = all).")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running training/eval.")
    p.add_argument(
        "--write-example-curriculum",
        type=Path,
        default=None,
        help="Write an example curriculum JSON and exit.",
    )
    p.add_argument(
        "--reuse-existing-sft",
        action="store_true",
        help="If a stage SFT checkpoint already exists under the output root, reuse it and skip SFT training.",
    )
    p.add_argument(
        "--reuse-existing-sft-eval-json",
        type=Path,
        default=None,
        help=(
            "Optional eval JSON to use for the stage's after_sft_eval gate when reusing an existing SFT checkpoint. "
            "Useful when responses were rescored from a saved eval file without rerunning inference."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_example_curriculum is not None:
        write_example_curriculum(args.write_example_curriculum)
        print(f"[done] wrote example curriculum -> {args.write_example_curriculum}")
        return

    if args.curriculum_file is None:
        raise ValueError("--curriculum-file is required unless --write-example-curriculum is used.")

    curriculum = load_curriculum(args.curriculum_file.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    python_exe = str(args.python_exe or curriculum.get("python_exe") or sys.executable)
    grpo_script = Path(
        args.grpo_script
        or curriculum.get("grpo_script")
        or "experiments/grpo.py"
    ).resolve()
    nproc_per_node = int(
        args.nproc_per_node if args.nproc_per_node is not None else curriculum.get("nproc_per_node", 1)
    )
    current_model = str(args.base_model or curriculum.get("base_model") or "")
    if not current_model:
        raise ValueError("No base model specified. Set --base-model or provide base_model in curriculum JSON.")
    shell_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_visible_devices = str(
        args.cuda_visible_devices
        if args.cuda_visible_devices is not None
        else (
            shell_cuda_visible_devices
            if shell_cuda_visible_devices is not None
            else curriculum.get("cuda_visible_devices", "0")
        )
    )
    gpu_env = os.environ.copy()
    gpu_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    default_eval_args = _parse_extra_args(curriculum.get("default_eval_args"))
    default_grpo_args = _parse_extra_args(curriculum.get("default_grpo_args"))
    stages = curriculum["stages"]

    stop_stage = int(args.stop_stage) if int(args.stop_stage) > 0 else len(stages)
    previous_stage_train_csvs: List[Path] = []
    run_summary: Dict[str, Any] = {
        "curriculum_file": str(args.curriculum_file.resolve()),
        "output_root": str(output_root),
        "base_model": current_model,
        "grpo_script": str(grpo_script),
        "nproc_per_node": nproc_per_node,
        "cuda_visible_devices": cuda_visible_devices,
        "stages": [],
    }

    if int(args.start_stage) > 1:
        for stage_idx, stage in enumerate(stages, start=1):
            if stage_idx >= int(args.start_stage):
                break
            stage_dir = output_root / f"{stage_idx:02d}_{stage['name']}"
            stage_result_path = stage_dir / "stage_result.json"
            if not stage_result_path.exists():
                raise FileNotFoundError(
                    f"Cannot resume from --start-stage {args.start_stage}: "
                    f"missing prior stage result {stage_result_path}"
                )
            prior = _load_json(stage_result_path)
            promoted = str(prior.get("promoted_model") or "").strip()
            if not promoted:
                raise ValueError(f"Prior stage result missing promoted_model: {stage_result_path}")
            current_model = promoted
            previous_stage_train_csvs.append(Path(str(stage["train_csv"])).resolve())
            run_summary["stages"].append(prior)
        run_summary["base_model"] = current_model

    for stage_idx, stage in enumerate(stages, start=1):
        if stage_idx < int(args.start_stage) or stage_idx > stop_stage:
            continue

        stage_name = str(stage["name"])
        task = str(stage["task"])
        train_csv = Path(stage["train_csv"]).resolve()
        eval_csv = Path(stage.get("eval_csv") or stage["train_csv"]).resolve()
        enable_sft = bool(stage.get("enable_sft", True))
        enable_grpo = bool(stage.get("enable_grpo", True))
        replay_ratio = float(stage.get("replay_ratio", 0.0))
        replay_seed = int(stage.get("replay_seed", 1234 + stage_idx))
        answer_mode = str(stage.get("sft_answer_mode", "payload"))
        think_text = str(stage.get("think_text") or default_short_think_text(task))
        prompt_text_col = str(stage.get("prompt_text_col", "prompt_text"))
        prompt_path_col = str(stage.get("prompt_path_col", "prompt_path"))
        answer_col = str(stage.get("answer_col", "answer"))
        answer_path_col = str(stage.get("answer_path_col", "answer_path"))
        wrap_system_prompt = bool(stage.get("cd_wrap_system_prompt", curriculum.get("cd_wrap_system_prompt", True)))
        append_format_hint = bool(stage.get("cd_append_format_hint", curriculum.get("cd_append_format_hint", True)))
        format_hint_text = str(
            stage.get("cd_format_hint_text")
            or curriculum.get("cd_format_hint_text")
            or default_format_hint_text(task)
        )
        gate_spec = dict(stage.get("gate") or {})
        enable_thinking = bool(stage.get("enable_thinking", curriculum.get("enable_thinking", False)))

        stage_dir = output_root / f"{stage_idx:02d}_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        mixed_train_csv = stage_dir / "train_mixed.csv"
        replay_summary = build_replay_mixed_csv(
            current_csv=train_csv,
            previous_stage_csvs=previous_stage_train_csvs,
            out_csv=mixed_train_csv,
            replay_ratio=replay_ratio,
            replay_seed=replay_seed,
        )

        stage_result: Dict[str, Any] = {
            "stage_idx": stage_idx,
            "name": stage_name,
            "task": task,
            "input_train_csv": str(train_csv),
            "input_eval_csv": str(eval_csv),
            "mixed_train_csv": str(mixed_train_csv),
            "replay": replay_summary,
            "start_model": current_model,
        }

        promoted_model = current_model

        if enable_sft:
            sft_cfg = dict(stage.get("sft") or {})
            sft_jsonl = stage_dir / "sft_train.jsonl"
            sft_output_dir = stage_dir / "sft"
            wrote, skipped = build_generic_sft_jsonl(
                in_csv=mixed_train_csv,
                out_jsonl=sft_jsonl,
                task=task,
                think_text=think_text,
                prompt_text_col=prompt_text_col,
                prompt_path_col=prompt_path_col,
                answer_col=answer_col,
                answer_path_col=answer_path_col,
                answer_mode=answer_mode,
                wrap_system_prompt=wrap_system_prompt,
                append_format_hint=append_format_hint,
                format_hint_text=format_hint_text,
            )
            sft_signature = {
                "task": task,
                "answer_mode": answer_mode,
                "think_text": think_text,
                "wrap_system_prompt": wrap_system_prompt,
                "append_format_hint": append_format_hint,
                "format_hint_text": format_hint_text,
                "input_csv": str(mixed_train_csv),
                "jsonl_sha256": _file_sha256(sft_jsonl),
            }
            sft_signature_path = stage_dir / "sft_signature.json"
            prior_sft_signature = _load_json(sft_signature_path) if sft_signature_path.exists() else None
            reuse_existing_sft = bool(
                args.reuse_existing_sft
                and sft_output_dir.exists()
                and prior_sft_signature == sft_signature
            )
            if bool(args.reuse_existing_sft and sft_output_dir.exists()) and not reuse_existing_sft:
                print(
                    f"[reuse] stage {stage_idx} disabled SFT checkpoint reuse because prompt/data signature changed"
                )
            _write_json(sft_signature_path, sft_signature)
            sft_validation = _validate_sft_jsonl(sft_jsonl)
            stage_result["sft_build"] = {
                "jsonl": str(sft_jsonl),
                "wrote": wrote,
                "skipped": skipped,
                "validation": sft_validation,
                "reused_checkpoint": reuse_existing_sft,
                "signature_path": str(sft_signature_path),
                "signature": sft_signature,
            }
            if wrote <= 0:
                raise ValueError(
                    f"Stage {stage_idx} ({stage_name}) produced no SFT rows from {mixed_train_csv}. "
                    f"skipped={skipped}"
                )
            if not sft_validation["ok"]:
                raise ValueError(
                    f"Stage {stage_idx} ({stage_name}) produced invalid SFT prompt/completion rows: "
                    f"{sft_validation['sample_issues']}"
                )
            if not args.dry_run and not reuse_existing_sft:
                prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                try:
                    run_sft(
                        base_model=current_model,
                        sft_jsonl=sft_jsonl,
                        sft_output_dir=sft_output_dir,
                        sft_epochs=float(sft_cfg.get("epochs", 1.0)),
                        sft_lr=float(sft_cfg.get("lr", 2e-5)),
                        sft_batch_size=int(sft_cfg.get("batch_size", 1)),
                        sft_grad_accum=int(sft_cfg.get("grad_accum", 4)),
                        sft_max_seq_length=int(sft_cfg.get("max_seq_length", 2048)),
                        sft_save_steps=int(sft_cfg.get("save_steps", 50)),
                        sft_logging_steps=int(sft_cfg.get("logging_steps", 5)),
                        sft_load_in_4bit=bool(sft_cfg.get("load_in_4bit", True)),
                        sft_use_unsloth_gc=bool(sft_cfg.get("use_unsloth_gc", True)),
                    )
                finally:
                    if prev_cvd is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
            elif reuse_existing_sft:
                print(f"[reuse] stage {stage_idx} using existing SFT checkpoint at {sft_output_dir}")
            stage_result["sft_output_dir"] = str(sft_output_dir)
            stage_result["sft_eval_json"] = str(stage_dir / "sft_eval.json")
            promoted_model = str(sft_output_dir)

            sft_eval_args = default_eval_args + _parse_extra_args(stage.get("sft_eval_args"))
            sft_eval_args = _append_flag_once(sft_eval_args, "--enable-thinking", enable_thinking)
            sft_eval_json = stage_dir / "sft_eval.json"
            sft_eval_gate_json = sft_eval_json
            if args.reuse_existing_sft_eval_json is not None:
                sft_eval_gate_json = args.reuse_existing_sft_eval_json.resolve()
                if not sft_eval_gate_json.exists():
                    raise FileNotFoundError(
                        f"--reuse-existing-sft-eval-json not found: {sft_eval_gate_json}"
                    )
                print(f"[reuse] stage {stage_idx} using existing SFT eval JSON at {sft_eval_gate_json}")
            else:
                _run_grpo_eval(
                    python_exe=python_exe,
                    grpo_script=grpo_script,
                    task=task,
                    eval_model=sft_output_dir,
                    eval_csv=eval_csv,
                    output_json=sft_eval_json,
                    extra_args=sft_eval_args,
                    env=gpu_env,
                    dry_run=args.dry_run,
                )
            if not args.dry_run and gate_spec.get("after_sft_eval"):
                sft_eval_payload = _load_json(sft_eval_gate_json)
                ok, messages = _check_gate(dict(sft_eval_payload.get("eval") or {}), dict(gate_spec["after_sft_eval"]))
                stage_result["after_sft_eval_gate"] = {
                    "passed": ok,
                    "checks": messages,
                    "source_json": str(sft_eval_gate_json),
                }
                if not ok:
                    stage_result["status"] = "failed_after_sft_eval_gate"
                    run_summary["stages"].append(stage_result)
                    run_summary["final_model"] = current_model
                    _write_json(stage_dir / "stage_result.json", stage_result)
                    _write_json(output_root / "run_summary.json", run_summary)
                    raise SystemExit(f"Stage {stage_idx} ({stage_name}) failed after_sft_eval gate.")

        if enable_grpo:
            grpo_output_dir = stage_dir / "grpo"
            grpo_args = default_grpo_args + _parse_extra_args(stage.get("grpo_args"))
            grpo_args = _append_flag_once(grpo_args, "--enable-thinking", enable_thinking)
            _run_grpo_train(
                python_exe=python_exe,
                nproc_per_node=nproc_per_node,
                grpo_script=grpo_script,
                task=task,
                model_id=promoted_model,
                train_csv=mixed_train_csv,
                eval_csv=eval_csv,
                output_dir=grpo_output_dir,
                extra_args=grpo_args,
                env=gpu_env,
                dry_run=args.dry_run,
            )
            stage_result["grpo_output_dir"] = str(grpo_output_dir)
            stage_result["grpo_eval_json"] = str(stage_dir / "grpo_eval.json")
            promoted_path = _latest_checkpoint_or_dir(grpo_output_dir)
            stage_result["grpo_promoted_model"] = str(promoted_path)
            promoted_model = str(promoted_path)

            if not args.dry_run and gate_spec.get("after_grpo_train"):
                train_metrics_path = grpo_output_dir / "grpo_log" / "train_metrics.jsonl"
                train_last = _load_last_jsonl(train_metrics_path)
                ok, messages = _check_gate(train_last, dict(gate_spec["after_grpo_train"]))
                stage_result["after_grpo_train_gate"] = {"passed": ok, "checks": messages}
                if not ok:
                    stage_result["status"] = "failed_after_grpo_train_gate"
                    run_summary["stages"].append(stage_result)
                    run_summary["final_model"] = current_model
                    _write_json(stage_dir / "stage_result.json", stage_result)
                    _write_json(output_root / "run_summary.json", run_summary)
                    raise SystemExit(f"Stage {stage_idx} ({stage_name}) failed after_grpo_train gate.")

            grpo_eval_args = default_eval_args + _parse_extra_args(stage.get("grpo_eval_args"))
            grpo_eval_args = _append_flag_once(grpo_eval_args, "--enable-thinking", enable_thinking)
            grpo_eval_json = stage_dir / "grpo_eval.json"
            _run_grpo_eval(
                python_exe=python_exe,
                grpo_script=grpo_script,
                task=task,
                eval_model=Path(promoted_model),
                eval_csv=eval_csv,
                output_json=grpo_eval_json,
                extra_args=grpo_eval_args,
                env=gpu_env,
                dry_run=args.dry_run,
            )
            if not args.dry_run and gate_spec.get("after_grpo_eval"):
                grpo_eval_payload = _load_json(grpo_eval_json)
                ok, messages = _check_gate(dict(grpo_eval_payload.get("eval") or {}), dict(gate_spec["after_grpo_eval"]))
                stage_result["after_grpo_eval_gate"] = {"passed": ok, "checks": messages}
                if not ok:
                    stage_result["status"] = "failed_after_grpo_eval_gate"
                    run_summary["stages"].append(stage_result)
                    run_summary["final_model"] = current_model
                    _write_json(stage_dir / "stage_result.json", stage_result)
                    _write_json(output_root / "run_summary.json", run_summary)
                    raise SystemExit(f"Stage {stage_idx} ({stage_name}) failed after_grpo_eval gate.")

        stage_result["status"] = "ok"
        stage_result["promoted_model"] = promoted_model
        current_model = promoted_model
        previous_stage_train_csvs.append(train_csv)
        _write_json(stage_dir / "stage_result.json", stage_result)
        run_summary["stages"].append(stage_result)
        run_summary["final_model"] = current_model
        _write_json(output_root / "run_summary.json", run_summary)
        print(f"[stage-ok] {stage_idx}: {stage_name} -> {current_model}")

    print(f"[done] final_model={current_model}")


if __name__ == "__main__":
    main()
