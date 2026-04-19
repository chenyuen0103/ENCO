#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent


def _run_eval(config_name: str, cmd: list[str]) -> tuple[dict[str, Any], Path, Path]:
    summary_path = Path(cmd[cmd.index("--summary-json") + 1]).resolve()
    output_path = Path(cmd[cmd.index("--output-jsonl") + 1]).resolve()
    print(f"[run] {config_name}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, output_path, summary_path


def _progress_value(summary: dict[str, Any], key: str) -> float:
    entry = (((summary.get("aggregate") or {}).get("format_progress") or {}).get(key) or {})
    try:
        return float(entry.get("rate", 0.0))
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval_grpo_vllm_rollouts.py in greedy and sampled modes and compare format compliance."
    )
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument(
        "--csv",
        action="append",
        default=None,
        help="Prompt CSV to sample from. Repeatable. Defaults to the evaluator defaults.",
    )
    parser.add_argument("--samples-per-csv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["auto", "causal_discovery", "cd_descendants"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm-gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--greedy-rollouts", type=int, default=1)
    parser.add_argument("--sampled-rollouts", type=int, default=8)
    parser.add_argument("--sampled-temperature", type=float, default=1.0)
    parser.add_argument("--sampled-top-p", type=float, default=1.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/out/grpo_vllm_rollouts_compare"),
        help="Directory for the two evaluator outputs plus the comparison JSON.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_script = _HERE / "eval_grpo_vllm_rollouts.py"

    base_cmd = [
        sys.executable,
        str(eval_script),
        "--model",
        str(args.model),
        "--samples-per-csv",
        str(int(args.samples_per_csv)),
        "--seed",
        str(int(args.seed)),
        "--task",
        str(args.task),
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--tensor-parallel-size",
        str(int(args.tensor_parallel_size)),
        "--vllm-dtype",
        str(args.vllm_dtype),
        "--vllm-gpu-mem-util",
        str(float(args.vllm_gpu_mem_util)),
    ]
    if args.vllm_max_model_len is not None:
        base_cmd.extend(["--vllm-max-model-len", str(int(args.vllm_max_model_len))])
    if args.vllm_enforce_eager:
        base_cmd.append("--vllm-enforce-eager")
    if args.trust_remote_code:
        base_cmd.append("--trust-remote-code")
    else:
        base_cmd.append("--no-trust-remote-code")
    for csv_path in args.csv or []:
        base_cmd.extend(["--csv", str(csv_path)])

    greedy_cmd = list(base_cmd)
    greedy_cmd.extend(
        [
            "--rollouts",
            str(int(args.greedy_rollouts)),
            "--temperature",
            "0.0",
            "--top-p",
            "1.0",
            "--output-jsonl",
            str((args.out_dir / "greedy.jsonl").resolve()),
            "--summary-json",
            str((args.out_dir / "greedy.summary.json").resolve()),
        ]
    )

    sampled_cmd = list(base_cmd)
    sampled_cmd.extend(
        [
            "--rollouts",
            str(int(args.sampled_rollouts)),
            "--temperature",
            str(float(args.sampled_temperature)),
            "--top-p",
            str(float(args.sampled_top_p)),
            "--output-jsonl",
            str((args.out_dir / "sampled.jsonl").resolve()),
            "--summary-json",
            str((args.out_dir / "sampled.summary.json").resolve()),
        ]
    )

    greedy_summary, greedy_output, greedy_summary_path = _run_eval("greedy", greedy_cmd)
    sampled_summary, sampled_output, sampled_summary_path = _run_eval("sampled", sampled_cmd)

    comparison = {
        "greedy": {
            "output_jsonl": str(greedy_output),
            "summary_json": str(greedy_summary_path),
            "reward_total_mean": float((greedy_summary.get("aggregate") or {}).get("reward_total_mean", 0.0)),
            "strict_format_ok_rate": _progress_value(greedy_summary, "strict_format_ok"),
            "parse_ok_rate": _progress_value(greedy_summary, "parse_ok"),
            "has_close_think_rate": _progress_value(greedy_summary, "has_close_think"),
            "has_open_answer_rate": _progress_value(greedy_summary, "has_open_answer"),
            "has_close_answer_rate": _progress_value(greedy_summary, "has_close_answer"),
            "has_adjacency_matrix_rate": _progress_value(greedy_summary, "has_adjacency_matrix"),
        },
        "sampled": {
            "output_jsonl": str(sampled_output),
            "summary_json": str(sampled_summary_path),
            "reward_total_mean": float((sampled_summary.get("aggregate") or {}).get("reward_total_mean", 0.0)),
            "strict_format_ok_rate": _progress_value(sampled_summary, "strict_format_ok"),
            "parse_ok_rate": _progress_value(sampled_summary, "parse_ok"),
            "has_close_think_rate": _progress_value(sampled_summary, "has_close_think"),
            "has_open_answer_rate": _progress_value(sampled_summary, "has_open_answer"),
            "has_close_answer_rate": _progress_value(sampled_summary, "has_close_answer"),
            "has_adjacency_matrix_rate": _progress_value(sampled_summary, "has_adjacency_matrix"),
        },
    }
    comparison_path = (args.out_dir / "comparison.json").resolve()
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n[compare] format compliance summary")
    print(f"comparison_json={comparison_path}")
    for name in ("greedy", "sampled"):
        section = comparison[name]
        print(
            f"{name}: reward_total_mean={section['reward_total_mean']:.4f} "
            f"strict_format_ok_rate={section['strict_format_ok_rate']:.3f} "
            f"parse_ok_rate={section['parse_ok_rate']:.3f} "
            f"close_think_rate={section['has_close_think_rate']:.3f} "
            f"open_answer_rate={section['has_open_answer_rate']:.3f} "
            f"close_answer_rate={section['has_close_answer_rate']:.3f} "
            f"adjacency_matrix_rate={section['has_adjacency_matrix_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
