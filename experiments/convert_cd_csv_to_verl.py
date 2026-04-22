import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from datasets import Dataset

try:
    from cd_generation.format import (
        default_format_hint_text,
        looks_like_chat_prompt,
        resolve_format_hint_text,
        system_prompt_for_task,
        append_format_hint_to_user_prompt,
    )
except ModuleNotFoundError:
    from experiments.cd_generation.format import (
        default_format_hint_text,
        looks_like_chat_prompt,
        resolve_format_hint_text,
        system_prompt_for_task,
        append_format_hint_to_user_prompt,
    )

try:
    from verifier_cd import build_cd_stage_targets
except ModuleNotFoundError:
    from experiments.verifier_cd import build_cd_stage_targets


def _set_csv_field_limit() -> None:
    limit = 1024 * 1024
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2
            if limit <= 0:
                raise


def _read_rows(paths: Iterable[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    _set_csv_field_limit()
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            out.extend(dict(row) for row in csv.DictReader(f))
    return out


def _unwrap_chat_user_prompt(text: str) -> str:
    lines = str(text or "").strip().splitlines()
    if not lines:
        return ""
    if lines[0].strip() != "system":
        return str(text or "").strip()
    try:
        user_idx = lines.index("user")
        assistant_idx = lines.index("assistant")
    except ValueError:
        return str(text or "").strip()
    if assistant_idx <= user_idx:
        return str(text or "").strip()
    return "\n".join(lines[user_idx + 1:assistant_idx]).strip()


def _sanitize_descendant_prompt_source(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    if looks_like_chat_prompt(s):
        s = _unwrap_chat_user_prompt(s)
    marker = "\n\nFormatting requirement:"
    if marker in s:
        s = s.split(marker, 1)[0].rstrip()
    marker = "\nFormatting requirement:"
    if marker in s:
        s = s.split(marker, 1)[0].rstrip()
    if "\nassistant" in s:
        s = s.split("\nassistant", 1)[0].rstrip()
    return s


def _build_prompt_messages(
    raw_prompt: str,
    *,
    task: str,
    wrap_system_prompt: bool,
    append_format_hint: bool,
    format_hint_text: str,
) -> list[dict[str, str]]:
    user_prompt = str(raw_prompt or "").strip()
    if task == "cd_descendants":
        user_prompt = _sanitize_descendant_prompt_source(user_prompt)
    elif looks_like_chat_prompt(user_prompt):
        user_prompt = _unwrap_chat_user_prompt(user_prompt)

    if append_format_hint:
        user_prompt = append_format_hint_to_user_prompt(
            user_prompt,
            resolve_format_hint_text(task, format_hint_text),
        )

    messages: list[dict[str, str]] = []
    if wrap_system_prompt:
        messages.append({"role": "system", "content": system_prompt_for_task(task)})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _reward_names_from_args(args: argparse.Namespace) -> list[str]:
    reward_names: list[str] = []
    if args.task == "math":
        reward_names.append("accuracy_reward_math")
        return reward_names

    if args.task == "causal_discovery":
        if args.cd_format_reward_scale > 0:
            reward_names.append("cd_format_reward")
        if args.cd_partial_format_reward_scale > 0:
            reward_names.append("cd_partial_format_reward")
        if args.cd_edge_f1_reward_scale > 0:
            reward_names.append("cd_edge_f1_reward")
        if args.cd_low_shd_reward_scale > 0:
            reward_names.append("cd_low_shd_reward")
        if args.cd_acyclic_reward_scale > 0:
            reward_names.append("cd_acyclic_reward")
        if args.cd_cot_structure_reward_scale > 0:
            reward_names.append("cd_cot_structure_reward")
        if args.cd_skeleton_f1_reward_scale > 0:
            reward_names.append("cd_skeleton_f1_reward")
        if args.cd_vstruct_f1_reward_scale > 0:
            reward_names.append("cd_vstruct_f1_reward")
        if args.cd_orientation_f1_reward_scale > 0:
            reward_names.append("cd_orientation_f1_reward")
        if args.cd_graph_reward_scale > 0:
            reward_names.append("cd_graph_reward")
    elif args.task == "cd_descendants":
        if args.cd_format_reward_scale > 0:
            reward_names.append("cd_format_reward")
        if args.cd_partial_format_reward_scale > 0:
            reward_names.append("cd_descendant_partial_format_reward")
        if args.cd_descendant_cot_structure_reward_scale > 0:
            reward_names.append("cd_descendant_cot_structure_reward")
        if args.cd_descendant_shift_ranking_reward_scale > 0:
            reward_names.append("cd_descendant_shift_ranking_reward")
        if args.cd_descendant_variable_classification_reward_scale > 0:
            reward_names.append("cd_descendant_variable_classification_reward")
        if args.cd_graph_reward_scale > 0:
            reward_names.append("cd_descendant_f1_reward")

    if args.length_penalty_coef > 0:
        reward_names.append("length_penalty_reward")
    return reward_names


def _row_to_verl(
    row: dict[str, str],
    *,
    args: argparse.Namespace,
    split_name: str,
    reward_names: list[str],
) -> dict[str, Any]:
    prompt_raw = (row.get("prompt_text") or row.get("prompt") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not prompt_raw or not answer:
        raise ValueError("Each row must contain non-empty prompt_text/prompt and answer fields.")

    prompt_messages = _build_prompt_messages(
        prompt_raw,
        task=args.task,
        wrap_system_prompt=bool(args.cd_wrap_system_prompt),
        append_format_hint=bool(args.cd_append_format_hint),
        format_hint_text=str(args.cd_format_hint_text),
    )

    extra_info: dict[str, Any] = {
        "split": split_name,
        "task": args.task,
        "prompt": prompt_messages[-1]["content"],
        "prompt_raw": prompt_raw,
        "answer": answer,
        "ground_truth": answer,
        "reward_names": list(reward_names),
        "cd_format_reward_scale": float(args.cd_format_reward_scale),
        "cd_partial_format_reward_scale": float(args.cd_partial_format_reward_scale),
        "cd_graph_reward_scale": float(args.cd_graph_reward_scale),
        "cd_edge_f1_reward_scale": float(args.cd_edge_f1_reward_scale),
        "cd_low_shd_reward_scale": float(args.cd_low_shd_reward_scale),
        "cd_acyclic_reward_scale": float(args.cd_acyclic_reward_scale),
        "cd_cot_structure_reward_scale": float(args.cd_cot_structure_reward_scale),
        "cd_skeleton_f1_reward_scale": float(args.cd_skeleton_f1_reward_scale),
        "cd_vstruct_f1_reward_scale": float(args.cd_vstruct_f1_reward_scale),
        "cd_orientation_f1_reward_scale": float(args.cd_orientation_f1_reward_scale),
        "cd_descendant_cot_structure_reward_scale": float(args.cd_descendant_cot_structure_reward_scale),
        "cd_descendant_shift_ranking_reward_scale": float(args.cd_descendant_shift_ranking_reward_scale),
        "cd_descendant_variable_classification_reward_scale": float(
            args.cd_descendant_variable_classification_reward_scale
        ),
        "cd_reward_require_dag": bool(args.cd_reward_require_dag),
        "cd_reward_dag_penalty": float(args.cd_reward_dag_penalty),
        "cd_reward_shd_weight": float(args.cd_reward_shd_weight),
        "length_penalty_coef": float(args.length_penalty_coef),
        "length_penalty_max_abs": float(args.length_penalty_max_abs),
    }
    if args.tokenizer_name_or_path:
        extra_info["tokenizer_name_or_path"] = args.tokenizer_name_or_path

    if args.task == "causal_discovery":
        extra_info.update(build_cd_stage_targets(prompt_raw, answer) or {})

    return {
        "data_source": args.data_source,
        "prompt": prompt_messages,
        "ability": args.task,
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": extra_info,
    }


def _select_by_split(
    rows: list[dict[str, str]],
    split_values: set[str],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        split = str(row.get("split") or "").strip().lower()
        if split in split_values:
            selected.append(row)
    return selected


def _write_dataset(rows: list[dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))
    print(f"[write] {len(rows)} rows -> {path}")


def _stable_bucket(row: dict[str, str], seed: int) -> float:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return float(value) / float(2**64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert ENCO causal-discovery CSVs into VERL parquet files.")
    p.add_argument("--task", type=str, default="causal_discovery", choices=["causal_discovery", "cd_descendants", "math"])
    p.add_argument("--data-source", type=str, default="causal_discovery")

    p.add_argument("--train-csv", action="append", default=[], help="Input CSV(s) for train parquet.")
    p.add_argument("--eval-csv", action="append", default=[], help="Input CSV(s) for eval parquet.")
    p.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help="Input CSV(s) containing a split column. Rows with split=train go to train; split=eval/test/val go to eval.",
    )
    p.add_argument("--train-out", type=str, required=True)
    p.add_argument("--eval-out", type=str, required=True)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="If no eval rows are provided explicitly, hold out this fraction from the combined input rows.",
    )
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--cd-wrap-system-prompt", dest="cd_wrap_system_prompt", action="store_true")
    p.add_argument("--no-cd-wrap-system-prompt", dest="cd_wrap_system_prompt", action="store_false")
    p.set_defaults(cd_wrap_system_prompt=True)
    p.add_argument("--cd-append-format-hint", dest="cd_append_format_hint", action="store_true")
    p.add_argument("--no-cd-append-format-hint", dest="cd_append_format_hint", action="store_false")
    p.set_defaults(cd_append_format_hint=True)
    p.add_argument("--cd-format-hint-text", type=str, default=default_format_hint_text("causal_discovery"))

    p.add_argument("--cd-reward-shd-weight", type=float, default=0.0)
    p.add_argument("--cd-reward-dag-penalty", type=float, default=0.1)
    p.add_argument("--cd-reward-require-dag", dest="cd_reward_require_dag", action="store_true")
    p.add_argument("--no-cd-reward-require-dag", dest="cd_reward_require_dag", action="store_false")
    p.set_defaults(cd_reward_require_dag=True)
    p.add_argument("--cd-graph-reward-scale", type=float, default=1.0)
    p.add_argument("--cd-format-reward-scale", type=float, default=0.2)
    p.add_argument("--cd-partial-format-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-edge-f1-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-low-shd-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-acyclic-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-cot-structure-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-skeleton-f1-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-vstruct-f1-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-orientation-f1-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-descendant-cot-structure-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-descendant-shift-ranking-reward-scale", type=float, default=0.0)
    p.add_argument("--cd-descendant-variable-classification-reward-scale", type=float, default=0.0)
    p.add_argument("--length-penalty-coef", type=float, default=0.0)
    p.add_argument("--length-penalty-max-abs", type=float, default=1.0)
    p.add_argument("--tokenizer-name-or-path", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reward_names = _reward_names_from_args(args)

    train_source_rows = _read_rows(args.train_csv)
    eval_source_rows = _read_rows(args.eval_csv)

    if args.input_csv:
        shared_rows = _read_rows(args.input_csv)
        train_source_rows.extend(_select_by_split(shared_rows, {"train"}))
        eval_source_rows.extend(_select_by_split(shared_rows, {"eval", "test", "val", "validation"}))

    if train_source_rows and not eval_source_rows:
        if not (0.0 < float(args.eval_fraction) < 1.0):
            raise ValueError("--eval-fraction must be in (0, 1) when creating a holdout split.")
        combined_rows = train_source_rows
        train_source_rows = []
        eval_source_rows = []
        for row in combined_rows:
            if _stable_bucket(row, seed=int(args.split_seed)) < float(args.eval_fraction):
                eval_source_rows.append(row)
            else:
                train_source_rows.append(row)
        print(
            f"[split] auto holdout with eval_fraction={args.eval_fraction} seed={args.split_seed}: "
            f"train={len(train_source_rows)} eval={len(eval_source_rows)}"
        )

    if not train_source_rows:
        raise ValueError("No training rows found. Use --train-csv or --input-csv with split=train.")
    if not eval_source_rows:
        raise ValueError("No eval rows found. Use --eval-csv or --input-csv with split=eval/test/val.")

    train_rows = [
        _row_to_verl(row, args=args, split_name="train", reward_names=reward_names)
        for row in train_source_rows[: args.max_train_samples or None]
    ]
    eval_rows = [
        _row_to_verl(row, args=args, split_name="eval", reward_names=reward_names)
        for row in eval_source_rows[: args.max_eval_samples or None]
    ]

    _write_dataset(train_rows, args.train_out)
    _write_dataset(eval_rows, args.eval_out)


if __name__ == "__main__":
    main()
