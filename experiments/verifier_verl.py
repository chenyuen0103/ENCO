import re
from functools import lru_cache
from typing import Any, Callable, Optional

try:
    from verifier_cd import (
        build_cd_acyclic_reward as _build_cd_acyclic_reward,
        build_cd_cot_structure_reward as _build_cd_cot_structure_reward,
        build_cd_descendant_answer_tail_partial_format_reward as _build_cd_descendant_answer_tail_partial_format_reward,
        build_cd_descendant_cot_structure_reward as _build_cd_descendant_cot_structure_reward,
        build_cd_descendant_f1_reward as _build_cd_descendant_f1_reward,
        build_cd_descendant_partial_format_reward as _build_cd_descendant_partial_format_reward,
        build_cd_descendant_shift_ranking_reward as _build_cd_descendant_shift_ranking_reward,
        build_cd_descendant_variable_classification_reward as _build_cd_descendant_variable_classification_reward,
        build_cd_edge_f1_reward as _build_cd_edge_f1_reward,
        build_cd_format_reward as _build_cd_format_reward,
        build_cd_graph_reward as _build_cd_graph_reward,
        build_cd_low_shd_reward as _build_cd_low_shd_reward,
        build_cd_orientation_f1_reward as _build_cd_orientation_f1_reward,
        build_cd_partial_format_reward as _build_cd_partial_format_reward,
        build_cd_skeleton_f1_reward as _build_cd_skeleton_f1_reward,
        build_cd_stage_targets,
        build_cd_vstruct_f1_reward as _build_cd_vstruct_f1_reward,
        build_length_penalty_reward as _build_length_penalty_reward,
        completion_to_text,
        extract_descendant_payload,
        target_matrix_from_answer,
    )
except ModuleNotFoundError:
    from experiments.verifier_cd import (
        build_cd_acyclic_reward as _build_cd_acyclic_reward,
        build_cd_cot_structure_reward as _build_cd_cot_structure_reward,
        build_cd_descendant_answer_tail_partial_format_reward as _build_cd_descendant_answer_tail_partial_format_reward,
        build_cd_descendant_cot_structure_reward as _build_cd_descendant_cot_structure_reward,
        build_cd_descendant_f1_reward as _build_cd_descendant_f1_reward,
        build_cd_descendant_partial_format_reward as _build_cd_descendant_partial_format_reward,
        build_cd_descendant_shift_ranking_reward as _build_cd_descendant_shift_ranking_reward,
        build_cd_descendant_variable_classification_reward as _build_cd_descendant_variable_classification_reward,
        build_cd_edge_f1_reward as _build_cd_edge_f1_reward,
        build_cd_format_reward as _build_cd_format_reward,
        build_cd_graph_reward as _build_cd_graph_reward,
        build_cd_low_shd_reward as _build_cd_low_shd_reward,
        build_cd_orientation_f1_reward as _build_cd_orientation_f1_reward,
        build_cd_partial_format_reward as _build_cd_partial_format_reward,
        build_cd_skeleton_f1_reward as _build_cd_skeleton_f1_reward,
        build_cd_stage_targets,
        build_cd_vstruct_f1_reward as _build_cd_vstruct_f1_reward,
        build_length_penalty_reward as _build_length_penalty_reward,
        completion_to_text,
        extract_descendant_payload,
        target_matrix_from_answer,
    )

try:
    from math_verify import LatexExtractionConfig, parse, verify

    _HAS_MATH_VERIFY = True
except Exception:
    LatexExtractionConfig = None
    parse = None
    verify = None
    _HAS_MATH_VERIFY = False


FORMAT_RE = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_answer_text(text: str) -> str:
    match = ANSWER_RE.search(text or "")
    if match:
        return match.group(1)
    s = text or ""
    if "</answer>" in s:
        return s.rsplit("</answer>", 1)[0].strip()
    return s


def _require_math_verify() -> None:
    if not _HAS_MATH_VERIFY:
        raise RuntimeError(
            "math_verify is required for math rewards in verifier_verl.py but is not installed."
        )


def _is_correct_math(answer_text: str, solution: str) -> float:
    _require_math_verify()
    gold = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    pred = parse(
        answer_text,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold) == 0:
        return 1.0
    try:
        return float(verify(pred, gold))
    except Exception:
        return 0.0


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _extra(extra_info: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in extra_info:
            return extra_info[key]
    return default


def _infer_task(data_source: Any, ground_truth: Any, extra_info: dict[str, Any]) -> str:
    task = _extra(extra_info, "task", "task_name", default=None)
    if task:
        return str(task)
    if extract_descendant_payload(str(ground_truth or "")) is not None:
        return "cd_descendants"
    if target_matrix_from_answer(ground_truth) is not None:
        return "causal_discovery"
    source = str(data_source or "").lower()
    if any(token in source for token in ("math", "gsm8k", "aime", "numina")):
        return "math"
    return "causal_discovery"


def _single_example_kwargs(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
) -> dict[str, list[Any]]:
    prompt = _extra(extra_info, "prompt", default=None)
    prompt_raw = _extra(extra_info, "prompt_raw", default=prompt)
    answer = _extra(extra_info, "answer", "ground_truth", default=ground_truth)
    answer_path = _extra(extra_info, "answer_path", default=None)

    kwargs: dict[str, list[Any]] = {
        "prompt": [prompt],
        "prompt_raw": [prompt_raw],
        "answer": [answer],
        "answer_path": [answer_path],
    }

    for key in (
        "solution",
        "target_stage1_skeleton_edges",
        "target_stage2_vstructures",
        "target_stage3_directed_edges",
    ):
        if key in extra_info:
            kwargs[key] = [extra_info[key]]

    if (
        "target_stage1_skeleton_edges" not in kwargs
        and prompt_raw is not None
        and answer is not None
    ):
        stage_targets = build_cd_stage_targets(str(prompt_raw), answer) or {}
        for key, value in stage_targets.items():
            kwargs[key] = [value]

    return kwargs


def _run_batch_reward(
    reward_fn: Callable[..., list[float]],
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    kwargs = _single_example_kwargs(solution_str, ground_truth, info)
    rewards = reward_fn([solution_str], **kwargs)
    return float(rewards[0]) if rewards else 0.0


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_name_or_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name_or_path)


def _resolve_tokenizer(extra_info: dict[str, Any]) -> Any:
    tokenizer = _extra(extra_info, "tokenizer", default=None)
    if tokenizer is not None:
        return tokenizer
    tokenizer_name_or_path = _extra(
        extra_info,
        "tokenizer_name_or_path",
        "model_id",
        "model_name_or_path",
        default=None,
    )
    if not tokenizer_name_or_path:
        raise ValueError(
            "length_penalty_reward requires extra_info['tokenizer'] or "
            "extra_info['tokenizer_name_or_path']."
        )
    return _load_tokenizer(str(tokenizer_name_or_path))


def format_reward_math(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    text = completion_to_text(solution_str)
    return 1.0 if FORMAT_RE.match(text) else 0.0


def accuracy_reward_math(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    answer_text = extract_answer_text(completion_to_text(solution_str))
    return _is_correct_math(answer_text, str(ground_truth or ""))


def cd_format_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_format_reward(
        scale=float(_extra(info, "cd_format_reward_scale", "scale", default=0.2))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_partial_format_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_partial_format_reward(
        scale=float(_extra(info, "cd_partial_format_reward_scale", "scale", default=0.25))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_graph_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_graph_reward(
        require_dag=_as_bool(_extra(info, "cd_reward_require_dag", default=True), True),
        dag_penalty=float(_extra(info, "cd_reward_dag_penalty", default=0.1)),
        shd_weight=float(_extra(info, "cd_reward_shd_weight", default=0.0)),
        scale=float(_extra(info, "cd_graph_reward_scale", "scale", default=1.0)),
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_descendant_partial_format_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_descendant_partial_format_reward(
        scale=float(_extra(info, "cd_partial_format_reward_scale", "scale", default=0.25))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_descendant_answer_tail_partial_format_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_descendant_answer_tail_partial_format_reward(
        scale=float(_extra(info, "cd_partial_format_reward_scale", "scale", default=0.25))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_descendant_f1_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_descendant_f1_reward(
        scale=float(_extra(info, "cd_graph_reward_scale", "scale", default=1.0))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_descendant_cot_structure_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_descendant_cot_structure_reward(
        scale=float(_extra(info, "cd_descendant_cot_structure_reward_scale", "scale", default=0.1))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_descendant_shift_ranking_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_descendant_shift_ranking_reward(
        scale=float(_extra(info, "cd_descendant_shift_ranking_reward_scale", "scale", default=0.2))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_descendant_variable_classification_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_descendant_variable_classification_reward(
        scale=float(
            _extra(info, "cd_descendant_variable_classification_reward_scale", "scale", default=0.2)
        )
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_edge_f1_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_edge_f1_reward(
        scale=float(_extra(info, "cd_edge_f1_reward_scale", "scale", default=1.0))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_low_shd_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_low_shd_reward(
        scale=float(_extra(info, "cd_low_shd_reward_scale", "scale", default=1.0))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_acyclic_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_acyclic_reward(
        scale=float(_extra(info, "cd_acyclic_reward_scale", "scale", default=0.1))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_cot_structure_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_cot_structure_reward(
        scale=float(_extra(info, "cd_cot_structure_reward_scale", "scale", default=0.1))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_skeleton_f1_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_skeleton_f1_reward(
        scale=float(_extra(info, "cd_skeleton_f1_reward_scale", "scale", default=0.2))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_vstruct_f1_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_vstruct_f1_reward(
        scale=float(_extra(info, "cd_vstruct_f1_reward_scale", "scale", default=0.15))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def cd_orientation_f1_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_cd_orientation_f1_reward(
        scale=float(_extra(info, "cd_orientation_f1_reward_scale", "scale", default=0.15))
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


def length_penalty_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    info = extra_info or {}
    reward_fn = _build_length_penalty_reward(
        tokenizer=_resolve_tokenizer(info),
        coef=float(_extra(info, "length_penalty_coef", "coef", default=0.0)),
        target_tokens=int(_extra(info, "length_penalty_target_tokens", "target_tokens", default=0)),
        max_abs=float(_extra(info, "length_penalty_max_abs", "max_abs", default=1.0)),
    )
    return _run_batch_reward(reward_fn, solution_str, ground_truth, info)


REWARD_REGISTRY: dict[str, Callable[[Any, str, Any, Optional[dict[str, Any]]], float]] = {
    "format_reward_math": format_reward_math,
    "accuracy_reward_math": accuracy_reward_math,
    "cd_format_reward": cd_format_reward,
    "cd_partial_format_reward": cd_partial_format_reward,
    "cd_graph_reward": cd_graph_reward,
    "cd_descendant_partial_format_reward": cd_descendant_partial_format_reward,
    "cd_descendant_answer_tail_partial_format_reward": cd_descendant_answer_tail_partial_format_reward,
    "cd_descendant_f1_reward": cd_descendant_f1_reward,
    "cd_descendant_cot_structure_reward": cd_descendant_cot_structure_reward,
    "cd_descendant_shift_ranking_reward": cd_descendant_shift_ranking_reward,
    "cd_descendant_variable_classification_reward": cd_descendant_variable_classification_reward,
    "cd_edge_f1_reward": cd_edge_f1_reward,
    "cd_low_shd_reward": cd_low_shd_reward,
    "cd_acyclic_reward": cd_acyclic_reward,
    "cd_cot_structure_reward": cd_cot_structure_reward,
    "cd_skeleton_f1_reward": cd_skeleton_f1_reward,
    "cd_vstruct_f1_reward": cd_vstruct_f1_reward,
    "cd_orientation_f1_reward": cd_orientation_f1_reward,
    "length_penalty_reward": length_penalty_reward,
}


def compute_reward_components(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    info = dict(extra_info or {})
    reward_names = _as_list(_extra(info, "reward_names", default=None))
    if not reward_names:
        reward_name = _extra(info, "reward_name", "reward_fn", "reward_type", default=None)
        if reward_name:
            reward_names = [str(reward_name)]
        else:
            task = _infer_task(data_source, ground_truth, info)
            if task == "math":
                reward_names = ["accuracy_reward_math"]
            elif task == "cd_descendants":
                reward_names = ["cd_descendant_f1_reward"]
            else:
                reward_names = ["cd_graph_reward"]

    scores: dict[str, float] = {}
    for reward_name in reward_names:
        key = str(reward_name)
        if key not in REWARD_REGISTRY:
            raise KeyError(
                f"Unknown reward '{key}'. Available rewards: {', '.join(sorted(REWARD_REGISTRY))}"
            )
        scores[key] = float(REWARD_REGISTRY[key](data_source, solution_str, ground_truth, info))
    return scores


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    scores = compute_reward_components(data_source, solution_str, ground_truth, extra_info)
    return float(sum(scores.values()))
