#!/usr/bin/env python3
"""
generate_reasoning.py

Generate SFT examples that teach the model to start reasoning immediately
after <think> in a chosen target style: answer-only formatting, concise
evidence-grounded reasoning, or full staged reasoning.

Three modes
-----------
Mode A  --graphs graph1 [graph2 ...]
    Generate data in-memory from BIF files.  Each (graph, obs, int) config
    is fed through iter_prompts_in_memory with --col-perms random column
    orderings so variable order is varied automatically.

Mode B  (default — no --graphs and no --perm-csv)
    Discover *_obs100_int10_anon_train.csv and *_randcol_seed*.csv files
    under --data-dir (or read explicit --csv files) and build one SFT record
    per CSV row.

Mode C  --perm-csv
    Like Mode B (reads the same CSV sources) but instead of one record per
    row, enumerates up to --max-perms variable-order permutations for each of
    --rows-per-source rows.  The prompt text is rewritten in-place: VARIABLES,
    OBSERVATIONAL DATA, INTERVENTIONAL DATA, num_states, and tv_change_vs_obs
    are all permuted consistently.  Supersedes the former
    collect_permuted_sft_data.py.

Usage examples
--------------
    # Mode A — in-memory BIF, 5 column permutations each
    python experiments/generate_reasoning.py \\
        --graphs cancer earthquake asia sachs \\
        --col-perms 5 --num-prompts-per-config 500

    # Mode A — concise evidence-grounded reasoning targets
    python experiments/generate_reasoning.py \\
        --graphs sachs cancer \\
        --obs-values 100 --int-values 50 \\
        --reasoning-target concise_evidence \\
        --output experiments/data/format_sft_concise.jsonl

    # Mode A — teacher-written reasoning from the gold adjacency matrix
    python experiments/generate_reasoning.py \\
        --graphs sachs \\
        --reasoning-target teacher_evidence \\
        --teacher-model Qwen/Qwen2.5-72B-Instruct \\
        --teacher-base-url http://localhost:8000/v1 \\
        --output experiments/data/format_sft_teacher.jsonl

    # Mode B — CSV discovery, 100 rows per source
    python experiments/generate_reasoning.py \\
        --data-dir experiments/data --n-per-source 100

    # Mode C — exhaustive permutation from existing CSV rows
    python experiments/generate_reasoning.py \\
        --perm-csv --rows-per-source 5 --max-perms 500

Output JSONL schema (one JSON object per line):
    {
      "prompt":  "<system>...<think>\\n",
      "answer":  "Stage 1 ...\\n\\n...Stage 3 ...</think><answer>{...}</answer>",
      "source":  "cancer_obs100_int10_anon_train",
      "graph":   "cancer"
    }

Compatible with train_sft.py --sft-jsonl.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    from cd_generation.format import validate_sft_example  # noqa: E402
    from cd_sft.staged_targets import (  # noqa: E402
        _load_adj,
        _extract_variables,
        _build_edge_effect_table,
        _directed_edges,
        _skeleton_edges,
        build_evidence_grounded_sections,
        build_staged_sections,
    )
    from cd_generation.prompt_utils import render_prompt_text  # noqa: E402
except ImportError:
    from experiments.cd_generation.format import validate_sft_example  # noqa: E402
    from experiments.cd_sft.staged_targets import (  # noqa: E402
        _load_adj,
        _extract_variables,
        _build_edge_effect_table,
        _directed_edges,
        _skeleton_edges,
        build_evidence_grounded_sections,
        build_staged_sections,
    )
    from experiments.cd_generation.prompt_utils import render_prompt_text  # noqa: E402
from generate_prompts import iter_prompts_in_memory  # noqa: E402

SUPPORTED_REASONING_TARGETS = {
    "answer_only",
    "concise_evidence",
    "stages",
    "stages_evidence",
    "teacher_evidence",
}


@dataclass
class TeacherConfig:
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: float = 120.0
    fallback_target: str = "concise_evidence"
    reasoning_effort: Optional[str] = None
    max_revisions: int = 1
    retry_until_valid: bool = False
    retry_sleep_s: float = 0.0


# ---------------------------------------------------------------------------
# CSV helpers (shared across all modes)
# ---------------------------------------------------------------------------

def _set_csv_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


def _iter_csv_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _sample_rows(path: Path, n: int, rng: random.Random) -> List[dict]:
    """Reservoir-sample n rows from a potentially large CSV."""
    reservoir: List[dict] = []
    for i, row in enumerate(_iter_csv_rows(path)):
        row["_source_row_idx"] = i
        if i < n:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = row
    rng.shuffle(reservoir)
    return reservoir


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _parse_reasoning_targets(raw_values: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw in raw_values or []:
        for tok in str(raw).split(","):
            val = tok.strip()
            if not val:
                continue
            if val not in SUPPORTED_REASONING_TARGETS:
                raise SystemExit(
                    f"Unsupported reasoning target {val!r}. "
                    f"Allowed: {sorted(SUPPORTED_REASONING_TARGETS)}"
                )
            out.append(val)
    return out or ["stages_evidence"]


def _build_think_sections(
    *,
    prompt_text: str,
    adj: List[List[int]],
    variables: List[str],
    think_style: str,
) -> Tuple[str, str, str]:
    if think_style == "strict":
        return build_staged_sections(adj, variables)
    if think_style == "evidence":
        return build_evidence_grounded_sections(prompt_text, adj, variables)
    raise ValueError(f"unsupported think_style={think_style!r}")


def _build_reasoning_text(
    *,
    prompt_text: str,
    adj: List[List[int]],
    variables: List[str],
    reasoning_target: str,
    teacher_config: Optional[TeacherConfig] = None,
) -> Tuple[str, str, str, str]:
    if reasoning_target == "answer_only":
        return "", "", "", ""
    if reasoning_target == "concise_evidence":
        return "", "", "", _build_concise_evidence_think(
            prompt_text=prompt_text,
            adj=adj,
            variables=variables,
        )
    if reasoning_target == "teacher_evidence":
        if teacher_config is None:
            raise ValueError("--reasoning-target teacher_evidence requires --teacher-model")
        return "", "", "", _build_teacher_evidence_think(
            prompt_text=prompt_text,
            adj=adj,
            variables=variables,
            teacher_config=teacher_config,
        )
    if reasoning_target == "stages":
        stage1_text, stage2_text, stage3_text = _build_think_sections(
            prompt_text=prompt_text,
            adj=adj,
            variables=variables,
            think_style="strict",
        )
    elif reasoning_target == "stages_evidence":
        stage1_text, stage2_text, stage3_text = _build_think_sections(
            prompt_text=prompt_text,
            adj=adj,
            variables=variables,
            think_style="evidence",
        )
    else:
        raise ValueError(f"unsupported reasoning_target={reasoning_target!r}")
    return stage1_text, stage2_text, stage3_text, f"{stage1_text}\n\n{stage2_text}\n\n{stage3_text}"


def _reasoning_style_label(reasoning_target: str) -> str:
    if reasoning_target == "teacher_evidence":
        return "teacher_evidence"
    if reasoning_target == "stages":
        return "strict"
    if reasoning_target == "stages_evidence":
        return "evidence"
    return reasoning_target


def _reasoning_guidance_for_target(reasoning_target: str) -> str:
    if reasoning_target in {"answer_only"}:
        return "none"
    if reasoning_target in {"concise_evidence", "teacher_evidence"}:
        return "concise"
    return "staged"


def _build_concise_evidence_think(
    *,
    prompt_text: str,
    adj: List[List[int]],
    variables: List[str],
) -> str:
    """
    Build a short, evidence-grounded think block.

    The goal is to teach the model to mention only the strongest causal evidence,
    keep the graph sparse, and transition quickly into the answer payload.
    """
    effect_table = _build_edge_effect_table(prompt_text, variables)
    directed_edges = _directed_edges(adj)
    supported_edges = [
        effect_table[(i, j)]
        for i, j in directed_edges
        if (i, j) in effect_table
    ]
    supported_edges.sort(key=lambda item: float(item["tv"]), reverse=True)

    lines: List[str] = []
    if supported_edges:
        top_edges = supported_edges[:3]
        edge_bits = [
            f"{item['cause_name']} -> {item['effect_name']} (TV={float(item['tv']):.2f})"
            for item in top_edges
        ]
        lines.append("Strongest intervention shifts support " + "; ".join(edge_bits) + ".")
    else:
        skeleton = _skeleton_edges(adj)
        if skeleton:
            pairs = [f"{variables[i]} -- {variables[j]}" for i, j in skeleton[:3]]
            lines.append("The strongest dependencies suggest edges among " + ", ".join(pairs) + ".")
        else:
            lines.append("The evidence supports a very sparse graph with no strong direct effects.")

    unsupported_true_edges = max(0, len(directed_edges) - len(supported_edges))
    if unsupported_true_edges > 0:
        lines.append("Keep the remaining structure sparse and only orient edges when the evidence clearly supports it.")
    else:
        lines.append("No extra direct edges are needed beyond the clearly supported effects.")

    if effect_table:
        lines.append("Choose the sparsest DAG consistent with these intervention patterns.")
    else:
        lines.append("Choose the sparsest DAG consistent with the observational and interventional evidence.")

    return "\n".join(lines)


def _matrix_edge_summary(adj: List[List[int]], variables: List[str]) -> str:
    edges = [
        f"{variables[i]} -> {variables[j]}"
        for i, j in _directed_edges(adj)
    ]
    return ", ".join(edges) if edges else "None"


def _answer_payload_for_adj(adj: List[List[int]]) -> str:
    return f"<answer>{json.dumps({'adjacency_matrix': adj}, ensure_ascii=False)}</answer>"


def _completion_for_adj(think_text: str, adj: List[List[int]]) -> str:
    return f"{think_text}</think>{_answer_payload_for_adj(adj)}"


def _parse_completion_adj(completion: str) -> Optional[List[List[int]]]:
    if "<answer>" not in completion or "</answer>" not in completion:
        return None
    raw = completion.split("<answer>", 1)[1].split("</answer>", 1)[0]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    mat = obj.get("adjacency_matrix") if isinstance(obj, dict) else None
    return mat if isinstance(mat, list) else None


def _hard_format_issues(
    *,
    think_text: str,
    completion: str,
    adj: List[List[int]],
) -> List[str]:
    expected = _completion_for_adj(think_text, adj)
    issues: List[str] = []
    if completion != expected:
        issues.append("completion is not exactly gold_think + </think><answer>{GT matrix}</answer>")
    parsed_adj = _parse_completion_adj(completion)
    if parsed_adj is None:
        issues.append("answer JSON did not parse as an adjacency_matrix payload")
    elif parsed_adj != adj:
        issues.append("answer adjacency_matrix does not equal ground truth")
    return issues


def _gt_directed_edge_set(adj: List[List[int]], variables: List[str]) -> set[Tuple[str, str]]:
    return {
        (variables[i], variables[j])
        for i, j in _directed_edges(adj)
    }


def _extract_mentioned_directed_edges(text: str, variables: List[str]) -> set[Tuple[str, str]]:
    if not text.strip() or not variables:
        return set()
    # Match only declared variable names on both sides to avoid collecting prose.
    alt = "|".join(re.escape(v) for v in sorted(variables, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w.-])({alt})\s*->\s*({alt})(?![\w.-])")
    return set(pattern.findall(text))


def _edge_consistency_issues(
    *,
    think_text: str,
    adj: List[List[int]],
    variables: List[str],
) -> List[str]:
    gt_edges = _gt_directed_edge_set(adj, variables)
    mentioned_edges = _extract_mentioned_directed_edges(think_text, variables)
    extra = sorted(mentioned_edges - gt_edges)

    issues: List[str] = []
    if extra:
        issues.append(
            "mentioned non-GT directed edges: "
            + ", ".join(f"{a} -> {b}" for a, b in extra)
        )
    return issues


def _teacher_reasoning_issues(
    *,
    think_text: str,
    adj: List[List[int]],
    variables: List[str],
) -> List[str]:
    completion = _completion_for_adj(think_text, adj)
    issues = _hard_format_issues(
        think_text=think_text,
        completion=completion,
        adj=adj,
    )
    issues.extend(
        _edge_consistency_issues(
            think_text=think_text,
            adj=adj,
            variables=variables,
        )
    )
    return issues


def _build_teacher_revision_prompt(
    *,
    base_prompt: str,
    previous_think: str,
    issues: List[str],
    adj: List[List[int]],
    variables: List[str],
) -> str:
    return (
        f"{base_prompt}\n\n"
        "The previous reasoning failed validation and must be revised.\n\n"
        "Previous reasoning:\n"
        f"{previous_think}\n\n"
        "Validation issues:\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\n\n"
        "Revision requirements:\n"
        "- Return only the revised reasoning text; no tags and no JSON.\n"
        "- Mention only directed edges that are in GROUND_TRUTH_EDGES when using A -> B notation.\n"
        "- It is okay to omit some ground-truth edges if the reasoning remains concise.\n"
        "- Make the reasoning evidence-grounded and concise.\n"
        f"- The only allowed directed edges are: {_matrix_edge_summary(adj, variables)}.\n\n"
        "Revised reasoning text only:"
    )


def _build_teacher_prompt(
    *,
    prompt_text: str,
    adj: List[List[int]],
    variables: List[str],
) -> str:
    matrix_json = json.dumps({"adjacency_matrix": adj}, ensure_ascii=False)
    variables_json = json.dumps(variables, ensure_ascii=False)
    return (
        "You are writing supervised fine-tuning target reasoning for a causal "
        "discovery model.\n\n"
        "The student prompt is below. The ground-truth answer is also provided. "
        "Write only the reasoning text that should appear inside <think>...</think>.\n\n"
        "Requirements:\n"
        "- Do not include <think>, </think>, <answer>, or JSON.\n"
        "- Do not predict a different graph; every directed edge you mention must be "
        "consistent with the ground-truth adjacency matrix.\n"
        "- Mention only directed edges that are in GROUND_TRUTH_EDGES when using A -> B notation.\n"
        "- It is okay to omit some ground-truth edges if the reasoning remains concise.\n"
        "- Keep it concise: 3 to 7 short lines.\n"
        "- Ground the reasoning in the observational/interventional evidence when the "
        "prompt contains such evidence.\n"
        "- Prefer clear causal language and avoid hedging about the known answer.\n\n"
        f"VARIABLES_IN_ORDER={variables_json}\n"
        f"GROUND_TRUTH_EDGES={_matrix_edge_summary(adj, variables)}\n"
        f"GROUND_TRUTH_ANSWER={matrix_json}\n\n"
        "STUDENT_PROMPT:\n"
        f"{prompt_text}\n\n"
        "Reasoning text only:"
    )


def _strip_teacher_think(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if "<think>" in text and "</think>" in text:
        text = text.split("<think>", 1)[1].split("</think>", 1)[0]
    if "<answer>" in text:
        text = text.split("<answer>", 1)[0]
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    text = text.strip()
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _call_teacher_openai_compatible(prompt: str, cfg: TeacherConfig) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return f"[ERROR] OpenAI SDK not available: {type(e).__name__}: {e}"

    api_key = os.getenv(cfg.api_key_env, "")
    if not api_key and cfg.base_url:
        api_key = "EMPTY"
    if not api_key:
        return f"[ERROR] Missing API key (set {cfg.api_key_env})."

    kwargs = {
        "api_key": api_key,
        "timeout": float(cfg.timeout),
        "max_retries": 0,
    }
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url.rstrip("/")
    client = OpenAI(**kwargs)

    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise, correct causal-discovery reasoning "
                        "targets from supplied ground truth."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(cfg.temperature),
            max_tokens=int(cfg.max_tokens),
        )
        msg = resp.choices[0].message
        return msg.content or ""
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


def _call_teacher_openai_responses(prompt: str, cfg: TeacherConfig) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return f"[ERROR] OpenAI SDK not available: {type(e).__name__}: {e}"

    api_key = os.getenv(cfg.api_key_env, "")
    if not api_key:
        return f"[ERROR] Missing API key (set {cfg.api_key_env})."

    client = OpenAI(
        api_key=api_key,
        timeout=float(cfg.timeout),
        max_retries=0,
    )

    try:
        req = {
            "model": cfg.model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You write concise, correct causal-discovery reasoning "
                        "targets from supplied ground truth."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": int(cfg.max_tokens),
        }
        if cfg.reasoning_effort:
            req["reasoning"] = {"effort": cfg.reasoning_effort}
        # GPT-5-family reasoning models often reject sampling controls unless
        # reasoning is disabled. Keep Responses teachers deterministic and broad.
        if float(cfg.temperature) != 0.0 and "gpt-5" not in cfg.model.lower():
            req["temperature"] = float(cfg.temperature)
        resp = client.responses.create(**req)
        text = getattr(resp, "output_text", None)
        if text:
            return text
        try:
            for item in getattr(resp, "output", []) or []:
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        t = getattr(part, "text", "") or ""
                        if t:
                            return t
        except Exception:
            pass
        status = getattr(resp, "status", None)
        incomplete = getattr(resp, "incomplete_details", None)
        return (
            "[ERROR] Empty OpenAI Responses output"
            f" (status={status}, incomplete_details={incomplete})."
        )
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


def _call_teacher_gemini(prompt: str, cfg: TeacherConfig) -> str:
    try:
        from query_vllm import call_gemini  # type: ignore
    except ImportError:
        try:
            from experiments.query_vllm import call_gemini  # type: ignore
        except ImportError as e:
            return f"[ERROR] Gemini helper not available: {type(e).__name__}: {e}"
    return call_gemini(cfg.model, prompt, temperature=float(cfg.temperature))


def _call_teacher(prompt: str, cfg: TeacherConfig) -> str:
    provider = cfg.provider.lower()
    if provider == "openai_responses":
        return _call_teacher_openai_responses(prompt, cfg)
    if provider in {"openai", "openai_compatible"}:
        return _call_teacher_openai_compatible(prompt, cfg)
    if provider == "gemini":
        return _call_teacher_gemini(prompt, cfg)
    return f"[ERROR] Unsupported teacher provider: {cfg.provider}"


def _build_teacher_evidence_think(
    *,
    prompt_text: str,
    adj: List[List[int]],
    variables: List[str],
    teacher_config: TeacherConfig,
) -> str:
    teacher_prompt = _build_teacher_prompt(
        prompt_text=prompt_text,
        adj=adj,
        variables=variables,
    )
    current_prompt = teacher_prompt
    raw = ""
    think = ""
    issues: List[str] = []

    attempt = 0
    while True:
        raw = _call_teacher(current_prompt, teacher_config)
        think = _strip_teacher_think(raw)
        if think and not think.startswith("[ERROR]"):
            issues = _teacher_reasoning_issues(
                think_text=think,
                adj=adj,
                variables=variables,
            )
            if not issues:
                return think
        else:
            issues = [f"teacher returned no usable reasoning: {raw[:200]}"]

        if (not teacher_config.retry_until_valid) and attempt >= int(teacher_config.max_revisions):
            break

        current_prompt = _build_teacher_revision_prompt(
            base_prompt=teacher_prompt,
            previous_think=think or raw,
            issues=issues,
            adj=adj,
            variables=variables,
        )
        attempt += 1
        if teacher_config.retry_sleep_s > 0:
            time.sleep(float(teacher_config.retry_sleep_s))

    if teacher_config.fallback_target == "none":
        reason = "; ".join(issues) if issues else raw
        raise ValueError(f"teacher generation failed validation: {reason}")

    print(
        "[warn] teacher reasoning failed; falling back to "
        f"{teacher_config.fallback_target}: {'; '.join(issues)[:200]}",
        file=sys.stderr,
    )
    _s1, _s2, _s3, fallback = _build_reasoning_text(
        prompt_text=prompt_text,
        adj=adj,
        variables=variables,
        reasoning_target=teacher_config.fallback_target,
        teacher_config=None,
    )
    return fallback


# ---------------------------------------------------------------------------
# Mode B: record builder
# ---------------------------------------------------------------------------

def _build_record(
    row: dict,
    source_name: str,
    graph_name: str,
    prompt_col: str,
    answer_col: str,
    reasoning_target: str,
    wrapper_mode: str,
    prompt_reasoning_guidance: Optional[str] = None,
    teacher_config: Optional[TeacherConfig] = None,
    source_row_idx: Optional[int] = None,
) -> Optional[dict]:
    """Return a validated SFT record or None if anything fails."""
    prompt_raw = (row.get(prompt_col) or "").strip()
    answer_raw = (row.get(answer_col) or "").strip()

    if not prompt_raw or not answer_raw:
        return None

    adj = _load_adj(answer_raw)
    if adj is None:
        return None

    variables = _extract_variables(prompt_raw)
    if variables is None or len(variables) != len(adj):
        n = len(adj)
        variables = [f"X{k+1}" for k in range(n)]

    try:
        stage1_text, stage2_text, stage3_text, think_text = _build_reasoning_text(
            prompt_text=prompt_raw,
            adj=adj,
            variables=variables,
            reasoning_target=reasoning_target,
            teacher_config=teacher_config,
        )
    except ValueError as e:
        print(
            f"  [warn] skipped {source_name} target={reasoning_target}: {e}",
            file=sys.stderr,
        )
        return None

    prompt = render_prompt_text(
        prompt_raw,
        task="causal_discovery",
        wrapper_mode=wrapper_mode,
        reasoning_guidance=prompt_reasoning_guidance or _reasoning_guidance_for_target(reasoning_target),
        prefill_think=True,
    )
    completion = _completion_for_adj(think_text, adj)
    hard_issues = _hard_format_issues(
        think_text=think_text,
        completion=completion,
        adj=adj,
    )
    if hard_issues:
        print(
            f"  [warn] skipped {source_name} target={reasoning_target}: "
            + "; ".join(hard_issues),
            file=sys.stderr,
        )
        return None

    issues = validate_sft_example(prompt, completion)
    if issues:
        return None

    row_graph_name = _resolve_graph_name_from_row(row, fallback=graph_name)

    rec = {
        "prompt": prompt,
        "answer": completion,
        "gold_think": think_text,
        "gold_stage1": stage1_text,
        "gold_stage2": stage2_text,
        "gold_stage3": stage3_text,
        "source": source_name,
        "graph": row_graph_name,
        "reasoning_target": reasoning_target,
        "think_style": _reasoning_style_label(reasoning_target),
    }
    if source_row_idx is not None:
        rec["source_row_idx"] = int(source_row_idx)
    return rec


def _resolve_variables_for_record(
    *,
    prompt_raw: str,
    adj: List[List[int]],
    item_variables: Optional[List[str]] = None,
    fallback_variables: Optional[List[str]] = None,
) -> List[str]:
    """Resolve the variable order for one record from the permuted prompt first."""
    n = len(adj)

    prompt_variables = _extract_variables(prompt_raw)
    if prompt_variables is not None and len(prompt_variables) == n:
        return [str(v) for v in prompt_variables]

    if item_variables is not None and len(item_variables) == n:
        return [str(v) for v in item_variables]

    if fallback_variables is not None and len(fallback_variables) == n:
        return [str(v) for v in fallback_variables]

    return [f"X{k+1}" for k in range(n)]


def _resolve_graph_name_from_row(row: dict, *, fallback: str) -> str:
    """Prefer row-level graph metadata over file-level source naming."""
    dataset = (row.get("dataset") or "").strip()
    if dataset:
        return dataset

    graph = (row.get("graph") or "").strip()
    if graph:
        return graph

    bif_file = (row.get("bif_file") or "").strip()
    if bif_file:
        return Path(bif_file).stem

    return fallback


# ---------------------------------------------------------------------------
# Mode A: in-memory generation from BIF files
# ---------------------------------------------------------------------------

def _build_records_in_memory(
    *,
    bif_file: Path,
    graph_name: str,
    prompt_style: str,
    obs_per_prompt: int,
    int_per_combo: int,
    num_prompts: int,
    seed: int,
    anonymize: bool,
    col_perms: int,
    reasoning_target: str,
    wrapper_mode: str,
    prompt_reasoning_guidance: Optional[str] = None,
    teacher_config: Optional[TeacherConfig] = None,
) -> List[dict]:
    """
    Generate SFT records directly from a BIF file without writing intermediate CSVs.

    col_perms controls column-order diversity:
      - col_perms=1  : original column order only (seed used as-is)
      - col_perms=N  : 1 original + (N-1) random column permutations
                       Each permutation uses seed+i so the column shuffle differs.
    """
    source_tag = f"{graph_name}_obs{obs_per_prompt}_int{int_per_combo}"
    records: List[dict] = []

    for perm_idx in range(col_perms):
        col_order = "original" if perm_idx == 0 else "random"
        perm_seed = seed + perm_idx

        try:
            _base_name, answer_obj, prompt_iter = iter_prompts_in_memory(
                bif_file=str(bif_file),
                num_prompts=num_prompts,
                shuffles_per_graph=1,
                seed=perm_seed,
                prompt_style=prompt_style,
                obs_per_prompt=obs_per_prompt,
                int_per_combo=int_per_combo,
                row_order="random",
                col_order=col_order,
                anonymize=anonymize,
                causal_rules=False,
                give_steps=False,
                def_int=False,
                intervene_vars="all",
                wrapper_mode="plain",
            )
        except Exception as e:
            print(f"  [warn] {source_tag} perm_idx={perm_idx}: iter_prompts_in_memory failed: {e}",
                  file=sys.stderr)
            continue

        adj = answer_obj["adjacency_matrix"]
        fallback_variables = [str(v) for v in answer_obj["variables"]]

        for item in prompt_iter:
            prompt_raw = (item.get("prompt_text") or "").strip()
            if not prompt_raw:
                continue
            variables = _resolve_variables_for_record(
                prompt_raw=prompt_raw,
                adj=adj,
                item_variables=[str(v) for v in (item.get("variables") or [])],
                fallback_variables=fallback_variables,
            )

            try:
                stage1_text, stage2_text, stage3_text, think_text = _build_reasoning_text(
                    prompt_text=prompt_raw,
                    adj=adj,
                    variables=variables,
                    reasoning_target=reasoning_target,
                    teacher_config=teacher_config,
                )
            except ValueError as e:
                print(
                    f"  [warn] skipped {source_tag} target={reasoning_target}: {e}",
                    file=sys.stderr,
                )
                continue

            prompt = render_prompt_text(
                prompt_raw,
                task="causal_discovery",
                wrapper_mode=wrapper_mode,
                reasoning_guidance=prompt_reasoning_guidance or _reasoning_guidance_for_target(reasoning_target),
                prefill_think=True,
            )
            completion = _completion_for_adj(think_text, adj)
            if _hard_format_issues(think_text=think_text, completion=completion, adj=adj):
                continue

            if validate_sft_example(prompt, completion):
                continue

            records.append({
                "prompt": prompt,
                "answer": completion,
                "gold_think": think_text,
                "gold_stage1": stage1_text,
                "gold_stage2": stage2_text,
                "gold_stage3": stage3_text,
                "source": source_tag,
                "graph": graph_name,
                "reasoning_target": reasoning_target,
                "think_style": _reasoning_style_label(reasoning_target),
            })

    return records


# ---------------------------------------------------------------------------
# Source discovery (Mode B and C)
# ---------------------------------------------------------------------------

def _discover_sources(data_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Return list of (csv_path, source_name, graph_name) tuples.

    Priority order:
      1. *_obs100_int10_anon_train.csv  — the canonical per-graph train splits
      2. *_randcol_seed*.csv            — column-permuted variants (larger)
    """
    sources: List[Tuple[Path, str, str]] = []

    for p in sorted(data_dir.glob("*_obs100_int10_anon_train.csv")):
        graph = p.stem.replace("_obs100_int10_anon_train", "")
        sources.append((p, p.stem, graph))

    for p in sorted(data_dir.glob("*_randcol_seed*.csv")):
        parts = p.stem.split("_randcol_seed")
        graph = parts[0]
        sources.append((p, p.stem, graph))

    return sources


def _record_retry_key(rec: dict) -> Optional[Tuple[str, str, int]]:
    source = str(rec.get("source") or "")
    target = str(rec.get("reasoning_target") or "")
    row_idx = rec.get("source_row_idx")
    if not source or not target or row_idx is None:
        return None
    try:
        return source, target, int(row_idx)
    except Exception:
        return None


def _existing_record_issues(rec: dict) -> List[str]:
    answer = str(rec.get("answer") or "")
    think = str(rec.get("gold_think") or "")
    adj = _parse_completion_adj(answer)
    if adj is None:
        return ["answer JSON did not parse as an adjacency_matrix payload"]

    issues = _hard_format_issues(
        think_text=think,
        completion=answer,
        adj=adj,
    )

    if rec.get("reasoning_target") == "teacher_evidence":
        variables = _extract_variables(str(rec.get("prompt") or ""))
        if variables is None or len(variables) != len(adj):
            variables = [f"X{k+1}" for k in range(len(adj))]
        issues.extend(
            _edge_consistency_issues(
                think_text=think,
                adj=adj,
                variables=variables,
            )
        )
    return issues


def _load_existing_output_records(path: Path) -> Tuple[List[dict], set[Tuple[str, str, int]]]:
    if not path.exists():
        return [], set()
    records: List[dict] = []
    keys: set[Tuple[str, str, int]] = set()
    dropped_invalid = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            issues = _existing_record_issues(rec)
            if issues:
                dropped_invalid += 1
                continue
            records.append(rec)
            key = _record_retry_key(rec)
            if key is not None:
                keys.add(key)
    if dropped_invalid:
        print(
            f"[resume] ignored {dropped_invalid} invalid existing records from {path}; "
            "they will be retried if their source rows are still selected.",
            file=sys.stderr,
        )
    return records, keys


# ---------------------------------------------------------------------------
# Mode C: prompt rewriting helpers
# ---------------------------------------------------------------------------

_VAR_BLOCK_RE = re.compile(r"(--- VARIABLES ---\n)(.*?)(\n---)", re.DOTALL)
_VAR_LINE_RE = re.compile(r"^(\s*\d+\s*:\s*)(\S+)", re.MULTILINE)
_OBS_BLOCK_RE = re.compile(
    r"(--- OBSERVATIONAL DATA ---\n.*?observational_data=\{)(.*?)(\}\s*\n---)", re.DOTALL
)
_INT_BLOCK_RE = re.compile(
    r"(--- INTERVENTIONAL DATA ---\n.*?interventional_data=\{)(.*?)(\}\s*(?:\n---|$))", re.DOTALL
)
_NUM_STATES_RE = re.compile(r"num_states=\[([^\]]+)\]")


def _extract_varnames_from_prompt(prompt: str) -> Optional[List[str]]:
    """Return variable name list parsed from the VARIABLES block, or None."""
    m = _VAR_BLOCK_RE.search(prompt)
    if not m:
        return None
    pairs = _VAR_LINE_RE.findall(m.group(2))
    return [name for _, name in pairs] if pairs else None


def _permute_assignment(assignment: List[int], perm: List[int]) -> List[int]:
    """new_assignment[i] = assignment[perm[i]]"""
    return [assignment[perm[i]] for i in range(len(perm))]


def _permute_obs_data(obs_data: dict, perm: List[int]) -> dict:
    new_hist = [[_permute_assignment(e[0], perm), e[1]] for e in obs_data["hist"]]
    out = {"n": obs_data["n"], "hist": new_hist}
    if "marginals" in obs_data:
        out["marginals"] = [obs_data["marginals"][perm[i]] for i in range(len(perm))]
    return out


def _permute_int_data(int_data: dict, perm: List[int], var_names: List[str]) -> dict:
    """Permute interventional data: rename do(Xi=v) keys and reorder hist slots."""
    inv_perm = [0] * len(perm)
    for new_i, old_i in enumerate(perm):
        inv_perm[old_i] = new_i

    new_int: dict = {}
    for key, val in int_data.items():
        m = re.match(r"do\((\w+)=(\d+)\)", key)
        if not m:
            new_int[key] = val
            continue
        old_var_name, v = m.group(1), m.group(2)
        if old_var_name not in var_names:
            new_int[key] = val
            continue
        new_idx = inv_perm[var_names.index(old_var_name)]
        new_key = f"do(X{new_idx + 1}={v})"
        new_hist = [[_permute_assignment(e[0], perm), e[1]] for e in val["hist"]]
        new_val = {"n": val["n"], "hist": new_hist}
        if "marginals" in val:
            new_val["marginals"] = [val["marginals"][perm[i]] for i in range(len(perm))]
        new_int[new_key] = new_val
    return new_int


def _permute_adj(adj: List[List[int]], perm: List[int]) -> List[List[int]]:
    n = len(perm)
    return [[adj[perm[i]][perm[j]] for j in range(n)] for i in range(n)]


def _fmt_float(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _obs_to_str(obs: dict) -> str:
    hist_parts = ", ".join(f"[{json.dumps(e[0])},{e[1]}]" for e in obs["hist"])
    parts = [f'"n": {obs["n"]}', f'"hist": [{hist_parts}]']
    if "marginals" in obs:
        marg_parts = ", ".join(
            f"[{', '.join(_fmt_float(v) for v in m)}]" for m in obs["marginals"]
        )
        parts.append(f'"marginals": [{marg_parts}]')
    return "{" + ", ".join(parts) + "}"


def _int_to_str(int_data: dict) -> str:
    parts = []
    for key in sorted(int_data.keys()):
        val = int_data[key]
        hist_parts = ", ".join(f"[{json.dumps(e[0])},{e[1]}]" for e in val["hist"])
        val_parts = [f'"n": {val["n"]}', f'"hist": [{hist_parts}]']
        if "marginals" in val:
            marg_parts = ", ".join(
                f"[{', '.join(_fmt_float(v) for v in m)}]" for m in val["marginals"]
            )
            val_parts.append(f'"marginals": [{marg_parts}]')
        v_str = "{" + ", ".join(val_parts) + "}"
        parts.append(f'"{key}": {v_str}')
    return "{" + ", ".join(parts) + "}"


def _get_permutations(n: int, max_perms: int, rng: random.Random) -> List[List[int]]:
    """All n! permutations if n! <= max_perms, else sample max_perms distinct ones."""
    total = math.factorial(n)
    base = list(range(n))
    if total <= max_perms:
        return [list(p) for p in itertools.permutations(base)]
    seen: set = set()
    result: List[List[int]] = []
    attempts = 0
    while len(result) < max_perms and attempts < max_perms * 20:
        p = base[:]
        rng.shuffle(p)
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            result.append(p)
        attempts += 1
    return result


def _rewrite_prompt(
    prompt: str,
    perm: List[int],
    var_names: List[str],
) -> Optional[str]:
    """
    Rewrite a CD prompt with permuted variable order.

    perm[new_idx] = old_idx — new position i gets the variable at old_idx.
    var_names: original variable names (X1, X2, ...) in original order.
    """
    n = len(perm)

    # 1. Rewrite VARIABLES block
    def _replace_var_block(m: re.Match) -> str:
        lines = [f"{new_i}: X{new_i + 1}" for new_i in range(n)]
        return m.group(1) + "\n".join(lines) + m.group(3)

    prompt2 = _VAR_BLOCK_RE.sub(_replace_var_block, prompt)

    # 2. Rewrite OBSERVATIONAL DATA
    obs_m = _OBS_BLOCK_RE.search(prompt2)
    if not obs_m:
        return None
    try:
        obs_data = json.loads("{" + obs_m.group(2) + "}")
    except json.JSONDecodeError:
        try:
            obs_data = json.loads(obs_m.group(2).strip())
        except Exception:
            return None

    obs_str = _obs_to_str(_permute_obs_data(obs_data, perm))
    prompt3 = prompt2[:obs_m.start(2)] + obs_str + prompt2[obs_m.end(2):]

    # 3. Rewrite INTERVENTIONAL DATA
    int_m = _INT_BLOCK_RE.search(prompt3)
    if not int_m:
        return None
    try:
        int_data = json.loads("{" + int_m.group(2).strip() + "}")
    except json.JSONDecodeError:
        try:
            int_data = json.loads(int_m.group(2).strip())
        except Exception:
            return None

    int_str = _int_to_str(_permute_int_data(int_data, perm, var_names))
    prompt4 = prompt3[:int_m.start(2)] + int_str + prompt3[int_m.end(2):]

    # 4. Update num_states order
    ns_m = _NUM_STATES_RE.search(prompt4)
    if ns_m:
        orig_states = [int(x.strip()) for x in ns_m.group(1).split(",")]
        new_states = [orig_states[perm[i]] for i in range(n)]
        prompt4 = (
            prompt4[:ns_m.start(1)]
            + ",".join(str(s) for s in new_states)
            + prompt4[ns_m.end(1):]
        )

    # 5. Update tv_change_vs_obs variable names.
    # Format: tv_change_vs_obs=[["X3",0.25],["X1",0.10],...]
    inv_perm = [0] * n
    for new_i, old_i in enumerate(perm):
        inv_perm[old_i] = new_i

    _TV_PREFIX = "tv_change_vs_obs="
    tv_pos = prompt4.find(_TV_PREFIX)
    if tv_pos != -1:
        val_start = tv_pos + len(_TV_PREFIX)
        try:
            decoder = json.JSONDecoder()
            tv_list, end_offset = decoder.raw_decode(prompt4, val_start)
            new_tv = []
            for entry in tv_list:
                var_name, tv_val = entry[0], entry[1]
                if (isinstance(var_name, str)
                        and var_name.startswith("X")
                        and var_name[1:].isdigit()):
                    old_i = int(var_name[1:]) - 1
                    var_name = f"X{inv_perm[old_i] + 1}"
                new_tv.append([var_name, tv_val])
            tv_json = json.dumps(new_tv, separators=(",", ":"), ensure_ascii=False)
            prompt4 = prompt4[:val_start] + tv_json + prompt4[val_start + end_offset:]
        except Exception:
            pass  # leave unchanged on parse failure

    return prompt4


def _build_records_perm_csv(
    csv_path: Path,
    source_name: str,
    graph_name: str,
    rows_per_source: int,
    max_perms: int,
    rng: random.Random,
    prompt_col: str = "prompt",
    answer_col: str = "answer",
    reasoning_target: str = "stages_evidence",
    wrapper_mode: str = "chat",
    prompt_reasoning_guidance: Optional[str] = None,
    teacher_config: Optional[TeacherConfig] = None,
) -> Tuple[List[dict], int, int]:
    """
    Build SFT records from one CSV by enumerating variable-order permutations.

    Returns (records, n_built, n_skipped).
    """
    rows = _sample_rows(csv_path, rows_per_source, rng)
    if not rows:
        return [], 0, 0

    var_names_0 = _extract_varnames_from_prompt((rows[0].get(prompt_col) or "").strip())
    if not var_names_0:
        return [], 0, len(rows)

    n = len(var_names_0)
    n_total = math.factorial(n)
    perms = _get_permutations(n, max_perms, rng)
    print(f"  {source_name}: n={n}, {n_total:,} total perms, "
          f"using {len(perms)} | rows={len(rows)}")

    records: List[dict] = []
    built = skipped = 0

    for row in rows:
        prompt_raw = (row.get(prompt_col) or "").strip()
        answer_raw = (row.get(answer_col) or "").strip()
        adj_orig = _load_adj(answer_raw)
        if adj_orig is None or len(adj_orig) != n:
            skipped += 1
            continue

        row_var_names = _extract_varnames_from_prompt(prompt_raw) or var_names_0
        row_graph_name = _resolve_graph_name_from_row(row, fallback=graph_name)
        row_seen: set = set()

        for perm in perms:
            adj_perm = _permute_adj(adj_orig, perm)
            adj_key = str(adj_perm)
            if adj_key in row_seen:
                continue  # automorphism: same graph under a different label

            prompt_perm = _rewrite_prompt(prompt_raw, perm, row_var_names)
            if prompt_perm is None:
                skipped += 1
                continue

            new_var_names = _extract_varnames_from_prompt(prompt_perm)
            if new_var_names is None or len(new_var_names) != n:
                skipped += 1
                continue

            prompt_final = render_prompt_text(
                prompt_perm,
                task="causal_discovery",
                wrapper_mode=wrapper_mode,
                reasoning_guidance=prompt_reasoning_guidance or _reasoning_guidance_for_target(reasoning_target),
                prefill_think=True,
            )
            try:
                stage1_text, stage2_text, stage3_text, think_text = _build_reasoning_text(
                    prompt_text=prompt_perm,
                    adj=adj_perm,
                    variables=new_var_names,
                    reasoning_target=reasoning_target,
                    teacher_config=teacher_config,
                )
            except ValueError as e:
                print(
                    f"  [warn] skipped {source_name} target={reasoning_target}: {e}",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            completion = _completion_for_adj(think_text, adj_perm)
            if _hard_format_issues(think_text=think_text, completion=completion, adj=adj_perm):
                skipped += 1
                continue

            if validate_sft_example(prompt_final, completion):
                skipped += 1
                continue

            row_seen.add(adj_key)
            records.append({
                "prompt": prompt_final,
                "answer": completion,
                "gold_think": think_text,
                "gold_stage1": stage1_text,
                "gold_stage2": stage2_text,
                "gold_stage3": stage3_text,
                "source": source_name,
                "graph": row_graph_name,
                "perm": perm,
                "reasoning_target": reasoning_target,
                "think_style": _reasoning_style_label(reasoning_target),
            })
            built += 1

    return records, built, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Collect format-teaching SFT examples for causal discovery. "
            "Mode A: in-memory BIF (--graphs). "
            "Mode B: CSV discovery [default]. "
            "Mode C: exhaustive permutation from CSV rows (--perm-csv)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("experiments/data/format_sft.jsonl"),
        help="Output JSONL path.",
    )

    # --- Mode A: in-memory BIF ---
    ap.add_argument(
        "--graphs", nargs="+", default=None,
        help=(
            "Graph names to generate from BIF files (Mode A). "
            "When set, data is generated in-memory — no CSV files needed."
        ),
    )
    ap.add_argument(
        "--graphs-dir", type=Path,
        default=Path("causal_graphs/real_data/small_graphs"),
        help="Directory containing *.bif files (Mode A).",
    )
    ap.add_argument(
        "--prompt-style",
        choices=["summary", "summary_joint", "matrix"],
        default="summary",
        help="Prompt style for in-memory generation (Mode A only).",
    )
    ap.add_argument(
        "--obs-values", nargs="+", type=int, default=[100],
        help="obs_per_prompt values (Mode A only).",
    )
    ap.add_argument(
        "--int-values", nargs="+", type=int, default=[10],
        help="int_per_combo values (Mode A only).",
    )
    ap.add_argument(
        "--num-prompts-per-config", type=int, default=500,
        help="Prompts per (graph, obs, int) config (Mode A only).",
    )
    ap.add_argument(
        "--col-perms", type=int, default=5,
        help=(
            "Column-order permutations per config (Mode A only). "
            "1 = original order only. N = 1 original + (N-1) random shuffles."
        ),
    )
    ap.add_argument(
        "--anonymize", action="store_true", default=False,
        help="Anonymize variable names to X1, X2, ... (Mode A only).",
    )
    ap.add_argument(
        "--wrapper-mode",
        choices=["plain", "chat"],
        default="chat",
        help="Prompt transport used for stored SFT prompts.",
    )

    # --- Mode B / C: CSV-based ---
    ap.add_argument(
        "--data-dir", type=Path, default=Path("experiments/data"),
        help="Directory containing source CSV files (Mode B/C).",
    )
    ap.add_argument(
        "--n-per-source", type=int, default=100,
        help="Max rows to sample per CSV source (Mode B only).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--answer-col", default="answer")
    ap.add_argument(
        "--reasoning-target",
        nargs="+",
        default=["stages_evidence"],
        help=(
            "Supervised completion target style(s). Accepts one or more values, "
            "either space-separated or comma-separated."
        ),
    )
    ap.add_argument(
        "--prompt-reasoning-guidance",
        choices=["auto", "none", "concise", "staged"],
        default="auto",
        help=(
            "Override the reasoning guidance rendered into prompts. "
            "'auto' uses the default guidance implied by each reasoning target."
        ),
    )
    ap.add_argument(
        "--teacher-model",
        default=None,
        help=(
            "Teacher model used by --reasoning-target teacher_evidence. "
            "The teacher writes only the <think> content; the final answer still "
            "uses the ground-truth adjacency matrix."
        ),
    )
    ap.add_argument(
        "--teacher-provider",
        choices=["openai", "openai_compatible", "openai_responses", "gemini"],
        default="openai_compatible",
        help="Teacher backend for --reasoning-target teacher_evidence.",
    )
    ap.add_argument(
        "--teacher-base-url",
        default=None,
        help=(
            "Optional OpenAI-compatible base URL for the teacher, for example "
            "a local vLLM server. Ignored for Gemini."
        ),
    )
    ap.add_argument(
        "--teacher-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the teacher API key.",
    )
    ap.add_argument(
        "--teacher-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for teacher reasoning generation.",
    )
    ap.add_argument(
        "--teacher-max-tokens",
        type=int,
        default=512,
        help="Maximum teacher output tokens for reasoning text.",
    )
    ap.add_argument(
        "--teacher-timeout",
        type=float,
        default=120.0,
        help="Teacher request timeout in seconds.",
    )
    ap.add_argument(
        "--teacher-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help=(
            "Optional Responses API reasoning effort for teacher models. "
            "For gpt-5-mini, 'minimal' or 'low' is usually enough for target generation."
        ),
    )
    ap.add_argument(
        "--teacher-max-revisions",
        type=int,
        default=1,
        help=(
            "Number of teacher repair attempts after edge/matrix validation fails. "
            "Set 0 to disable revision."
        ),
    )
    ap.add_argument(
        "--teacher-retry-until-valid",
        action="store_true",
        help="Keep asking the teacher to revise until reasoning validation passes.",
    )
    ap.add_argument(
        "--teacher-retry-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between teacher retries when --teacher-retry-until-valid is set.",
    )
    ap.add_argument(
        "--teacher-fallback-target",
        choices=["none", "concise_evidence", "stages", "stages_evidence"],
        default="concise_evidence",
        help=(
            "Deterministic target to use if a teacher request fails. "
            "Use 'none' to skip/fail that record instead."
        ),
    )
    ap.add_argument(
        "--csv", action="append", default=[], metavar="PATH[:GRAPH]",
        help=(
            "Explicit CSV file to include (repeatable). "
            "Optionally append :GRAPH_NAME (e.g. data/cancer.csv:cancer). "
            "When provided, auto-discovery from --data-dir is skipped unless "
            "--also-discover is also set."
        ),
    )
    ap.add_argument(
        "--also-discover", action="store_true",
        help="When --csv is given, also run auto-discovery from --data-dir.",
    )
    ap.add_argument(
        "--graph-filter", nargs="*",
        help="Only include sources whose graph name matches one of these.",
    )
    ap.add_argument(
        "--sources-only", action="store_true",
        help="List discovered sources and exit without writing output.",
    )
    ap.add_argument(
        "--resume-existing-output",
        action="store_true",
        help=(
            "Load existing --output JSONL records and retry only missing CSV source rows "
            "for the requested reasoning target(s). Useful after teacher [ERROR] skips."
        ),
    )
    ap.add_argument(
        "--append-records-as-they-complete",
        dest="append_records_as_they_complete",
        action="store_true",
        default=True,
        help=(
            "Write each successful CSV-mode record to --output immediately. "
            "This is the default so interrupted teacher runs can be resumed."
        ),
    )
    ap.add_argument(
        "--no-append-records-as-they-complete",
        dest="append_records_as_they_complete",
        action="store_false",
        help="Disable immediate CSV-mode writes and restore final write-at-end behavior.",
    )

    # --- Mode C: perm-csv ---
    ap.add_argument(
        "--perm-csv", action="store_true",
        help=(
            "Enable exhaustive variable-order permutation mode (Mode C). "
            "Reads CSV sources (same discovery logic as Mode B) and generates "
            "up to --max-perms permutations for each of --rows-per-source rows."
        ),
    )
    ap.add_argument(
        "--rows-per-source", type=int, default=5,
        help="CSV rows to sample per source (Mode C only).",
    )
    ap.add_argument(
        "--max-perms", type=int, default=500,
        help=(
            "Max permutations per row (Mode C only). "
            "All n! permutations are used when n! <= this value."
        ),
    )

    args = ap.parse_args()
    if args.prompt_style == "summary_joint":
        args.prompt_style = "summary"
    reasoning_targets = _parse_reasoning_targets(args.reasoning_target)
    teacher_config: Optional[TeacherConfig] = None
    if "teacher_evidence" in reasoning_targets:
        if not args.teacher_model:
            sys.exit("ERROR: --reasoning-target teacher_evidence requires --teacher-model")
        teacher_config = TeacherConfig(
            provider=str(args.teacher_provider),
            model=str(args.teacher_model),
            base_url=args.teacher_base_url,
            api_key_env=str(args.teacher_api_key_env),
            temperature=float(args.teacher_temperature),
            max_tokens=int(args.teacher_max_tokens),
            timeout=float(args.teacher_timeout),
            fallback_target=str(args.teacher_fallback_target),
            reasoning_effort=args.teacher_reasoning_effort,
            max_revisions=int(args.teacher_max_revisions),
            retry_until_valid=bool(args.teacher_retry_until_valid),
            retry_sleep_s=float(args.teacher_retry_sleep),
        )
    prompt_reasoning_guidance = (
        None
        if str(args.prompt_reasoning_guidance) == "auto"
        else str(args.prompt_reasoning_guidance)
    )
    _set_csv_limit()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_records: List[dict] = []
    stream_csv_records = (
        bool(args.append_records_as_they_complete)
        and not bool(args.graphs)
        and not bool(args.perm_csv)
    )
    existing_retry_keys: set[Tuple[str, str, int]] = set()
    if bool(args.resume_existing_output):
        existing_records, existing_retry_keys = _load_existing_output_records(args.output)
        all_records.extend(existing_records)
        print(
            f"[resume] loaded {len(existing_records)} existing records "
            f"({len(existing_retry_keys)} retry keys) from {args.output}"
        )
        if stream_csv_records:
            with args.output.open("w", encoding="utf-8") as fout:
                for rec in existing_records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elif stream_csv_records:
        args.output.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Mode A: in-memory BIF generation                                    #
    # ------------------------------------------------------------------ #
    if args.graphs:
        graph_filter = set(args.graph_filter) if args.graph_filter else None
        config_seed = args.seed
        for graph_name in args.graphs:
            if graph_filter and graph_name not in graph_filter:
                continue
            bif_file = args.graphs_dir / f"{graph_name}.bif"
            if not bif_file.exists():
                print(f"  [warn] BIF not found: {bif_file}", file=sys.stderr)
                continue
            for obs_n in args.obs_values:
                for int_n in args.int_values:
                    recs: List[dict] = []
                    for reasoning_target in reasoning_targets:
                        recs.extend(
                            _build_records_in_memory(
                                bif_file=bif_file,
                                graph_name=graph_name,
                                prompt_style=args.prompt_style,
                                obs_per_prompt=obs_n,
                                int_per_combo=int_n,
                                num_prompts=args.num_prompts_per_config,
                                seed=config_seed,
                                anonymize=args.anonymize,
                                col_perms=args.col_perms,
                                reasoning_target=reasoning_target,
                                wrapper_mode=args.wrapper_mode,
                                prompt_reasoning_guidance=prompt_reasoning_guidance,
                                teacher_config=teacher_config,
                            )
                        )
                    config_seed += 1000
                    all_records.extend(recs)
                    print(
                        f"  {graph_name:12s}  obs={obs_n:4d}  int={int_n:3d}"
                        f"  col_perms={args.col_perms}  targets={','.join(reasoning_targets)}"
                        f"  → {len(recs)} records"
                    )

    # ------------------------------------------------------------------ #
    # Mode B / C: CSV-based                                               #
    # ------------------------------------------------------------------ #
    else:
        # Resolve CSV sources
        sources: List[Tuple[Path, str, str]] = []
        explicit_csvs = args.csv or []

        if explicit_csvs and not args.also_discover:
            for spec in explicit_csvs:
                if ":" in spec:
                    path_str, graph = spec.rsplit(":", 1)
                else:
                    path_str = spec
                    graph = Path(spec).stem.split("_")[0]
                p = Path(path_str)
                if not p.exists():
                    sys.exit(f"ERROR: --csv file not found: {p}")
                sources.append((p, p.stem, graph))
        else:
            data_dir = args.data_dir
            if not data_dir.is_dir():
                sys.exit(f"ERROR: data directory not found: {data_dir}")
            sources = _discover_sources(data_dir)
            for spec in explicit_csvs:
                if ":" in spec:
                    path_str, graph = spec.rsplit(":", 1)
                else:
                    path_str = spec
                    graph = Path(spec).stem.split("_")[0]
                p = Path(path_str)
                if not p.exists():
                    sys.exit(f"ERROR: --csv file not found: {p}")
                sources.append((p, p.stem, graph))

        if not sources:
            sys.exit("ERROR: no CSV sources found (check --data-dir or --csv)")

        if args.graph_filter:
            keep = set(args.graph_filter)
            sources = [(p, s, g) for p, s, g in sources if g in keep]
            if not sources:
                sys.exit(f"ERROR: no sources match graph filter {args.graph_filter}")

        if args.sources_only:
            for p, s, g in sources:
                print(f"  graph={g:20s}  source={s:45s}  path={p}")
            print(f"\n{len(sources)} sources total")
            return

        # ---- Mode C: exhaustive permutation ----
        if args.perm_csv:
            print(f"Mode C (perm-csv): {len(sources)} sources, "
                  f"rows_per_source={args.rows_per_source}, max_perms={args.max_perms}\n")
            unique_answers: set = set()
            for csv_path, source_name, graph_name in _progress(
                sources,
                desc="sources",
                unit="source",
                leave=True,
            ):
                recs: List[dict] = []
                built = skipped = 0
                for reasoning_target in reasoning_targets:
                    target_recs, target_built, target_skipped = _build_records_perm_csv(
                        csv_path=csv_path,
                        source_name=source_name,
                        graph_name=graph_name,
                        rows_per_source=args.rows_per_source,
                        max_perms=args.max_perms,
                        rng=rng,
                        prompt_col=args.prompt_col,
                        answer_col=args.answer_col,
                        reasoning_target=reasoning_target,
                        wrapper_mode=args.wrapper_mode,
                        prompt_reasoning_guidance=prompt_reasoning_guidance,
                        teacher_config=teacher_config,
                    )
                    recs.extend(target_recs)
                    built += target_built
                    skipped += target_skipped
                all_records.extend(recs)
                for rec in recs:
                    unique_answers.add(rec["answer"].split("<answer>")[-1])
                print(f"    → built={built}, skipped={skipped}, "
                      f"unique answers so far={len(unique_answers)}")

        # ---- Mode B: one record per CSV row ----
        else:
            print(f"Mode B (csv): {len(sources)} sources, "
                  f"n_per_source={args.n_per_source}, seed={args.seed}\n")
            stream_fout = args.output.open("a", encoding="utf-8") if stream_csv_records else None
            for csv_path, source_name, graph_name in _progress(
                sources,
                desc="sources",
                unit="source",
                leave=True,
            ):
                sampled = _sample_rows(csv_path, args.n_per_source, rng)
                built = skipped = 0
                pbar = _progress(
                    sampled,
                    desc=f"{source_name}",
                    unit="row",
                    leave=True,
                )
                for row in pbar:
                    for reasoning_target in reasoning_targets:
                        row_idx_raw = row.get("_source_row_idx")
                        row_idx = int(row_idx_raw) if row_idx_raw is not None else None
                        if row_idx is not None and (source_name, reasoning_target, row_idx) in existing_retry_keys:
                            continue
                        rec = _build_record(
                            row,
                            source_name=source_name,
                            graph_name=graph_name,
                            prompt_col=args.prompt_col,
                            answer_col=args.answer_col,
                            reasoning_target=reasoning_target,
                            wrapper_mode=args.wrapper_mode,
                            prompt_reasoning_guidance=prompt_reasoning_guidance,
                            teacher_config=teacher_config,
                            source_row_idx=row_idx,
                        )
                        if rec is None:
                            skipped += 1
                        else:
                            all_records.append(rec)
                            if stream_fout is not None:
                                stream_fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                stream_fout.flush()
                            built += 1
                    if tqdm is not None:
                        pbar.set_postfix(built=built, skipped=skipped)
                print(f"  {source_name:50s}  built={built:4d}  skipped={skipped}")
            if stream_fout is not None:
                stream_fout.close()

    # Final shuffle + write. CSV Mode B streams records by default so long
    # teacher generations survive interruption; downstream merge scripts do
    # the final global shuffle.
    if not stream_csv_records:
        rng.shuffle(all_records)
        with args.output.open("w", encoding="utf-8") as fout:
            for rec in all_records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_graph: dict[str, int] = {}
    for rec in all_records:
        by_graph[rec["graph"]] = by_graph.get(rec["graph"], 0) + 1

    print(f"\nWrote {len(all_records)} records to {args.output}")
    print("\nBreakdown by graph:")
    for g, cnt in sorted(by_graph.items()):
        print(f"  {g:20s}  {cnt}")

    if all_records:
        ex = all_records[0]
        print("\n--- Example completion (first record) ---")
        print(ex["answer"][:500])
        print("..." if len(ex["answer"]) > 500 else "")

    if not all_records:
        sys.exit("ERROR: no records written — check CSV format and column names")


if __name__ == "__main__":
    main()
