from __future__ import annotations

import re


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant reasons step by step inside <think> tags, following three explicit stages:\n"
    "  Stage 1 (Skeleton): List each directly connected variable pair on its own line as \"X -- Y\". "
    "If none, write \"None\".\n"
    "  Stage 2 (V-structures): List each unshielded collider as \"(parent1, collider, parent2)\" on its own line. "
    "If none, write \"None\".\n"
    "  Stage 3 (Orientation): List each directed edge on its own line as \"X -> Y\". "
    "If none, write \"None\".\n"
    "After reasoning, the assistant outputs the final adjacency matrix inside <answer> tags. "
    "The adjacency matrix is an N×N array of integers (0 or 1) in the order variables are listed under VARIABLES. "
    "Entry [i][j]=1 means variable i directly causes variable j; [i][j]=0 means no direct edge. "
    "Must be a DAG (acyclic)."
)

DEFAULT_FORMAT_HINT_TEXT = (
    "Reason in three stages inside <think>: "
    "Stage 1 (Skeleton) - one \"X -- Y\" per line; "
    "Stage 2 (V-structures) - one \"(parent1, collider, parent2)\" per line; "
    "Stage 3 (Orientation) - one \"X -> Y\" per line. "
    "Write \"None\" for any empty stage. "
    "Then output: <answer>{\"adjacency_matrix\": [[0,1,...],[0,0,...],...]}</answer> "
    "where the matrix is N×N with integer entries 0 or 1 in VARIABLES order, "
    "and [i][j]=1 means variable i directly causes variable j."
)

DESCENDANT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant is an expert in causal inference and reasons from empirical evidence "
    "to identify causal relationships."
)

DEFAULT_DESCENDANT_FORMAT_HINT_TEXT = (
    "Output exactly: <think>brief reasoning</think>"
    "<answer>{\"target\": \"Xk\", \"descendants\": [\"Xi\", ...]}</answer>. "
    "Use exact variable names. "
    "If there are no descendants, use an empty list: \"descendants\": []. "
    "No extra text outside the two blocks."
)

_CHAT_PROMPT_RE = re.compile(r"(?s)^\s*(system\n|<\|im_start\|>system\b)")
_ASSISTANT_SUFFIX_RE = re.compile(r"(?s)(assistant\s*)$")
_CHAT_USER_BLOCK_RE = re.compile(r"(?s)^\s*system\n.*?\nuser\n(.*?)\nassistant(?:\n.*)?\s*$")
_DESCENDANT_STALE_MARKERS = (
    "After reasoning, the assistant outputs the final adjacency matrix inside <answer> tags.",
    '"adjacency_matrix": [...]',
    "Stage 1 (Skeleton)",
)


def default_short_think_text(task: str) -> str:
    if task == "cd_descendants":
        return "I compare the intervention shifts and keep only downstream variables."
    return (
        "Stage 1 (Skeleton):\n"
        "Stage 2 (V-structures):\n"
        "Stage 3 (Orientation):"
    )


def system_prompt_for_task(task: str) -> str:
    if task == "cd_descendants":
        return DESCENDANT_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def default_format_hint_text(task: str) -> str:
    if task == "cd_descendants":
        return DEFAULT_DESCENDANT_FORMAT_HINT_TEXT
    return DEFAULT_FORMAT_HINT_TEXT


def resolve_format_hint_text(task: str, format_hint_text: str) -> str:
    hint = str(format_hint_text or "").strip()
    if not hint:
        return default_format_hint_text(task)
    if task == "cd_descendants" and hint == DEFAULT_FORMAT_HINT_TEXT:
        return DEFAULT_DESCENDANT_FORMAT_HINT_TEXT
    return hint


def looks_like_chat_prompt(text: str) -> bool:
    return bool(_CHAT_PROMPT_RE.match(str(text or "")))


def _unwrap_chat_user_prompt(text: str) -> str:
    s = str(text or "").strip()
    m = _CHAT_USER_BLOCK_RE.match(s)
    if m:
        return m.group(1).strip()
    return s


def _strip_formatting_requirement(text: str) -> str:
    s = str(text or "").strip()
    marker = "\n\nFormatting requirement:"
    if marker in s:
        return s.split(marker, 1)[0].rstrip()
    marker = "\nFormatting requirement:"
    if marker in s:
        return s.split(marker, 1)[0].rstrip()
    return s


def _sanitize_descendant_prompt_source(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    if looks_like_chat_prompt(s):
        s = _unwrap_chat_user_prompt(s)
    s = _strip_formatting_requirement(s)
    if "\nassistant" in s:
        s = s.split("\nassistant", 1)[0].rstrip()
    return s


def append_format_hint_to_user_prompt(prompt_text: str, format_hint_text: str) -> str:
    base = str(prompt_text or "").rstrip()
    hint = str(format_hint_text or "").strip()
    if not hint:
        return base
    if hint in base:
        return base
    if "--- OUTPUT INSTRUCTIONS ---" in base:
        return base
    if not base:
        return f"Formatting requirement: {hint}"
    return f"{base}\n\nFormatting requirement: {hint}"


def build_chat_prompt(
    user_prompt: str,
    *,
    task: str = "causal_discovery",
    prefill_think: bool,
    prefill_answer: bool = False,
    think_text: str = "",
) -> str:
    prompt = f"system\n{system_prompt_for_task(task)}\nuser\n{str(user_prompt or '').rstrip()}\nassistant\n"
    if prefill_answer:
        think = str(think_text or "").strip()
        prompt += "<think>\n"
        if think:
            prompt += f"{think}</think><answer>"
        else:
            prompt += "</think><answer>"
    elif prefill_think:
        prompt += "<think>\n"
    return prompt


def ensure_assistant_think_prefill(prompt_text: str) -> str:
    s = str(prompt_text or "").rstrip()
    if not s:
        return "<think>\n"
    if s.endswith("<think>") or s.endswith("<think>\n"):
        return s if s.endswith("\n") else (s + "\n")
    if _ASSISTANT_SUFFIX_RE.search(s):
        return s + "<think>\n" if s.endswith("\n") else (s + "\n<think>\n")
    return s + "\n<think>\n"


def ensure_assistant_answer_prefill(prompt_text: str, think_text: str) -> str:
    s = str(prompt_text or "").rstrip()
    if "<answer>" in s:
        return s
    think = str(think_text or "").strip()
    if _ASSISTANT_SUFFIX_RE.search(s):
        prefix = s + ("" if s.endswith("\n") else "\n")
    else:
        prefix = s + "\n"
    return prefix + "<think>\n" + (f"{think}</think><answer>" if think else "</think><answer>")


def canonicalize_cd_prompt(
    raw_prompt: str,
    *,
    task: str = "causal_discovery",
    wrap_system_prompt: bool,
    append_format_hint: bool,
    format_hint_text: str,
    prefill_think: bool,
    prefill_answer: bool = False,
    think_text: str = "",
) -> str:
    if prefill_think and prefill_answer:
        raise ValueError("prefill_think and prefill_answer cannot both be true")
    prompt = str(raw_prompt or "").strip()
    if not prompt:
        return ""

    if task == "cd_descendants":
        prompt = _sanitize_descendant_prompt_source(prompt)

    if looks_like_chat_prompt(prompt):
        if prefill_answer:
            return ensure_assistant_answer_prefill(prompt, think_text)
        if prefill_think:
            return ensure_assistant_think_prefill(prompt)
        return prompt.rstrip() + "\n"

    user_prompt = prompt
    if append_format_hint:
        user_prompt = append_format_hint_to_user_prompt(
            user_prompt,
            resolve_format_hint_text(task, format_hint_text),
        )
    if wrap_system_prompt:
        return build_chat_prompt(
            user_prompt,
            task=task,
            prefill_think=prefill_think,
            prefill_answer=prefill_answer,
            think_text=think_text,
        )
    if prefill_answer:
        return ensure_assistant_answer_prefill(user_prompt, think_text)
    if prefill_think:
        return ensure_assistant_think_prefill(user_prompt)
    return user_prompt.rstrip() + "\n"


def build_payload_completion(payload_text: str, *, think_text: str, task: str) -> str:
    think = str(think_text or default_short_think_text(task)).strip()
    if not think:
        think = default_short_think_text(task)
    return f"{think}</think><answer>{payload_text}</answer>"


_STALE_SYSTEM_PROMPT_MARKERS = (
    "Identify which pairs of variables are directly connected",
    "determine if Z is a collider",
    "Orient remaining undirected edges using Meek rules",
)
_STALE_FORMAT_HINT_MARKERS = (
    "list adjacent variable pairs",
    "identify colliders in unshielded triples",
    "orient remaining edges. Then output",
)
_SYSTEM_BLOCK_RE = re.compile(r"(?s)^(system\n)(.*?)(\nuser\n)")
_FORMAT_HINT_RE = re.compile(
    r"\nFormatting requirement:.*?(?=\nassistant\b|\Z)", re.DOTALL
)
_OUTPUT_INSTRUCTIONS_RE = re.compile(
    r"\n--- OUTPUT INSTRUCTIONS ---.*?(?=\nassistant\b|\Z)", re.DOTALL
)
_DAG_DUPLICATE_RE = re.compile(r"(Must be a DAG \(acyclic\)\.)\n\1")


def update_prompt_to_current_format(prompt_text: str) -> str:
    """
    Replace stale system-prompt text and format-hint with current canonical
    versions from SYSTEM_PROMPT and DEFAULT_FORMAT_HINT_TEXT.

    Safe to call on already-current prompts (no-op if nothing stale found).
    """
    s = str(prompt_text or "")

    # Detect staleness before any modifications
    has_stale_system = any(marker in s for marker in _STALE_SYSTEM_PROMPT_MARKERS)
    has_stale_hint = any(marker in s for marker in _STALE_FORMAT_HINT_MARKERS)
    has_output_instructions = "--- OUTPUT INSTRUCTIONS ---" in s
    has_current_hint = DEFAULT_FORMAT_HINT_TEXT in s

    # 1. Replace stale system prompt block
    if has_stale_system:
        def _replace_sys(m: re.Match) -> str:
            return m.group(1) + SYSTEM_PROMPT + m.group(3)
        s = _SYSTEM_BLOCK_RE.sub(_replace_sys, s, count=1)

    # 2. Strip stale inline format hint before removing OUTPUT INSTRUCTIONS
    #    (the hint may be inside the OUTPUT INSTRUCTIONS block)
    s = _FORMAT_HINT_RE.sub("", s)

    # 3. Strip --- OUTPUT INSTRUCTIONS --- block (often contains duplicates)
    s = _OUTPUT_INSTRUCTIONS_RE.sub("", s)

    # 4. Fix duplicate DAG constraint line
    s = _DAG_DUPLICATE_RE.sub(r"\1", s)

    # 5. Add current format hint before the assistant turn if it's missing
    if not has_current_hint:
        s = re.sub(
            r"(\nassistant\b)",
            f"\n\nFormatting requirement: {DEFAULT_FORMAT_HINT_TEXT}\\1",
            s,
            count=1,
        )

    return s


def validate_sft_example(prompt_text: str, completion_text: str) -> list[str]:
    issues: list[str] = []
    prompt = str(prompt_text or "")
    completion = str(completion_text or "")
    if "<think>" not in prompt:
        issues.append("prompt missing <think> prefill")
    if not prompt.rstrip().endswith("<think>"):
        issues.append("prompt does not end at assistant <think> boundary")
    if "</think>" not in completion:
        issues.append("completion missing </think>")
    if "<answer>" not in completion or "</answer>" not in completion:
        issues.append("completion missing <answer> block")
    return issues
