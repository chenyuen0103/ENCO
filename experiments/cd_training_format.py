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
    "After reasoning, the assistant outputs the final adjacency matrix inside <answer> tags."
)

DEFAULT_FORMAT_HINT_TEXT = (
    "Reason in three stages inside <think>: "
    "Stage 1 (Skeleton) - one \"X -- Y\" per line; "
    "Stage 2 (V-structures) - one \"(parent1, collider, parent2)\" per line; "
    "Stage 3 (Orientation) - one \"X -> Y\" per line. "
    "Write \"None\" for any empty stage. "
    "Then output: <answer>{\"adjacency_matrix\": [...]}</answer>."
)

_CHAT_PROMPT_RE = re.compile(r"(?s)^\s*(system\n|<\|im_start\|>system\b)")
_ASSISTANT_SUFFIX_RE = re.compile(r"(?s)(assistant\s*)$")


def default_short_think_text(task: str) -> str:
    if task == "cd_descendants":
        return "I compare the intervention shifts and keep only downstream variables."
    return (
        "Stage 1 (Skeleton):\n"
        "Stage 2 (V-structures):\n"
        "Stage 3 (Orientation):"
    )


def looks_like_chat_prompt(text: str) -> bool:
    return bool(_CHAT_PROMPT_RE.match(str(text or "")))


def append_format_hint_to_user_prompt(prompt_text: str, format_hint_text: str) -> str:
    base = str(prompt_text or "").rstrip()
    hint = str(format_hint_text or "").strip()
    if not hint:
        return base
    if hint in base:
        return base
    if not base:
        return f"Formatting requirement: {hint}"
    return f"{base}\n\nFormatting requirement: {hint}"


def build_chat_prompt(
    user_prompt: str,
    *,
    prefill_think: bool,
    prefill_answer: bool = False,
    think_text: str = "",
) -> str:
    prompt = f"system\n{SYSTEM_PROMPT}\nuser\n{str(user_prompt or '').rstrip()}\nassistant\n"
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

    if looks_like_chat_prompt(prompt):
        if prefill_answer:
            return ensure_assistant_answer_prefill(prompt, think_text)
        if prefill_think:
            return ensure_assistant_think_prefill(prompt)
        return prompt.rstrip() + "\n"

    user_prompt = prompt
    if append_format_hint:
        user_prompt = append_format_hint_to_user_prompt(user_prompt, format_hint_text)
    if wrap_system_prompt:
        return build_chat_prompt(
            user_prompt,
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
