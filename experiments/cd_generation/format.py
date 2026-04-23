from __future__ import annotations

import re


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the solution and then provides the final answer."
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

DEFAULT_FORMAT_HINT_TEXT_CONCISE = (
    "Reason however you want inside <think>, but keep it concise. "
    "Then output: <answer>{\"adjacency_matrix\": [[0,1,...],[0,0,...],...]}</answer> "
    "where the matrix is N×N with integer entries 0 or 1 in VARIABLES order, "
    "and [i][j]=1 means variable i directly causes variable j."
)

DEFAULT_FORMAT_HINT_TEXT_NONE = (
    "Output exactly: <think>...</think><answer>{\"adjacency_matrix\": [[0,1,...],[0,0,...],...]}</answer> "
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
_OUTPUT_INSTRUCTIONS_RE = re.compile(
    r"\n--- OUTPUT INSTRUCTIONS ---.*?(?=\n(?:---|assistant\b|Formatting requirement:)|\Z)",
    re.DOTALL,
)


def default_short_think_text(task: str, reasoning_guidance: str = "staged") -> str:
    if task == "cd_descendants":
        return "I compare the intervention shifts and keep only downstream variables."
    if reasoning_guidance == "concise":
        return "I infer the causal graph from the observational and interventional evidence."
    if reasoning_guidance == "none":
        return ""
    return (
        "Stage 1 (Skeleton):\n"
        "Stage 2 (V-structures):\n"
        "Stage 3 (Orientation):"
    )


def system_prompt_for_task(task: str, response_format: str = "think_answer") -> str:
    if task == "cd_descendants":
        return DESCENDANT_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def default_format_hint_text(
    task: str,
    response_format: str = "think_answer",
    reasoning_guidance: str = "staged",
) -> str:
    if task == "cd_descendants":
        return DEFAULT_DESCENDANT_FORMAT_HINT_TEXT
    if reasoning_guidance == "concise":
        return DEFAULT_FORMAT_HINT_TEXT_CONCISE
    if reasoning_guidance == "none":
        return DEFAULT_FORMAT_HINT_TEXT_NONE
    return DEFAULT_FORMAT_HINT_TEXT


def resolve_format_hint_text(
    task: str,
    format_hint_text: str,
    response_format: str = "think_answer",
    reasoning_guidance: str = "staged",
) -> str:
    hint = str(format_hint_text or "").strip()
    if not hint:
        return default_format_hint_text(
            task,
            response_format=response_format,
            reasoning_guidance=reasoning_guidance,
        )
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


def _extract_formatting_requirement(text: str) -> str:
    s = str(text or "").strip()
    for marker in ("\n\nFormatting requirement:", "\nFormatting requirement:"):
        if marker in s:
            return s.split(marker, 1)[1].strip()
    if s.startswith("Formatting requirement:"):
        return s.split(":", 1)[1].strip()
    return ""


def _strip_output_instructions_block(text: str) -> str:
    s = str(text or "").rstrip()
    return _OUTPUT_INSTRUCTIONS_RE.sub("", s).rstrip()


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
    task: str = "causal_discovery",
    response_format: str = "think_answer",
    prefill_think: bool,
    prefill_answer: bool = False,
    think_text: str = "",
) -> str:
    prompt = (
        f"system\n{system_prompt_for_task(task, response_format=response_format)}\n"
        f"user\n{str(user_prompt or '').rstrip()}\nassistant\n"
    )
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
    response_format: str = "think_answer",
    wrap_system_prompt: bool,
    append_format_hint: bool,
    format_hint_text: str,
    reasoning_guidance: str = "staged",
    prefill_think: bool,
    prefill_answer: bool = False,
    think_text: str = "",
    strip_output_instructions: bool = False,
) -> str:
    if prefill_think and prefill_answer:
        raise ValueError("prefill_think and prefill_answer cannot both be true")
    prompt = str(raw_prompt or "").strip()
    if not prompt:
        return ""

    if strip_output_instructions:
        prompt = _strip_output_instructions_block(prompt)

    if looks_like_chat_prompt(prompt):
        prompt = update_prompt_to_current_format(
            prompt,
            task=task,
            response_format=response_format,
            append_format_hint=append_format_hint,
            format_hint_text=format_hint_text,
            reasoning_guidance=reasoning_guidance,
        )
        if prefill_answer:
            return ensure_assistant_answer_prefill(prompt, think_text)
        if prefill_think:
            return ensure_assistant_think_prefill(prompt)
        return prompt.rstrip() + "\n"

    user_prompt = prompt
    if append_format_hint:
        user_prompt = append_format_hint_to_user_prompt(
            user_prompt,
            resolve_format_hint_text(
                task,
                format_hint_text,
                response_format=response_format,
                reasoning_guidance=reasoning_guidance,
            ),
        )
    if wrap_system_prompt:
        return build_chat_prompt(
            user_prompt,
            task=task,
            response_format=response_format,
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


def update_prompt_to_current_format(
    prompt_text: str,
    *,
    task: str = "causal_discovery",
    response_format: str = "think_answer",
    append_format_hint: bool = True,
    format_hint_text: str = "",
    reasoning_guidance: str = "staged",
) -> str:
    prompt = str(prompt_text or "").strip()
    if not prompt or not looks_like_chat_prompt(prompt):
        return prompt

    user_prompt = _unwrap_chat_user_prompt(prompt)
    existing_hint = _extract_formatting_requirement(user_prompt)
    user_prompt = _strip_output_instructions_block(_strip_formatting_requirement(user_prompt))
    if append_format_hint or existing_hint:
        user_prompt = append_format_hint_to_user_prompt(
            user_prompt,
            existing_hint
            or resolve_format_hint_text(
                task,
                format_hint_text,
                response_format=response_format,
                reasoning_guidance=reasoning_guidance,
            ),
        )
    return build_chat_prompt(
        user_prompt,
        task=task,
        response_format=response_format,
        prefill_think=False,
        prefill_answer=False,
    )


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
