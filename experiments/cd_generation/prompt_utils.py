"""Shared prompt utilities used by generate_prompts.py and cd_generation/names_only.py."""
from __future__ import annotations

from typing import List, Optional

try:
    from .format import canonicalize_cd_prompt, default_format_hint_text
except Exception:
    from experiments.cd_generation.format import canonicalize_cd_prompt, default_format_hint_text

LARGE_GRAPH_EDGE_LIST_THRESHOLD = 100
PROMPT_WRAPPER_MODES = ("plain", "chat")
REASONING_GUIDANCE_MODES = ("staged", "concise", "none")


def get_topological_sort(adj_matrix: List[List[int]]) -> List[int]:
    """Returns a list of node indices in topological order (Causes -> Effects) via Kahn's algorithm."""
    n = len(adj_matrix)
    in_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                in_degree[j] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    queue.sort()
    topo_order = []

    in_degree_curr = list(in_degree)
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in range(n):
            if adj_matrix[u][v] == 1:
                in_degree_curr[v] -= 1
                if in_degree_curr[v] == 0:
                    queue.append(v)

    if len(topo_order) != n:
        return list(range(n))
    return topo_order


def normalize_variable_names(graph) -> List[str]:
    return [v.name for v in graph.variables]


def _build_output_contract_lines(
    *,
    output_edge_list: bool,
    require_think_answer_blocks: bool = True,
) -> List[str]:
    if output_edge_list:
        json_key = "edges"
        json_field_desc = '- "edges": [["source","target"], ...] using exact variable names.'
    else:
        json_key = "adjacency_matrix"
        json_field_desc = '- "adjacency_matrix": N x N 0/1 matrix in declared variable order.'

    return [
        "Output exactly: <think>...</think><answer>...</answer>.",
        "Keep <think> concise (minimal necessary reasoning only).",
        f'Inside <answer>, output exactly one JSON object with key "{json_key}".',
        json_field_desc,
        "No extra text before, between, or after the two blocks.",
        'The JSON in <answer> must start with "{" and end with "}".',
    ]


def resolve_wrapper_mode(
    *,
    wrapper_mode: Optional[str] = None,
) -> str:
    mode = str(wrapper_mode or "").strip().lower()
    if not mode:
        return "plain"
    if mode not in PROMPT_WRAPPER_MODES:
        raise ValueError(
            f"Unsupported wrapper_mode={wrapper_mode!r}. "
            f"Expected one of {PROMPT_WRAPPER_MODES}."
        )
    return mode


def resolve_reasoning_guidance(
    *,
    reasoning_guidance: Optional[str] = None,
) -> str:
    mode = str(reasoning_guidance or "").strip().lower()
    if not mode:
        return "staged"
    if mode not in REASONING_GUIDANCE_MODES:
        raise ValueError(
            f"Unsupported reasoning_guidance={reasoning_guidance!r}. "
            f"Expected one of {REASONING_GUIDANCE_MODES}."
        )
    return mode

def build_causal_graph_assumption_lines(
    *,
    include_intervention_assumption: bool = True,
) -> List[str]:
    lines = [
        "- The true graph is a DAG (no directed cycles).",
        "- Causal sufficiency holds (no unobserved confounders among these variables).",
    ]
    if include_intervention_assumption:
        lines.append("- Interventions are perfect do-interventions (surgical): do(X=v) cuts all incoming edges into X.")
    return lines


def build_causal_discovery_reminder_lines() -> List[str]:
    return [
        "- Use observational dependence/independence to suggest which pairs may be connected (skeleton).",
        "- Use interventions to orient edges: if Y changes under do(X=v), that supports X being an ancestor of Y.",
        "- Prefer directions that are consistent across all intervention blocks and keep the graph acyclic.",
    ]


def build_intervention_semantics_lines() -> List[str]:
    return [
        "- Each block or summary labeled do(X = v) contains samples where X is externally set to v.",
        "- In that regime, X's usual causes are disabled (incoming edges into X are cut).",
        "- Only descendants of X can change in distribution because of do(X=v); non-descendants should remain invariant up to sampling noise.",
        "- Treat values as categorical labels (do NOT assume numeric ordering of codes).",
    ]


def build_causal_discovery_method_lines(
    *,
    has_observational_data: bool,
    has_interventional_data: bool,
    require_think_answer_blocks: bool,
) -> List[str]:
    lines = [
        "\n--- METHOD ---",
        (
            "Follow these steps in the <think>...</think> block."
            if require_think_answer_blocks
            else "Follow these steps internally; do not output the reasoning."
        ),
    ]
    if has_observational_data and has_interventional_data:
        lines.extend([
            "1) Use observational evidence to narrow plausible adjacencies and conditional structure.",
            "2) Use interventional shifts to identify descendants and orient edges when possible.",
            "3) Resolve collider and direction choices using patterns consistent across all evidence while keeping the graph acyclic.",
            "4) Choose the sparsest DAG consistent with the data and any known edges.",
        ])
    elif has_observational_data:
        lines.extend([
            "1) Use observational evidence to narrow plausible adjacencies and conditional structure.",
            "2) Resolve collider and direction choices using patterns consistent with the data while keeping the graph acyclic.",
            "3) Choose the sparsest DAG consistent with the data and any known edges.",
        ])
    elif has_interventional_data:
        lines.extend([
            "1) Use interventional shifts to identify descendants and orient edges when possible.",
            "2) Combine evidence across intervention regimes into one acyclic graph.",
            "3) Choose the sparsest DAG consistent with the data and any known edges.",
        ])
    else:
        lines.extend([
            "1) Use the provided variable semantics and background causal knowledge to propose plausible direct causes.",
            "2) Prefer a sparse acyclic graph that best matches the available evidence.",
        ])
    return lines


def build_causal_discovery_reasoning_guidance_lines(
    *,
    reasoning_guidance: str,
    has_observational_data: bool,
    has_interventional_data: bool,
    require_think_answer_blocks: bool,
) -> List[str]:
    mode = resolve_reasoning_guidance(reasoning_guidance=reasoning_guidance)
    if mode == "none":
        return []
    if mode == "concise":
        return [
            "\n--- REASONING GUIDANCE ---",
            (
                "Reason however you want in the <think>...</think> block, but keep it concise and focused on the causal evidence."
                if require_think_answer_blocks
                else "Reason however you want internally, but keep it concise and focused on the causal evidence."
            ),
        ]
    return build_causal_discovery_method_lines(
        has_observational_data=has_observational_data,
        has_interventional_data=has_interventional_data,
        require_think_answer_blocks=require_think_answer_blocks,
    )


def render_prompt_text(
    prompt_text: str,
    *,
    task: str = "causal_discovery",
    wrapper_mode: str = "plain",
    append_format_hint: Optional[bool] = None,
    format_hint_text: Optional[str] = None,
    reasoning_guidance: str = "staged",
    prefill_think: Optional[bool] = None,
    prefill_answer: bool = False,
    think_text: str = "",
) -> str:
    mode = resolve_wrapper_mode(wrapper_mode=wrapper_mode)
    if append_format_hint is None:
        append_format_hint = (mode == "chat")
    resolved_format_hint_text = (
        default_format_hint_text(task)
        if format_hint_text is None
        else str(format_hint_text)
    )
    if prefill_think is None:
        prefill_think = (mode == "chat" and not prefill_answer)
    return canonicalize_cd_prompt(
        prompt_text,
        task=task,
        response_format="think_answer",
        wrap_system_prompt=(mode == "chat"),
        append_format_hint=bool(append_format_hint),
        format_hint_text=resolved_format_hint_text,
        reasoning_guidance=resolve_reasoning_guidance(reasoning_guidance=reasoning_guidance),
        prefill_think=bool(prefill_think),
        prefill_answer=bool(prefill_answer),
        think_text=think_text,
        strip_output_instructions=(mode == "chat"),
    )
