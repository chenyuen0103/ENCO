#!/usr/bin/env python3
"""
Smoke test for "LLM tool use during inference".

This script runs a tiny OpenAI function-calling loop where the model may (optionally)
call a local Python "tool" to compute statistics from a CSV dataset. It is intended as
infrastructure you can adapt for LLM-in-the-loop causal discovery (e.g., LLM proposes
edits; Python tool computes a score/CI tests; repeat).

Examples (from repo root):
  # Dry-run (no API calls)
  python experiments/tool_use_smoke_test.py --dry-run

  # Real run (requires OPENAI_API_KEY and network access)
  python experiments/tool_use_smoke_test.py --model gpt-5-mini --with-tools
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_csv_path(path_str: str) -> Path:
    """
    Resolve a user-provided CSV path in a friendly way.

    Search order:
      1) as-is (absolute or relative to cwd)
      2) relative to repo root
      3) relative to this script's directory
    """
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p

    cand1 = (Path.cwd() / p).resolve()
    if cand1.exists():
        return cand1

    cand2 = (_repo_root() / p).resolve()
    if cand2.exists():
        return cand2

    cand3 = (Path(__file__).resolve().parent / p).resolve()
    if cand3.exists():
        return cand3

    # Best-effort: return the cwd-resolved path and let the caller error.
    return cand1


def _read_csv_discrete(path: Path, *, max_rows: Optional[int] = None) -> Tuple[List[str], List[List[float]]]:
    # Minimal CSV reader to avoid adding deps; expects header row.
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[float]] = []
        for i, r in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            if not r or all(not (c or "").strip() for c in r):
                continue
            rows.append([float(x) for x in r])
    return header, rows


def _corr_matrix(data: List[List[float]]) -> List[List[float]]:
    # Pure-Python Pearson correlation to avoid BLAS/OpenMP issues on some clusters.
    n = len(data)
    if n < 2:
        raise ValueError(f"Need at least 2 rows to compute correlation; got n={n}.")
    m = len(data[0])
    if any(len(r) != m for r in data):
        raise ValueError("Ragged rows: all rows must have the same number of columns.")

    # Column means
    means = [0.0] * m
    for r in data:
        for j, x in enumerate(r):
            means[j] += x
    means = [s / n for s in means]

    # Column std (sample, ddof=1)
    var = [0.0] * m
    for r in data:
        for j, x in enumerate(r):
            d = x - means[j]
            var[j] += d * d
    denom = float(n - 1)
    std = [(v / denom) ** 0.5 for v in var]

    corr: List[List[float]] = [[0.0] * m for _ in range(m)]
    for i in range(m):
        corr[i][i] = 1.0
        for j in range(i + 1, m):
            cov = 0.0
            for r in data:
                cov += (r[i] - means[i]) * (r[j] - means[j])
            cov /= denom
            if std[i] == 0.0 or std[j] == 0.0:
                c = 0.0
            else:
                c = cov / (std[i] * std[j])
            corr[i][j] = c
            corr[j][i] = c
    return corr


def compute_stats_tool(*, csv_path: str, max_rows: int = 2000) -> Dict[str, Any]:
    """
    Local "tool" callable by the model. Returns JSON-serializable stats.
    """
    p = _resolve_csv_path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    header, rows = _read_csv_discrete(p, max_rows=max_rows)
    return {
        "csv_path": str(p),
        "n_rows": len(rows),
        "n_cols": len(header),
        "columns": header,
        "corr": _corr_matrix(rows) if len(rows) >= 2 else None,
    }


@dataclass
class ToolCall:
    tool_call_id: str
    name: str
    arguments: Dict[str, Any]


def _extract_tool_calls(message: Any) -> List[ToolCall]:
    """
    Normalize OpenAI SDK message tool calls across minor schema differences.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []
    out: List[ToolCall] = []
    for tc in tool_calls:
        tc_id = getattr(tc, "id", None) or getattr(tc, "tool_call_id", None) or ""
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn is not None else getattr(tc, "name", None)
        args_raw = getattr(fn, "arguments", None) if fn is not None else getattr(tc, "arguments", None)
        if not tc_id or not name:
            continue
        try:
            arguments = json.loads(args_raw) if isinstance(args_raw, str) and args_raw.strip() else {}
        except json.JSONDecodeError:
            arguments = {}
        out.append(ToolCall(tool_call_id=tc_id, name=str(name), arguments=arguments))
    return out


def run_openai_tool_loop(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    allow_tools: bool,
    max_rounds: int,
    dry_run: bool,
    trace: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> str:
    if dry_run:
        return "[dry-run] (no model call)"

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenAI SDK not available. Install with: pip install openai"
        ) from e

    client = OpenAI()

    tools = []
    if allow_tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "compute_stats",
                    "description": "Compute basic dataset stats and a correlation matrix for a CSV file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "csv_path": {
                                "type": "string",
                                "description": "Path to a local CSV file with header and numeric entries.",
                            },
                            "max_rows": {
                                "type": "integer",
                                "description": "Max number of rows to read (for speed).",
                                "default": 2000,
                                "minimum": 2,
                            },
                        },
                        "required": ["csv_path"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    if trace is not None:
        trace(
            "start",
            {
                "model": model,
                "allow_tools": allow_tools,
                "max_rounds": max_rounds,
                "messages": messages,
                "tools": tools,
            },
        )

    for _round in range(max_rounds):
        if trace is not None:
            trace("request", {"round": _round, "messages": messages})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if allow_tools else None,
            tool_choice="auto" if allow_tools else "none",
        )

        msg = resp.choices[0].message
        if trace is not None:
            trace(
                "response",
                {
                    "round": _round,
                    "assistant_message": {
                        "role": getattr(msg, "role", "assistant"),
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.tool_call_id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in _extract_tool_calls(msg)
                        ],
                    },
                },
            )
        tool_calls = _extract_tool_calls(msg)
        if not tool_calls:
            # Keep the conversation consistent even on the final turn.
            messages.append({"role": "assistant", "content": (msg.content or "")})
            return (msg.content or "").strip()

        # IMPORTANT: In the Chat Completions API, any "tool" role message must be a
        # response to a preceding assistant message that contains "tool_calls".
        # So we must append the assistant message (with tool_calls) to the history
        # before we append tool responses.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # If there are tool calls, execute them and continue the loop.
        for tc in tool_calls:
            if tc.name != "compute_stats":
                tool_out = {"error": f"Unknown tool: {tc.name}"}
            else:
                try:
                    tool_out = compute_stats_tool(**tc.arguments)
                except Exception as e:
                    tool_out = {"error": f"{type(e).__name__}: {e}"}

            if trace is not None:
                trace(
                    "tool_result",
                    {
                        "round": _round,
                        "tool_name": tc.name,
                        "tool_call_id": tc.tool_call_id,
                        "tool_arguments": tc.arguments,
                        "tool_output": tool_out,
                    },
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.tool_call_id,
                    "content": json.dumps(tool_out, ensure_ascii=False),
                }
            )

    raise TimeoutError(f"Tool loop exceeded max_rounds={max_rounds} without producing a final answer.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument(
        "--data-csv",
        default="synthetic_obs.csv",
        help="Dataset CSV (local path; relative paths are resolved against repo root).",
    )
    ap.add_argument("--max-rows", type=int, default=2000, help="Max rows the tool can read.")
    ap.add_argument("--with-tools", action="store_true", help="Allow the model to call compute_stats().")
    ap.add_argument(
        "--summary-in-prompt",
        action="store_true",
        help="Compute summary stats locally and embed them directly in the prompt (disables tool calling).",
    )
    ap.add_argument(
        "--corr-round",
        type=int,
        default=6,
        help="Rounding precision for correlation values when embedding in the prompt.",
    )
    ap.add_argument("--max-rounds", type=int, default=8, help="Max tool-call rounds.")
    ap.add_argument("--dry-run", action="store_true", help="Do not call any model API.")
    ap.add_argument(
        "--print-trace",
        action="store_true",
        help="Print the full prompt/response/tool trace to stderr.",
    )
    ap.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the generated prompt (system+user) and exit without querying any API.",
    )
    ap.add_argument(
        "--prompt-format",
        choices=["text", "json"],
        default="text",
        help="How to print the prompt when --print-prompt is set.",
    )
    ap.add_argument(
        "--trace-jsonl",
        default=None,
        help="Optional path to write a JSONL trace (events: start/request/response/tool_result).",
    )
    args = ap.parse_args()

    csv_path = _resolve_csv_path(args.data_csv)

    data_csv = str(csv_path.resolve())

    trace_file = Path(args.trace_jsonl).resolve() if args.trace_jsonl else None
    trace_fh = trace_file.open("w", encoding="utf-8") if trace_file is not None else None

    def trace(event: str, payload: Dict[str, Any]) -> None:
        rec = {"event": event, **payload}
        if args.print_trace:
            print(json.dumps(rec, ensure_ascii=False), file=sys.stderr)
        if trace_fh is not None:
            trace_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            trace_fh.flush()

    if args.summary_in_prompt:
        stats = compute_stats_tool(csv_path=str(csv_path), max_rows=int(args.max_rows))
        corr = stats["corr"]
        if isinstance(corr, list):
            corr = [
                [round(float(x), int(args.corr_round)) for x in row]
                for row in corr
            ]
        summary_obj = {"columns": stats["columns"], "corr": corr}
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant with strong causal discovery skills."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here are summary statistics computed from a dataset.\n"
                    "Use them to answer.\n\n"
                    "SUMMARY_JSON:\n"
                    f"{json.dumps(summary_obj, ensure_ascii=False)}\n\n"
                    "Return ONLY valid JSON with keys: columns, corr (exactly as in SUMMARY_JSON)."
                ),
            },
        ]
        allow_tools = False
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "When tools are available, use them to compute statistics rather than guessing."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Compute the Pearson correlation matrix for the dataset at:\n"
                    f"{data_csv}\n\n"
                    "Return ONLY valid JSON with keys: columns, corr.\n"
                    f"If tools are available, call compute_stats(csv_path=..., max_rows={int(args.max_rows)})."
                ),
            },
        ]
        allow_tools = bool(args.with_tools)

    # Offline modes (no API calls)
    if args.print_prompt:
        if args.prompt_format == "json":
            print(json.dumps(messages, ensure_ascii=False, indent=2))
        else:
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                print(f"--- {role.upper()} ---")
                print(content)
                print()
        return 0

    # In dry-run mode, still run the local tool so you can verify the "tool side"
    # works even without API keys/network access.
    if args.dry_run:
        stats = compute_stats_tool(csv_path=str(csv_path), max_rows=int(args.max_rows))
        out = {"columns": stats["columns"], "corr": stats["corr"]}
        print(json.dumps(out, ensure_ascii=False))
        return 0

    try:
        text = run_openai_tool_loop(
            model=args.model,
            messages=messages,
            allow_tools=allow_tools,
            max_rounds=int(args.max_rounds),
            dry_run=bool(args.dry_run),
            trace=trace,
        )
        print(text)
        return 0
    finally:
        if trace_fh is not None:
            trace_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
