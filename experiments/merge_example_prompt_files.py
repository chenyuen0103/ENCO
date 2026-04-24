#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


META_MARKER = "=== META ==="
PROMPT_MARKER = "=== PROMPT ==="


@dataclass(frozen=True)
class PromptSource:
    path: Path
    dataset: str | None
    base_name: str
    model: str | None
    prompt_text: str
    prompt_sha256: str


def _parse_example_prompt(path: Path) -> PromptSource | None:
    text = path.read_text(encoding="utf-8")
    if META_MARKER not in text or PROMPT_MARKER not in text:
        return None

    _, rest = text.split(META_MARKER, 1)
    meta_text, prompt_text = rest.split(PROMPT_MARKER, 1)
    meta = json.loads(meta_text.strip())

    # Skip files emitted by this merge tool to keep reruns idempotent.
    if isinstance(meta, dict) and meta.get("merged_from_files"):
        return None

    if not isinstance(meta, dict):
        return None
    base_name = meta.get("base_name")
    if not isinstance(base_name, str) or not base_name:
        return None

    dataset = meta.get("dataset")
    if not isinstance(dataset, str):
        dataset = None

    model = meta.get("model")
    if not isinstance(model, str):
        model = None

    normalized_prompt = prompt_text.lstrip("\n").rstrip("\n")
    return PromptSource(
        path=path,
        dataset=dataset,
        base_name=base_name,
        model=model,
        prompt_text=normalized_prompt,
        prompt_sha256=hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest(),
    )


def _default_output_path(example_dir: Path, base_name: str, prompt_sha256: str, variant_count: int) -> Path:
    stem = base_name
    if variant_count > 1:
        stem = f"{stem}_sha{prompt_sha256[:10]}"
    return example_dir / f"{stem}_example_prompt.txt"


def _build_output_text(
    *,
    dataset: str | None,
    base_name: str,
    prompt_sha256: str,
    sources: list[PromptSource],
) -> str:
    models = sorted({src.model for src in sources if src.model})
    meta: dict[str, Any] = {
        "dataset": dataset,
        "base_name": base_name,
        "prompt_sha256": prompt_sha256,
        "source_count": len(sources),
        "merged_from_files": [src.path.name for src in sorted(sources, key=lambda s: s.path.name)],
    }
    if models:
        meta["models"] = models
    return (
        f"{META_MARKER}\n"
        f"{json.dumps(meta, ensure_ascii=False, indent=2)}\n\n"
        f"{PROMPT_MARKER}\n"
        f"{sources[0].prompt_text}\n"
    )


def merge_example_prompts(example_dir: Path, *, dry_run: bool) -> tuple[int, int]:
    by_base_and_hash: dict[str, dict[str, list[PromptSource]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(example_dir.glob("*_example_prompt.txt")):
        parsed = _parse_example_prompt(path)
        if parsed is None:
            continue
        by_base_and_hash[parsed.base_name][parsed.prompt_sha256].append(parsed)

    written = 0
    groups = 0
    for base_name in sorted(by_base_and_hash):
        prompt_groups = by_base_and_hash[base_name]
        variant_count = len(prompt_groups)
        for prompt_sha256, sources in sorted(prompt_groups.items()):
            groups += 1
            datasets = sorted({src.dataset for src in sources if src.dataset})
            dataset = datasets[0] if len(datasets) == 1 else None
            out_path = _default_output_path(example_dir, base_name, prompt_sha256, variant_count)
            output_text = _build_output_text(
                dataset=dataset,
                base_name=base_name,
                prompt_sha256=prompt_sha256,
                sources=sources,
            )
            if dry_run:
                print(
                    json.dumps(
                        {
                            "base_name": base_name,
                            "prompt_sha256": prompt_sha256,
                            "variants_for_base": variant_count,
                            "sources": [src.path.name for src in sources],
                            "out_path": str(out_path),
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            out_path.write_text(output_text, encoding="utf-8")
            written += 1

    return written, groups


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge model-suffixed example prompt files into canonical config-based files. "
            "If same-base files contain different prompt bodies, keep one merged file per unique prompt body."
        )
    )
    ap.add_argument(
        "example_dir",
        nargs="?",
        default="experiments/prompts/sachs/example_prompts",
        help="Directory containing *_example_prompt.txt files.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing files.")
    args = ap.parse_args()

    example_dir = Path(args.example_dir)
    if not example_dir.exists():
        raise SystemExit(f"Directory not found: {example_dir}")

    written, groups = merge_example_prompts(example_dir, dry_run=bool(args.dry_run))
    if args.dry_run:
        print(f"[dry-run] grouped prompt variants: {groups}")
    else:
        print(f"[done] wrote {written} merged prompt files from {groups} grouped prompt variants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
