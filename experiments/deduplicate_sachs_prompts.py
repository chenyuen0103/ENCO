#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

from merge_example_prompt_files import _parse_example_prompt, merge_example_prompts


def _delete_legacy_example_prompts(example_dir: Path, *, dry_run: bool) -> int:
    deleted = 0
    for path in sorted(example_dir.glob("*_example_prompt.txt")):
        parsed = _parse_example_prompt(path)
        if parsed is None:
            continue
        if dry_run:
            print(f"[dry-run] delete legacy example prompt: {path}")
        else:
            path.unlink()
        deleted += 1
    return deleted


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hardlink_exact_duplicates(root: Path, *, dry_run: bool) -> tuple[int, int]:
    by_sig: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        sig = (int(stat.st_size), _file_sha256(path))
        by_sig[sig].append(path)

    groups = 0
    relinked = 0
    for (_, _), paths in sorted(by_sig.items(), key=lambda item: (str(item[1][0]), len(item[1]))):
        if len(paths) <= 1:
            continue
        groups += 1
        canonical = sorted(paths)[0]
        canonical_stat = canonical.stat()
        for dup in sorted(paths)[1:]:
            dup_stat = dup.stat()
            if dup_stat.st_ino == canonical_stat.st_ino and dup_stat.st_dev == canonical_stat.st_dev:
                continue
            if dry_run:
                print(f"[dry-run] hardlink {dup} -> {canonical}")
            else:
                dup.unlink()
                os.link(canonical, dup)
            relinked += 1
    return relinked, groups


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Deduplicate Sachs prompt artifacts by merging example prompts to canonical config-based files, "
            "removing legacy model-suffixed example prompts, and hardlinking any remaining byte-identical files."
        )
    )
    ap.add_argument(
        "--root",
        default="experiments/prompts/sachs",
        help="Root Sachs prompt directory to deduplicate.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Prompt root not found: {root}")

    example_dir = root / "example_prompts"
    if example_dir.exists():
        merge_example_prompts(example_dir, dry_run=bool(args.dry_run))
        deleted = _delete_legacy_example_prompts(example_dir, dry_run=bool(args.dry_run))
    else:
        deleted = 0

    relinked, groups = _hardlink_exact_duplicates(root, dry_run=bool(args.dry_run))
    mode = "dry-run" if args.dry_run else "done"
    print(
        f"[{mode}] deleted_legacy_example_prompts={deleted} "
        f"hardlinked_files={relinked} duplicate_groups={groups}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
