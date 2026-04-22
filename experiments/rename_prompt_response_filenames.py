#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path


WRAPPER_MARKERS = {"thinktags", "wrapchat", "wrapplain"}
FORMAT_HINT_MARKERS = {"cothint", "fmthint"}


@dataclass(frozen=True)
class RenameOp:
    source: Path
    target: Path


@dataclass(frozen=True)
class ResolvedCollision:
    target: Path
    keep_source: Path | None
    drop_sources: tuple[Path, ...]
    target_already_matches: bool


def normalize_filename(name: str) -> str:
    new = str(name)

    # Remove deprecated output-format marker.
    new = new.replace("respthink_answer", "")

    # Canonical prompt-style naming.
    new = new.replace("summary_joint", "summary")
    new = new.replace("summary_join", "summary")

    parts = new.split("_")
    out: list[str] = []
    inserted_wrapchat = False
    inserted_format_hint = False
    for part in parts:
        if not part:
            continue
        if part in WRAPPER_MARKERS:
            if part in {"thinktags", "wrapchat"} and not inserted_wrapchat:
                out.append("wrapchat")
                inserted_wrapchat = True
            continue
        if part in FORMAT_HINT_MARKERS:
            if not inserted_format_hint:
                out.append("fmthint")
                inserted_format_hint = True
            continue
        out.append(part)

    new = "_".join(out)
    while "__" in new:
        new = new.replace("__", "_")
    new = new.replace("_.", ".")
    return new.strip("_")


def iter_candidate_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def build_plan(roots: list[Path]) -> tuple[list[RenameOp], dict[Path, list[Path]]]:
    ops: list[RenameOp] = []
    collisions: dict[Path, list[Path]] = defaultdict(list)
    for root in roots:
        for source in iter_candidate_files(root):
            target_name = normalize_filename(source.name)
            if target_name == source.name:
                continue
            target = source.with_name(target_name)
            ops.append(RenameOp(source=source, target=target))
            collisions[target].append(source)
    return ops, {target: sources for target, sources in collisions.items() if len(sources) > 1}


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_identical_collisions(
    collisions: dict[Path, list[Path]],
) -> tuple[dict[Path, ResolvedCollision], dict[Path, list[Path]]]:
    resolved: dict[Path, ResolvedCollision] = {}
    unresolved: dict[Path, list[Path]] = {}

    for target, sources in collisions.items():
        digest_groups: dict[str, list[Path]] = defaultdict(list)
        for source in sorted(sources):
            digest_groups[file_digest(source)].append(source)

        if len(digest_groups) != 1:
            unresolved[target] = sorted(sources)
            continue

        digest = next(iter(digest_groups))
        ordered_sources = sorted(sources)
        target_already_matches = target.exists() and file_digest(target) == digest

        if target.exists() and not target_already_matches:
            unresolved[target] = ordered_sources
            continue

        if target_already_matches:
            resolved[target] = ResolvedCollision(
                target=target,
                keep_source=None,
                drop_sources=tuple(ordered_sources),
                target_already_matches=True,
            )
        else:
            resolved[target] = ResolvedCollision(
                target=target,
                keep_source=ordered_sources[0],
                drop_sources=tuple(ordered_sources[1:]),
                target_already_matches=False,
            )

    return resolved, unresolved


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize legacy prompt/response filenames under experiments/prompts "
            "and experiments/responses."
        )
    )
    ap.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[
            Path("experiments/responses"),
            Path("experiments/prompts"),
        ],
        help="Root directories to scan recursively.",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renames. Default is dry-run.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing target file.",
    )
    ap.add_argument(
        "--dedupe-identical-collisions",
        action="store_true",
        help=(
            "Automatically resolve collision groups when all colliding files have identical contents. "
            "Keeps one canonical file and drops the redundant duplicates."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    roots = [p.resolve() for p in args.roots]
    missing = [p for p in roots if not p.exists()]
    if missing:
        missing_text = ", ".join(str(p) for p in missing)
        raise SystemExit(f"Missing root(s): {missing_text}")

    ops, collisions = build_plan(roots)
    skipped = 0
    executed = 0
    deleted = 0

    resolved_collisions: dict[Path, ResolvedCollision] = {}
    if args.dedupe_identical_collisions and collisions:
        resolved_collisions, collisions = resolve_identical_collisions(collisions)

    collision_sources = {source for sources in collisions.values() for source in sources}
    resolved_keep_sources = {
        info.keep_source for info in resolved_collisions.values() if info.keep_source is not None
    }
    resolved_drop_sources = {
        source for info in resolved_collisions.values() for source in info.drop_sources
    }
    if collisions:
        print("[collisions]")
        for target, sources in sorted(collisions.items()):
            print(f"  target={target}")
            for source in sorted(sources):
                print(f"    source={source}")
    if resolved_collisions:
        print("[dedupe-identical-collisions]")
        for target, info in sorted(resolved_collisions.items()):
            if info.target_already_matches:
                print(f"  target={target} already exists with identical content")
            else:
                print(f"  target={target} keep={info.keep_source}")
            for source in info.drop_sources:
                print(f"    drop={source}")

    print(f"[plan] {len(ops)} file(s) need renaming across {len(roots)} root(s).")
    for op in ops:
        if op.source in resolved_drop_sources:
            label = "delete" if args.apply else "dry-run-delete"
            print(f"[{label}] {op.source}")
            if args.apply:
                op.source.unlink()
                deleted += 1
            continue
        if op.source in resolved_keep_sources:
            info = resolved_collisions[op.target]
            label = "rename" if args.apply else "dry-run"
            print(f"[{label}] {op.source} -> {op.target}")
            if args.apply:
                if op.target.exists() and op.target != op.source:
                    if not args.overwrite:
                        raise SystemExit(
                            f"Refusing to overwrite existing target without --overwrite: {op.target}"
                        )
                    op.target.unlink()
                op.source.rename(op.target)
                executed += 1
            continue
        if op.source in collision_sources:
            print(f"[skip-collision] {op.source} -> {op.target}")
            skipped += 1
            continue
        if op.target.exists() and op.target != op.source and not args.overwrite:
            print(f"[skip-exists] {op.source} -> {op.target}")
            skipped += 1
            continue

        label = "rename" if args.apply else "dry-run"
        print(f"[{label}] {op.source} -> {op.target}")
        if args.apply:
            if op.target.exists() and op.target != op.source:
                op.target.unlink()
            op.source.rename(op.target)
            executed += 1

    if args.apply:
        print(
            f"[done] renamed={executed} deleted={deleted} skipped={skipped} "
            f"collisions={len(collisions)} resolved_collisions={len(resolved_collisions)}"
        )
    else:
        print(
            f"[done] planned={len(ops) - skipped} skipped={skipped} "
            f"collisions={len(collisions)} resolved_collisions={len(resolved_collisions)}"
        )


if __name__ == "__main__":
    main()
