#!/usr/bin/env python3
"""Consolidate response artifacts into scripts/responses/<graph>/.

The MICAD paper tooling treats scripts/responses as the canonical response
root. This utility copies response CSVs and their evaluation sidecars from
legacy locations such as experiments/responses or benchmark_runs/**/responses
into that canonical layout.

By default the script is a dry run. Pass --apply to copy files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOTS = [
    Path("experiments/responses"),
    Path("experiments/experiments/responses"),
    Path("responses"),
    Path("benchmark_runs"),
]
DEFAULT_DEST_ROOT = Path("scripts/responses")
DEFAULT_MANIFEST = Path("experiments/out/response_consolidation_manifest.csv")
KNOWN_GRAPH_NAMES = {
    "alarm",
    "asia",
    "cancer",
    "child",
    "diabetes",
    "earthquake",
    "pigs",
    "sachs",
    "sachs_old",
}


@dataclass(frozen=True)
class Artifact:
    path: Path
    graph: str
    source_root: Path
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Legacy response root to scan. Repeatable. Defaults to "
            "experiments/responses, experiments/experiments/responses, responses, and benchmark_runs."
        ),
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help="Canonical response root. Defaults to scripts/responses.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest recording every copy/skip/conflict decision.",
    )
    parser.add_argument("--graphs", nargs="*", default=None, help="Optional graph-name filter.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy files. Without this flag, only write a dry-run manifest.",
    )
    parser.add_argument(
        "--conflict",
        choices=["report", "suffix", "overwrite"],
        default="report",
        help=(
            "What to do when destination exists with different bytes. "
            "Default: report conflict and leave both files untouched."
        ),
    )
    parser.add_argument(
        "--include-aggregates",
        action="store_true",
        help="Also copy eval_summary.csv and <graph>_summary.csv files. Usually regenerate these after consolidation.",
    )
    parser.add_argument(
        "--include-plots",
        action="store_true",
        help="Also copy response diagnostic plot PDFs. Default keeps CSV/JSON artifacts only.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_kind(path: Path, include_aggregates: bool, include_plots: bool) -> str | None:
    name = path.name
    if name.startswith(("responses_", "predictions_")):
        if name.endswith(".per_row.csv"):
            return "per_row_csv"
        if name.endswith(".summary.json"):
            return "summary_json"
        if ".consensus_tau" in name and name.endswith(".json"):
            return "consensus_json"
        if name.endswith(".csv"):
            return "primary_csv"
        if include_plots and name.endswith(".pdf"):
            return "diagnostic_plot"
    if include_aggregates and (name == "eval_summary.csv" or name.endswith("_summary.csv")):
        return "aggregate_csv"
    return None


def infer_graph(path: Path, source_root: Path) -> str | None:
    rel_parts = path.relative_to(source_root).parts
    if not rel_parts:
        return None

    if source_root.name == "responses" and len(rel_parts) > 1:
        return rel_parts[0]

    if "responses" in rel_parts:
        idx = rel_parts.index("responses")
        if idx + 1 < len(rel_parts):
            candidate = rel_parts[idx + 1]
            if candidate in KNOWN_GRAPH_NAMES:
                return candidate

    for part in rel_parts[:-1]:
        if part in KNOWN_GRAPH_NAMES:
            return part

    parent = path.parent.name
    return parent if parent in KNOWN_GRAPH_NAMES else None


def iter_artifacts(
    source_roots: Iterable[Path],
    graphs: set[str] | None,
    include_aggregates: bool,
    include_plots: bool,
) -> Iterable[Artifact]:
    for root in source_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            kind = artifact_kind(path, include_aggregates, include_plots)
            if kind is None:
                continue
            graph = infer_graph(path, root)
            if graph is None:
                yield Artifact(path=path, graph="", source_root=root, kind="unknown_graph")
                continue
            if graphs is not None and graph not in graphs:
                continue
            yield Artifact(path=path, graph=graph, source_root=root, kind=kind)


def conflict_suffix(source_root: Path) -> str:
    rel = source_root.relative_to(REPO_ROOT) if source_root.is_relative_to(REPO_ROOT) else source_root
    return "__from_" + "_".join(str(rel).split("/")).replace(".", "_")


def suffixed_dest(dest: Path, source_root: Path) -> Path:
    suffix = conflict_suffix(source_root)
    dest = dest.parent.parent / "_conflicts" / dest.parent.name / dest.name
    if dest.name.endswith(".summary.json"):
        return dest.with_name(dest.name.removesuffix(".summary.json") + suffix + ".summary.json")
    if dest.name.endswith(".per_row.csv"):
        return dest.with_name(dest.name.removesuffix(".per_row.csv") + suffix + ".per_row.csv")
    if ".consensus_tau" in dest.name and dest.name.endswith(".json"):
        stem, ext = dest.name.rsplit(".", 1)
        return dest.with_name(f"{stem}{suffix}.{ext}")
    return dest.with_name(dest.stem + suffix + dest.suffix)


def copy_artifact(artifact: Artifact, dest_root: Path, apply: bool, conflict: str) -> dict[str, str]:
    source = artifact.path
    if not artifact.graph:
        return row(artifact, "", "skip_unknown_graph", "could not infer graph directory")

    dest = dest_root / artifact.graph / source.name
    source_hash = sha256(source)
    final_dest = dest
    action = "dry_run_copy"
    reason = "destination missing"

    if dest.exists():
        dest_hash = sha256(dest)
        if dest_hash == source_hash:
            return row(artifact, str(dest), "skip_same", "destination already has identical bytes", source_hash)
        if conflict == "report":
            return row(artifact, str(dest), "conflict", "destination exists with different bytes", source_hash)
        if conflict == "suffix":
            final_dest = suffixed_dest(dest, artifact.source_root)
            action = "dry_run_copy_suffixed"
            reason = "destination conflict; using source-root suffix"
        elif conflict == "overwrite":
            action = "dry_run_overwrite"
            reason = "destination conflict; overwrite requested"

    if apply:
        final_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, final_dest)
        action = action.replace("dry_run_", "")
        if action == "dry_run_copy":
            action = "copy"

    return row(artifact, str(final_dest), action, reason, source_hash)


def row(artifact: Artifact, dest: str, action: str, reason: str, source_hash: str = "") -> dict[str, str]:
    source_root = (
        str(artifact.source_root.relative_to(REPO_ROOT))
        if artifact.source_root.is_relative_to(REPO_ROOT)
        else str(artifact.source_root)
    )
    source = str(artifact.path.relative_to(REPO_ROOT)) if artifact.path.is_relative_to(REPO_ROOT) else str(artifact.path)
    return {
        "action": action,
        "reason": reason,
        "graph": artifact.graph,
        "kind": artifact.kind,
        "source_root": source_root,
        "source_path": source,
        "dest_path": dest,
        "source_sha256": source_hash,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["action", "reason", "graph", "kind", "source_root", "source_path", "dest_path", "source_sha256"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_roots = [resolve(path) for path in (args.source_root or DEFAULT_SOURCE_ROOTS)]
    dest_root = resolve(args.dest_root)
    manifest = resolve(args.manifest)
    graph_filter = {graph.lower() for graph in args.graphs} if args.graphs else None

    rows: list[dict[str, str]] = []
    for artifact in iter_artifacts(source_roots, graph_filter, args.include_aggregates, args.include_plots):
        rows.append(copy_artifact(artifact, dest_root, args.apply, args.conflict))

    write_manifest(manifest, rows)
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["action"]] = counts.get(item["action"], 0) + 1
    summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "no artifacts"
    mode = "apply" if args.apply else "dry-run"
    print(f"[{mode}] {summary}")
    print(f"[write] {manifest}")


if __name__ == "__main__":
    main()
