from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


_VAR_STATES_RE = re.compile(
    r"variable\s+([^\s\{]+)\s*\{[^\{\}]*\{\s*([^\}]*)\s*\}[^\}]*\}",
    flags=re.S,
)
_PROB_RE = re.compile(r"probability\s*\(\s*([^\)\|]+?)\s*(?:\|\s*([^\)]+))?\)\s*\{", flags=re.S)


def _parse_bif_summary(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    variables: list[str] = []
    max_states = 0
    for name, states_str in _VAR_STATES_RE.findall(text):
        name = name.strip()
        variables.append(name)
        states = [s.strip() for s in states_str.split(",") if s.strip()]
        if len(states) > max_states:
            max_states = len(states)

    node_set = set(variables)
    edges: set[tuple[str, str]] = set()
    for child, parents in _PROB_RE.findall(text):
        child = child.strip()
        if child not in node_set or not parents:
            continue
        for parent in parents.split(","):
            parent = parent.strip()
            if parent in node_set:
                edges.add((parent, child))

    return len(variables), len(edges), max_states


def build_table(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bif_path in sorted(root.rglob("*.bif")):
        num_nodes, num_edges, max_states = _parse_bif_summary(bif_path)
        rows.append(
            {
                "graph": bif_path.stem,
                "path": str(bif_path),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "max_node_states": max_states,
            }
        )
    rows.sort(key=lambda r: (int(r["num_nodes"]), str(r["graph"])))
    return rows


def _write_latex_table(rows: list[dict[str, object]], out_tex: Path) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"% Requires \usepackage{booktabs}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Graph & Nodes & Edges & Max states \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            f"{row['graph']} & {int(row['num_nodes'])} & {int(row['num_edges'])} & {int(row['max_node_states'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Summary of real-data Bayesian networks in \texttt{causal\_graphs/real\_data}.}")
    lines.append(r"\label{tab:real-data-graph-summary}")
    lines.append(r"\end{table}")
    lines.append("")
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("causal_graphs/real_data"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("experiments/out/real_data_graph_summary.csv"),
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("experiments/out/real_data_graph_summary.tex"),
    )
    args = parser.parse_args()

    rows = build_table(args.root)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["graph", "path", "num_nodes", "num_edges", "max_node_states"],
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_latex_table(rows, args.out_tex)

    print(f"[done] Wrote {args.out_csv} and {args.out_tex} ({len(rows)} graphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
