#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    dedup: dict[int, dict] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = int(row.get("global_step", -1))
            if step < 0:
                continue
            dedup[step] = row
    return [dedup[step] for step in sorted(dedup)]


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)
    out: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        denom = min(idx + 1, window)
        out.append(running / denom)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot train/eval loss curves from JSONL trainer logs.")
    ap.add_argument("metrics_jsonl", type=Path, help="Path to train_metrics.jsonl")
    ap.add_argument("--output", type=Path, default=None, help="Output image path (default: <jsonl>.png)")
    ap.add_argument("--smoothing-window", type=int, default=5, help="Moving-average window for train loss.")
    args = ap.parse_args()

    rows = _load_rows(args.metrics_jsonl)
    if not rows:
        raise SystemExit(f"no metric rows found in {args.metrics_jsonl}")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to plot metrics") from exc

    steps = [int(row["global_step"]) for row in rows]
    train_points = [(int(row["global_step"]), float(row["loss"])) for row in rows if "loss" in row]
    eval_points = [(int(row["global_step"]), float(row["eval_loss"])) for row in rows if "eval_loss" in row]

    output_path = args.output or args.metrics_jsonl.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    if train_points:
        train_steps = [step for step, _ in train_points]
        train_loss = [loss for _, loss in train_points]
        ax.plot(train_steps, train_loss, color="#9aa5b1", alpha=0.45, linewidth=1.1, label="train loss")
        ax.plot(
            train_steps,
            _moving_average(train_loss, max(1, int(args.smoothing_window))),
            color="#0b7285",
            linewidth=2.0,
            label=f"train loss (ma={max(1, int(args.smoothing_window))})",
        )
    if eval_points:
        ax.plot(
            [step for step, _ in eval_points],
            [loss for _, loss in eval_points],
            color="#c92a2a",
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            label="eval loss",
        )

    ax.set_title(args.metrics_jsonl.parent.name)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_xlim(min(steps), max(steps))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
