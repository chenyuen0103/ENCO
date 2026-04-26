#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --overwrite)
      echo "Refusing --overwrite: this runner is intended to reuse completed outputs and only run missing work." >&2
      exit 2
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--dry-run]" >&2
      exit 2
      ;;
  esac
done

CONFIGS=(
  "paper_slices/sachs_full_grid.json"
  "experiments/configs/mv_six_graph_semantic_floor.json"
)

for config in "${CONFIGS[@]}"; do
  echo
  echo "==> Running $config"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    python3 scripts/run_paper_slice.py --config "$config" --dry-run
  else
    python3 scripts/run_paper_slice.py --config "$config"
  fi
done
