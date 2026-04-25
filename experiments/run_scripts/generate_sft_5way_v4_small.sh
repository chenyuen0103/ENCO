#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_FILE="${CONFIG_FILE:-experiments/configs/sft_source_v4_small.json}"
SOURCE_CSV="${SOURCE_CSV:-experiments/data/sft_source_v4_small.csv}"
OUT_JSONL="${OUT_JSONL:-experiments/data/format_sft_5way_v4_small.jsonl}"
SHARD_DIR="${SHARD_DIR:-experiments/data/sft_5way_v4_small_shards}"
ROWS_PER_SOURCE="${ROWS_PER_SOURCE:-999999}"
BASE_SEED="${BASE_SEED:-42}"
WRAPPER_MODE="${WRAPPER_MODE:-chat}"

TEACHER_PROVIDER="${TEACHER_PROVIDER:-openai_compatible}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:8001/v1}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-768}"
TEACHER_MAX_REVISIONS="${TEACHER_MAX_REVISIONS:-1}"
TEACHER_FALLBACK_TARGET="${TEACHER_FALLBACK_TARGET:-none}"
TEACHER_REASONING_EFFORT="${TEACHER_REASONING_EFFORT:-}"
RESUME_TEACHER_SHARDS="${RESUME_TEACHER_SHARDS:-0}"

mkdir -p "$(dirname "${SOURCE_CSV}")" "${SHARD_DIR}" "$(dirname "${OUT_JSONL}")"

echo "[1/7] Generating source prompt/answer CSV: ${SOURCE_CSV}"
python experiments/generate_prompt_answer_csv.py \
  --config-file "${CONFIG_FILE}" \
  --output-csv "${SOURCE_CSV}"

echo "[2/7] none + answer_only"
python experiments/generate_reasoning.py \
  --csv "${SOURCE_CSV}:sft_source_v4_small" \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target answer_only \
  --prompt-reasoning-guidance none \
  --wrapper-mode "${WRAPPER_MODE}" \
  --n-per-source "${ROWS_PER_SOURCE}" \
  --seed "${BASE_SEED}" \
  --output "${SHARD_DIR}/01_none_answer_only.jsonl"

echo "[3/7] none + concise_evidence"
python experiments/generate_reasoning.py \
  --csv "${SOURCE_CSV}:sft_source_v4_small" \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target concise_evidence \
  --prompt-reasoning-guidance none \
  --wrapper-mode "${WRAPPER_MODE}" \
  --n-per-source "${ROWS_PER_SOURCE}" \
  --seed "$((BASE_SEED + 1))" \
  --output "${SHARD_DIR}/02_none_concise_evidence.jsonl"

teacher_args=(
  --teacher-provider "${TEACHER_PROVIDER}"
  --teacher-model "${TEACHER_MODEL}"
  --teacher-max-tokens "${TEACHER_MAX_TOKENS}"
  --teacher-max-revisions "${TEACHER_MAX_REVISIONS}"
  --teacher-fallback-target "${TEACHER_FALLBACK_TARGET}"
)
if [[ -n "${TEACHER_BASE_URL}" ]]; then
  teacher_args+=(--teacher-base-url "${TEACHER_BASE_URL}")
fi
if [[ -n "${TEACHER_REASONING_EFFORT}" ]]; then
  teacher_args+=(--teacher-reasoning-effort "${TEACHER_REASONING_EFFORT}")
fi
if [[ "${RESUME_TEACHER_SHARDS}" == "1" ]]; then
  teacher_args+=(--resume-existing-output)
fi

echo "[4/7] none + teacher_evidence (${TEACHER_MODEL})"
python experiments/generate_reasoning.py \
  --csv "${SOURCE_CSV}:sft_source_v4_small" \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target teacher_evidence \
  --prompt-reasoning-guidance none \
  "${teacher_args[@]}" \
  --wrapper-mode "${WRAPPER_MODE}" \
  --n-per-source "${ROWS_PER_SOURCE}" \
  --seed "$((BASE_SEED + 2))" \
  --output "${SHARD_DIR}/03_none_teacher.jsonl"

echo "[5/7] concise + concise_evidence"
python experiments/generate_reasoning.py \
  --csv "${SOURCE_CSV}:sft_source_v4_small" \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target concise_evidence \
  --prompt-reasoning-guidance concise \
  --wrapper-mode "${WRAPPER_MODE}" \
  --n-per-source "${ROWS_PER_SOURCE}" \
  --seed "$((BASE_SEED + 3))" \
  --output "${SHARD_DIR}/04_concise_concise_evidence.jsonl"

echo "[6/7] concise + teacher_evidence (${TEACHER_MODEL})"
python experiments/generate_reasoning.py \
  --csv "${SOURCE_CSV}:sft_source_v4_small" \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target teacher_evidence \
  --prompt-reasoning-guidance concise \
  "${teacher_args[@]}" \
  --wrapper-mode "${WRAPPER_MODE}" \
  --n-per-source "${ROWS_PER_SOURCE}" \
  --seed "$((BASE_SEED + 4))" \
  --output "${SHARD_DIR}/05_concise_teacher.jsonl"

echo "[7/7] Merging and shuffling -> ${OUT_JSONL}"
python - "$SHARD_DIR" "$OUT_JSONL" "$BASE_SEED" <<'PY'
import json
import random
import sys
from pathlib import Path

shard_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
seed = int(sys.argv[3])
paths = [
    shard_dir / "01_none_answer_only.jsonl",
    shard_dir / "02_none_concise_evidence.jsonl",
    shard_dir / "03_none_teacher.jsonl",
    shard_dir / "04_concise_concise_evidence.jsonl",
    shard_dir / "05_concise_teacher.jsonl",
]

rows = []
counts = {}
for path in paths:
    with path.open("r", encoding="utf-8") as f:
        shard_rows = [json.loads(line) for line in f]
    counts[path.name] = len(shard_rows)
    rows.extend(shard_rows)

random.Random(seed).shuffle(rows)
with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[done] wrote {len(rows)} rows -> {out_path}")
for name, count in counts.items():
    print(f"  {name}: {count}")
PY
