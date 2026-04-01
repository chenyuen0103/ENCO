#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERR] line=$LINENO cmd=$BASH_COMMAND exit=$?"' ERR

PY=/home/yuen_chen/.conda/envs/enco/bin/python
GRPO=/home/yuen_chen/ENCO/experiments/grpo_unsloth.py
BIF=/home/yuen_chen/ENCO/causal_graphs/real_data/small_graphs/sachs.bif
CKPT="Qwen/Qwen3-4B-Thinking-2507"
OUT_ROOT=/home/yuen_chen/ENCO/experiments/checkpoints/grpo_sachs_guarded
CFG_TMP=/tmp/sachs_curriculum_guarded
mkdir -p "$CFG_TMP" "$OUT_ROOT"

# Include obs=0,int=0 warmup once, then continue curriculum.
STAGES=("0 0" "50 0" "100 0" "200 0" "500 0" "0 10" "50 10" "100 10" "200 10" "500 10" "0 50" "50 50" "100 50" "200 50" "500 50" "0 100" "50 100" "100 100" "200 100" "500 100")
FMT_MIN_DEFAULT=0.70; CD_MIN_DEFAULT=0.05; CLIP_MAX_DEFAULT=0.20

check_stage () {
  local metrics_jsonl="$1"
  local fmt_min="$2"
  local cd_min="$3"
  local clip_max="$4"
  "$PY" - "$metrics_jsonl" "$fmt_min" "$cd_min" "$clip_max" <<'PY'
import json, sys
p, fmt_min, cd_min, clip_max = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
lines=[json.loads(x) for x in open(p) if x.strip()]
x=lines[-1]
fmt=float(x.get("rewards/cd_format_reward/mean",0.0) or 0.0)
cd=float(x.get("rewards/cd_graph_reward/mean",0.0) or 0.0)
clip=float(x.get("completions/clipped_ratio",1.0) or 1.0)
ok=(fmt>=fmt_min) and (cd>=cd_min) and (clip<=clip_max)
print(f"last: fmt={fmt:.4f} cd={cd:.4f} clip={clip:.4f} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
}

stage_id=0
for pair in "${STAGES[@]}"; do
  stage_id=$((stage_id+1))
  OBS=$(echo "$pair" | awk '{print $1}')
  INT=$(echo "$pair" | awk '{print $2}')
  if [[ "$INT" -eq 0 ]]; then MAX_LEN=4096; LEN_COEF=0.00008; else MAX_LEN=3072; LEN_COEF=0.00012; fi
  # Stage-specific safety.
  FMT_MIN="$FMT_MIN_DEFAULT"; CD_MIN="$CD_MIN_DEFAULT"; CLIP_MAX="$CLIP_MAX_DEFAULT"
  if   [[ "$INT" -eq 0   ]]; then PFMT=0.15; SHDW=0.05
  elif [[ "$INT" -eq 10  ]]; then PFMT=0.10; SHDW=0.10
  elif [[ "$INT" -eq 50  ]]; then PFMT=0.07; SHDW=0.15
  else                         PFMT=0.05; SHDW=0.20
  fi

  CFG="$CFG_TMP/stage_${stage_id}_obs${OBS}_int${INT}.json"
  cat > "$CFG" <<EOF
{"configs":[
{"style":"summary_joint","anonymize":false,"obs_per_prompt":${OBS},"int_per_combo":${INT},"row_order":"random","col_order":"original","shuffles_per_graph":1},
{"style":"summary_joint","anonymize":true,"obs_per_prompt":${OBS},"int_per_combo":${INT},"row_order":"random","col_order":"original","shuffles_per_graph":1},
{"style":"matrix","anonymize":false,"obs_per_prompt":${OBS},"int_per_combo":${INT},"row_order":"random","col_order":"original","shuffles_per_graph":1},
{"style":"matrix","anonymize":true,"obs_per_prompt":${OBS},"int_per_combo":${INT},"row_order":"random","col_order":"original","shuffles_per_graph":1}
]}
EOF

  OUT_DIR="$OUT_ROOT/stage_${stage_id}_obs${OBS}_int${INT}"
  echo "============================================================"
  echo "Stage $stage_id: obs=$OBS int=$INT"
  echo "Resume from: $CKPT"
  echo "Output dir : $OUT_DIR"
  echo "max_len=$MAX_LEN len_coef=$LEN_COEF pfmt=$PFMT shdw=$SHDW"
  echo "guard: fmt>=$FMT_MIN cd>=$CD_MIN clip<=$CLIP_MAX"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES=3 "$PY" "$GRPO" \
    --task causal_discovery \
    --model_id "$CKPT" \
    --cd-config-file "$CFG" \
    --cd-bif-file "$BIF" \
    --cd-config-num-prompts 5 \
    --cd-config-seed 0 \
    --output_dir "$OUT_DIR" \
    --no-use-vllm \
    --no-unsloth-fast-inference \
    --no-unsloth-vllm-standby \
    --max_completion_length "$MAX_LEN" \
    --num_generations 2 \
    --gradient_accumulation_steps 2 \
    --cd-partial-format-reward-scale "$PFMT" \
    --cd-reward-shd-weight "$SHDW" \
    --length_penalty_coef "$LEN_COEF" \
    --length_penalty_max_abs 0 \
    --report_to none \
    --save-eval-responses \
    --no-eval-responses-include-text

  METRICS="$OUT_DIR/grpo_log/train_metrics.jsonl"
  check_stage "$METRICS" "$FMT_MIN" "$CD_MIN" "$CLIP_MAX" || { echo "Stage failed guard; stopping."; exit 2; }
  CKPT=$(ls -d "$OUT_DIR"/checkpoint-* | sort -V | tail -1)
done

echo "Done. Final checkpoint: $CKPT"
