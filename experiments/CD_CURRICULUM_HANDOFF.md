# CD Curriculum Handoff

## Overview
This repo now has a staged curriculum launcher for causal-discovery experiments:

- [run_cd_curriculum.py](/home/yuen_chen/ENCO/experiments/run_cd_curriculum.py)
- cancer smoke-test manifest: [cd_curriculum_debug_cancer.json](/home/yuen_chen/ENCO/experiments/cd_curriculum_debug_cancer.json)
- prompt template catalog: [causal_curriculum_prompt_templates.txt](/home/yuen_chen/ENCO/experiments/prompts/causal_curriculum_prompt_templates.txt)

The launcher handles:
- stage handoff
- replay mixing
- eval gating
- failure logging
- GPU selection via `CUDA_VISIBLE_DEVICES`

GPU selection precedence in the launcher is:
1. `--cuda-visible-devices`
2. shell `CUDA_VISIBLE_DEVICES`
3. curriculum JSON `cuda_visible_devices`
4. fallback default

## Main Files
- [run_cd_curriculum.py](/home/yuen_chen/ENCO/experiments/run_cd_curriculum.py): curriculum orchestrator
- [run_sft_then_grpo.py](/home/yuen_chen/ENCO/experiments/run_sft_then_grpo.py): SFT helper used by the launcher
- [export_cd_train_eval_csv.py](/home/yuen_chen/ENCO/experiments/export_cd_train_eval_csv.py): exports train/eval CSVs from config JSONs
- [cd_curriculum_debug_cancer.json](/home/yuen_chen/ENCO/experiments/cd_curriculum_debug_cancer.json): small 2-stage cancer debug curriculum

## What Was Changed
- Added `run_cd_curriculum.py`.
- Added stage-level replay mixing and promotion gates.
- Added `stage_result.json` and `run_summary.json` writes on both success and failure.
- Updated `run_sft_then_grpo.py` so the SFT path is less brittle with current TRL behavior.
- Updated launcher GPU handling so shell `CUDA_VISIBLE_DEVICES` is respected by default.

## Smoke Test
Use the repo conda env.

### 1. Export tiny cancer train/eval CSVs
```bash
mkdir -p experiments/prompts/cancer_debug

python experiments/export_cd_train_eval_csv.py \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --config-file experiments/cancer_summary_joint_matrix_obs100_int10_nonanon.json \
  --num-prompts 2 \
  --train-seed 11 \
  --eval-seed 22 \
  --train-csv experiments/prompts/cancer_debug/cancer_obs100_int10_train.csv \
  --eval-csv experiments/prompts/cancer_debug/cancer_obs100_int10_eval.csv

python experiments/export_cd_train_eval_csv.py \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --config-file experiments/cancer_summary_joint_matrix_obs500_int20_nonanon.json \
  --num-prompts 2 \
  --train-seed 33 \
  --eval-seed 44 \
  --train-csv experiments/prompts/cancer_debug/cancer_obs500_int20_train.csv \
  --eval-csv experiments/prompts/cancer_debug/cancer_obs500_int20_eval.csv
```

### 2. Dry run the launcher
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_cd_curriculum.py \
  --curriculum-file experiments/cd_curriculum_debug_cancer.json \
  --output-root /tmp/cd_curriculum_debug_cancer \
  --dry-run
```

### 3. Run the real smoke test
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_cd_curriculum.py \
  --curriculum-file experiments/cd_curriculum_debug_cancer.json \
  --output-root /tmp/cd_curriculum_debug_cancer
```

## What To Check After Running
- [run_summary.json](/tmp/cd_curriculum_debug_cancer/run_summary.json)
- [stage_result.json](/tmp/cd_curriculum_debug_cancer/01_cancer_obs100_int10_debug/stage_result.json)
- [stage_result.json](/tmp/cd_curriculum_debug_cancer/02_cancer_obs500_int20_debug/stage_result.json)
- [train_mixed.csv](/tmp/cd_curriculum_debug_cancer/02_cancer_obs500_int20_debug/train_mixed.csv)

Expected behavior:
- stage 1 runs SFT, SFT eval, GRPO, GRPO eval
- stage 2 resumes from stage 1's promoted model
- stage 2 mixed train CSV includes replay rows from stage 1
- failures should still leave `stage_result.json` and `run_summary.json`

## Known Caveat
The most likely future failures are in the SFT leg inside [run_sft_then_grpo.py](/home/yuen_chen/ENCO/experiments/run_sft_then_grpo.py), due to TRL/Transformers behavior changes. If a run breaks, check that traceback first.

## Quick Recovery Tips
- If GPU selection looks wrong, check shell `CUDA_VISIBLE_DEVICES` first.
- If SFT fails, try the same curriculum with `"enable_sft": false` for the debug stage to isolate whether the failure is in SFT or GRPO.
- If a stage fails, inspect the stage-local `stage_result.json` before rerunning.
