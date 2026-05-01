# LLM Causal Discovery Benchmark and Training Repo

This repository is for generating causal-discovery prompts and datasets, training
SFT / GRPO models, querying external LLMs, and evaluating both LLM and classical
causal-discovery baselines.

The codebase includes:

- prompt and dataset generation under `experiments/`
- benchmark specs/cards/runs under `benchmark_specs/`, `benchmark_cards/`, and `benchmark_runs/`
- benchmark orchestration CLIs under `scripts/`
- finetuning / evaluation utilities for causal-discovery reasoning
- the original ENCO structure-learning code under `causal_discovery/` and `causal_graphs/`

## Provenance

This repo was built on top of the original ENCO codebase by Phillip Lippe, Taco
Cohen, and Efstratios Gavves.

In particular, the observational and interventional data generation logic used by
our prompt and dataset builders is adapted from the ENCO codebase and then wrapped
for prompt construction, LLM evaluation, and training.

The original ENCO implementation is still present here for baseline comparison and
for graph / data utilities, but this repository should be read primarily as our
LLM causal-discovery benchmark and training repo.

## Repo Layout

- `experiments/`: prompt/data generation, SFT/GRPO training, reward eval, query/eval helpers
- `benchmark_specs/`: benchmark configs
- `benchmark_cards/`: benchmark descriptions and claims
- `benchmark_runs/`: generated benchmark artifacts and summaries
- `benchmark_builder/`: reusable benchmark schema, adapters, orchestration, evaluation
- `causal_graphs/`: graph definitions, BIF assets, ENCO-derived data utilities
- `causal_discovery/`: original ENCO learner
- `scripts/`: benchmark, paper-slice, baseline, and config-eval CLIs

## Setup

For the LLM / benchmark workflow, use a Python environment based on
`requirements.txt`. The current working environment for this repo is
`python 3.12.12` in the `enco` conda env.

```bash
conda create -n enco-llm python=3.12 -y
conda activate enco-llm
conda install -c conda-forge graphviz -y
pip install -r requirements.txt
```

Or:

```bash
./setup.sh enco-llm 3.12
```

If you want to run the original ENCO code directly, `environment.yml` is still
available for that older workflow.

Environment variables:

- `OPENAI_API_KEY` for OpenAI-backed querying
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` for Gemini-backed querying
- `ILLINOIS_CHAT_API` for Illinois-backed querying
- optional `ILLINOIS_CHAT_COURSE` (defaults to `llm_cd`)

## Data Assets

The repo already includes a small set of bundled real-data BIF graphs under:

- `causal_graphs/real_data/small_graphs/`
- `causal_graphs/real_data/large_graphs/`

To download the broader benchmark graph assets from the original ENCO setup:

```bash
bash download_datasets.sh
```

To download the full discrete `bnlearn` BIF collection:

```bash
python causal_graphs/real_data/download_bnlearn_bifs.py
```

## Quick Orientation

Pick the workflow that matches your goal:

| Goal | Entry point |
|------|-------------|
| Evaluate an API model (GPT, Gemini) end-to-end | `scripts/run_cd_eval_pipeline.py` |
| Evaluate a specific baseline method (TakayamaSCP, CausalLLMData, …) | `scripts/run_external_llm.py` + `experiments/evaluate.py` |
| Score an existing prediction CSV | `experiments/evaluate.py` |
| Build training data for SFT / GRPO | `experiments/generate_prompt_answer_csv.py` → `experiments/generate_reasoning.py` |
| Evaluate a finetuned / local model | `experiments/eval_sft_on_jsonl.py` |
| Generate prompts only (single graph, no querying) | `experiments/generate_prompts.py` |
| Query a prompt CSV without the full pipeline | `experiments/query_api.py` |

## Recommended Start for New Researchers

If you are new to the repo, start with the config-driven workflow rather than
the older script-by-script pipeline.

1. Set up the environment.
2. Run the focused Sachs paper slice:

```bash
python scripts/run_paper_slice.py --config paper_slices/sachs_main.json
```

3. Inspect the reusable benchmark suite before running it:

```bash
python scripts/run_benchmark.py --config benchmark_specs/reference_suite.json --dry-run
```

4. Run the small validation suite before editing configs or adapters:

```bash
python -m unittest \
  tests.test_benchmark_spec \
  tests.test_paper_slices \
  tests.test_external_llm_baselines \
  tests.test_takayama_scd
```

This is the shortest path to the current architecture:

- `benchmark_specs/`: reusable benchmark definitions
- `paper_slices/`: frozen paper-facing configs
- `benchmark_builder/`: schema, adapters, orchestration, evaluation
- `scripts/run_benchmark.py`: reusable entrypoint
- `scripts/run_paper_slice.py`: paper-facing compatibility wrapper

## Main Workflows

### 1. Generate prompts for a single graph

`experiments/generate_prompts.py` writes prompt CSVs for one BIF graph. Use this when you
want raw prompt files without immediately querying a model.

```bash
python experiments/generate_prompts.py \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --num-prompts 5 \
  --obs-per-prompt 100 \
  --int-per-combo 10 \
  --prompt-style summary \
  --out-dir experiments/prompts/cancer
```

Key flags: `--anonymize` to strip node names, `--shuffles-per-graph N` for
row-order replicates, `--wrapper-mode chat` for system/user/assistant format.
Output goes to `experiments/prompts/<graph>/` by default.

### 2. Query a prompt CSV directly

`experiments/query_api.py` queries a CSV of prompts against OpenAI, Gemini, or HF backends
and writes a response CSV. Use this when you already have a prompt CSV and want
to skip the full pipeline.

```bash
python experiments/query_api.py \
  --csv experiments/prompts/cancer/prompts_obs100_int10.csv \
  --model gpt-4o-mini \
  --provider openai \
  --prompt-col prompt_text \
  --out-csv experiments/responses/cancer/predictions_obs100_int10.csv
```

For Gemini: `--model gemini-2.5-flash --provider gemini`.
For a local HF model: `--model Qwen/Qwen3-4B --provider hf --hf-trust-remote-code`.

### 3. Run the end-to-end LLM pipeline

`scripts/run_cd_eval_pipeline.py` combines prompt generation, querying, evaluation,
and summary table aggregation in one command. This is the recommended starting
point for evaluating an API-backed model.

```bash
python scripts/run_cd_eval_pipeline.py \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --dataset cancer \
  --model gpt-4o-mini \
  --shuffles-per-graph 1 --shuffles-per-graph 3
```

Outputs:

- prompts under `experiments/prompts/`
- model responses under `experiments/responses/`
- aggregated summaries under `experiments/out/experiment1/`

`--shuffles-per-graph` controls row-order replicates and is used for
ordering-bias analysis.

### 3a. Run config-based in-memory evaluation

`scripts/eval_cd_configs.py` is the lightweight path when you already have a
prompt config JSON and want to query a model without first writing prompt CSV
assets to disk.

OpenAI API example:

```bash
export OPENAI_API_KEY=...

python scripts/eval_cd_configs.py \
  --bif-file causal_graphs/real_data/small_graphs/sachs.bif \
  --config-file ./experiments/configs/eval_configs.json \
  --model gpt-5-mini \
  --provider openai \
  --temperature 0.0 \
  --request-timeout-s 600 \
  --log-calls
```

Local Hugging Face example:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_cd_configs.py \
  --bif-file causal_graphs/real_data/small_graphs/sachs.bif \
  --config-file ./experiments/configs/eval_configs_obs1000.json \
  --model /scratch/yuen_chen/models/Qwen2.5-72B-Instruct-AWQ \
  --provider hf \
  --temperature 0.0 \
  --hf-dtype bf16 \
  --hf-device-map auto \
  --hf-max-new-tokens 4096 \
  --hf-batch-size 1 \
  --hf-kv-vram-fraction 0.80 \
  --hf-context-limit 131072
```

Notes:

- Use `--provider openai` for Responses API models such as `gpt-5.2-pro`.
- The OpenAI path does not use local GPUs, so `CUDA_VISIBLE_DEVICES` is not needed.
- Use `--provider hf` plus the `--hf-*` flags for local checkpoints or Hub models.

### 4. Run benchmark-native LLM baselines

`scripts/run_external_llm.py` implements structured baselines such as
`TakayamaSCP`, `JiralerspongBFS`, `CausalLLMPrompt`, and `CausalLLMData`.

```bash
python scripts/run_external_llm.py \
  --method CausalLLMData \
  --graph_files causal_graphs/real_data/small_graphs/cancer.bif \
  --sample_size_obs 100 \
  --sample_size_inters 10 \
  --prompt_mode summary \
  --naming_regime real \
  --model gpt-4o-mini \
  --provider openai
```

Score the output CSV with:

```bash
python experiments/evaluate.py \
  --csv experiments/responses/cancer/predictions_obs100_int10_CausalLLMData.csv
```

`experiments/evaluate.py` prints mean precision, recall, F1, and SHD (structural Hamming
distance) to stdout and writes a per-row metrics CSV alongside the input file.
Pass `--summary-csv path/to/summary.csv` to accumulate results across runs.

`TakayamaSCP` is implemented separately in `scripts/takayama_scd.py`.
It is observational-only, supports checkpoint/resume for the pairwise LLM stage,
and can run with:

- `provider=openai` for the faithful logprob-based path
- `provider=illinois` for a practical fallback that parses yes/no text answers and retries transient HTTP failures

### 5. Build mixed prompt/answer CSV datasets

`experiments/generate_prompt_answer_csv.py` generates prompt CSVs with gold adjacency answers
across multiple graphs and observation/intervention sizes.

```bash
python experiments/generate_prompt_answer_csv.py \
  --output-csv experiments/data/grpo_mix_named.csv \
  --graph-names cancer,earthquake,asia,sachs \
  --prompt-style summary \
  --obs-values 100 \
  --int-values 10 \
  --num-prompts-per-config 5 \
  --shuffles-per-graph 1
```

For anonymized prompts add `--anonymize`.

#### Current GRPO source pool

The current GRPO source config is:

```text
experiments/configs/grpo_source_v4_mixed.json
```

It uses canonical prompt styles (`summary`, `matrix`) and only the current GRPO
reasoning-guidance mix (`concise`, `none`). Regenerate the CSV from that config
when `experiments/data/grpo_source_v4_mixed_train.csv` is stale:

```bash
mv experiments/data/grpo_source_v4_mixed_train.csv \
   experiments/data/grpo_source_v4_mixed_train.stale_with_staged.csv

python experiments/generate_prompt_answer_csv.py \
  --config-file experiments/configs/grpo_source_v4_mixed.json \
  --output-csv experiments/data/grpo_source_v4_mixed_train.csv
```

Expected quick check:

```bash
python - <<'PY'
import csv
from collections import Counter
p = "experiments/data/grpo_source_v4_mixed_train.csv"
c = Counter()
rows = 0
with open(p, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows += 1
        c[r["reasoning_guidance"]] += 1
print("rows:", rows)
print("reasoning_guidance:", dict(c))
PY
```

Expected output:

```text
rows: 3840
reasoning_guidance: {'concise': 1920, 'none': 1920}
```

Older files such as `grpo_mix_anon*.csv` and `grpo_mix_named*.csv` may still
contain legacy `summary_joint` prompts and should be treated as historical
artifacts unless a run explicitly depends on them.

### 6. Export train/eval CSV splits from a config file

`export_cd_train_eval_csv.py` produces reproducible train/eval splits from an
explicit prompt grid.

```bash
python experiments/export_cd_train_eval_csv.py \
  --bif-file causal_graphs/real_data/small_graphs/sachs.bif \
  --config-file experiments/configs/sachs_qwen_configs.json \
  --num-prompts 20 \
  --train-seed 42 \
  --eval-seed 1337 \
  --train-csv experiments/data/grpo_sachs_train.csv \
  --eval-csv experiments/data/sft_eval_child.csv
```

### 7. Generate SFT train/eval data in one command

`experiments/run_scripts/generate_sft_data.sh` wraps the full Sachs eval export plus JSONL
conversion pipeline. By default it writes:

- `experiments/data/grpo_sachs_train.csv`
- `experiments/data/sft_eval_child.csv`
- `experiments/data/format_sft_stages_v4_mixed.jsonl`
- `experiments/data/sft_eval.jsonl`

```bash
bash experiments/run_scripts/generate_sft_data.sh
```

Useful overrides:

```bash
REASONING_TARGET=stages \
TRAIN_JSONL=experiments/data/format_sft_strict_v4_mixed.jsonl \
EVAL_JSONL=experiments/data/sft_eval.jsonl \
bash experiments/run_scripts/generate_sft_data.sh
```

The Sachs config uses `col_order=random` for the names-only prompt so the
train/eval export stays leak-free across different seeds.

### 8. Convert prompt CSVs into SFT JSONL

`experiments/generate_reasoning.py` converts prompt/answer CSV rows into
chat-formatted SFT records with `<think>...</think><answer>...</answer>`.

```bash
python experiments/generate_reasoning.py \
  --output experiments/data/format_sft_stages_v4_mixed.jsonl \
  --csv experiments/data/grpo_mix_anon.csv:grpo_mix_anon \
  --csv experiments/data/grpo_mix_named.csv:grpo_mix_named \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target stages \
  --wrapper-mode chat \
  --n-per-source 1725 \
  --seed 42
```

Held-out eval JSONL from the exported Sachs CSV:

```bash
python experiments/generate_reasoning.py \
  --output experiments/data/sft_eval.jsonl \
  --csv experiments/data/sft_eval_child.csv:sft_eval_child \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target stages \
  --wrapper-mode chat \
  --n-per-source 999999 \
  --seed 1337
```

### 9. Evaluate a finetuned SFT / LoRA model

`experiments/eval_sft_on_jsonl.py` evaluates a local adapter or HF model on a JSONL or CSV
eval set.

```bash
python experiments/eval_sft_on_jsonl.py \
  --model experiments/checkpoints/qwen3_4b_cd_format_v5 \
  --jsonl experiments/data/format_sft_stages_v4_mixed.jsonl \
  --n 100 \
  --output-jsonl experiments/checkpoints/qwen3_4b_cd_format_v5/eval_format_sft.jsonl
```

## Script Reference

The repo now separates workflow scripts by responsibility:

- `experiments/` holds prompt/data generation, SFT/GRPO training, reward eval, and lower-level query/eval helpers.
- `scripts/` holds benchmark, config-eval, paper-slice, and baseline entry points.
- `benchmark_builder/`, `benchmark_specs/`, and `benchmark_cards/` hold the reusable benchmark framework, benchmark definitions, and benchmark documentation.

### Prompt and dataset generation

| Script | Purpose |
|--------|---------|
| `experiments/generate_prompts.py` | Prompt CSVs from a single BIF graph (obs + interventional) |
| `experiments/cd_generation/names_only.py` | Names-only prompts with no sampled data |
| `experiments/generate_prompt_answer_csv.py` | Mixed prompt/answer CSVs across multiple graphs and data sizes |
| `experiments/export_cd_train_eval_csv.py` | Leak-free train/eval CSV splits from a config file |
| `experiments/generate_reasoning.py` | Converts prompt CSVs to SFT JSONL with `<think>…</think><answer>…</answer>` |
| `experiments/collect_descendant_sft_data.py` | SFT data for the descendant-identification task |
| `experiments/run_scripts/generate_sft_data.sh` | Convenience wrapper to export Sachs eval CSVs and build train/eval causal-discovery SFT JSONL |

### Querying and evaluation

| Script | Purpose |
|--------|---------|
| `scripts/run_cd_eval_pipeline.py` | End-to-end: generate → query → evaluate → summarize |
| `scripts/eval_cd_configs.py` | Same as above but without writing prompt CSV assets first |
| `experiments/query_api.py` | Query a prompt CSV against OpenAI, Gemini, or HF backends |
| `scripts/run_prompt_csv_models.py` | Legacy prompt-CSV batch wrapper for API/HF models |
| `scripts/run_external_llm.py` | Structured LLM baselines (TakayamaSCP, JiralerspongBFS, CausalLLMPrompt, CausalLLMData) |
| `experiments/evaluate.py` | Score a prediction CSV; outputs precision, recall, F1, SHD |
| `experiments/eval_sft_on_jsonl.py` | Evaluate a finetuned SFT / LoRA model on JSONL or CSV |
| `experiments/eval_grpo_vllm_rollouts.py` | Fast vLLM rollout sampling and reward scoring for merged/full models |

### Baselines and training

| Script | Purpose |
|--------|---------|
| `scripts/run_classical.py` | Classical baselines: PC, GES, ENCO |
| `experiments/run_exported_graphs.py` | Original ENCO learner on exported graphs |
| `experiments/run_generated_graphs.py` | ENCO on newly generated synthetic graphs |
| `experiments/train_sft.py` | Supervised finetuning |
| `experiments/grpo.py` | GRPO training / evaluation utilities |

## Prompt Variants

`experiments/prompt_variants.md` documents the current prompt rendering variants,
including:

- `plain` vs `chat` wrapper mode
- `summary`, `matrix`, and `names-only` content styles
- SFT reasoning targets

## Benchmark Builder

This repo also includes a config-driven benchmark builder.

Core directories:

- `benchmark_builder/`
- `benchmark_specs/`
- `benchmark_cards/`
- `benchmark_runs/`
- `paper_slices/`

Main CLIs:

```bash
scripts/build-benchmark --config benchmark_specs/reference_suite.json
scripts/run-benchmark --config benchmark_specs/reference_suite.json
scripts/summarize-benchmark --config benchmark_specs/reference_suite.json
scripts/clean-benchmark-prompts --config benchmark_specs/reference_suite.json --yes
```

Useful configs:

- `benchmark_specs/reference_suite.json`
- `benchmark_specs/sachs_child_eval_configs.json`
- `benchmark_specs/smoke_suite.json`
- `benchmark_specs/synthetic_ladder.json`
- `benchmark_specs/authoring_demo.json`

To regenerate the benchmark-data CSV assets for the real graphs plus the
representative controlled synthetic challenge family:

```bash
python scripts/build_benchmark_data_suite.py
python scripts/summarize_reference_suite.py
python scripts/plot_dataset_composition.py
```

See also `docs/tutorials/benchmark_authoring.md`.

Compatibility shims remain at several older `experiments/run_*` paths, but new
benchmark and baseline commands should use `scripts/`.

## Classical Baselines

`scripts/run_classical.py` supports `PC`, `GES`, and `ENCO`.

```bash
python scripts/run_classical.py \
  --method PC \
  --graph_files causal_graphs/real_data/small_graphs/cancer.bif \
  --sample_size_obs 5000 \
  --seed 42
```

The original ENCO learner is also still available:

```bash
python experiments/run_exported_graphs.py \
  --graph_files causal_graphs/real_data/small_graphs/cancer.bif \
  --sample_size_obs 5000 \
  --sample_size_inters 200 \
  --max_inters -1 \
  --seed 42
```

## GRPO with verl

This repo also contains verl-based GRPO training support.

Typical workflow:

1. prepare prompt / training parquet or JSONL data
2. launch the Apptainer container
3. run `verl.trainer.main_ppo`

Example container launch:

```bash
apptainer shell --nv --bind /work:/work /work/hdd/bdeb/chenyuen0103/verl.sif
```

Inside the container:

```bash
export HF_HOME=/work/hdd/bdeb/chenyuen0103/.cache
```

The local verl code lives under `verl/`.

## Citation and Attribution

If you use this repository, cite the original ENCO work for the base code and
data-generation foundation:

```bibtex
@inproceedings{lippe2022enco,
  author = {Lippe, Phillip and Cohen, Taco and Gavves, Efstratios},
  booktitle = {International Conference on Learning Representations},
  title = {Efficient Neural Causal Discovery without Acyclicity Constraints},
  url = {https://openreview.net/forum?id=eYciPrLuUhG},
  year = {2022}
}
```
