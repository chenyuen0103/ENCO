# LLM Causal Discovery Benchmark and Training Repo

This repository is for generating causal-discovery prompts and datasets, training
SFT / GRPO models, querying external LLMs, and evaluating both LLM and classical
causal-discovery baselines.

The codebase includes:

- prompt and dataset generation under `experiments/`
- benchmark manifests and run orchestration under `benchmark_specs/` and `scripts/`
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

- `experiments/`: prompt generation, dataset building, querying, evaluation, training
- `benchmark_specs/`: benchmark manifests
- `benchmark_cards/`: benchmark descriptions and claims
- `benchmark_runs/`: generated benchmark artifacts and summaries
- `causal_graphs/`: graph definitions, BIF assets, ENCO-derived data utilities
- `causal_discovery/`: original ENCO learner
- `scripts/`: benchmark-builder CLIs

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
| Evaluate an API model (GPT, Gemini) end-to-end | `run_experiment1_pipeline.py` |
| Evaluate a specific baseline method (TakayamaSCP, CausalLLMData, …) | `run_external_llm_baselines.py` + `evaluate.py` |
| Score an existing prediction CSV | `evaluate.py` |
| Build training data for SFT / GRPO | `build_grpo_cd_mix_dataset.py` → `collect_format_sft_data.py` |
| Evaluate a finetuned / local model | `eval_sft_on_jsonl.py` |
| Generate prompts only (single graph, no querying) | `generate_prompts.py` |
| Query a prompt CSV without the full pipeline | `query_api.py` |

## Recommended Start for New Researchers

If you are new to the repo, start with the manifest-driven workflow rather than
the older script-by-script pipeline.

1. Set up the environment.
2. Run the focused Sachs paper slice:

```bash
python scripts/run_paper_slice.py --manifest paper_slices/sachs_main.json
```

3. Inspect the reusable benchmark suite before running it:

```bash
python scripts/run_benchmark.py --manifest benchmark_specs/reference_suite.json --dry-run
```

4. Run the small validation suite before editing manifests or adapters:

```bash
python -m unittest \
  tests.test_benchmark_spec \
  tests.test_paper_slices \
  tests.test_external_llm_baselines \
  tests.test_takayama_scd
```

This is the shortest path to the current architecture:

- `benchmark_specs/`: reusable benchmark definitions
- `paper_slices/`: frozen paper-facing manifests
- `benchmark_builder/`: schema, adapters, orchestration, evaluation
- `scripts/run_benchmark.py`: reusable entrypoint
- `scripts/run_paper_slice.py`: paper-facing compatibility wrapper

## Main Workflows

### 1. Generate prompts for a single graph

`generate_prompts.py` writes prompt CSVs for one BIF graph. Use this when you
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

`query_api.py` queries a CSV of prompts against OpenAI, Gemini, or HF backends
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

`run_experiment1_pipeline.py` combines prompt generation, querying, evaluation,
and summary table aggregation in one command. This is the recommended starting
point for evaluating an API-backed model.

```bash
python experiments/run_experiment1_pipeline.py \
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

### 4. Run benchmark-native LLM baselines

`run_external_llm_baselines.py` implements structured baselines such as
`TakayamaSCP`, `JiralerspongBFS`, `CausalLLMPrompt`, and `CausalLLMData`.

```bash
python experiments/run_external_llm_baselines.py \
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

`evaluate.py` prints mean precision, recall, F1, and SHD (structural Hamming
distance) to stdout and writes a per-row metrics CSV alongside the input file.
Pass `--summary-csv path/to/summary.csv` to accumulate results across runs.

`TakayamaSCP` is implemented separately in `experiments/run_takayama_scd.py`.
It is observational-only, supports checkpoint/resume for the pairwise LLM stage,
and can run with:

- `provider=openai` for the faithful logprob-based path
- `provider=illinois` for a practical fallback that parses yes/no text answers and retries transient HTTP failures

### 5. Build mixed prompt/answer CSV datasets

`build_grpo_cd_mix_dataset.py` generates prompt CSVs with gold adjacency answers
across multiple graphs and observation/intervention sizes.

```bash
python experiments/build_grpo_cd_mix_dataset.py \
  --output-csv experiments/data/grpo_mix_named.csv \
  --graph-names cancer,earthquake,asia,sachs \
  --prompt-style summary \
  --obs-values 100 \
  --int-values 10 \
  --num-prompts-per-config 5 \
  --shuffles-per-graph 1
```

For anonymized prompts add `--anonymize`.

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

`run_scripts/generate_sft_data.sh` wraps the full Sachs eval export plus JSONL
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

`collect_format_sft_data.py` converts prompt/answer CSV rows into
chat-formatted SFT records with `<think>...</think><answer>...</answer>`.

```bash
python experiments/collect_format_sft_data.py \
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
python experiments/collect_format_sft_data.py \
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

`eval_sft_on_jsonl.py` evaluates a local adapter or HF model on a JSONL or CSV
eval set.

```bash
python experiments/eval_sft_on_jsonl.py \
  --model experiments/checkpoints/qwen3_4b_cd_format_v5 \
  --jsonl experiments/data/format_sft_stages_v4_mixed.jsonl \
  --n 100 \
  --output-jsonl experiments/checkpoints/qwen3_4b_cd_format_v5/eval_format_sft.jsonl
```

## Script Reference

All scripts live under `experiments/`.

### Prompt and dataset generation

| Script | Purpose |
|--------|---------|
| `generate_prompts.py` | Prompt CSVs from a single BIF graph (obs + interventional) |
| `cd_generation/names_only.py` | Names-only prompts with no sampled data |
| `build_grpo_cd_mix_dataset.py` | Mixed prompt/answer CSVs across multiple graphs and data sizes |
| `export_cd_train_eval_csv.py` | Leak-free train/eval CSV splits from a config file |
| `collect_format_sft_data.py` | Converts prompt CSVs to SFT JSONL with `<think>…</think><answer>…</answer>` |
| `collect_descendant_sft_data.py` | SFT data for the descendant-identification task |
| `run_scripts/generate_sft_data.sh` | Convenience wrapper to export Sachs eval CSVs and build train/eval causal-discovery SFT JSONL |

### Querying and evaluation

| Script | Purpose |
|--------|---------|
| `run_experiment1_pipeline.py` | End-to-end: generate → query → evaluate → summarize |
| `run_experiment1_in_memory.py` | Same as above but without writing prompt CSV assets first |
| `query_api.py` | Query a prompt CSV against OpenAI, Gemini, or HF backends |
| `run_external_llm_baselines.py` | Structured LLM baselines (TakayamaSCP, JiralerspongBFS, CausalLLMPrompt, CausalLLMData) |
| `evaluate.py` | Score a prediction CSV; outputs precision, recall, F1, SHD |
| `eval_sft_on_jsonl.py` | Evaluate a finetuned SFT / LoRA model on JSONL or CSV |

### Baselines and training

| Script | Purpose |
|--------|---------|
| `run_classical_baselines.py` | Classical baselines: PC, GES, ENCO |
| `run_exported_graphs.py` | Original ENCO learner on exported graphs |
| `run_generated_graphs.py` | ENCO on newly generated synthetic graphs |
| `run_sft.py` | Supervised finetuning |
| `grpo.py` | GRPO training / evaluation utilities |
| `grpo_unsloth.py` | Unsloth-oriented GRPO path |

## Prompt Variants

`experiments/prompt_variants.md` documents the current prompt rendering variants,
including:

- `plain` vs `chat` wrapper mode
- `summary`, `matrix`, and `names-only` content styles
- SFT reasoning targets

## Benchmark Builder

This repo also includes a manifest-driven benchmark builder.

Core directories:

- `benchmark_specs/`
- `benchmark_cards/`
- `benchmark_runs/`
- `paper_slices/`

Main CLIs:

```bash
scripts/build-benchmark --manifest benchmark_specs/reference_suite.json
scripts/run-benchmark --manifest benchmark_specs/reference_suite.json
scripts/summarize-benchmark --manifest benchmark_specs/reference_suite.json
scripts/clean-benchmark-prompts --manifest benchmark_specs/reference_suite.json --yes
```

Useful manifests:

- `benchmark_specs/reference_suite.json`
- `benchmark_specs/smoke_suite.json`
- `benchmark_specs/synthetic_ladder.json`
- `benchmark_specs/authoring_demo.json`

See also `docs/tutorials/benchmark_authoring.md`.

## Classical Baselines

`experiments/run_classical_baselines.py` supports `PC`, `GES`, and `ENCO`.

```bash
cd experiments

python run_classical_baselines.py \
  --method PC \
  --graph_files ../causal_graphs/real_data/small_graphs/cancer.bif \
  --sample_size_obs 5000 \
  --seed 42
```

The original ENCO learner is also still available:

```bash
cd experiments

python run_exported_graphs.py \
  --graph_files ../causal_graphs/real_data/small_graphs/cancer.bif \
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
