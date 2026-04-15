# Experiments

Experiments can be run two modi, denoted by different files.
* `run_exported_graphs.py` applies ENCO to a causal graph that has been saved on the disk. The graph can be saved in the following three formats:
   * `.bif` format as from the BnLearn repository
   * `.npz` format as generated when running [`causal_graphs/graph_export.py`](../causal_graphs/graph_export.py)
   * `.pt` format as when saving a `CausalDAG` object to disk ([`CausalDAG.save_to_file`](../causal_graphs/graph_definition.py))
  The graph files can be specified using the parser argument `--graph_files`. Multiple files can be specified when ENCO should be tested on all those graphs in sequence. Example usage:
  ```bash
  python run_exported_graphs.py --graph_files ../causal_graphs/real_data/small_graphs/sachs.bif
  ```
* `run_generated_graphs.py` applies ENCO to newly generated causal graphs. It takes additional arguments for the graph(s) to generate, and is mostly meant for prototyping on various graph structures such as in the sythetic dataset. Example usage:
  ```bash
  python run_generated_graphs.py --graph_type random --num_vars 25 --edge_prob 0.3 --num_graphs 2
  ```

For all experiments, checkpoint folders are created that store logging information and the final, predicted graph. By default, those are created under the folder `checkpoints` with a date and time folder string. To specify a different checkpoint directory, use the argument `--checkpoint_dir`.

## Experiments from the paper
The commands to reproduce the experiments in the paper are summarized in the folder [run_scripts](run_scripts).
See the corresponding README for details.

## Experiment 1 (LLM causal discovery prompts): end-to-end pipeline

This repo also includes an Experiment 1 pipeline that:
1) generates prompt CSVs + `.txt` prompt files,
2) queries a model for each prompt (`OpenAI`, `Gemini`, or `HF`),
3) evaluates predictions and builds consensus artifacts, and
4) aggregates results and computes an ordering-bias summary.

### Key scripts

- Prompt generation (grid orchestrator): `experiments/generate_prompt_files.py`
  - Calls `experiments/generate_prompts.py` (data prompts) and `experiments/generate_prompts_names_only.py` (names-only, `N=0`).
  - Writes to `experiments/prompts/experiment1/<dataset>/...`.
- Model querying (per-CSV): `experiments/query_gemini.py`
  - Despite the filename, this is the unified runner.
  - For OpenAI models (e.g. `gpt-*`, `o1-*`, `o3-*`) it uses the OpenAI Responses API (requires `OPENAI_API_KEY`).
  - Writes outputs under `experiments/responses/<dataset>/...` (when run from `experiments/`).
- Model querying (batch over a dataset folder): `experiments/run_api_models.py`
  - Runs `query_gemini.py` over discovered prompt CSVs.
- Evaluation: `experiments/evaluate.py`
  - Writes per-row metrics to `<csv>.per_row.csv` (or appends in-place with `--inplace`) and writes `<csv>.summary.json` (skips outputs if no valid rows).
  - Produces a probability/consensus artifact per CSV (JSON + a `.probplot.pdf`).
- End-to-end wrapper (recommended): `experiments/run_experiment1_pipeline.py`

### Tool-use during inference (smoke test)

If you want to test an “LLM+tools” setup (where the model can call local Python during inference),
use: `experiments/tool_use_smoke_test.py`.
It demonstrates OpenAI function-calling with a local tool that computes stats from a CSV.

### “Shuffles per graph” dimension (ordering bias)

`--shuffles-per-graph S` controls how many independent row-order shuffles are generated per sampled dataset.
It is encoded into filenames as `_shufS` and produces multiple rows per `data_idx` with `shuffle_idx=0..S-1`.

`experiments/run_experiment1_pipeline.py` aggregates an ordering-bias summary in:
`experiments/out/experiment1/<dataset>_ordering_bias.csv`.

### Quickstart

Run from the repo root; the wrapper will execute sub-steps under `experiments/`.

1) Dry-run (prints commands only):
```bash
python experiments/run_experiment1_pipeline.py --dry-run \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --dataset cancer \
  --model gpt-5-mini \
  --shuffles-per-graph 1 --shuffles-per-graph 3 --shuffles-per-graph 10
```

2) Full run (generate → query → evaluate → analyze):
```bash
python experiments/run_experiment1_pipeline.py \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --dataset cancer \
  --model gpt-5-mini \
  --shuffles-per-graph 1 --shuffles-per-graph 3 --shuffles-per-graph 10
```

3) Resume behavior
- Querying: `experiments/query_gemini.py` resumes by default and skips rows with an existing non-`[ERROR]` `raw_response` and a non-empty `prediction`. Use `--overwrite` to re-run.
- Evaluation: `experiments/run_experiment1_pipeline.py` skips response CSVs that already have `<csv>.summary.json` unless `--overwrite-eval` is set.

### Outputs (relative to `experiments/`)

- Prompts: `experiments/prompts/experiment1/<dataset>/...`
- Responses: `experiments/responses/<dataset>/responses_..._<model>.csv`
- Per-CSV evaluation artifacts:
  - `experiments/responses/<dataset>/*.csv.summary.json`
  - `experiments/responses/<dataset>/*.csv.consensus_tau*.json`
  - `experiments/responses/<dataset>/*.csv.probplot.pdf`
- Aggregated analysis tables:
  - `experiments/out/experiment1/<dataset>_summary.csv`
  - `experiments/out/experiment1/<dataset>_ordering_bias.csv`

### Running all table dimensions (recommended workflow)

The LaTeX table dimensions map onto two batches of runs:

1) **Core grid** (semantics/volume/type/representation/robustness + shuffle replicates)
2) **Inductive-bias ablations** (rules/steps/intervention definition/examples), typically on a small subset

#### 1) Core grid (covers most of the table)

This is the “main sweep” and is what you usually mean by “run the table”:
- **Semantics**: real vs anonymized variable names
- **Data Volume**: multiple `N` (observational) and intervention sizes
- **Data Type**: observational-only vs interventional
- **Data Granularity / Representation**: serialized samples (`cases`) vs summary statistics (`matrix`)
- **Robustness**: ordering variants (currently generated only for the baseline config to control cost)
- **Shuffle Replicates**: `shuffles_per_graph` sweep (`_shuf1/_shuf3/_shuf10` etc.)

Run:
```bash
python experiments/run_experiment1_pipeline.py \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --dataset cancer \
  --model gpt-5-mini \
  --shuffles-per-graph 1 --shuffles-per-graph 3 --shuffles-per-graph 10
```

#### 2) Inductive-bias ablations (rules/steps/definitions/examples)

These are additional prompt sets you generate explicitly (often on the baseline condition) and then re-run
the pipeline from the `run` step so the model is queried + evaluated + analyzed.

Example: add causal rules + steps + a definition of interventions (only relevant when `int-per-combo > 0`):
```bash
python experiments/generate_prompts.py \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --out-dir experiments/prompts/experiment1/cancer/cases_real_obs5000_int200 \
  --prompt-style cases --obs-per-prompt 5000 --int-per-combo 200 --intervene-vars all \
  --shuffles-per-graph 10 \
  --causal-rules --give-steps --def-int
```

If you want “examples / partial supervision”, use known edges:
```bash
python experiments/generate_prompts.py \
  --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
  --out-dir experiments/prompts/experiment1/cancer/cases_real_obs5000_int200 \
  --prompt-style cases --obs-per-prompt 5000 --int-per-combo 200 --intervene-vars all \
  --shuffles-per-graph 10 \
  --given-edge-frac 0.2
```

Then query/evaluate/analyze the newly generated CSVs:
```bash
python experiments/run_experiment1_pipeline.py \
  --steps run,evaluate,analyze \
  --dataset cancer \
  --model gpt-5-mini
```

#### Notes

- If you truly want **robustness ordering variants for every single grid point**, you’ll need to relax the
  filtering in `experiments/generate_prompt_files.py` that currently restricts topo/reverse ordering to a
  baseline config (this is intentional to reduce runtime/cost).

### Summary-statistics prompts (shorter)

`experiments/generate_prompts.py` supports `--prompt-style summary` to embed compact summary statistics
(e.g., observational correlation + intervention mean shifts) instead of listing all samples.
This usually yields much shorter prompts and avoids having the LLM call tools during inference.

To build an ENCO-style LaTeX grid table from evaluated LLM runs, use:
`experiments/make_llm_baseline_table.py`.

## Classical baselines

The original ENCO repo ships ENCO as the native interventional baseline. This benchmark-builder extension
adds representative observational baselines through `experiments/run_classical_baselines.py`:

- `PC` (constraint-based)
- `GES` (score-based)
- `ENCO` (interventional anchor from the original repo)

### Run PC or GES on one config

Run from `experiments/`:
```bash
cd experiments

python run_classical_baselines.py \
  --method PC \
  --graph_files ../causal_graphs/real_data/small_graphs/cancer.bif \
  --sample_size_obs 5000 \
  --seed 42
```

Swap `--method GES` to run GES instead. Both methods write predictions under
`experiments/responses/<dataset>/predictions_obs*_int0_<METHOD>.csv`.

`run_classical_baselines.py` requires `pgmpy`.

### Run ENCO on one config

Use ENCO on the same underlying graph(s) with sampled observational/interventional data sizes that match
the “Data Volume” / “Data Type” dimensions.

Example: cancer, `obs=5000`, `int=200`.

Run from `experiments/`:
```bash
cd experiments

python run_exported_graphs.py \
  --graph_files ../causal_graphs/real_data/small_graphs/cancer.bif \
  --sample_size_obs 5000 \
  --sample_size_inters 200 \
  --max_inters -1 \
  --seed 42
```

This produces:
- checkpoints/logs under `experiments/checkpoints/...`
- a predictions CSV under `experiments/responses/cancer/predictions_obs5000_int200_ENCO.csv`

### Sweep ENCO over a grid (obs N × intervention M)

```bash
cd experiments

for N in 0 100 1000 5000 8000; do
  for M in 0 50 100 200 500; do
    # skip the empty-data case
    if [ "$N" -eq 0 ] && [ "$M" -eq 0 ]; then
      continue
    fi

    python run_exported_graphs.py \
      --graph_files ../causal_graphs/real_data/small_graphs/cancer.bif \
      --sample_size_obs "$N" \
      --sample_size_inters "$M" \
      --max_inters -1 \
      --seed 42 \
      --checkpoint_dir "checkpoints/enco_grid/cancer/obs${N}_int${M}"
  done
done
```

Notes:
- Observational-only baseline: set `--sample_size_inters 0`
- Interventional-only baseline: set `--sample_size_obs 0`

### Evaluate ENCO predictions

```bash
cd experiments
python evaluate.py --csv responses/cancer/predictions_obs5000_int200_ENCO.csv
```

This writes `responses/cancer/predictions_obs5000_int200_ENCO.csv.summary.json` and writes per-row metrics
to `<csv>.per_row.csv` by default (or in-place with `--inplace`). ENCO outputs are typically 1-row CSVs.

### Include ENCO in the aggregated analysis tables

After ENCO CSVs exist under `experiments/responses/<dataset>/`, you can re-run:
```bash
python run_experiment1_pipeline.py --steps evaluate,analyze --dataset cancer
```

This will include `predictions_obs*_int*_ENCO.csv` in `experiments/out/experiment1/cancer_summary.csv`.

## GRPO Training

This section covers fine-tuning a causal discovery model with GRPO (Group Relative Policy Optimization)
after an initial SFT phase.

### SFT readiness check

Before launching GRPO, verify the SFT checkpoint produces correct output format and has partial (not
perfect) task accuracy. Run `eval_sft_on_jsonl.py` on a small held-out set:

```bash
cd /u/chenyuen0103/ENCO

python experiments/eval_sft_on_jsonl.py \
    --model experiments/checkpoints/staged_sft_v3 \
    --jsonl experiments/data/permuted_sft.jsonl \
    --n 20 \
    --graph-filter cancer earthquake \
    --output-jsonl experiments/checkpoints/staged_sft_v3/eval_permuted_20.jsonl
```

**Readiness criteria:**
- `tags_correct = 100%` — every rollout is scoreable; no zero-reward completions due to format failures
- `adj_matrix_present = 100%` — adjacency matrix is always parseable
- `exact_match` between 5–50% — model is trying but not memorised; GRPO has a gradient to work with
- `prompt_copy = 0%` — no degenerate hallucination

The `prompt_model_input` field in the output JSONL records exactly what tokens the model received, which
you can compare against the raw `prompt` field to verify the tokenizer is not adding unexpected structure.

### Step 1 — Generate a fresh GRPO dataset

GRPO needs data the model has **not memorised** from SFT. Generate fresh observational/interventional
samples from the same graphs but with a different random seed. Start with the two simplest graphs
(cancer, earthquake — 5 nodes each) before introducing harder ones.

Run from the repo root:

```bash
python experiments/build_grpo_cd_mix_dataset.py \
    --graph-names cancer,earthquake \
    --output-csv experiments/data/grpo_v1_cancer_earthquake_seed200.csv \
    --obs-values 100 \
    --int-values 10 \
    --num-prompts-per-config 500 \
    --shuffles-per-graph 3 \
    --seed 200 \
    --anonymize
```

**Why these settings:**
- `--seed 200` — outside all SFT seeds (42) and prior GRPO seeds (42–46); different Bayesian network
  samples so the model cannot memorise
- `--obs-values 100 --int-values 10` — same data regime as SFT to keep difficulty consistent
- `--num-prompts-per-config 500` × `--shuffles-per-graph 3` = 3000 rows per graph (6000 total)
- `--graph-names cancer,earthquake` — curriculum: master simple 5-node graphs first before adding
  asia (8 nodes) or sachs (11 nodes)

Sanity-check the output:

```bash
python3 -c "
import csv
with open('experiments/data/grpo_v1_cancer_earthquake_seed200.csv') as f:
    rows = list(csv.DictReader(f))
graphs = {}
for r in rows:
    g = r.get('dataset', '?')
    graphs[g] = graphs.get(g, 0) + 1
print('Total rows:', len(rows))
print('By graph:', graphs)
print('Columns:', list(rows[0].keys()))
print('Prompt tail:', rows[0].get('prompt_text', '')[-200:])
"
```

The prompt tail should end with `assistant\n<think>` — this is the prefill boundary where the model
begins generating.

### Step 2 — Launch GRPO training

```bash
torchrun --standalone --nproc_per_node=4 experiments/grpo.py \
    --mode train \
    --task causal_discovery \
    --model_id experiments/checkpoints/staged_sft_v3 \
    --cd-train-csv experiments/data/grpo_v1_cancer_earthquake_seed200.csv \
    --cd-test-csv experiments/data/cancer_randcol_seed44.csv \
    --output_dir experiments/checkpoints/grpo_v1_cancer_earthquake \
    --no-use-vllm \
    --max_completion_length 8192 \
    --max_prompt_tokens 4096 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --cd-format-reward-scale 0.05 \
    --cd-cot-structure-reward-scale 0.05 \
    --cd-skeleton-f1-reward-scale 0.10 \
    --cd-vstruct-f1-reward-scale 0.10 \
    --cd-orientation-f1-reward-scale 0.10 \
    --cd-edge-f1-reward-scale 0.30 \
    --cd-low-shd-reward-scale 0.20 \
    --cd-graph-reward-scale 0.50 \
    --cd-append-format-hint \
    --save_steps 20 \
    --logging_steps 1 \
    --sample_completions_every 10 \
    --report_to none
```

### Step 3 — Monitor training

Check `experiments/checkpoints/grpo_v1_cancer_earthquake/grpo_log/sample_completions_rank0.jsonl`
for the sampled completions. Key fields to watch:

| Field | What to look for |
|---|---|
| `format_ok` | Should be 1 on nearly all rows within a few steps |
| `cd_edge_f1_reward` | Should increase over training steps |
| `cd_graph_reward` | Sparse signal; look for trend over 50+ steps |
| `completion` | Should start with `Stage 1 (Skeleton):`, not hallucinated data |

If `cd_edge_f1_reward` for one graph is consistently 0 across many steps, that graph may be too hard
for the current policy — consider temporarily removing it from `--cd-train-csv`.

### Curriculum progression

Once cancer+earthquake edge-F1 is stable (typically > 0.6), add harder graphs:

```bash
torchrun --standalone --nproc_per_node=4 experiments/grpo.py \
    --mode train \
    --task causal_discovery \
    --model_id experiments/checkpoints/grpo_v1_cancer_earthquake \
    --cd-train-csv experiments/data/grpo_v1_cancer_earthquake_seed200.csv \
    --cd-train-csv experiments/data/asia_randcol_seed45.csv \
    --cd-test-csv experiments/data/cancer_randcol_seed44.csv \
    --output_dir experiments/checkpoints/grpo_v2_with_asia \
    ... (same reward flags)
```

Generate asia/sachs GRPO data the same way as Step 1, substituting `--graph-names asia` or
`--graph-names sachs` and keeping `--seed 200`.

## GRPO Training — Descendant Task

This section covers fine-tuning from an SFT checkpoint on the `cd_descendants` task using GRPO.

### Step 1 — Evaluate the SFT checkpoint

```bash
python experiments/eval_sft_on_jsonl.py \
    --model experiments/checkpoints/descendant_sft_v1 \
    --jsonl experiments/data/descendant_sft.jsonl \
    --n 100 \
    --max-new-tokens 4096 \
    --output-jsonl experiments/checkpoints/descendant_sft_v1/eval_descendant_100.jsonl
```

To verify GRPO multi-GPU compatibility before launching training:

```bash
python experiments/eval_sft_on_jsonl.py \
    --model experiments/checkpoints/descendant_sft_v1 \
    --jsonl experiments/data/descendant_sft.jsonl \
    --check-grpo-compat
```

### Step 2 — Generate GRPO prompt data

Split CSVs already exist under `experiments/prompts/cd_descendants/sachs/splits/`. To regenerate:

```bash
python experiments/collect_descendant_sft_data.py \
    --output experiments/data/descendant_sft.jsonl \
    --graphs cancer earthquake asia sachs \
    --obs-values 50 100 \
    --int-values 10 50 \
    --num-prompts 10 \
    --seed 42
```

### Step 3 — Launch GRPO training from the SFT checkpoint

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  experiments/grpo.py \
  --mode train \
  --task cd_descendants \
  --model_id experiments/checkpoints/descendant_sft_v1 \
  --cd-train-csv experiments/prompts/cd_descendants/sachs/splits/stage_1_anon_obs50_int10_train.csv \
  --cd-test-csv  experiments/prompts/cd_descendants/sachs/splits/stage_1_anon_obs50_int10_eval.csv \
  --output_dir experiments/checkpoints/descendant_grpo_v1 \
  --no-use-vllm \
  --max_completion_length 4096 \
  --num_generations 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --cd-format-reward-scale 1.0 \
  --cd-partial-format-reward-scale 1.0 \
  --cd-graph-reward-scale 1.0 \
  --length_penalty_coef 0.00002 \
  --length_penalty_max_abs 0 \
  --save_steps 20 \
  --logging_steps 1 \
  --sample_completions_every 10 \
  --sample_completions_max_chars 0 \
  --report_to none
```

## Behavioral teacher attribution

To support the research direction of asking which classical procedure an LLM is
behaviorally closest to, this repo now includes:

- `experiments/attribute_teacher_behavior.py`

This script is a post-hoc analysis step. It does not run PC/GES/etc. for you;
instead, it expects one evaluated LLM CSV plus one or more teacher CSVs that already
contain graph predictions for the same instances.

### What it computes

- nearest-teacher matching per row
- disagreement-set attribution (rows where teachers disagree)
- pairwise graph similarity metrics between the LLM and each teacher:
  - directed-edge SHD
  - edge F1
  - skeleton F1
  - orientation accuracy on shared skeleton edges
  - collider F1

### Row alignment

Teacher rows are aligned to LLM rows in this order:

1. `(data_idx, shuffle_idx)`
2. `data_idx` alone, but only when unique in the teacher CSV
3. ground-truth graph hash parsed from `answer` / `answer_path`
4. optional single-row fallback via `--allow-single-row-fallback`

The last fallback is mainly for diagnostics. For the actual attribution study in your
research plan, teacher outputs should ideally be generated on the same per-instance
datasets as the LLM prompts.

### Example

```bash
python experiments/attribute_teacher_behavior.py \
  --llm-csv experiments/responses/cancer/responses_obs200_int0_shuf3_anon_gpt-4o-mini.csv \
  --teacher-csv path/to/pc_outputs.csv --teacher-name PC \
  --teacher-csv path/to/ges_outputs.csv --teacher-name GES \
  --teacher-csv path/to/igsp_outputs.csv --teacher-name IGSP \
  --metric pair_shd
```

Outputs:

- `<llm_csv>.teacher_attr.rows.csv`: one row per `(llm row, teacher)` pair
- `<llm_csv>.teacher_attr.row_summary.csv`: one row per LLM example with nearest-teacher info
- `<llm_csv>.teacher_attr.summary.json`: aggregate attribution summary and alignment stats
