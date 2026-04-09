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
