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
  - Appends per-row metrics to each responses CSV and writes `<csv>.summary.json`.
  - Produces a probability/consensus artifact per CSV (JSON + a `.probplot.pdf`).
- End-to-end wrapper (recommended): `experiments/run_experiment1_pipeline.py`

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
