# Benchmark Authoring Tutorial

This repository now treats benchmark authoring as a manifest-driven workflow.

## Goal

Define a new LLM causal-discovery benchmark without editing evaluator or model code.

## Example Manifest

Use [`benchmark_specs/authoring_demo.json`](../../benchmark_specs/authoring_demo.json) as the reference example. It mixes:

- one public real graph: `asia`
- one synthetic graph materialized from manifest parameters
- two prompt families: `summary_joint` and `matrix`
- both observational and intervention-bearing benchmark cells
- one contamination control: anonymization
- one prior-only control: `names_only`
- in-memory prompt execution with example-only prompt retention

## Commands

Build only:

```bash
scripts/build-benchmark --manifest benchmark_specs/authoring_demo.json
```

Run the full workflow:

```bash
scripts/run-benchmark --manifest benchmark_specs/authoring_demo.json
```

Summarize completed runs:

```bash
scripts/summarize-benchmark --manifest benchmark_specs/authoring_demo.json
```

Clean up prompt files after a run:

```bash
scripts/clean-benchmark-prompts --manifest benchmark_specs/authoring_demo.json --yes
```

If you also kept one example prompt per configuration in in-memory mode, add `--example-prompts`.

## What You Get

- `benchmark_runs/authoring_demo/prompt_bundle.json`
- `benchmark_runs/authoring_demo/response_bundle.json`
- `benchmark_runs/authoring_demo/evaluation_summary.csv`
- `benchmark_runs/authoring_demo/consensus_summary.csv`
- `benchmark_runs/authoring_demo/contamination_audit.csv`

Because this manifest uses in-memory prompting, the full prompt CSVs are not written to disk. Instead:

- prompts are generated lazily, sent to the model, and discarded
- one example prompt per configuration is kept for debugging
- response CSVs remain the stable evaluation artifact

## Required Manifest Fields

- benchmark identity: `name`, `role`, `description`, `task_family`
- dataset definitions: `datasets[]`
- prompt definitions: `prompt_cells[]`
- controls: `names_only`
- execution roster: `models[]`, `baselines[]`
- optional classical-baseline controls: `pc_variant`, `pc_ci_test`, `pc_max_cond_vars`, `ges_scoring_method`
- optional external-LLM-baseline controls: `model`, `provider`, `temperature`, `max_new_tokens`,
  `num_samples`, `edge_threshold`, `takayama_pattern`, `takayama_bootstrap_samples`
- evaluation settings: `evaluator`
- execution policy: `execution`
- provenance metadata: `provenance`

## Notes

- Synthetic datasets are materialized as `.pt` causal graphs under `benchmark_runs/<name>/graphs/`.
- Optional classical baselines require `pgmpy` for `PC` and `GES`.
- Optional external LLM baselines are available as `TakayamaSCP`, `JiralerspongBFS`,
  `CausalLLMPrompt`, and `CausalLLMData`. `CausalLLMPrompt` is a semantic names-only
  baseline, `JiralerspongBFS` is an observational-summary querying baseline, and
  `CausalLLMData` is a one-shot data-backed prompting baseline.
- `TakayamaSCP` is observational-only. `provider: "openai"` is the faithful path because it
  uses token logprobs for yes/no probability extraction. `provider: "illinois"` is also
  supported as a pragmatic fallback that parses yes/no text responses and uses checkpointed resume.
- The demo is for authoring evidence, not for headline paper claims.

## Execution Policy

Set this in the manifest:

```json
"execution": {
  "prompt_storage": "in_memory",
  "prompt_retention": "example"
}
```

Options:

- `prompt_storage: "disk"`: write full prompt CSVs before querying
- `prompt_storage: "in_memory"`: generate prompts lazily and discard them after response capture
- `prompt_retention: "full"`: keep all prompts on disk; only valid with `disk`
- `prompt_retention: "example"`: keep one example prompt per configuration for debugging
- `prompt_retention: "none"`: keep no prompt text after querying
