# Research Contract: ENCO as a Reproducible Sachs-Centered Causal-Discovery Benchmark Artifact for LLMs

## Selected Idea

- **Description**: turn the current ENCO-derived LLM experimentation code into a NeurIPS evaluation-and-dataset paper centered on a reproducible framework for generating causal-discovery datasets, prompt tasks, benchmark slices, and evaluation protocols for LLMs.
- **Source**: existing `ENCO` repository state plus the experiment pipeline already present under `experiments/`.
- **Selection rationale**: the repository already contains graph-backed prompt generation, model querying, evaluation, aggregation, and a classical ENCO anchor baseline. The highest-leverage paper is therefore an artifact-and-benchmark paper, not a new causal-discovery method paper.

## Problem Anchor

- **Bottom-line problem**: LLM causal-discovery results are currently hard to trust because prompt construction, data budgets, naming choices, and evaluation procedures are inconsistent across papers and repos.
- **Must-solve bottleneck**: this repo needs to become a release-ready benchmark artifact with one realistic, reproducible main-paper slice and explicit supporting controls, rather than a loose collection of prompt scripts and partial runs.
- **Non-goals**:
  - a new mainline causal-discovery algorithm
  - a broad claim that the benchmark measures unrestricted causal reasoning
  - a training-heavy GRPO or curriculum paper as the main contribution
  - a broad multi-dataset sweep in the main paper
  - presenting `cancer` as core scientific evidence
- **Constraints**:
  - stay close to what the current repo already runs reliably
  - keep the main paper on one coherent benchmark family
  - use ENCO as the classical scientific anchor, not as the novelty claim
  - prioritize reproducibility and release clarity over breadth
  - keep the paper body on one decisive Sachs slice plus a small number of controls
- **Success condition**: another researcher can reproduce a canonical Sachs slice from the repo, compare LLM outputs against ENCO on the same graph/data regime, and observe that benchmark design choices materially change conclusions.

## Frozen Benchmark Scope

- **Primary dataset**: `sachs`
  - rationale: `sachs` is the smallest realistic non-toy slice in this repo that is still structurally rich enough to make representation and naming controls matter; it also already appears in repo-local prompt, response, and curriculum artifacts
- **Debug-only dataset**: `cancer`
  - role: smoke test and pipeline validation only
  - exclusion: no main-paper tables, no headline claims, no dataset-selection rationale based on `cancer`
- **Primary model for the first decisive slice**:
  - `gpt-5-mini`
- **Supporting second model only after the core Sachs story is stable**:
  - `gemini-2.5-flash`
- **Primary non-LLM anchor**:
  - `ENCO`
- **Main data-backed prompt family**:
  - `summary_joint`
  - `matrix`
- **Supporting controls in the paper body**:
  - anonymized-variable versions of the same Sachs slice
  - `names_only` on Sachs as a prior-only reference condition
- **Single best Sachs benchmark slice to run first**:
  - dataset: `sachs`
  - prompt representation: `summary_joint` vs `matrix`
  - variable semantics: real names
  - data regime: low-budget `obs100/int50`
  - model family: `gpt-5-mini`
  - classical anchor: matched `ENCO`
  - shuffle setting: `shuf1` for the first pass
- **Cancer runs retained only as validation/debugging support**:
  - one end-to-end smoke run on `cancer` with `summary_joint`, real names, `obs100/int50`, `gpt-5-mini`, `shuf1`, and a minimal prompt count
  - one parser/evaluation smoke run on `cancer` `names_only` with `gpt-5-mini`
  - optional `cancer` `matrix` smoke run only when debugging matrix serialization or evaluator regressions
- **Explicitly deferred from the main benchmark body**:
  - `cases`, `summary`, `summary_probs`, `payload`, `payload_topk`
  - a large model zoo or many-provider sweep
  - curriculum / GRPO as a main claim
  - new causal-discovery method development
  - broad ordering-variant sweeps before the core Sachs table is stable
  - any `cancer` result beyond debugging support

## Core Claims

1. The repository provides a reusable and reproducible framework for generating controlled causal-discovery benchmark slices, not just ad hoc prompts.
2. In one tightly matched `sachs` slice, conclusions already depend on prompt representation, so the benchmark is doing more than measuring one prompt template.
3. The artifact enables scientifically grounded comparison because LLMs and ENCO can be evaluated on matched graph/data settings with one shared evaluation pipeline.
4. Naming and prior-knowledge controls on `sachs` matter for interpretation, but `cancer` is only used to validate plumbing and should not carry paper claims.

## Method Summary

The paper is centered on a benchmark artifact, not a new causal-discovery algorithm. The repository already supports:

- graph-backed prompt generation through `experiments/generate_prompts.py` and `experiments/cd_generation/names_only.py`
- end-to-end execution via `experiments/run_experiment1_pipeline.py`
- querying across OpenAI, Gemini, and HF / local model backends
- evaluation and aggregation through `experiments/evaluate.py` and `experiments/collect_results_table.py`
- classical ENCO baselines on the same underlying graph families
- optional curriculum and subproblem scaffolding in `experiments/CD_CURRICULUM_HANDOFF.md` and `experiments/CD_SUBPROBLEM_CURRICULUM_BLUEPRINT.md`

The paper's job is to formalize this repository as a benchmark artifact:

1. define one canonical Sachs-centered benchmark family that the repo can generate and evaluate reproducibly
2. standardize benchmark slices, configs, and output conventions
3. compare a compact primary LLM set against ENCO on matched settings
4. show which benchmark choices change the qualitative conclusion
5. release the artifact in a form other researchers can reuse directly

## Experiment Design

- **Primary artifact**: codebase for generating benchmark slices, prompt CSVs / text files, response files, evaluation summaries, and aggregate result tables
- **Main-paper benchmark slice**:
  - `summary_joint` and `matrix`
  - one primary graph: `sachs`
  - real names
  - low-budget `obs100/int50`
  - `gpt-5-mini`
  - matched `ENCO`
- **Supporting controls in the paper body**:
  - anonymized names on the same Sachs `obs100/int50` slice
  - `names_only` on Sachs as a prior-only reference probe
- **Debug-only support outside the paper evidence chain**:
  - `cancer` smoke runs for prompt generation, API querying, parser extraction, evaluation, and aggregation checks
- **Deferred until the core Sachs slice is working end-to-end**:
  - `gemini-2.5-flash`
  - higher-budget Sachs confirmations
  - curriculum/subproblem task demonstrations
  - legacy prompt-style families
- **Main-paper table shape**:
  - primary panel rows: `summary_joint` and `matrix` in the matched `sachs`, real-name, `obs100/int50` slice
  - primary panel columns: `gpt-5-mini` and `ENCO`
  - supporting panel: anonymized-name variants and `names_only` on Sachs
- **Not part of the paper body**:
  - `cancer` tables
  - large dataset sweeps
  - large provider sweeps

## Release Surface

The benchmark release should make the following first-class:

- benchmark definition document and canonical slice list
- generation commands or configs for each canonical Sachs slice
- generated prompt CSVs / prompt text files
- response CSV schema and evaluation schema
- aggregate summary tables under `experiments/out/experiment1/`
- matched ENCO baseline outputs
- reproduction instructions for:
  - one Sachs slice
  - the full main-body table
  - adding a new model backend
- explicit documentation that `cancer` is for smoke tests only

## Baselines

| System | Role | Status | Source |
|--------|------|--------|--------|
| ENCO | classical anchor baseline | implemented | `causal_discovery/`, `experiments/run_exported_graphs.py` |
| `gpt-5-mini` | primary closed-model benchmark runner | implemented | `experiments/run_experiment1_pipeline.py`, `experiments/query_api.py` |
| `gemini-2.5-flash` | second API benchmark runner | implemented | `experiments/run_experiment1_pipeline.py`, `experiments/query_api.py` |
| Curriculum launcher | appendix-level extensibility evidence | partially implemented | `experiments/run_cd_curriculum.py` |

## Current Results

| Area | Status | Notes |
|------|--------|-------|
| prompt generation | present | graph-backed generation exists |
| API querying | present | OpenAI and Gemini paths exist |
| evaluation | present but not frozen on the paper slice | per-CSV metrics exist, but the canonical `sachs` aggregate artifact is not yet in place |
| classical baseline | present | ENCO runs can be matched to graph/data settings |
| Sachs artifacts | partial | repo-local Sachs responses and curriculum scaffolding exist, but not yet as a canonical main-paper table |
| cancer smoke path | useful but non-evidentiary | keep only for end-to-end validation and debugging |
| benchmark story | now tighter | the first decisive slice is `sachs` + `summary_joint` / `matrix` + real names + `obs100/int50` + `gpt-5-mini` + `ENCO`; supporting controls stay on Sachs; `cancer` is debug-only |
| release packaging | incomplete | needs canonical slice configs, output docs, and artifact-facing README cleanup |
| curriculum extensibility | partial | keep out of the main paper body unless needed later |

## Key Decisions

- The paper's dominant contribution is the **artifact and benchmark**, not a new causal-discovery method.
- The main paper is anchored on **one Sachs slice**, not on a toy-first `cancer` result.
- The first decisive benchmark slice stays on data-backed `summary_joint` and `matrix`; `names_only` is a supporting control, not a co-equal main condition.
- `cancer` is retained only for smoke tests and debugging support; it must not carry core paper claims.
- The first-pass comparison set is intentionally minimal: `gpt-5-mini` and `ENCO`; `gemini-2.5-flash` is supporting confirmation only after the core Sachs slice works.
- Curriculum/subproblem tasks remain appendix-level extensibility evidence unless the artifact story is already complete.
- Legacy prompt styles remain in the repo, but they are not part of the canonical NeurIPS benchmark claim.
- The paper must explicitly separate:
  - what the codebase generates
  - what the benchmark evaluates
  - what the results imply about current LLM capabilities
  - what was run only for validation/debugging

## Status

- [x] Repo exists with core generation and evaluation pipeline
- [x] Paper direction chosen
- [x] Research contract created
- [x] Benchmark scope frozen
- [x] First decisive Sachs benchmark slice specified
- [x] Cancer demoted to smoke-test/debug-only status
- [x] Missing release/documentation components identified
- [ ] Canonical Sachs benchmark configs added
- [ ] Supporting Sachs controls frozen after the first slice is run
- [ ] Main tables complete
- [ ] Narrative report drafted
- [ ] Paper writing started
