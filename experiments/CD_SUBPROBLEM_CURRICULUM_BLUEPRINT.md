# Causal Discovery Subproblem Curriculum Blueprint

This note turns the earlier curriculum discussion into a concrete training plan for this repo.

The design target here is:

- same core assumptions as ENCO for the main graph task
- DAG
- causal sufficiency
- perfect do-interventions
- observational + interventional summaries as the input modality
- supervised warm start first, GRPO later

The goal is not to replace the existing `causal_discovery` task immediately. The goal is to add a sequence of easier, denser supervision tasks that transfer into it.

## Why This Fits The Current Repo

The current code already supports:

- staged curricula via `experiments/run_cd_curriculum.py`
- one subproblem task via `cd_descendants`
- causal-discovery prompts based on the same summary/matrix evidence later used for full graph prediction
- GRPO reward plumbing by task

The missing pieces are:

- more task generators
- more task-specific parsers/rewards
- a curriculum that ramps task difficulty, not only `obs/int`

## Recommended Task Inventory

Use these task ids.

| Task id | Output target | Label source | Why it transfers |
| --- | --- | --- | --- |
| `cd_intervention_effects` | changed variable set under `do(X=v)` | compare observational vs interventional marginals | teaches intervention parsing and causal effect semantics |
| `cd_descendants` | descendant set of intervention target | transitive closure of ground-truth DAG | teaches ancestor-descendant asymmetry |
| `cd_edge_exists` | whether a direct edge exists between `X,Y` | adjacency matrix | teaches skeleton recovery locally |
| `cd_edge_orientation` | `X->Y`, `Y->X`, `none`, optional `ambiguous` | adjacency matrix | teaches edge direction directly |
| `cd_vstructures` | whether `X->Z<-Y` holds | adjacency matrix | teaches collider structure |
| `causal_discovery` | full DAG adjacency matrix | adjacency matrix | final ENCO-style target |

Optional later tasks if you broaden beyond ENCO assumptions:

| Task id | Output target | Use only if |
| --- | --- | --- |
| `cd_cpdag` | equivalence-class graph | observational-only or mixed-identifiability regime |
| `cd_ambiguity` | abstain/identified flags | finite-sample uncertainty becomes a first-class objective |
| `cd_ci` | CI statement truth value | you expose enough information for CI judgments, or generate explicit CI summaries |

## Task Schemas

Keep all tasks in the same `<think>...</think><answer>...</answer>` wrapper used today. Only the JSON payload changes by task.

### 1. `cd_intervention_effects`

```json
{
  "target": "PKC",
  "intervention_value": "HIGH",
  "changed": ["PKA", "Raf", "Mek"],
  "unchanged": ["Plcg", "P38", "Jnk"],
  "evidence_metric": "tv_marginal",
  "threshold": 0.05
}
```

Notes:

- `unchanged` is optional. Start with `target` + `changed` only if you want the easiest version.
- This task can be generated from prompt summaries alone, without asking the model to infer graph structure.

### 2. `cd_descendants`

Already implemented. Keep the current schema:

```json
{
  "target": "PKC",
  "descendants": ["PKA", "Raf", "Mek", "Erk"]
}
```

### 3. `cd_edge_exists`

```json
{
  "source": "PKC",
  "target": "Raf",
  "direct_edge": true
}
```

Variant with abstention if you later relax assumptions:

```json
{
  "source": "PKC",
  "target": "Raf",
  "relation": "edge"
}
```

where `relation` is one of `edge`, `no_edge`, `ambiguous`.

### 4. `cd_edge_orientation`

```json
{
  "source": "PKC",
  "target": "Raf",
  "orientation": "source_to_target"
}
```

Use:

- `source_to_target`
- `target_to_source`
- `no_edge`
- optional `ambiguous` if you broaden the regime later

### 5. `cd_vstructures`

```json
{
  "x": "PKC",
  "z": "Mek",
  "y": "P38",
  "is_v_structure": false
}
```

Positive means:

- `x -> z`
- `y -> z`
- no edge between `x` and `y`

### 6. `causal_discovery`

Keep the current schema under ENCO assumptions:

```json
{
  "adjacency_matrix": [[0,1,0],[0,0,1],[0,0,0]]
}
```

## Label Generation Rules From The Existing Simulator

All of these labels can be generated from the existing prompt pipeline plus the current DAG answer object.

### `cd_intervention_effects`

Inputs:

- observational marginals
- interventional marginals for `do(X=v)`

Rule:

- compute per-variable marginal TV distance
- mark `changed` if `TV(obs_j, do_j) > epsilon`

This is the easiest new task to add because it only needs prompt-side statistics and no extra graph algorithm.

### `cd_descendants`

Inputs:

- adjacency matrix

Rule:

- take graph transitive closure from the intervened node

This is already implemented in `experiments/build_cd_descendant_tasks.py`.

### `cd_edge_exists`

Inputs:

- adjacency matrix

Rule:

- for pair `(i, j)`, label true if `adj[i][j] == 1 or adj[j][i] == 1`

Sampling:

- half positive pairs
- half negative pairs
- keep pair order randomized so the model cannot overfit to `(parent, child)` ordering

### `cd_edge_orientation`

Inputs:

- adjacency matrix

Rule:

- `source_to_target` if `adj[i][j] == 1`
- `target_to_source` if `adj[j][i] == 1`
- `no_edge` otherwise

Sampling:

- use more positives than negatives at first
- later add harder negatives where there is a path but no direct edge

### `cd_vstructures`

Inputs:

- adjacency matrix

Rule:

- positive iff `adj[x][z] == 1`, `adj[y][z] == 1`, `adj[z][x] == 0`, `adj[z][y] == 0`, and no adjacency between `x` and `y`

Sampling:

- oversample positives because they are rarer

## Suggested Prompt Shapes

Use the same evidence family across tasks so transfer is real.

Start with `summary_joint`.

Why:

- it is shorter than raw rows
- it already encodes the causal assumptions explicitly
- it is the same family used in the current full-graph task

Then add `matrix`.

Do not start with anonymized prompts. Add anonymization only after the model is competent on named variables.

## Stage-By-Stage Curriculum

This is the recommended curriculum under ENCO-style assumptions.

### Phase A: Intervention Semantics

Goal:

- make the model reliably read intervention blocks and map them to downstream effects

Stages:

1. `cd_intervention_effects`, named, `summary_joint`, `obs50/int10`, SFT only
2. `cd_descendants`, named, `summary_joint`, `obs50/int10`, SFT only
3. `cd_descendants`, named, `summary_joint`, `obs100/int10`, SFT + light GRPO
4. `cd_descendants`, named, `summary_joint`, `obs100/int50`, SFT + GRPO

Promotion gate:

- `format_rate >= 0.95`
- task accuracy clearly above trivial baseline

### Phase B: Local Graph Structure

Goal:

- move from downstream-effect reasoning to local graph judgments

Stages:

5. `cd_edge_exists`, named, `summary_joint`, `obs100/int10`, SFT only
6. `cd_edge_orientation`, named, `summary_joint`, `obs100/int10`, SFT + light GRPO
7. `cd_vstructures`, named, `summary_joint`, `obs100/int50`, SFT + GRPO
8. mixed replay of `cd_descendants` + `cd_edge_exists` + `cd_edge_orientation`

Promotion gate:

- high format compliance
- pairwise accuracy stable on held-out data

### Phase C: Full Graph Recovery

Goal:

- compose local decisions into ENCO-style DAG output

Stages:

9. `causal_discovery`, named, `summary_joint`, `obs50/int10`, SFT only
10. `causal_discovery`, named, `summary_joint`, `obs100/int10`, SFT + light GRPO
11. `causal_discovery`, named, `summary_joint`, `obs100/int50`, SFT + GRPO
12. `causal_discovery`, named, `matrix`, `obs100/int50`, SFT + GRPO
13. `causal_discovery`, anonymized, mixed `summary_joint`/`matrix`, `obs100/int50`, SFT + GRPO
14. `causal_discovery`, named, mixed styles, `obs200/int100`, SFT + GRPO

Promotion gate:

- `format_rate >= 0.95`
- graph accuracy or graph reward improving
- clipped ratio under control

### Phase D: Robustness

Goal:

- make the model less brittle to representation shifts

Add:

- row-order variations
- anonymization
- larger sample sizes
- mixed prompt styles

Keep replay within the same semantic neighborhood:

- local tasks replay into early full-graph stages
- full-graph replay into later robustness stages

## Runner And Code Changes Needed

These are the minimal code changes to support the above.

### 1. Extend supported tasks

File:

- `experiments/run_cd_curriculum.py`

Change:

- extend `SUPPORTED_TASKS` with:
  - `cd_intervention_effects`
  - `cd_edge_exists`
  - `cd_edge_orientation`
  - `cd_vstructures`

### 2. Add task builders

Best pattern:

- mirror `experiments/build_cd_descendant_tasks.py`

Suggested new files:

- `experiments/build_cd_intervention_effect_tasks.py`
- `experiments/build_cd_edge_tasks.py`
- `experiments/build_cd_vstructure_tasks.py`

### 3. Add prompt formatters

File:

- `experiments/generate_prompts.py`

Add:

- `format_prompt_intervention_effects_summary(...)`
- `format_prompt_edge_exists_summary(...)`
- `format_prompt_edge_orientation_summary(...)`
- `format_prompt_vstructure_summary(...)`

Keep their tone and evidence blocks close to `format_prompt_descendants_summary(...)`.

### 4. Add verifier parsers and rewards

Files:

- `experiments/verifier_cd.py`
- `experiments/grpo.py`

Add:

- payload extractors for each new schema
- task-specific partial-format rewards
- task-specific accuracy/F1 rewards

Recommended reward shapes:

- `cd_intervention_effects`: set F1 on changed variables
- `cd_edge_exists`: binary accuracy
- `cd_edge_orientation`: categorical accuracy
- `cd_vstructures`: binary accuracy

### 5. Keep SFT-first handoff

File:

- `experiments/run_sft_then_grpo.py`

Current behavior is already compatible with task-specific JSON payloads as long as the conversion path is generalized beyond adjacency matrices.

## Immediate Priority Order

Do these in this order.

1. `cd_intervention_effects`
2. `cd_edge_exists`
3. `cd_edge_orientation`
4. `cd_vstructures`
5. mixed curriculum manifest

Reason:

- `cd_intervention_effects` is the cheapest new task to implement
- `cd_edge_exists` and `cd_edge_orientation` provide the most direct transfer into full DAG prediction
- `cd_vstructures` adds the main observational structure pattern without requiring a full graph decoder

## Minimal Runnable Curriculum Today

With current code, the usable transfer curriculum is:

1. existing `cd_descendants` stages from `experiments/cd_descendants_sachs_train.json`
2. hand off the promoted checkpoint into a `causal_discovery` manifest using the existing summary-joint DAG data

That is weaker than the full plan above, but it already moves training from:

- graph-only

to:

- intervention semantics first
- full graph second

## Example Future Manifest Shape

This is the manifest shape to use once the new tasks are implemented.

```json
{
  "grpo_script": "experiments/grpo.py",
  "base_model": "unsloth/qwen3-4b-thinking-2507-unsloth-bnb-4bit",
  "task": "cd_intervention_effects",
  "stages": [
    {
      "name": "a1_intervention_effects_named_obs50_int10",
      "task": "cd_intervention_effects",
      "train_csv": "experiments/prompts/cd_intervention_effects/sachs/stage_1_train.csv",
      "eval_csv": "experiments/prompts/cd_intervention_effects/sachs/stage_1_eval.csv",
      "enable_sft": true,
      "enable_grpo": false,
      "replay_ratio": 0.0
    },
    {
      "name": "a2_descendants_named_obs100_int10",
      "task": "cd_descendants",
      "train_csv": "experiments/prompts/cd_descendants/sachs/splits/stage_2_named_obs100_int10_train.csv",
      "eval_csv": "experiments/prompts/cd_descendants/sachs/splits/stage_2_named_obs100_int10_eval.csv",
      "enable_sft": true,
      "enable_grpo": true,
      "replay_ratio": 0.1
    },
    {
      "name": "b1_edge_exists_named_obs100_int10",
      "task": "cd_edge_exists",
      "train_csv": "experiments/prompts/cd_edge_exists/sachs/stage_1_train.csv",
      "eval_csv": "experiments/prompts/cd_edge_exists/sachs/stage_1_eval.csv",
      "enable_sft": true,
      "enable_grpo": false,
      "replay_ratio": 0.1
    },
    {
      "name": "b2_edge_orientation_named_obs100_int10",
      "task": "cd_edge_orientation",
      "train_csv": "experiments/prompts/cd_edge_orientation/sachs/stage_1_train.csv",
      "eval_csv": "experiments/prompts/cd_edge_orientation/sachs/stage_1_eval.csv",
      "enable_sft": true,
      "enable_grpo": true,
      "replay_ratio": 0.15
    },
    {
      "name": "b3_vstructures_named_obs100_int50",
      "task": "cd_vstructures",
      "train_csv": "experiments/prompts/cd_vstructures/sachs/stage_1_train.csv",
      "eval_csv": "experiments/prompts/cd_vstructures/sachs/stage_1_eval.csv",
      "enable_sft": true,
      "enable_grpo": true,
      "replay_ratio": 0.15
    },
    {
      "name": "c1_graph_named_obs100_int10",
      "task": "causal_discovery",
      "train_csv": "experiments/prompts/experiment1/sachs/summary_real_obs100_int10/prompts_obs100_int10_shuf1_p100_thinktags_summary_joint.csv",
      "eval_csv": "experiments/prompts/experiment1/sachs/summary_real_obs100_int10/prompts_obs100_int10_shuf1_p100_thinktags_summary_joint.csv",
      "enable_sft": true,
      "enable_grpo": true,
      "replay_ratio": 0.2
    }
  ]
}
```

## Recommended Next Implementation Step

If you want the highest-value next change, implement `cd_edge_exists` first.

Why:

- simplest label generation after descendants
- strong transfer to graph recovery
- much denser and less ambiguous supervision than full-DAG prediction
- easy reward function

