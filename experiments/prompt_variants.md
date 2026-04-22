# Prompt Variants

Prompt rendering is now split into:

- `content`: the actual causal-discovery prompt body
- `wrapper-mode`: how that body is transported to the model
- `reasoning-target`: what completion SFT supervises

The same prompt body is used across evaluation, SFT, and GRPO. The only intended
differences are:

- evidence representation: `summary`, `matrix`, or `names-only`
- wrapper transport: `plain` or `chat`
- completion supervision: training may prefill `<think>` and choose a reasoning target

## Unified Settings

| Setting | Values | Used by | Purpose |
|---|---|---|---|
| `--wrapper-mode` | `plain` · `chat` | prompt generators, SFT builder | Controls whether prompts are raw text or `system/user/assistant` chat |
| `--append-format-hint` | flag | prompt generators, in-memory eval runner | Appends the canonical `Formatting requirement:` line; for causal discovery this enables the optional staged reasoning instructions |
| `--reasoning-target` | `answer_only` · `stages` · `stages_evidence` | `collect_format_sft_data.py` | Controls the supervised completion style |

### Training support note

Prompt generation now always uses the `think_answer` contract. `collect_format_sft_data.py`
and GRPO causal-discovery tasks therefore expect
`<think>...</think><answer>...</answer>` outputs.

## Querying Prompts

The examples below were generated from the current code using:

```bash
--bif-file causal_graphs/real_data/small_graphs/cancer.bif
--seed 42
--num-prompts 1
--shuffles-per-graph 1
--obs-per-prompt 8
--int-per-combo 4
--row-order sorted
```

### Summary, `wrapper-mode=plain`

```bash
python experiments/generate_prompts.py \
    --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
    --prompt-style summary \
    --wrapper-mode plain \
    --obs-per-prompt 8 \
    --int-per-combo 4 \
    --row-order sorted \
    --seed 42 \
    --num-prompts 1 \
    --shuffles-per-graph 1 \
    --out-dir experiments/prompts/cancer_summary_plain_think
```

```text
ROLE: You are an expert in causal discovery from observational and interventional data.
TASK: Infer the directed causal graph over the variables.
The following are empirical distributions computed from data sampled from a Bayesian network named cancer.
ASSUMPTIONS:
- The true graph is a DAG (no directed cycles).
- Causal sufficiency holds (no unobserved confounders among these variables).
- Interventions are perfect do-interventions (surgical): do(X=v) cuts all incoming edges into X.

--- VARIABLE ORDER (ORDER MATTERS) ---
0: Pollution
1: Smoker
2: Cancer
3: Xray
4: Dyspnoea

--- DATA FORMAT ---
Each payload summarizes empirical samples in the declared variable order.
num_states=[2,2,2,2,2]. hist entries use [assignment, count]; assignments follow the declared variable order.
observational_data summarizes samples with no intervention. interventional_data maps each do(X=v) regime to its empirical summary. marginals[j][s]=P(variable_j=s). Unlisted assignments may be absent due to finite sampling.

--- STATE LEGEND ---
Pollution: 0->low, 1->high
Smoker: 0->True, 1->False
Cancer: 0->True, 1->False
Xray: 0->positive, 1->negative
Dyspnoea: 0->True, 1->False

--- OBSERVATIONAL DATA ---
observational_data={
  "n": 8,
  "hist": [[[0,0,1,0,1],2],[[0,1,1,1,1],2],[[0,0,1,0,0],1],[[0,0,1,1,1],1],[[0,1,1,1,0],1],[[1,1,1,1,1],1]],
  "marginals": [[0.875,0.125],[0.5,0.5],[0.0,1.0],[0.375,0.625],[0.25,0.75]]
}

--- INTERVENTIONAL DATA ---
interventional_data={
  "do(Cancer=False)": {"n":2,"hist":[[[0,0,1,0,1],1],[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.5,0.5],[0.0,1.0],[0.5,0.5],[0.0,1.0]]},
  "do(Cancer=True)": {"n":2,"hist":[[[0,0,0,0,0],1],[[0,1,0,0,0],1]],"marginals":[[1.0,0.0],[0.5,0.5],[1.0,0.0],[1.0,0.0],[1.0,0.0]]},
  "do(Dyspnoea=False)": {"n":3,"hist":[[[0,0,1,0,1],1],[[0,0,1,1,1],1],[[1,0,1,1,1],1]],"marginals":[[0.666667,0.333333],[1.0,0.0],[0.0,1.0],[0.333333,0.666667],[0.0,1.0]]},
  "do(Dyspnoea=True)": {"n":1,"hist":[[[0,0,1,1,0],1]],"marginals":[[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]},
  "do(Pollution=high)": {"n":3,"hist":[[[1,0,1,1,1],2],[[1,1,1,0,1],1]],"marginals":[[0.0,1.0],[0.666667,0.333333],[0.0,1.0],[0.333333,0.666667],[0.0,1.0]]},
  "do(Pollution=low)": {"n":1,"hist":[[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]},
  "do(Smoker=False)": {"n":3,"hist":[[[0,1,1,0,1],1],[[0,1,1,1,0],1],[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.333333,0.666667],[0.333333,0.666667]]},
  "do(Smoker=True)": {"n":1,"hist":[[[0,0,1,1,0],1]],"marginals":[[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]},
  "do(Xray=negative)": {"n":2,"hist":[[[0,1,1,1,1],2]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]},
  "do(Xray=positive)": {"n":2,"hist":[[[0,1,1,0,0],1],[[0,1,1,0,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0],[0.5,0.5]]}
}

--- END OF DATA ---

--- OUTPUT INSTRUCTIONS ---
Output exactly: <think>...</think><answer>...</answer>.
Keep <think> concise (minimal necessary reasoning only).
Inside <answer>, output exactly one JSON object with key "adjacency_matrix".
- "adjacency_matrix": N x N 0/1 matrix in declared variable order.
No extra text before, between, or after the two blocks.
The JSON in <answer> must start with "{" and end with "}".
```

### Summary, `wrapper-mode=chat`

```bash
python experiments/generate_prompts.py \
    --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
    --prompt-style summary \
    --wrapper-mode chat \
    --append-format-hint \
    --obs-per-prompt 8 \
    --int-per-combo 4 \
    --row-order sorted \
    --seed 42 \
    --num-prompts 1 \
    --shuffles-per-graph 1 \
    --out-dir experiments/prompts/cancer_summary_chat_think
```

With `--append-format-hint`, chat mode strips the redundant `--- OUTPUT INSTRUCTIONS ---`
block and replaces it with a compact `Formatting requirement:` line in the wrapped prompt:

```text
system
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the solution and then provides the final answer.
user
ROLE: You are an expert in causal discovery from observational and interventional data.
TASK: Infer the directed causal graph over the variables.
The following are empirical distributions computed from data sampled from a Bayesian network named cancer.
ASSUMPTIONS:
- The true graph is a DAG (no directed cycles).
- Causal sufficiency holds (no unobserved confounders among these variables).
- Interventions are perfect do-interventions (surgical): do(X=v) cuts all incoming edges into X.

--- VARIABLE ORDER (ORDER MATTERS) ---
0: Pollution
1: Smoker
2: Cancer
3: Xray
4: Dyspnoea

--- DATA FORMAT ---
Each payload summarizes empirical samples in the declared variable order.
num_states=[2,2,2,2,2]. hist entries use [assignment, count]; assignments follow the declared variable order.
observational_data summarizes samples with no intervention. interventional_data maps each do(X=v) regime to its empirical summary. marginals[j][s]=P(variable_j=s). Unlisted assignments may be absent due to finite sampling.

--- STATE LEGEND ---
Pollution: 0->low, 1->high
Smoker: 0->True, 1->False
Cancer: 0->True, 1->False
Xray: 0->positive, 1->negative
Dyspnoea: 0->True, 1->False

--- OBSERVATIONAL DATA ---
observational_data={
  "n": 8,
  "hist": [[[0,0,1,0,1],2],[[0,1,1,1,1],2],[[0,0,1,0,0],1],[[0,0,1,1,1],1],[[0,1,1,1,0],1],[[1,1,1,1,1],1]],
  "marginals": [[0.875,0.125],[0.5,0.5],[0.0,1.0],[0.375,0.625],[0.25,0.75]]
}

--- INTERVENTIONAL DATA ---
interventional_data={
  "do(Cancer=False)": {"n":2,"hist":[[[0,0,1,0,1],1],[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.5,0.5],[0.0,1.0],[0.5,0.5],[0.0,1.0]]},
  "do(Cancer=True)": {"n":2,"hist":[[[0,0,0,0,0],1],[[0,1,0,0,0],1]],"marginals":[[1.0,0.0],[0.5,0.5],[1.0,0.0],[1.0,0.0],[1.0,0.0]]},
  "do(Dyspnoea=False)": {"n":3,"hist":[[[0,0,1,0,1],1],[[0,0,1,1,1],1],[[1,0,1,1,1],1]],"marginals":[[0.666667,0.333333],[1.0,0.0],[0.0,1.0],[0.333333,0.666667],[0.0,1.0]]},
  "do(Dyspnoea=True)": {"n":1,"hist":[[[0,0,1,1,0],1]],"marginals":[[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]},
  "do(Pollution=high)": {"n":3,"hist":[[[1,0,1,1,1],2],[[1,1,1,0,1],1]],"marginals":[[0.0,1.0],[0.666667,0.333333],[0.0,1.0],[0.333333,0.666667],[0.0,1.0]]},
  "do(Pollution=low)": {"n":1,"hist":[[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]},
  "do(Smoker=False)": {"n":3,"hist":[[[0,1,1,0,1],1],[[0,1,1,1,0],1],[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.333333,0.666667],[0.333333,0.666667]]},
  "do(Smoker=True)": {"n":1,"hist":[[[0,0,1,1,0],1]],"marginals":[[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]},
  "do(Xray=negative)": {"n":2,"hist":[[[0,1,1,1,1],2]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]},
  "do(Xray=positive)": {"n":2,"hist":[[[0,1,1,0,0],1],[[0,1,1,0,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0],[0.5,0.5]]}
}

--- END OF DATA ---

Formatting requirement: Reason in three stages inside <think>: Stage 1 (Skeleton) - one "X -- Y" per line; Stage 2 (V-structures) - one "(parent1, collider, parent2)" per line; Stage 3 (Orientation) - one "X -> Y" per line. Write "None" for any empty stage. Then output: <answer>{"adjacency_matrix": [[0,1,...],[0,0,...],...]}</answer> where the matrix is N×N with integer entries 0 or 1 in VARIABLES order, and [i][j]=1 means variable i directly causes variable j.
assistant
<think>
```

### Matrix, `wrapper-mode=plain`

```bash
python experiments/generate_prompts.py \
    --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
    --prompt-style matrix \
    --wrapper-mode plain \
    --obs-per-prompt 8 \
    --int-per-combo 4 \
    --row-order sorted \
    --seed 42 \
    --num-prompts 1 \
    --shuffles-per-graph 1 \
    --out-dir experiments/prompts/cancer_matrix_plain_think
```

```text
ROLE: You are an expert in causal discovery from observational and interventional data.
TASK: Infer the directed causal graph over the variables.
The following are empirical distributions computed from data sampled from a Bayesian network named cancer.
ASSUMPTIONS:
- The true graph is a DAG (no directed cycles).
- Causal sufficiency holds (no unobserved confounders among these variables).
- Interventions are perfect do-interventions (surgical): do(X=v) cuts all incoming edges into X.

--- VARIABLE ORDER (ORDER MATTERS) ---
0: Pollution
1: Smoker
2: Cancer
3: Xray
4: Dyspnoea

--- OBSERVATIONAL DATA (no intervention) ---
Each row is one observed case.
Pollution | Smoker | Cancer | Xray | Dyspnoea
high | False | False | negative | False
low | False | False | negative | False
low | False | False | negative | False
low | False | False | negative | True
low | True | False | negative | False
low | True | False | positive | False
low | True | False | positive | False
low | True | False | positive | True

--- INTERVENTIONAL DATA ---
Each block corresponds to samples collected under a perfect intervention do(X = value).

[Intervention: do(Cancer = False)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
low | False | False | negative | False
low | True | False | positive | False

[Intervention: do(Cancer = True)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
low | False | True | positive | True
low | True | True | positive | True

[Intervention: do(Dyspnoea = False)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
high | True | False | negative | False
low | True | False | negative | False
low | True | False | positive | False

[Intervention: do(Dyspnoea = True)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
low | True | False | negative | True

[Intervention: do(Pollution = high)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
high | False | False | positive | False
high | True | False | negative | False
high | True | False | negative | False

[Intervention: do(Pollution = low)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
low | False | False | negative | False

[Intervention: do(Smoker = False)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
low | False | False | negative | False
low | False | False | negative | True
low | False | False | positive | False

[Intervention: do(Smoker = True)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
low | True | False | negative | True

[Intervention: do(Xray = negative)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
low | False | False | negative | False
low | False | False | negative | False

[Intervention: do(Xray = positive)]
Pollution | Smoker | Cancer | Xray | Dyspnoea
low | False | False | positive | False
low | False | False | positive | True

--- END OF DATA ---

--- OUTPUT INSTRUCTIONS ---
Output exactly: <think>...</think><answer>...</answer>.
Keep <think> concise (minimal necessary reasoning only).
Inside <answer>, output exactly one JSON object with key "adjacency_matrix".
- "adjacency_matrix": N x N 0/1 matrix in declared variable order.
No extra text before, between, or after the two blocks.
The JSON in <answer> must start with "{" and end with "}".
```

### Names-only, `wrapper-mode=plain`

```bash
python experiments/cd_generation/names_only.py \
    --bif-file causal_graphs/real_data/small_graphs/cancer.bif \
    --wrapper-mode plain \
    --out-dir experiments/prompts/cancer_names_only_plain_think \
    --seed 42 \
    --num-prompts 1
```

```text
ROLE: You are an expert in causal discovery from variable semantics and background knowledge.
TASK: Infer the directed causal graph over the variables.
We are studying a system called 'cancer'.
No observational or interventional data are provided for this case. Use only the variable meanings and relevant background causal knowledge.
ASSUMPTIONS:
- The true graph is a DAG (no directed cycles).
- Causal sufficiency holds (no unobserved confounders among these variables).

--- VARIABLE ORDER (ORDER MATTERS) ---
0: Pollution
1: Smoker
2: Cancer
3: Xray
4: Dyspnoea

--- AVAILABLE EVIDENCE ---
No sampled data are provided. The graph must be inferred from the variable names and domain knowledge alone.

--- OUTPUT INSTRUCTIONS ---
Output exactly: <think>...</think><answer>...</answer>.
Keep <think> concise (minimal necessary reasoning only).
Inside <answer>, output exactly one JSON object with key "adjacency_matrix".
- "adjacency_matrix": N x N 0/1 matrix in declared variable order.
No extra text before, between, or after the two blocks.
The JSON in <answer> must start with "{" and end with "}".
```

## Training Prompts (SFT / GRPO)

SFT now uses the same underlying prompt body as evaluation and only changes:

- wrapper transport
- assistant prefill through `<think>`
- supervised completion style via `--reasoning-target`

GRPO still exposes `--cd-wrapper-mode` and `--cd-response-format`, but causal-discovery
training currently supports only `think_answer`.

### SFT example, `wrapper-mode=chat`, `reasoning-target=stages`

```bash
python experiments/collect_format_sft_data.py \
    --graphs cancer \
    --prompt-style summary \
    --anonymize \
    --wrapper-mode chat \
    --reasoning-target stages \
    --obs-values 8 \
    --int-values 4 \
    --col-perms 1 \
    --num-prompts-per-config 1 \
    --output experiments/data/format_sft_example.jsonl
```

**Generated JSONL `prompt` field**

```text
system
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the solution and then provides the final answer.
user
ROLE: You are an expert in causal discovery from observational and interventional data.
TASK: Infer the directed causal graph over the variables.
The following are empirical distributions computed from data sampled from an anonymized Bayesian network.
ASSUMPTIONS:
- The true graph is a DAG (no directed cycles).
- Causal sufficiency holds (no unobserved confounders among these variables).
- Interventions are perfect do-interventions (surgical): do(X=v) cuts all incoming edges into X.

--- VARIABLE ORDER (ORDER MATTERS) ---
0: X1
1: X2
2: X3
3: X4
4: X5

--- DATA FORMAT ---
Each payload summarizes empirical samples in the declared variable order.
num_states=[2,2,2,2,2]. hist entries use [assignment, count]; assignments follow the declared variable order.
observational_data summarizes samples with no intervention. interventional_data maps each do(X=v) regime to its empirical summary. marginals[j][s]=P(variable_j=s). Unlisted assignments may be absent due to finite sampling.

--- OBSERVATIONAL DATA ---
observational_data={
  "n": 8,
  "hist": [[[0,0,1,0,1],2],[[0,1,1,1,1],2],[[0,0,1,0,0],1],[[0,0,1,1,1],1],[[0,1,1,1,0],1],[[1,1,1,1,1],1]],
  "marginals": [[0.875,0.125],[0.5,0.5],[0.0,1.0],[0.375,0.625],[0.25,0.75]]
}

--- INTERVENTIONAL DATA ---
interventional_data={
  "do(X1=0)": {"n":1,"hist":[[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]},
  "do(X1=1)": {"n":3,"hist":[[[1,0,1,1,1],2],[[1,1,1,0,1],1]],"marginals":[[0.0,1.0],[0.666667,0.333333],[0.0,1.0],[0.333333,0.666667],[0.0,1.0]]},
  "do(X2=0)": {"n":1,"hist":[[[0,0,1,1,0],1]],"marginals":[[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]},
  "do(X2=1)": {"n":3,"hist":[[[0,1,1,0,1],1],[[0,1,1,1,0],1],[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.333333,0.666667],[0.333333,0.666667]]},
  "do(X3=0)": {"n":2,"hist":[[[0,0,0,0,0],1],[[0,1,0,0,0],1]],"marginals":[[1.0,0.0],[0.5,0.5],[1.0,0.0],[1.0,0.0],[1.0,0.0]]},
  "do(X3=1)": {"n":2,"hist":[[[0,0,1,0,1],1],[[0,1,1,1,1],1]],"marginals":[[1.0,0.0],[0.5,0.5],[0.0,1.0],[0.5,0.5],[0.0,1.0]]},
  "do(X4=0)": {"n":2,"hist":[[[0,1,1,0,0],1],[[0,1,1,0,1],1]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0],[0.5,0.5]]},
  "do(X4=1)": {"n":2,"hist":[[[0,1,1,1,1],2]],"marginals":[[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]},
  "do(X5=0)": {"n":1,"hist":[[[0,0,1,1,0],1]],"marginals":[[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]},
  "do(X5=1)": {"n":3,"hist":[[[0,0,1,0,1],1],[[0,0,1,1,1],1],[[1,0,1,1,1],1]],"marginals":[[0.666667,0.333333],[1.0,0.0],[0.0,1.0],[0.333333,0.666667],[0.0,1.0]]}
}

--- END OF DATA ---

Formatting requirement: Reason in three stages inside <think>: Stage 1 (Skeleton) - one "X -- Y" per line; Stage 2 (V-structures) - one "(parent1, collider, parent2)" per line; Stage 3 (Orientation) - one "X -> Y" per line. Write "None" for any empty stage. Then output: <answer>{"adjacency_matrix": [[0,1,...],[0,0,...],...]}</answer> where the matrix is N×N with integer entries 0 or 1 in VARIABLES order, and [i][j]=1 means variable i directly causes variable j.
assistant
<think>
```

**Generated JSONL `answer` field**

```text
Stage 1 (Skeleton):
X1 -- X3
X2 -- X3
X3 -- X4
X3 -- X5

Stage 2 (V-structures):
(X1, X3, X2)

Stage 3 (Orientation):
X1 -> X3
X2 -> X3
X3 -> X4
X3 -> X5</think><answer>{"adjacency_matrix": [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]}</answer>
```
