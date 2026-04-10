# NeurIPS Evaluations & Datasets Checklist For A Causal Discovery Benchmark

This checklist is tailored to this repo's research direction:

- causal discovery from observational and interventional data
- LLM and non-LLM methods
- synthetic and real-data regimes
- evaluation of both graph recovery and downstream causal usefulness

Use this as a paper-planning and release-readiness checklist.

## 1. Contribution Framing

State these clearly in the paper introduction.

- What gap in existing causal discovery benchmarks are we fixing?
- Why is the gap important for current methods, especially foundation models or LLM-based systems?
- Is the main contribution:
  - a new dataset
  - a new benchmark/evaluation protocol
  - both
- What scientific claims will this benchmark let the field test that existing benchmarks do not?

Minimum bar:

- The benchmark must expose something nontrivial that changes scientific conclusions, not just add more instances.

## 2. Benchmark Scope

Define the benchmark target precisely.

- What is the prediction target for each regime?
  - DAG
  - CPDAG
  - I-MEC
  - PAG
- What assumptions hold in each regime?
  - acyclicity
  - causal sufficiency
  - perfect interventions
  - known intervention targets
  - no selection bias
- Which settings are included?
  - observational only
  - observational + interventions
  - perfect interventions
  - soft interventions
  - unknown targets
  - latent confounding
- Is the benchmark intended to evaluate:
  - classical causal discovery methods
  - LLM prompting
  - fine-tuned foundation models
  - hybrid systems

Minimum bar:

- The benchmark must not silently mix incompatible identifiability regimes.

## 3. Dataset Inventory

List every dataset family and why it exists.

For each dataset family:

- name
- synthetic or real
- domain
- number of variables
- variable types
  - binary
  - categorical
  - continuous
  - mixed
- graph density/sparsity
- intervention types
- sample sizes
- whether latent confounding is present
- whether ground-truth graph is available
- intended role
  - train
  - validation
  - test
  - transfer-only

Minimum bar:

- Each dataset family must have a clear purpose in the benchmark, not just be included because it exists.

## 4. Data Provenance And Construction

For synthetic data:

- SCM generation procedure
- graph generation procedure
- edge density distribution
- function families
- noise families
- intervention generation procedure
- random seeds and versioning

For real data:

- original source
- licensing and redistribution rights
- preprocessing steps
- how ground truth or partial ground truth is established
- what assumptions are likely violated

Minimum bar:

- Another group should be able to regenerate the synthetic benchmark exactly from the release.

## 5. Splits And Leakage Control

Specify the split logic for all datasets.

- train/dev/test split definitions
- graph-level split versus sample-level split
- whether test graphs are structurally distinct from train graphs
- whether there are out-of-prior test sets
- whether real datasets are held out from training entirely
- contamination checks for internet-scale models
- benchmark leakage checks across prompt files, answer files, and published graphs

For LLM evaluation:

- ensure labels are not embedded in prompt metadata
- ensure benchmark names do not trivially reveal canonical graphs
- check whether famous benchmark graphs like Sachs create memorization shortcuts

Minimum bar:

- You need an explicit contamination and memorization discussion if LLMs are part of the target audience.

## 6. Task Definitions

Document every task separately.

Examples relevant to this repo:

- full causal discovery
- descendants from intervention
- intervention effect detection
- edge existence
- edge orientation
- v-structure detection
- ambiguity/abstention

For each task:

- exact input format
- exact output schema
- scoring rule
- whether target is point-identified or equivalence-class valued

Minimum bar:

- The benchmark should distinguish full-graph tasks from local causal reasoning tasks.

## 7. Metrics

Use both structural and causal metrics.

Structural metrics:

- SHD
- edge precision/recall/F1
- skeleton F1
- orientation accuracy
- v-structure accuracy

Causal-effect metrics:

- SID
- causal-effect distance or equivalent intervention-response distance
- descendant-set F1 for subproblem tasks

Reliability metrics:

- format validity
- abstention quality
- calibration if probabilities/confidence are emitted

Minimum bar:

- Do not rely on edge-F1 alone.

## 8. Baselines

Include strong and diverse baselines.

Classical:

- ENCO
- PC / GES / NOTEARS / IGSP or equivalent depending on regime

LLM baselines:

- zero-shot prompting
- chain-of-thought or think-tag prompting if used
- few-shot prompting if relevant

Fine-tuned baselines:

- SFT-only
- SFT + GRPO / verifier-backed RL if used
- subproblem curriculum versus graph-only training

Hybrid baselines:

- LLM + explicit causal routines
- tool-augmented systems if they are part of the claim

Minimum bar:

- The benchmark paper should not only report the authors' own method.

## 9. Difficulty Calibration

Show that the benchmark is neither trivial nor pathological.

Include:

- easy/medium/hard slices
- scaling by graph size
- scaling by intervention count
- scaling by sample size
- robustness to anonymization and prompt format
- robustness to partial or noisy interventions

Minimum bar:

- The benchmark needs interpretable difficulty axes.

## 10. Statistical Rigor

Report uncertainty in benchmark results.

- multiple seeds
- confidence intervals
- significance testing where relevant
- variance across prompt shuffles and decoding randomness
- variance across graph instances within the same family

Minimum bar:

- If LLM variance is material, you must report it rather than a single number.

## 11. Reproducibility Package

Release these artifacts.

- dataset files or public download scripts
- benchmark generation scripts
- evaluation code
- exact metrics code
- baseline training and inference scripts
- environment specification
- fixed seeds
- prompt generation scripts if prompts are part of the benchmark
- answer schema documentation

For NeurIPS ED:

- dataset/code available at submission time
- machine-readable metadata in Croissant format
- Responsible AI metadata completed

Minimum bar:

- Reviewers must be able to run evaluation without reverse-engineering the repo.

## 12. Documentation

Prepare these documents.

- dataset card
- benchmark card
- task specification
- evaluation protocol
- leaderboard policy
- versioning and changelog

Dataset card should include:

- provenance
- intended use
- out-of-scope use
- sensitive content
- access and licensing
- known limitations

Benchmark card should include:

- task regimes
- metrics
- baseline results
- failure modes
- interpretation guidance

## 13. Ethics, Risk, And Governance

NeurIPS reviewers will look for this explicitly.

- privacy or consent issues for real datasets
- biomedical risk if using pathway datasets like Sachs
- misuse risk from overstating causal validity
- limitations under assumption violations
- demographic or subgroup bias if human data are involved
- release restrictions if needed
- maintenance owner and contact point
- takedown/update policy

Minimum bar:

- The paper must clearly separate benchmark utility from claims of real-world causal truth.

## 14. Paper Structure

Recommended paper outline:

1. Motivation and benchmark gap
2. Related benchmarks and why they are insufficient
3. Benchmark design principles
4. Dataset families and construction
5. Task definitions and evaluation regimes
6. Metrics and protocol
7. Baselines
8. Benchmark results and analyses
9. Limitations, ethics, and maintenance

## 15. What This Project Specifically Needs

For this repo, the most important benchmark elements are:

- Explicit regime separation
  - observational-only should not be pooled with observational+interventional
- Explicit target object
  - if you keep ENCO assumptions, DAG output is fine for that regime
  - if you broaden beyond ENCO assumptions, add equivalence-class targets
- Both graph and subproblem tasks
  - this repo already supports `cd_descendants`
  - benchmarking only final graph output would miss useful signals
- Structural and effect metrics together
- LLM contamination discussion
  - especially for named benchmark graphs like Sachs
- Prompt-format robustness
  - summary versus matrix
  - named versus anonymized
  - row order / shuffle sensitivity
- Strong classical baselines
  - ENCO should be included prominently

## 16. Must-Have Tables And Figures

Prepare these for the paper.

Tables:

- benchmark inventory table
- per-regime baseline comparison table
- structural versus causal-effect metric table
- robustness table
- contamination/memorization stress-test table

Figures:

- benchmark design diagram
- regime taxonomy
- performance versus intervention count
- performance versus graph size
- failure mode examples

## 17. Minimal Release Checklist

Before submission, verify all of the following.

- The benchmark has a clear scientific claim.
- Every dataset has provenance and licensing documented.
- Every task has an explicit JSON or formal output schema.
- Splits are fixed and reproducible.
- Metrics are implemented and documented.
- At least one strong classical and one strong LLM baseline are included.
- Variance across runs is reported.
- Ethics and limitations sections are specific, not boilerplate.
- Croissant metadata is prepared.
- Code and data are accessible to reviewers by the submission deadline.

## 18. Suggested Benchmark Narrative For This Project

If you want a concise benchmark thesis, this is the defensible version:

"We introduce a causal discovery benchmark that evaluates both full-graph recovery and decomposed causal reasoning from observational and interventional evidence, across explicitly separated identifiability regimes, with metrics covering both structural correctness and causal-effect fidelity."

That is much stronger than:

"We release more causal discovery prompts and compare LLMs."

## 19. Immediate Next Steps

In order:

1. Freeze the benchmark regimes.
2. Freeze the task inventory.
3. Freeze the metrics.
4. Decide which datasets are benchmark-train versus benchmark-test only.
5. Add contamination and memorization checks.
6. Define the release package and metadata.
7. Only then finalize the paper framing.

