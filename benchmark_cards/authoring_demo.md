# Authoring Demo Card

## Intended Claims
- A researcher can author a new benchmark family from a config that mixes existing public graphs and a synthetic generator.
- The framework does not require evaluator or model-code edits for new benchmark definitions.

## Unsupported Claims
- This demo is not a paper-quality benchmark by itself.
- The demo does not prove cross-model robustness.

## Applicable Baselines
- `PC`, `ENCO`

## Known Failure Modes
- Optional baseline dependencies may be missing locally.
- Synthetic and real graph naming conventions may need explicit interpretation in downstream analysis.
