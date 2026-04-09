# Alarm Reference Card

## Intended Claims
- Larger real graph for pressure-testing the benchmark framework and ranking sensitivity.
- Supports the claim that benchmark design choices still matter beyond tiny graphs.

## Unsupported Claims
- Not intended as the only scale claim in the paper.
- Not suitable for low-context-window providers without budgeting care.

## Applicable Baselines
- `PC`, `GES`, `ENCO`

## Known Failure Modes
- Matrix prompts can become expensive.
- Some models may fail formatting before reasoning quality becomes measurable.
