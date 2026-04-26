# Reference Suite Card

## Intended Claims
- The framework can express a reusable causal-discovery benchmark family over multiple real graphs with one config.
- Prompt representation, naming regime, and intervention budget can change empirical conclusions.
- Matched classical baselines and LLM runs can be audited under one evaluator and provenance contract.

## Unsupported Claims
- This suite does not prove general causal reasoning ability.
- This suite does not make biomedical claims about Sachs, Child, or Alarm.
- This suite is not a fairness or contamination-complete audit by itself.

## Applicable Baselines
- `PC` and `GES` for observational slices.
- `ENCO` for intervention-bearing slices.

## Known Failure Modes
- Large prompts can exceed context limits on some providers.
- Real variable names may leak prior knowledge.
- `alarm` can be expensive for matrix prompts and some API models.
