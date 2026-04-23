# Paper Slices

This directory freezes the benchmark slices used by the NeurIPS artifact paper.

`paper_slices/` is now a compatibility layer over the general manifest-driven benchmark framework in
`benchmark_specs/`.

Each JSON file defines one tightly scoped benchmark slice. A slice is small on purpose:

- one dataset
- one model
- one or a few prompt configurations
- optional matched baselines (`ENCO`, `CausalLLMPrompt`, `JiralerspongBFS`, `TakayamaSCP`, ...)
- optional `names_only` control
- a clear paper role such as `main`, `control`, or `smoke`

Use the runner:

```bash
python scripts/run_paper_slice.py --manifest paper_slices/sachs_main.json
```

The current `sachs_main.json` slice is intentionally small and benchmark-native:

- `summary_joint` and `matrix` at `obs=1000, int=50`
- observational `summary_joint` controls at `obs=1000, int=0` for real and anonymized names
- `names_only` enabled
- baselines: `PC`, `GES`, `ENCO`, `CausalLLMPrompt`, `JiralerspongBFS`, `TakayamaSCP`

Additional compact breadth slices are available for:

- `asia_compact.json`
- `child_compact.json`
- `alarm_compact.json`

These are intended for the multi-graph breadth block in the paper. They keep
only `summary_joint` observational/interventional cells at `obs=1000` plus
`names_only` so they are cheaper to run than the Sachs main slice.

For new benchmark definitions, prefer:

```bash
scripts/run-benchmark --manifest benchmark_specs/reference_suite.json
```

After each run, the runner updates:

- `refine-logs/latest_slice_summary.md`
- `refine-logs/NEXT_SLICE.md`

Those files are the stable handoff surface for ARIS review iterations.
