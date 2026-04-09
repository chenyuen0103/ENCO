# Paper Slices

This directory freezes the benchmark slices used by the NeurIPS artifact paper.

`paper_slices/` is now a compatibility layer over the general manifest-driven benchmark framework in
`benchmark_specs/`.

Each JSON file defines one tightly scoped benchmark slice. A slice is small on purpose:

- one dataset
- one model
- one or a few prompt configurations
- optional matched ENCO baseline
- a clear paper role such as `main`, `control`, or `smoke`

Use the runner:

```bash
python scripts/run_paper_slice.py --manifest paper_slices/sachs_main.json
```

For new benchmark definitions, prefer:

```bash
scripts/run-benchmark --manifest benchmark_specs/reference_suite.json
```

After each run, the runner updates:

- `refine-logs/latest_slice_summary.md`
- `refine-logs/NEXT_SLICE.md`

Those files are the stable handoff surface for ARIS review iterations.
