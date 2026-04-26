# Benchmark Specs

This directory contains reusable configs for the benchmark builder.

Primary configs:

- `reference_suite.json`: main real-graph suite for the paper
- `smoke_suite.json`: fast smoke/debug suite
- `synthetic_ladder.json`: synthetic appendix suite
- `authoring_demo.json`: tutorial benchmark used in the authoring guide

Run any config with:

```bash
scripts/run-benchmark --config benchmark_specs/reference_suite.json
```
