## Bash scripts
The scripts in this folder contain the commands for running the experiments presented in the paper. Note:
* Download the datasets via `download_datasets.sh` before running any experiment except `experiment_scale.sh`.
* The scripts can be run from anywhere (they `cd` to the repo’s `experiments/` directory internally).
* To run with a different random seed, set `SEED` (default: 42), e.g. `SEED=0 bash experiment_synthetic.sh`.
* To run the whole ENCO baseline suite in one go, use `run_all_enco_baselines.sh` (also respects `SEED`).
* To run ENCO on the Experiment 1 “table range” (obs/int grid), use `experiment1_enco_table_range.sh`.
  * One graph (default `cancer.bif`): `SEED=0 bash experiment1_enco_table_range.sh`
  * Sachs convenience wrapper: `SEED=0 bash sachs_enco_baseline.sh`
  * All real_data graphs: `ALL_REAL_DATA=1 REAL_DATA_SCOPE=all SEED=0 bash experiment1_enco_table_range.sh`
  * OOM tuning knobs (env vars): `MAX_GRAPH_STACKING_LARGE`, `BATCH_SIZE_LARGE`, `GF_NUM_BATCHES_LARGE`, `HIDDEN_SIZE_LARGE` (and `*_SMALL` variants)
* To run both the paper baselines and the table-range sweep, use `run_all_enco_paper_and_table.sh`.
* To avoid rerunning completed scripts, each script writes a marker file under `experiments/checkpoints/enco_baselines/.done/` and will skip if it exists. Use `FORCE=1` to rerun.
* The experiments have been designed for a single GPU device with 24GB memory. The full memory is only necessary for experiments with large graphs (>100 variables). If you have a smaller GPU and run out of memory during the graph fitting stage, adjust the argument `max_graph_stacking` to a smaller value. The used memory scales approximately linearly with `max_graph_stacking`. In case you run out of memory during the distribution fitting stage, adjust the argument `batch_size` and increase `GF_num_batches` to keep the overall batch size for the graph fitting stage constant.
* The experiments run by default in `cluster` mode which reduces the output. If you remove the `cluster` argument in the bash scripts, the training progress will be shown with progress bars.
* All checkpoints and logging information can be found in `experiments/checkpoints/...` by default.
