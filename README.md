# LLM Patch Failure Prediction

This project studies fast failure detection for LLM-generated patches using SWE-smith-trajectories.

## Current setup

- Dataset: `SWE-bench/SWE-smith-trajectories`
- Split used: `tool`
- Current subset: non-empty patch only
- Cross-validation: leave-one-repository-out
- Training setup: leave-one-repository-out, with one held-out repo as test and all remaining repos as train
- Baseline model: XGBoost

## Repository structure

- `src/`: reusable code for loading data, extracting features, splitting, modeling, and evaluation
- `scripts/`: runnable experiment entry points
- `notebooks/`: optional exploratory notebooks
- `results/`: saved outputs

## How to run

### 1. Build the feature table
```bash
python -m scripts.build_feature_table

### 2. Run leave-one-repository-out evaluation
```bash
python -m scripts.run_leave_one_repo_out
