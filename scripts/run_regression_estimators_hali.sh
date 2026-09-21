#!/usr/bin/env bash
# Submit one SLURM array per regressor and resample; each task handles one target.
set -euo pipefail

repo_dir="${REPO_DIR:-$HOME/Code/RamanBench}"
tsml_eval_dir="${TSML_EVAL_DIR:-$HOME/Code/tsml-eval}"
data_dir="${DATA_DIR:-$HOME/Data/RamanBench/regression}"
results_dir="${RESULTS_DIR:-$HOME/Results/RamanBench}"
dataset_list="${DATASET_LIST:-$results_dir/ramanbench_regression_problems.txt}"
account="${SLURM_ACCOUNT:-cmp}"
partition="${SLURM_PARTITION:-compute}"
qos="${SLURM_QOS:-uea-core-default}"
memory_mb="${MEMORY_MB:-32000}"
time_limit="${TIME_LIMIT:-7-00:00:00}"
module_name="${HALI_MODULE:-python/anaconda/2024.10/3.12.7}"
conda_sh="${CONDA_SH:-/gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh}"
env_name="${CONDA_ENV:-tsml-eval}"
drcif_estimators="${DRCIF_ESTIMATORS:-200}"
# Example: REGRESSORS=DrCIF DRCIF_ESTIMATORS=500 bash scripts/run_regression_estimators_hali.sh
read -r -a regressors <<< "${REGRESSORS:-DrCIF QUANT}"
for regressor in "${regressors[@]}"; do
  case "$regressor" in
    DrCIF|QUANT) ;;
    *) echo "Unknown regressor: $regressor" >&2; exit 1 ;;
  esac
done

[[ -d "$data_dir" ]] || { echo "Data directory not found: $data_dir" >&2; exit 1; }
[[ -f "$repo_dir/scripts/run_regression_estimators.py" ]] || {
  echo "Runner not found: $repo_dir/scripts/run_regression_estimators.py" >&2; exit 1;
}

mkdir -p "$(dirname "$dataset_list")" "$results_dir/slurm_logs" "$results_dir/summaries"
find "$data_dir" -type f -name '*_TRAIN.ts' -print0 \
  | while IFS= read -r -d '' train_file; do
      basename "$(dirname "$train_file")"
    done | sort -u > "$dataset_list"
count=$(wc -l < "$dataset_list")
if [[ "$count" -eq 0 ]]; then
  echo "No regression problems with TRAIN .ts files found in $data_dir" >&2
  exit 1
fi

for regressor in "${regressors[@]}"; do
  output_name="$regressor"
  [[ "$regressor" != DrCIF ]] || output_name="DrCIF-${drcif_estimators}"
  for resample in 0 1 2; do
    sbatch \
      --account="$account" --partition="$partition" --qos="$qos" \
      --time="$time_limit" --mem="${memory_mb}M" --cpus-per-task=1 \
      --array="1-${count}" --job-name="RamanReg-${output_name}-r${resample}" \
      --output="$results_dir/slurm_logs/%x-%A_%a.out" \
      --error="$results_dir/slurm_logs/%x-%A_%a.err" \
      --wrap="
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module add '$module_name'
source '$conda_sh'
conda activate '$env_name'
problem=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" '$dataset_list')
python -u '$repo_dir/scripts/run_regression_estimators.py' \\
  --data-dir '$data_dir' --results-dir '$results_dir' --tsml-eval '$tsml_eval_dir' \\
  --regressor '$regressor' --problem \"\$problem\" --resamples '$resample' \\
  --drcif-estimators '$drcif_estimators' --train-files \\
  --summary \"$results_dir/summaries/${output_name}_\${problem}_resample${resample}.json\"
"
  done
done

echo "Submitted ${regressors[*]} for $count target problems, resamples 0-2."
