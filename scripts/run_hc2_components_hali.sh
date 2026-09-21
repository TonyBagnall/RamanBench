#!/usr/bin/env bash
# Submit one HALI array per HC2 component and resample. Each task runs one
# problem/resample and writes tsml-style
# trainResampleN.csv/testResampleN.csv files.
set -euo pipefail

repo_dir="${REPO_DIR:-$HOME/Code/RamanBench}"
tsml_eval_dir="${TSML_EVAL_DIR:-$HOME/Code/tsml-eval}"
data_dir="${DATA_DIR:-$HOME/Data/RamanBench/classification}"
results_dir="${RESULTS_DIR:-$HOME/Results/RamanBench}"
dataset_list="${DATASET_LIST:-$results_dir/ramanbench_classification_problems.txt}"
account="${SLURM_ACCOUNT:-cmp}"
partition="${SLURM_PARTITION:-compute}"
qos="${SLURM_QOS:-uea-core-default}"
memory_mb="${MEMORY_MB:-32000}"
time_limit="${TIME_LIMIT:-7-00:00:00}"
module_name="${HALI_MODULE:-python/anaconda/2024.10/3.12.7}"
conda_sh="${CONDA_SH:-/gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh}"
env_name="${CONDA_ENV:-tsml-eval}"
# Example: CLASSIFIERS=QUANT bash scripts/run_hc2_components_hali.sh
read -r -a classifiers <<< "${CLASSIFIERS:-Arsenal DrCIF TDE STC}"
for classifier in "${classifiers[@]}"; do
    case "$classifier" in
        Arsenal|DrCIF|TDE|STC|HC2|MRHydra|QUANT) ;;
        *) echo "Unknown classifier: $classifier" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$data_dir" ]]; then
    echo "Data directory not found: $data_dir" >&2
    exit 1
fi
if [[ ! -f "$repo_dir/scripts/run_hc2_components.py" ]]; then
    echo "Runner not found: $repo_dir/scripts/run_hc2_components.py" >&2
    exit 1
fi

mkdir -p "$(dirname "$dataset_list")" "$results_dir/slurm_logs"
find "$data_dir" -mindepth 1 -maxdepth 1 -type d -print0 \
    | while IFS= read -r -d '' directory; do
        problem=$(basename "$directory")
        compgen -G "$directory/*_TRAIN.ts" >/dev/null && printf '%s\n' "$problem"
      done | sort > "$dataset_list"

count=$(wc -l < "$dataset_list")
if [[ "$count" -ne 21 ]]; then
    echo "Expected 21 problems, found $count in $dataset_list" >&2
    exit 1
fi

for classifier in "${classifiers[@]}"; do
  for resample in 0 1 2; do
    job_name="RamanBench-${classifier}-r${resample}"
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --qos="$qos" \
        --time="$time_limit" \
        --mem="${memory_mb}M" \
        --cpus-per-task=1 \
        --array="1-${count}" \
        --job-name="$job_name" \
        --output="$results_dir/slurm_logs/%x-%A_%a.out" \
        --error="$results_dir/slurm_logs/%x-%A_%a.err" \
        --wrap="
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module add "$module_name"
source "$conda_sh"
conda activate "$env_name"
problem=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" \"$dataset_list\")
python -u \"$repo_dir/scripts/run_hc2_components.py\" \\
  --data-dir \"$data_dir\" \\
  --results-dir \"$results_dir\" \\
  --tsml-eval \"$tsml_eval_dir\" \\
  --classifier \"$classifier\" \\
  --problem \"\$problem\" \\
  --resamples \"$resample\" \\
  --train-files \\
  --summary \"$results_dir/summaries/${classifier}_\${problem}_resample${resample}.json\"
"
  done
done

echo "Submitted ${classifiers[*]}: $((count * 3 * ${#classifiers[@]})) tasks, one classifier/problem/resample per task."
