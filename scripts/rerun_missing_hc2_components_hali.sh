#!/usr/bin/env bash
# Retry incomplete HC2 component results with 128 GB and verbose output.
# Each classifier/problem/resample is an independent job.
# Run with --dry-run to print the plan without submitting jobs.
set -euo pipefail

dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
    dry_run=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi

repo_dir="${REPO_DIR:-$HOME/Code/RamanBench}"
tsml_eval_dir="${TSML_EVAL_DIR:-$HOME/Code/tsml-eval}"
data_dir="${DATA_DIR:-$HOME/Data/RamanBench/classification}"
results_dir="${RESULTS_DIR:-$HOME/Results/RamanBench}"
dataset_list="${DATASET_LIST:-$results_dir/ramanbench_classification_problems.txt}"
module_name="${HALI_MODULE:-python/anaconda/2024.10/3.12.7}"
conda_sh="${CONDA_SH:-/gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh}"
env_name="${CONDA_ENV:-tsml-eval}"

# Preserve the original list/order: its line numbers identify existing array tasks.
[[ -f "$dataset_list" ]] || { echo "Missing original problem list: $dataset_list" >&2; exit 1; }
mapfile -t problems < "$dataset_list"
[[ ${#problems[@]} -eq 21 ]] || { echo "Expected 21 problems in $dataset_list" >&2; exit 1; }

active_jobs=""
if command -v squeue >/dev/null; then
    active_jobs=$(squeue --array --noheader --user "$(id -un)" --format '%i|%j')
elif ! $dry_run; then
    echo "squeue is required: run this script on HALI." >&2
    exit 1
else
    echo "squeue unavailable: dry-run cannot show dependencies on active jobs."
fi

if ! $dry_run; then
    mkdir -p "$results_dir/slurm_logs" "$results_dir/summaries"
fi
jobs=0
runs=0
for classifier in Arsenal DrCIF TDE STC; do
    output_name="$classifier"
    [[ "$classifier" != DrCIF ]] || output_name=DrCIF-500
    for index in "${!problems[@]}"; do
        problem="${problems[$index]%$'\r'}"
        missing=()
        for resample in 0 1 2; do
            predictions="$results_dir/$output_name/Predictions/$problem"
            if [[ ! -f "$predictions/trainResample${resample}.csv" || ! -f "$predictions/testResample${resample}.csv" ]]; then
                missing+=("$resample")
            fi
        done
        [[ ${#missing[@]} -gt 0 ]] || continue
      for resample in "${missing[@]}"; do
        job_name="RamanBench-retry-${classifier}-${problem}-r${resample}"
        dependencies=()
        while IFS='|' read -r job_id active_name; do
            if [[ "$active_name" == "$job_name" || "$active_name" == "RamanBench-retry-${classifier}-${problem}" ]] ||
               [[ ( "$active_name" == "RamanBench-${classifier}" || "$active_name" == "RamanBench-${classifier}-r${resample}" ) && "$job_id" == *_$((index + 1)) ]]; then
                dependencies+=("$job_id")
            fi
        done <<< "$active_jobs"
        dependency_args=()
        if [[ ${#dependencies[@]} -gt 0 ]]; then
            dependency_args+=("--dependency=afterany:$(IFS=:; echo "${dependencies[*]}")")
        fi
        echo "$classifier $problem resample=$resample memory=128G ${dependency_args[*]}"
        jobs=$((jobs + 1))
        runs=$((runs + 1))
        $dry_run && continue

        # Quote each value for the job shell, including problem names with parentheses.
        printf -v command_line '%q ' python -u "$repo_dir/scripts/run_hc2_components.py" \
            --data-dir "$data_dir" --results-dir "$results_dir" --tsml-eval "$tsml_eval_dir" \
            --classifier "$classifier" --problem "$problem" --resamples "$resample" \
            --train-files --verbose --summary "$results_dir/summaries/retry_${classifier}_${problem}_resample${resample}.json"
        printf -v setup 'module add %q\nsource %q\nconda activate %q\n' "$module_name" "$conda_sh" "$env_name"
        sbatch --account="${SLURM_ACCOUNT:-cmp}" --partition="${SLURM_PARTITION:-compute}" \
            --qos="${SLURM_QOS:-uea-core-default}" --time="${TIME_LIMIT:-7-00:00:00}" \
            --mem=128G --cpus-per-task=1 --job-name="$job_name" \
            --output="$results_dir/slurm_logs/%x-%j.out" --error="$results_dir/slurm_logs/%x-%j.err" \
            "${dependency_args[@]}" --wrap="$(printf 'set -euo pipefail\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n%s%s\n' "$setup" "$command_line")"
      done
    done
done
echo "Jobs: $jobs; incomplete component/resample pairs: $runs; dry-run: $dry_run"
