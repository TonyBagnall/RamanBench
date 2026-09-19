# Regression downloads and aeon export

From the repository root, run:

```powershell
python scripts/download_regression_aeon.py
```

Outputs are under `results/regression`. The short `.rd` cache path avoids
Windows path-length problems. Neither directory is tracked by Git.

Each downloaded dataset has `data.npz` containing `X`, `y`, `raman_shifts`, and
`target_names`. Each benchmark-eligible target gets three TRAIN/TEST `.ts` pairs,
using the seeds, exclusions, and test fraction in `benchmark_v0.1.json` and
RamanBenchmark's grouped regression splitting. No additional preprocessing is
applied. Multiple targets become separate scalar regression problems.

Every split is loaded back using aeon and numerically compared with its source
before its dataset's `manifest.json` is written. Files without a completed
manifest may belong to an interrupted run and must not be treated as verified.
Datasets excluded by the benchmark retain their NPZ arrays but no split exports.

```python
from aeon.datasets import load_from_ts_file
X, y = load_from_ts_file(
    'results/regression/amino_acids_glycine/amino_acids_glycine_0/'
    'amino_acids_glycine_00_TRAIN.ts'
)
# X: (cases, 1, wavenumbers); y: continuous scalar targets
```

Retry one dataset using `--dataset NAME`. Rerunning the full command skips
datasets with completed manifests. `--report-only` refreshes the overall
download manifest from completed per-dataset manifests without downloading.
The default per-dataset time limit is 1200 seconds (`--timeout` changes it).

The script contains process-local compatibility fixes for a Kaggle ZIP wrapper,
Hugging Face's Windows symlink probe, and a zenodo-get HTTP transport mismatch.
It does not modify installed packages. Source availability is still required;
an HTML browser challenge returned by RWTH is not valid spectral data.
