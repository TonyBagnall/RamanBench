#!/usr/bin/env python
"""Load one RamanBench dataset and inspect its standard split.

Usage
-----
python scripts/load_one_ramanbench_dataset.py
python scripts/load_one_ramanbench_dataset.py --dataset wheat_lines --task classification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_bench.benchmark import RamanBenchmark
from raman_data import raman_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one RamanBench dataset")
    parser.add_argument(
        "--dataset",
        default="diabetes_skin_ear_lobe",
        help="raman-data dataset key",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Prediction task type for the benchmark split",
    )
    parser.add_argument("--target-idx", type=int, default=0, help="Target index to inspect")
    parser.add_argument("--cache-dir", default=".cache", help="Dataset/split cache directory")
    args = parser.parse_args()

    try:
        dataset = raman_data(args.dataset, cache_dir=f"{args.cache_dir}/datasets_raw")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {args.dataset!r} via raman-data. This failed in the upstream "
            "dataset downloader/reader before RamanBench could create a split. Try a different "
            "dataset key, or use a cache that already contains the downloaded dataset."
        ) from exc
    print(f"Dataset: {args.dataset}")
    print(f"Spectra shape: {dataset.spectra.shape}")
    print(f"Targets shape: {dataset.targets.shape}")
    print(f"Target names: {dataset.target_names}")
    print(f"Raman shifts shape: {dataset.raman_shifts.shape}")

    benchmark = RamanBenchmark(
        dataset_names_classification=[args.dataset] if args.task == "classification" else [],
        dataset_names_regression=[args.dataset] if args.task == "regression" else [],
        cache_dir=args.cache_dir,
    )

    key = benchmark.get_key(args.dataset, args.target_idx)
    train_df, test_df = benchmark._load_dataset_from_key(key)
    if train_df is None or test_df is None:
        raise RuntimeError(f"{key} could not be loaded after RamanBench filtering")

    X_train = train_df.iloc[:, :-1].to_numpy()
    y_train = train_df.iloc[:, -1].to_numpy()
    X_test = test_df.iloc[:, :-1].to_numpy()
    y_test = test_df.iloc[:, -1].to_numpy()

    print(f"\nProblem key: {key}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"First target values: {y_train[:5]}")


if __name__ == "__main__":
    main()
