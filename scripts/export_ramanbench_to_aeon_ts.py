#!/usr/bin/env python
"""Export RamanBench classification problems to aeon ``.ts`` files.

The exporter reuses :class:`raman_bench.benchmark.RamanBenchmark` so dataset
enumeration, target expansion, rare-class filtering, and train/test splitting
match the benchmark pipeline. Raman spectra are exported as univariate,
equal-length collections using aeon's official writer.

Usage
-----
::

    python scripts/export_ramanbench_to_aeon_ts.py
    python scripts/export_ramanbench_to_aeon_ts.py --config configs/benchmark_v0.1.json
    python scripts/export_ramanbench_to_aeon_ts.py --all-resamples
    python scripts/export_ramanbench_to_aeon_ts.py --seed-index 1
    python scripts/export_ramanbench_to_aeon_ts.py --output-dir C:\\Data\\RamanBench
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

MANIFEST_COLUMNS = [
    "problem_name",
    "resample",
    "seed",
    "task_type",
    "original_dataset_name",
    "original_target_name",
    "ramanbench_key",
    "target_idx",
    "n_train",
    "n_test",
    "n_timepoints",
    "n_classes",
    "y_dtype",
    "train_file",
    "test_file",
    "export_status",
    "error_message",
]


@dataclass(frozen=True)
class Writer:
    function: Any
    name: str


@dataclass(frozen=True)
class Problem:
    dataset_name: str
    target_name: str
    target_idx: int
    key: str
    task_type: Any
    task_label: str
    problem_name: str = ""
    enumeration_error: str = ""


def _default_config() -> str:
    return str(ROOT / "configs" / "benchmark_v0.1.json")


def _default_classification_metrics() -> str:
    return str(ROOT / "data" / "precomputed" / "classification_metrics.csv")


def _get_num_targets_and_names(dataset_name: str, cache_dir_raw: str) -> tuple[int, list[str]]:
    from raman_data import raman_data

    dataset = raman_data(dataset_name, cache_dir=cache_dir_raw)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return 0, []
    num_targets = 1 if targets.ndim == 1 else int(targets.shape[1])
    target_names = getattr(dataset, "target_names", None)
    names = []
    for target_idx in range(num_targets):
        if target_names is not None and target_idx < len(target_names):
            target_name = target_names[target_idx]
            names.append(str(target_name) if str(target_name).strip() else f"target_{target_idx}")
        else:
            names.append(f"target_{target_idx}")
    return num_targets, names


def _task_name(task_type: Any) -> str:
    return str(getattr(task_type, "name", task_type)).lower()


def _is_classification_task(task_type: Any) -> bool:
    return _task_name(task_type) == "classification"


def _is_regression_task(task_type: Any) -> bool:
    return _task_name(task_type) == "regression"


def _select_writer() -> Writer:
    import aeon.datasets as aeon_datasets

    if hasattr(aeon_datasets, "save_to_ts_file"):
        return Writer(aeon_datasets.save_to_ts_file, "save_to_ts_file")
    if hasattr(aeon_datasets, "write_to_ts_file"):
        return Writer(aeon_datasets.write_to_ts_file, "write_to_ts_file")
    raise ImportError("Neither aeon.datasets.save_to_ts_file nor write_to_ts_file is available")


def _dataframe_to_xy(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if data is None or data.empty:
        raise ValueError("split is empty")
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    return X, y


def _is_discrete_labels(y: np.ndarray) -> bool:
    series = pd.Series(y)
    if is_bool_dtype(series) or is_integer_dtype(series):
        return True
    if not is_numeric_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    values = numeric.to_numpy(dtype=float)
    return bool(np.all(np.isfinite(values)) and np.all(np.equal(values, np.floor(values))))


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = pd.Series(y).value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _warnings_for_values(name: str, values: np.ndarray) -> list[str]:
    warnings: list[str] = []
    if pd.isna(values).any():
        warnings.append(f"{name} contains missing values")
    if is_numeric_dtype(pd.Series(values.ravel())):
        arr = values.astype(float, copy=False)
        if np.isnan(arr).any():
            warnings.append(f"{name} contains NaNs")
        if not np.isfinite(arr).all():
            warnings.append(f"{name} contains non-finite values")
    return warnings


def _validate_problem(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: Any,
) -> list[str]:
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D, got shape {X_test.shape}")
    if y_train.ndim != 1:
        raise ValueError(f"y_train must be 1D, got shape {y_train.shape}")
    if y_test.ndim != 1:
        raise ValueError(f"y_test must be 1D, got shape {y_test.shape}")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("train/test split must not be empty")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different sample counts")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test have different sample counts")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "train and test have different numbers of wavenumber points "
            f"({X_train.shape[1]} != {X_test.shape[1]})"
        )

    warnings = []
    for label, values in [
        ("X_train", X_train),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_test", y_test),
    ]:
        warnings.extend(_warnings_for_values(label, values))

    if _is_classification_task(task_type):
        if not _is_discrete_labels(y_train) or not _is_discrete_labels(y_test):
            raise ValueError("classification labels must be discrete")
    elif _is_regression_task(task_type):
        if not is_numeric_dtype(pd.Series(y_train)) or not is_numeric_dtype(pd.Series(y_test)):
            raise ValueError("regression targets must be numeric")
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    return warnings


def _write_ts(
    writer: Writer,
    X: np.ndarray,
    y: np.ndarray,
    label_type: str,
    problem_dir: Path,
    problem_name: str,
    file_suffix: str,
) -> None:
    problem_dir.mkdir(parents=True, exist_ok=True)
    if writer.name == "save_to_ts_file":
        writer.function(
            X,
            y,
            label_type=label_type,
            path=str(problem_dir),
            problem_name=problem_name,
            file_suffix=file_suffix,
        )
        return

    kwargs = {
        "path": str(problem_dir),
        "problem_name": problem_name,
        "file_suffix": file_suffix,
        "regression": label_type == "regression",
    }
    signature = inspect.signature(writer.function)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    writer.function(X, y, **kwargs)


def _empty_record(
    problem_name: str,
    resample: int,
    seed: int,
    task_label: str,
    dataset_name: str,
    target_name: str,
    key: str,
    target_idx: int,
) -> dict[str, Any]:
    return {
        "problem_name": problem_name,
        "resample": resample,
        "seed": seed,
        "task_type": task_label,
        "original_dataset_name": dataset_name,
        "original_target_name": target_name,
        "ramanbench_key": key,
        "target_idx": target_idx,
        "n_train": "",
        "n_test": "",
        "n_timepoints": "",
        "n_classes": "",
        "y_dtype": "",
        "train_file": "",
        "test_file": "",
        "export_status": "pending",
        "error_message": "",
    }


def _write_manifests(
    records: list[dict[str, Any]],
    output_dir: Path,
    writer_name: str,
    export_complete: bool = False,
) -> None:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    clf_df = all_df[all_df["task_type"] == "classification"]
    reg_df = all_df[all_df["task_type"] == "regression"]

    all_df.to_csv(manifest_dir / "all_problems.csv", index=False)
    clf_df.to_csv(manifest_dir / "classification_problems.csv", index=False)
    reg_df.to_csv(manifest_dir / "regression_problems.csv", index=False)

    summary = {
        "writer": writer_name,
        "n_total": int(len(all_df)),
        "n_classification": int(len(clf_df)),
        "n_regression": int(len(reg_df)),
        "n_passed": int((all_df["export_status"] == "pass").sum()),
        "n_skipped": int((all_df["export_status"] == "skip").sum()),
        "n_failed": int((all_df["export_status"] == "fail").sum()),
        "output_dir": str(output_dir),
        "export_complete": export_complete,
    }
    with open(manifest_dir / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _has_completed_export(output_dir: Path) -> bool:
    summary_path = output_dir / "manifests" / "export_summary.json"
    manifest_path = output_dir / "manifests" / "all_problems.csv"
    if not summary_path.exists():
        return False
    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        manifest = pd.read_csv(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    try:
        n_total = int(summary.get("n_total", 0))
        n_failed = int(summary.get("n_failed", 0))
        n_done = int(summary.get("n_passed", 0)) + int(summary.get("n_skipped", 0))
    except (TypeError, ValueError):
        return False
    if summary.get("export_complete") is False:
        return False
    if n_total <= 0 or n_failed != 0 or n_done < n_total:
        return False

    required_columns = {"train_file", "test_file"}
    if not required_columns.issubset(manifest.columns):
        return False
    exported = manifest[["train_file", "test_file"]].fillna("")
    if len(exported) < n_total:
        return False
    for train_file, test_file in exported.itertuples(index=False, name=None):
        if not train_file or not test_file:
            return False
        if not Path(str(train_file)).exists() or not Path(str(test_file)).exists():
            return False
    return n_total > 0 and n_failed == 0 and n_done >= n_total


def _resolve_classification_dataset_names(config: dict[str, Any], cache_dir_raw: str) -> list[str]:
    from raman_data import TASK_TYPE, raman_data

    classification = config.get("dataset_names_classification")
    if classification is None:
        classification = raman_data(task_type=TASK_TYPE.Classification, cache_dir=cache_dir_raw)
    return list(classification)


def _load_metric_problem_keys(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"classification metrics CSV not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    required = {"key", "dataset", "target_idx"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(
            f"classification metrics CSV missing required columns: {sorted(missing)}"
        )
    if "problem_name" not in metrics.columns:
        metrics = metrics.copy()
        metrics["problem_name"] = metrics["key"]
    problems = (
        metrics[["problem_name", "key", "dataset", "target_idx"]]
        .drop_duplicates()
        .sort_values(["problem_name", "key", "target_idx"])
        .reset_index(drop=True)
    )
    logger.info(
        "Loaded %d classification problem names from %s",
        len(problems),
        metrics_path,
    )
    return problems


def _metric_resamples(metrics_path: Path, seeds: list[int]) -> list[tuple[int, int]]:
    metrics = pd.read_csv(metrics_path)
    if "seed" not in metrics.columns:
        raise ValueError("classification metrics CSV missing required column: seed")
    metric_seeds = sorted({int(seed) for seed in metrics["seed"].dropna().unique()})
    seed_to_resample = {int(seed): resample for resample, seed in enumerate(seeds)}
    missing = [seed for seed in metric_seeds if seed not in seed_to_resample]
    if missing:
        raise ValueError(
            "classification metrics contains seed value(s) not present in "
            f"get_seeds(config): {missing}"
        )
    return [(seed_to_resample[seed], seed) for seed in metric_seeds]


def _iter_classification_problems(benchmark: Any, metric_problems: pd.DataFrame):
    from raman_data import TASK_TYPE

    for _, row in metric_problems.iterrows():
        key = str(row["key"])
        problem_name = str(row["problem_name"])
        dataset_name, target_idx = benchmark.split_key(key)
        try:
            num_targets, target_names = _get_num_targets_and_names(
                dataset_name, benchmark.cache_dir_raw
            )
            benchmark._index[dataset_name] = num_targets
            benchmark._save_index()
            if target_idx >= num_targets:
                raise ValueError(
                    f"Target index {target_idx} out of range for dataset {dataset_name}"
                )
            target_name = (
                str(target_names[target_idx])
                if target_idx < len(target_names) and str(target_names[target_idx]).strip()
                else f"target_{target_idx}"
            )
            yield Problem(
                dataset_name=dataset_name,
                target_name=target_name,
                target_idx=target_idx,
                key=key,
                task_type=TASK_TYPE.Classification,
                task_label="classification",
                problem_name=problem_name,
            )
        except Exception as exc:
            logger.error("Failed to enumerate %s: %s", dataset_name, exc)
            target_name = "unknown"
            yield Problem(
                dataset_name=dataset_name,
                target_name=target_name,
                target_idx=target_idx,
                key=key,
                task_type=TASK_TYPE.Classification,
                task_label="classification",
                problem_name=problem_name,
                enumeration_error=str(exc),
            )


def _load_problem_split(
    benchmark: Any, problem: Problem, use_processed_cache: bool = True
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if use_processed_cache and benchmark._has_dataset_in_cache(problem.key):
        return benchmark._load_dataset_from_cache(problem.key)
    data_train, data_test = benchmark._load_dataset_from_key(problem.key)
    if use_processed_cache and data_train is not None:
        benchmark._save_dataset(problem.key, data_train, data_test)
    return data_train, data_test


def export(
    config: dict[str, Any],
    output_dir: Path,
    classification_metrics: Path,
    seed_index: int | None = None,
    all_resamples: bool = False,
) -> int:
    from raman_bench.benchmark import RamanBenchmark
    from raman_bench.seeds import get_seeds

    writer = _select_writer()
    seeds = get_seeds(config)
    if seed_index is not None:
        resample = seed_index
        if resample < 0 or resample >= len(seeds):
            raise IndexError(f"seed index {resample} out of range for {len(seeds)} seed(s)")
        resamples = [(resample, seeds[resample])]
    else:
        resamples = list(enumerate(seeds))
    use_processed_cache = False

    cache_dir = config.get("cache_dir") or ".cache"
    cache_dir_raw = os.path.join(cache_dir, "datasets_raw")
    metric_problems = _load_metric_problem_keys(classification_metrics)
    classification_names = sorted(
        {RamanBenchmark.split_key(str(key))[0] for key in metric_problems["key"]}
    )
    configured_names = set(_resolve_classification_dataset_names(config, cache_dir_raw))
    unknown_names = sorted(set(classification_names) - configured_names)
    if unknown_names:
        logger.warning(
            "Metrics CSV contains %d classification dataset(s) not present in the "
            "configured dataset list: %s",
            len(unknown_names),
            unknown_names,
        )
    logger.info(
        "Processed split cache disabled for export; raw cache remains %s",
        cache_dir_raw,
    )
    logger.info(
        "Exporting configured resamples: %s",
        ", ".join(f"{resample} (seed {seed})" for resample, seed in resamples),
    )

    records: list[dict[str, Any]] = []
    for resample, seed in resamples:
        logger.info("--- Resample %d | seed %s ---", resample, seed)
        benchmark = RamanBenchmark(
            dataset_names_classification=classification_names,
            dataset_names_regression=[],
            test_size=config["test_size"],
            random_state=seed,
            cache_dir=cache_dir,
            min_samples_per_class=config.get("min_samples_per_class", 9),
            group_regression_splits=config.get("group_regression_splits", True),
        )

        for problem in _iter_classification_problems(benchmark, metric_problems):
            problem_name = problem.problem_name
            problem_file_stem = f"{problem_name}{resample}"
            task_label = problem.task_label
            problem_dir = output_dir / task_label / problem_name
            train_file = problem_dir / f"{problem_file_stem}_TRAIN.ts"
            test_file = problem_dir / f"{problem_file_stem}_TEST.ts"
            record = _empty_record(
                problem_name,
                resample,
                seed,
                task_label,
                problem.dataset_name,
                problem.target_name,
                problem.key,
                problem.target_idx,
            )

            try:
                if problem.enumeration_error:
                    raise ValueError(f"dataset enumeration failed: {problem.enumeration_error}")
                if train_file.exists() and test_file.exists():
                    record.update(
                        {
                            "train_file": str(train_file),
                            "test_file": str(test_file),
                            "export_status": "skip",
                            "error_message": "already exported",
                        }
                    )
                    logger.info(
                        "Skipping %s resample %d; export files already exist",
                        problem_name,
                        resample,
                    )
                    records.append(record)
                    _write_manifests(records, output_dir, writer.name, export_complete=False)
                    continue

                logger.info(
                    "Downloading/exporting %s resample %d seed %s (%s: %s)",
                    problem_name,
                    resample,
                    seed,
                    problem.dataset_name,
                    problem.target_name,
                )
                data_train, data_test = _load_problem_split(
                    benchmark,
                    problem,
                    use_processed_cache=use_processed_cache,
                )
                if data_train is None or data_test is None:
                    raise ValueError("benchmark returned an empty problem")

                X_train, y_train = _dataframe_to_xy(data_train)
                X_test, y_test = _dataframe_to_xy(data_test)
                warnings = _validate_problem(X_train, y_train, X_test, y_test, problem.task_type)
                for warning in warnings:
                    logger.warning("%s resample %d: %s", problem_name, resample, warning)

                _write_ts(
                    writer,
                    X_train,
                    y_train,
                    task_label,
                    problem_dir,
                    problem_file_stem,
                    "_TRAIN",
                )
                _write_ts(
                    writer,
                    X_test,
                    y_test,
                    task_label,
                    problem_dir,
                    problem_file_stem,
                    "_TEST",
                )

                record.update(
                    {
                        "n_train": int(X_train.shape[0]),
                        "n_test": int(X_test.shape[0]),
                        "n_timepoints": int(X_train.shape[1]),
                        "n_classes": int(pd.Series(np.concatenate([y_train, y_test])).nunique()),
                        "y_dtype": str(np.asarray(y_train).dtype),
                        "train_file": str(train_file),
                        "test_file": str(test_file),
                        "export_status": "pass",
                        "error_message": "",
                    }
                )
                logger.info(
                    (
                        "Exported %s resample %d seed %s: "
                        "X_train=%s, y_train=%s, X_test=%s, y_test=%s, "
                        "train_classes=%s, test_classes=%s"
                    ),
                    problem_name,
                    resample,
                    seed,
                    X_train.shape,
                    y_train.shape,
                    X_test.shape,
                    y_test.shape,
                    _class_distribution(y_train),
                    _class_distribution(y_test),
                )
            except Exception as exc:
                record["export_status"] = "fail"
                record["error_message"] = str(exc)
                logger.error(
                    "Failed to export %s resample %d seed %s: %s",
                    problem_name,
                    resample,
                    seed,
                    exc,
                )

            records.append(record)
            _write_manifests(records, output_dir, writer.name, export_complete=False)

    _write_manifests(records, output_dir, writer.name, export_complete=True)
    failed = sum(record["export_status"] == "fail" for record in records)
    skipped = sum(record["export_status"] == "skip" for record in records)
    logger.info(
        "Export complete: %d passed, %d skipped, %d failed. Manifests: %s",
        len(records) - skipped - failed,
        skipped,
        failed,
        output_dir / "manifests",
    )
    return 1 if failed else 0


def main() -> int:
    from raman_bench.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_default_config(), help="Path to benchmark config JSON")
    parser.add_argument(
        "--classification-metrics",
        default=_default_classification_metrics(),
        help="Path to RamanBench precomputed classification metrics CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\Data\RamanBench",
        help="Directory for exported .ts files and manifests",
    )
    parser.add_argument("--cache-dir", default=None, help="Override RamanBench cache directory")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed-index",
        type=int,
        default=None,
        help=(
            "Export only this zero-based seed index from the config; by default, "
            "export all configured resamples"
        ),
    )
    seed_group.add_argument(
        "--all-resamples",
        action="store_true",
        help="Export every resample returned by get_seeds(config); this is the default",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    config = load_config(args.config)
    if args.cache_dir:
        config["cache_dir"] = args.cache_dir

    return export(
        config,
        Path(args.output_dir),
        Path(args.classification_metrics),
        seed_index=args.seed_index,
        all_resamples=args.all_resamples,
    )


if __name__ == "__main__":
    raise SystemExit(main())
