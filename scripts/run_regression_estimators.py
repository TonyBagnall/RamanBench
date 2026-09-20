#!/usr/bin/env python
"""Run aeon DrCIFRegressor and QUANTRegressor on RamanBench regression .ts splits.

Writes tsml-eval-style testResampleN.csv results and, by default, 10-fold
trainResampleN.csv estimates under <results>/<Regressor>/Predictions/<problem>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from aeon.datasets import load_from_ts_file

LOG = logging.getLogger("run_regression_estimators")
REGRESSORS = ("DrCIF", "QUANT")


def _parse_resamples(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("resamples must be non-negative integers")
    return values


def _problem_files(directory: Path, problem: str, resample: int) -> tuple[Path, Path]:
    stem = f"{problem}{resample}"
    train = directory / f"{stem}_TRAIN.ts"
    test = directory / f"{stem}_TEST.ts"
    if not train.is_file() or not test.is_file():
        raise FileNotFoundError(f"Missing pair for {problem} resample {resample}: {train}, {test}")
    return train, test


def _make_regressor(name: str, seed: int, drcif_estimators: int):
    if name == "DrCIF":
        from aeon.regression.interval_based import DrCIFRegressor

        return DrCIFRegressor(n_estimators=drcif_estimators, random_state=seed)
    from aeon.regression.interval_based import QUANTRegressor

    return QUANTRegressor(random_state=seed)


def _run_one(problem_dir: Path, problem: str, resample: int, regressor_name: str,
             results_dir: Path, drcif_estimators: int, train_files: bool,
             overwrite: bool, benchmark_time: bool) -> dict:
    from tsml_eval.experiments import run_regression_experiment

    output_name = f"DrCIF-{drcif_estimators}" if regressor_name == "DrCIF" else "QUANT"
    prediction_dir = results_dir / output_name / "Predictions" / problem
    train_result = prediction_dir / f"trainResample{resample}.csv"
    test_result = prediction_dir / f"testResample{resample}.csv"
    if test_result.exists() and (not train_files or train_result.exists()) and not overwrite:
        LOG.info("Skipping existing results: %s %s resample %d", output_name, problem, resample)
        return {"regressor": output_name, "problem": problem, "resample": resample,
                "status": "skip"}

    train_file, test_file = _problem_files(problem_dir, problem, resample)
    X_train, y_train = load_from_ts_file(str(train_file))
    X_test, y_test = load_from_ts_file(str(test_file))
    y_train, y_test = np.asarray(y_train, dtype=float), np.asarray(y_test, dtype=float)
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(f"Expected aeon 3D arrays, got {X_train.shape} and {X_test.shape}")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError(f"Non-finite spectra in {problem} resample {resample}")
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError(f"Non-finite targets in {problem} resample {resample}")

    estimator = _make_regressor(regressor_name, resample, drcif_estimators)
    LOG.info("Starting %s %s resample %d: X_train=%s, X_test=%s, train_files=%s",
             output_name, problem, resample, X_train.shape, X_test.shape, train_files)
    started = time.perf_counter()
    run_regression_experiment(
        X_train, y_train, X_test, y_test, estimator, str(results_dir),
        regressor_name=output_name, dataset_name=problem, resample_id=resample,
        build_test_file=True, build_train_file=train_files,
        benchmark_time=benchmark_time,
    )
    elapsed = round(time.perf_counter() - started, 3)
    LOG.info("Completed %s %s resample %d in %.3fs", output_name, problem, resample, elapsed)
    return {"regressor": output_name, "problem": problem, "resample": resample,
            "status": "pass", "seconds": elapsed, "train_file": str(train_file),
            "test_file": str(test_file)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / "Data/RamanBench/regression")
    parser.add_argument("--results-dir", type=Path,
                        default=Path.home() / "Results/RamanBench")
    parser.add_argument("--tsml-eval", type=Path,
                        default=Path.home() / "Code/tsml-eval",
                        help="tsml-eval checkout to add to PYTHONPATH")
    parser.add_argument("--regressor", action="append", choices=REGRESSORS,
                        required=True, help="repeat to select both regressors")
    parser.add_argument("--problem", action="append", default=None)
    parser.add_argument("--resamples", type=_parse_resamples, default=[0, 1, 2])
    parser.add_argument("--drcif-estimators", type=int, default=200)
    parser.add_argument("--train-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    if args.drcif_estimators < 1:
        parser.error("--drcif-estimators must be positive")

    sys.path.insert(0, str(args.tsml_eval.resolve()))
    data_dir, results_dir = args.data_dir.resolve(), args.results_dir.resolve()
    if not data_dir.is_dir():
        parser.error(f"data directory does not exist: {data_dir}")
    problem_dirs = {}
    duplicates = set()
    for train_file in data_dir.rglob("*_TRAIN.ts"):
        problem = train_file.parent.name
        if problem in problem_dirs and problem_dirs[problem] != train_file.parent:
            duplicates.add(problem)
        problem_dirs[problem] = train_file.parent
    duplicates = sorted(duplicates)
    if duplicates:
        parser.error(f"duplicate target/problem directory names: {duplicates}")
    problems = sorted(problem_dirs)
    if args.problem:
        unknown = sorted(set(args.problem) - set(problems))
        if unknown:
            parser.error(f"problem directories not found: {unknown}")
        problems = [problem for problem in problems if problem in set(args.problem)]
    if not problems:
        parser.error(f"no regression .ts split directories found in {data_dir}")

    results = []
    for name in args.regressor:
        for problem in problems:
            for resample in args.resamples:
                try:
                    results.append(_run_one(problem_dirs[problem], problem, resample, name,
                        results_dir, args.drcif_estimators, args.train_files,
                        args.overwrite, args.benchmark_time))
                except Exception as exc:
                    LOG.exception("Failed %s %s resample %d", name, problem, resample)
                    results.append({"regressor": name, "problem": problem,
                                    "resample": resample, "status": "fail", "error": str(exc)})
    summary = args.summary or results_dir / "summaries/regression_estimators_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = sum(row["status"] == "pass" for row in results)
    skipped = sum(row["status"] == "skip" for row in results)
    failed = sum(row["status"] == "fail" for row in results)
    LOG.info("Finished: %d passed, %d skipped, %d failed; summary=%s",
             passed, skipped, failed, summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
