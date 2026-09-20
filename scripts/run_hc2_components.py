#!/usr/bin/env python
"""Run HC2 component classifiers on RamanBench classification ``.ts`` files.

The output layout matches tsml-eval::

    <results>/<Classifier>/Predictions/<problem>/trainResample0.csv
    <results>/<Classifier>/Predictions/<problem>/testResample0.csv

The train result is a 10-fold estimate where supported by tsml-eval. The test
result is produced after fitting on the complete TRAIN file. The input files
are never changed.

Example on HALI::

    python scripts/run_hc2_components.py \
        --data-dir /gpfs/home/ajb/Data/RamanBench/classification \
        --results-dir /gpfs/home/ajb/Results/RamanBench \
        --tsml-eval /gpfs/home/ajb/Code/tsml-eval \
        --classifier Arsenal --classifier DrCIF --classifier TDE --classifier STC \
        --resamples 0,1,2 --train-files

``DrCIF`` means ``DrCIF-500`` by default, matching the HC2 component settings.
Use ``--drcif-estimators 200`` for the standard aeon DrCIF default instead.
Estimator verbosity is enabled where supported; use ``--no-verbose`` to disable it.
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

LOG = logging.getLogger("run_hc2_components")
CLASSIFIERS = ("Arsenal", "DrCIF", "TDE", "STC", "HC2", "MRHydra")


def _parse_resamples(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("resamples must be non-negative integers")
    return values


def _problem_files(directory: Path, problem: str, resample: int) -> tuple[Path, Path]:
    # Problem directories already include the target suffix, e.g. ``alzheimer_0``.
    # RamanBench files append the resample index directly: ``alzheimer_00``.
    stem = f"{problem}{resample}"
    train = directory / f"{stem}_TRAIN.ts"
    test = directory / f"{stem}_TEST.ts"
    if not train.is_file() or not test.is_file():
        raise FileNotFoundError(f"Missing pair for {problem} resample {resample}: {train}, {test}")
    return train, test


def _classifier_name(name: str, drcif_estimators: int) -> str:
    return "DrCIF-500" if name == "DrCIF" and drcif_estimators == 500 else name


def _make_classifier(name: str, seed: int, n_jobs: int, fit_contract: int,
                     drcif_estimators: int, verbose: bool = True):
    from tsml_eval.experiments import get_classifier_by_name

    requested = _classifier_name(name, drcif_estimators)
    estimator = get_classifier_by_name(
        requested,
        random_state=seed,
        n_jobs=n_jobs,
        fit_contract=fit_contract,
    )
    # Not all aeon components expose verbosity (currently HC2 and STC do).
    if "verbose" in estimator.get_params(deep=False):
        estimator.set_params(verbose=verbose)
    return requested, estimator


def _run_one(
    problem_dir: Path,
    problem: str,
    resample: int,
    classifier: str,
    results_dir: Path,
    n_jobs: int,
    fit_contract: int,
    drcif_estimators: int,
    train_files: bool,
    overwrite: bool,
    benchmark_time: bool,
    verbose: bool = True,
) -> dict:
    from tsml_eval.experiments.experiments import run_classification_experiment

    output_name = _classifier_name(classifier, drcif_estimators)
    prediction_dir = results_dir / output_name / "Predictions" / problem
    train_result = prediction_dir / f"trainResample{resample}.csv"
    test_result = prediction_dir / f"testResample{resample}.csv"
    if test_result.exists() and (not train_files or train_result.exists()) and not overwrite:
        LOG.info("Skipping existing results: %s %s resample %d", output_name, problem, resample)
        return {"classifier": output_name, "problem": problem, "resample": resample, "status": "skip"}

    train_file, test_file = _problem_files(problem_dir, problem, resample)
    LOG.info("Loading %s and %s", train_file, test_file)
    X_train, y_train = load_from_ts_file(str(train_file))
    X_test, y_test = load_from_ts_file(str(test_file))
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(f"Expected aeon 3D arrays, got {X_train.shape} and {X_test.shape}")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError(f"Non-finite values in {problem} resample {resample}")

    output_name, estimator = _make_classifier(
        classifier, resample, n_jobs, fit_contract, drcif_estimators, verbose
    )
    LOG.info("Starting %s: train=%s, test=%s, classes=%d, train_results=%s, verbose=%s",
             output_name, X_train.shape, X_test.shape, len(np.unique(y_train)),
             train_files, estimator.get_params(deep=False).get("verbose", "unsupported"))
    started = time.perf_counter()
    run_classification_experiment(
        X_train,
        np.asarray(y_train),
        X_test,
        np.asarray(y_test),
        estimator,
        str(results_dir),
        classifier_name=output_name,
        dataset_name=problem,
        resample_id=resample,
        build_test_file=True,
        build_train_file=train_files,
        benchmark_time=benchmark_time,
    )
    elapsed = round(time.perf_counter() - started, 3)
    LOG.info("Completed %s %s resample %d in %.3fs; test results: %s",
             output_name, problem, resample, elapsed, test_result)
    return {
        "classifier": output_name,
        "problem": problem,
        "resample": resample,
        "status": "pass",
        "seconds": elapsed,
        "train_file": str(train_file),
        "test_file": str(test_file),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / "Data/RamanBench/classification")
    parser.add_argument("--results-dir", type=Path,
                        default=Path.home() / "Results/RamanBench")
    parser.add_argument("--tsml-eval", type=Path,
                        default=Path.home() / "Code/tsml-eval",
                        help="tsml-eval checkout to add to PYTHONPATH")
    parser.add_argument("--classifier", action="append", choices=CLASSIFIERS,
                        required=True, help="repeat to select multiple components")
    parser.add_argument("--problem", action="append", default=None,
                        help="select one or more problem directory names")
    parser.add_argument("--resamples", type=_parse_resamples, default=[0, 1, 2])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--fit-contract", type=int, default=0,
                        help="fit limit in minutes; 0 means no limit")
    parser.add_argument("--drcif-estimators", type=int, choices=[200, 500], default=500)
    parser.add_argument("--train-files", action=argparse.BooleanOptionalAction, default=True,
                        help="write tsml trainResample files; enabled by default")
    parser.add_argument("--benchmark-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                        help="enable estimator progress output where supported (default: on)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    if args.tsml_eval:
        sys.path.insert(0, str(args.tsml_eval.resolve()))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.n_jobs))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.n_jobs))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.n_jobs))

    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    if not data_dir.is_dir():
        parser.error(f"data directory does not exist: {data_dir}")
    problems = sorted(
        directory.name for directory in data_dir.iterdir()
        if directory.is_dir() and any(directory.glob("*_TRAIN.ts"))
    )
    if args.problem:
        unknown = sorted(set(args.problem) - set(problems))
        if unknown:
            parser.error(f"problem directories not found: {unknown}")
        problems = [problem for problem in problems if problem in set(args.problem)]
    if not args.problem and len(problems) != 21:
        LOG.warning("Found %d problems; expected 21", len(problems))
    classifiers = args.classifier
    results = []
    for classifier in classifiers:
        for problem in problems:
            problem_dir = data_dir / problem
            for resample in args.resamples:
                LOG.info("%s %s resample %d", classifier, problem, resample)
                try:
                    results.append(_run_one(
                        problem_dir, problem, resample, classifier, results_dir,
                        args.n_jobs, args.fit_contract, args.drcif_estimators,
                        args.train_files, args.overwrite, args.benchmark_time,
                        args.verbose,
                    ))
                except Exception as exc:  # keep the batch moving across datasets
                    LOG.exception("Failed %s %s resample %d", classifier, problem, resample)
                    results.append({
                        "classifier": _classifier_name(classifier, args.drcif_estimators),
                        "problem": problem, "resample": resample,
                        "status": "fail", "error": str(exc),
                    })

    summary = args.summary or results_dir / "hc2_components_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = sum(row["status"] == "pass" for row in results)
    skipped = sum(row["status"] == "skip" for row in results)
    failed = sum(row["status"] == "fail" for row in results)
    LOG.info("Finished: %d passed, %d skipped, %d failed; summary=%s", passed, skipped, failed, summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
