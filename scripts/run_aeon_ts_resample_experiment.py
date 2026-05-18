#!/usr/bin/env python
"""Run aeon classifiers on exported RamanBench ``.ts`` classification splits."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_bench.seeds import get_seeds

logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

RESULT_COLUMNS = [
    "classifier_name",
    "problem_name",
    "ramanbench_key",
    "target_idx",
    "original_dataset_name",
    "original_target_name",
    "resample",
    "seed",
    "n_train",
    "n_test",
    "n_timepoints",
    "n_classes",
    "f1_macro",
    "balanced_accuracy",
    "accuracy",
    "repo_model",
    "repo_f1_score",
    "repo_balanced_accuracy",
    "f1_macro_diff",
    "balanced_accuracy_diff",
    "train_time_seconds",
    "predict_time_seconds",
    "train_file",
    "test_file",
    "status",
    "error_message",
]

PREDICTION_COLUMNS = [
    "problem_name",
    "ramanbench_key",
    "resample",
    "seed",
    "classifier_name",
    "case_index",
    "y_true",
    "y_pred",
]

REPO_MODEL_NAMES = {
    "rocket": "ROCKET",
    "random_forest": "RF",
    "ridge": "LR",
    "rotation_forest": "Rotation Forest",
}


@dataclass(frozen=True)
class ProblemRun:
    """Description of one exported problem/resample split to evaluate."""

    problem_name: str
    ramanbench_key: str
    target_idx: Any
    original_dataset_name: str
    original_target_name: str
    resample: int
    seed: int
    train_file: Path
    test_file: Path


@dataclass(frozen=True)
class LoadedSplit:
    """Loaded train/test arrays in both aeon and sklearn-compatible layouts."""

    X_train_aeon: np.ndarray
    X_test_aeon: np.ndarray
    X_train_2d: np.ndarray
    X_test_2d: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_timepoints: int


def make_classifier(name: str, random_state: int) -> Any:
    """Create a supported classifier with a reproducible random state."""
    normalized = name.strip().lower()
    if normalized == "rocket":
        try:
            from aeon.classification.convolution_based import RocketClassifier

            return _instantiate_with_supported_kwargs(
                RocketClassifier,
                {"random_state": random_state, "n_jobs": -1},
            )
        except Exception as rocket_error:
            try:
                from aeon.classification.convolution_based import MiniRocketClassifier

                return _instantiate_with_supported_kwargs(
                    MiniRocketClassifier,
                    {"random_state": random_state, "n_jobs": -1},
                )
            except Exception as mini_error:
                raise ImportError(
                    "Could not construct an aeon ROCKET classifier. Tried "
                    "RocketClassifier and MiniRocketClassifier."
                ) from mini_error or rocket_error

    if normalized == "rotation_forest":
        try:
            from aeon.classification.sklearn import RotationForestClassifier

            return _instantiate_with_supported_kwargs(
                RotationForestClassifier,
                {"random_state": random_state, "n_jobs": -1},
            )
        except Exception as exc:
            raise ImportError(
                "Could not import aeon.classification.sklearn.RotationForestClassifier"
            ) from exc

    if normalized == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=500, random_state=random_state, n_jobs=-1
        )

    if normalized == "ridge":
        from sklearn.linear_model import RidgeClassifier

        return RidgeClassifier(random_state=random_state)

    raise ValueError(
        f"Unsupported classifier {name!r}. Expected one of: "
        "rocket, rotation_forest, random_forest, ridge."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line experiment."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    ts_dir = _resolve_path(args.ts_dir)
    manifest_path = _resolve_path(args.manifest or ts_dir / "manifests" / "classification_problems.csv")
    repo_results_path = _resolve_path(args.repo_results)
    config_path = _resolve_path(args.config)
    output_path = _resolve_path(args.output)
    predictions_dir = _resolve_path(args.predictions_dir)

    classifier_key = args.classifier.strip().lower()
    classifier_name = args.classifier_name or REPO_MODEL_NAMES.get(
        classifier_key, classifier_key
    )
    repo_model = REPO_MODEL_NAMES.get(classifier_key, classifier_name)

    logger.info("Selected classifier: %s (output name: %s)", classifier_key, classifier_name)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    seeds = get_seeds(config)

    manifest = _read_manifest(manifest_path)
    selected_problems = _parse_optional_list(args.problems)
    selected_resamples = _parse_resamples(args.resamples)
    runs = _build_problem_runs(manifest, ts_dir, seeds, selected_problems, selected_resamples)

    unique_problems = {run.problem_name for run in runs}
    unique_resamples = {run.resample for run in runs}
    logger.info("Number of problems: %d", len(unique_problems))
    logger.info("Number of resamples: %d", len(unique_resamples))
    logger.info("Number of problem/resample runs: %d", len(runs))

    repo_df, repo_status = _load_repo_results(repo_results_path)
    existing = _load_existing_results(output_path)
    completed = _completed_keys(existing, classifier_name) if not args.overwrite else set()

    result_rows: list[dict[str, Any]] = []
    for run in runs:
        row_key = (classifier_name, run.problem_name, int(run.resample))
        if row_key in completed:
            logger.info(
                "Skipping completed %s resample %d for %s",
                run.problem_name,
                run.resample,
                classifier_name,
            )
            continue

        result_row = _base_result_row(run, classifier_name)
        try:
            logger.info(
                "Starting %s resample %d seed %s", run.problem_name, run.resample, run.seed
            )
            _set_random_state(run.seed)
            loaded = _load_split(run.train_file, run.test_file)
            logger.info("Train class distribution: %s", _class_distribution(loaded.y_train))
            logger.info("Test class distribution: %s", _class_distribution(loaded.y_test))

            classifier = make_classifier(classifier_key, run.seed)
            X_train, X_test = _select_classifier_input(
                classifier_key, loaded, classifier
            )

            start = time.perf_counter()
            classifier.fit(X_train, loaded.y_train)
            train_time = time.perf_counter() - start

            start = time.perf_counter()
            y_pred = classifier.predict(X_test)
            predict_time = time.perf_counter() - start

            metrics = _compute_metrics(loaded.y_test, y_pred)
            repo_values = _align_repo_result(
                repo_df, repo_status, run.ramanbench_key, run.seed, repo_model, metrics
            )
            _write_predictions(
                predictions_dir, run, classifier_name, loaded.y_test, y_pred
            )

            result_row.update(
                {
                    "n_train": int(len(loaded.y_train)),
                    "n_test": int(len(loaded.y_test)),
                    "n_timepoints": int(loaded.n_timepoints),
                    "n_classes": int(pd.Series(loaded.y_train).nunique(dropna=False)),
                    "f1_macro": metrics["f1_macro"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "accuracy": metrics["accuracy"],
                    "repo_model": repo_values["repo_model"],
                    "repo_f1_score": repo_values["repo_f1_score"],
                    "repo_balanced_accuracy": repo_values["repo_balanced_accuracy"],
                    "f1_macro_diff": repo_values["f1_macro_diff"],
                    "balanced_accuracy_diff": repo_values["balanced_accuracy_diff"],
                    "repo_match_status": repo_values["repo_match_status"],
                    "train_time_seconds": train_time,
                    "predict_time_seconds": predict_time,
                    "status": "pass",
                    "error_message": "",
                }
            )
            logger.info(
                "Finished %s resample %d: f1_macro=%.6f balanced_accuracy=%.6f "
                "accuracy=%.6f repo_match=%s",
                run.problem_name,
                run.resample,
                metrics["f1_macro"],
                metrics["balanced_accuracy"],
                metrics["accuracy"],
                repo_values["repo_match_status"],
            )
        except Exception as exc:
            result_row.update({"status": "fail", "error_message": str(exc)})
            logger.exception(
                "Failed %s resample %d: %s", run.problem_name, run.resample, exc
            )

        result_rows.append(result_row)
        _save_results(output_path, existing, result_rows, args.overwrite)

    _save_results(output_path, existing, result_rows, args.overwrite)
    logger.info("Saved results to %s", output_path)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts-dir", type=Path, default=Path(r"C:\Data\RamanBench"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--repo-results",
        type=Path,
        default=ROOT / "data" / "precomputed" / "classification_metrics.csv",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "benchmark_v0.1.json"
    )
    parser.add_argument(
        "--classifier",
        default="rocket",
        choices=["rocket", "rotation_forest", "random_forest", "ridge"],
    )
    parser.add_argument("--classifier-name", default=None)
    parser.add_argument("--resamples", default=None)
    parser.add_argument("--problems", default=None)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "aeon_ts_resample_results.csv"
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=ROOT / "results" / "predictions"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = pd.read_csv(path)
    if "problem_name" not in manifest.columns:
        raise ValueError(f"Manifest {path} is missing required column problem_name")
    if "ramanbench_key" not in manifest.columns:
        raise ValueError(f"Manifest {path} is missing required column ramanbench_key")
    return manifest


def _parse_optional_list(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_resamples(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _build_problem_runs(
    manifest: pd.DataFrame,
    ts_dir: Path,
    seeds: list[int],
    selected_problems: set[str] | None,
    selected_resamples: list[int] | None,
) -> list[ProblemRun]:
    records = []
    task_dir = ts_dir / "classification"
    for _, row in manifest.iterrows():
        if selected_problems and (
            str(row.get("problem_name", "")) not in selected_problems
            and str(row.get("ramanbench_key", "")) not in selected_problems
        ):
            continue
        if str(row.get("export_status", "pass")).lower() == "fail":
            continue

        if "resample" in manifest.columns and not pd.isna(row.get("resample")):
            candidate_resamples = [int(row["resample"])]
        elif selected_resamples is not None:
            candidate_resamples = selected_resamples
        else:
            candidate_resamples = _infer_resamples(task_dir, str(row["problem_name"]))

        for resample in candidate_resamples:
            if resample < 0 or resample >= len(seeds):
                raise IndexError(
                    f"resample {resample} out of range for configured seed list "
                    f"of length {len(seeds)}"
                )
            train_file, test_file = _resolve_split_files(task_dir, row, resample)
            records.append(
                ProblemRun(
                    problem_name=str(row["problem_name"]),
                    ramanbench_key=str(row["ramanbench_key"]),
                    target_idx=row.get("target_idx", ""),
                    original_dataset_name=str(row.get("original_dataset_name", "")),
                    original_target_name=str(row.get("original_target_name", "")),
                    resample=int(resample),
                    seed=int(seeds[resample]),
                    train_file=train_file,
                    test_file=test_file,
                )
            )
    return records


def _infer_resamples(task_dir: Path, problem_name: str) -> list[int]:
    problem_dir = task_dir / problem_name
    if not problem_dir.exists():
        return []
    resamples = []
    prefix = problem_name
    suffix = "_TRAIN.ts"
    for train_path in problem_dir.glob(f"{problem_name}*_TRAIN.ts"):
        name = train_path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        raw = name[len(prefix) : -len(suffix)]
        if raw == "":
            resample = 0
        elif raw.isdigit():
            resample = int(raw)
        else:
            continue
        test_path = train_path.with_name(f"{problem_name}{raw}_TEST.ts")
        if test_path.exists():
            resamples.append(resample)
    return sorted(set(resamples))


def _resolve_split_files(task_dir: Path, row: pd.Series, resample: int) -> tuple[Path, Path]:
    problem_name = str(row["problem_name"])
    if "resample" in row.index and not pd.isna(row.get("resample")):
        train_file = _manifest_file(row.get("train_file"))
        test_file = _manifest_file(row.get("test_file"))
        if train_file and test_file:
            return train_file, test_file

    problem_dir = task_dir / problem_name
    candidates = [(f"{problem_name}{resample}_TRAIN.ts", f"{problem_name}{resample}_TEST.ts")]
    if resample == 0:
        candidates.append((f"{problem_name}_TRAIN.ts", f"{problem_name}_TEST.ts"))
        train_file = _manifest_file(row.get("train_file"))
        test_file = _manifest_file(row.get("test_file"))
        if train_file and test_file and train_file.exists() and test_file.exists():
            candidates.insert(0, (train_file.name, test_file.name))

    for train_name, test_name in candidates:
        train_file = problem_dir / train_name
        test_file = problem_dir / test_name
        if train_file.exists() and test_file.exists():
            return train_file, test_file
    return problem_dir / candidates[0][0], problem_dir / candidates[0][1]


def _manifest_file(value: Any) -> Path | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return Path(str(value))


def _load_split(train_file: Path, test_file: Path) -> LoadedSplit:
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    X_train, y_train = _load_ts_file(train_file)
    X_test, y_test = _load_ts_file(test_file)
    X_train_aeon = _to_aeon_3d(X_train)
    X_test_aeon = _to_aeon_3d(X_test)
    X_train_2d = _to_2d(X_train_aeon)
    X_test_2d = _to_2d(X_test_aeon)
    if X_train_2d.shape[1] != X_test_2d.shape[1]:
        raise ValueError(
            f"Train/test timepoint mismatch: {X_train_2d.shape[1]} != {X_test_2d.shape[1]}"
        )
    return LoadedSplit(
        X_train_aeon=X_train_aeon,
        X_test_aeon=X_test_aeon,
        X_train_2d=X_train_2d,
        X_test_2d=X_test_2d,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        n_timepoints=int(X_train_2d.shape[1]),
    )


def _load_ts_file(path: Path) -> tuple[Any, np.ndarray]:
    from aeon.datasets import load_from_ts_file

    signature = inspect.signature(load_from_ts_file)
    kwargs: dict[str, Any] = {}
    if "return_type" in signature.parameters:
        kwargs["return_type"] = "numpy3d"
    loaded = load_from_ts_file(str(path), **kwargs)
    if len(loaded) == 3:
        X, y, _ = loaded
    else:
        X, y = loaded
    return X, np.asarray(y)


def _to_aeon_3d(X: Any) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim == 2:
        return arr[:, np.newaxis, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D or 3D series array, got shape {arr.shape}")


def _to_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D aeon array, got shape {X.shape}")
    if X.shape[1] != 1:
        raise ValueError(f"Expected univariate series, got {X.shape[1]} channels")
    return X[:, 0, :]


def _select_classifier_input(
    classifier_key: str, loaded: LoadedSplit, classifier: Any
) -> tuple[np.ndarray, np.ndarray]:
    if classifier_key in {"random_forest", "ridge"}:
        return loaded.X_train_2d, loaded.X_test_2d
    module = type(classifier).__module__
    if ".sklearn" in module:
        return loaded.X_train_2d, loaded.X_test_2d
    return loaded.X_train_aeon, loaded.X_test_aeon


def _instantiate_with_supported_kwargs(cls: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(cls)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return cls(**supported)


def _set_random_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = pd.Series(y).value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _write_predictions(
    predictions_dir: Path,
    run: ProblemRun,
    classifier_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    safe_classifier = str(classifier_name).replace("/", "_").replace("\\", "_")
    path = predictions_dir / (
        f"{run.problem_name}__resample_{run.resample}__{safe_classifier}.csv"
    )
    df = pd.DataFrame(
        {
            "problem_name": run.problem_name,
            "ramanbench_key": run.ramanbench_key,
            "resample": run.resample,
            "seed": run.seed,
            "classifier_name": classifier_name,
            "case_index": np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df.to_csv(path, index=False, columns=PREDICTION_COLUMNS)


def _load_repo_results(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        logger.warning("Repository results file missing: %s", path)
        return None, "repo_file_missing"
    df = pd.read_csv(path)
    required = {"seed", "key", "model", "f1_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Repository results missing required columns: {sorted(missing)}")
    logger.info("Repository result columns: %s", ", ".join(df.columns))
    return df, "loaded"


def _align_repo_result(
    repo_df: pd.DataFrame | None,
    repo_status: str,
    ramanbench_key: str,
    seed: int,
    repo_model: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    values = {
        "repo_model": repo_model,
        "repo_f1_score": np.nan,
        "repo_balanced_accuracy": np.nan,
        "f1_macro_diff": np.nan,
        "balanced_accuracy_diff": np.nan,
        "repo_match_status": repo_status,
    }
    if repo_df is None:
        return values

    problem_seed = repo_df[
        (repo_df["key"].astype(str) == str(ramanbench_key))
        & (repo_df["seed"].astype(int) == int(seed))
    ]
    if problem_seed.empty:
        values["repo_match_status"] = "no_repo_result_for_problem_seed"
        return values
    match = problem_seed[problem_seed["model"].astype(str) == str(repo_model)]
    if match.empty:
        values["repo_match_status"] = "no_repo_result_for_model"
        return values

    row = match.iloc[0]
    repo_f1 = float(row["f1_score"])
    values["repo_f1_score"] = repo_f1
    values["f1_macro_diff"] = float(metrics["f1_macro"] - repo_f1)
    if "balanced_accuracy" in match.columns and not pd.isna(row["balanced_accuracy"]):
        repo_bal_acc = float(row["balanced_accuracy"])
        values["repo_balanced_accuracy"] = repo_bal_acc
        values["balanced_accuracy_diff"] = float(metrics["balanced_accuracy"] - repo_bal_acc)
    values["repo_match_status"] = "matched"
    return values


def _load_existing_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS + ["repo_match_status"])
    return pd.read_csv(path)


def _completed_keys(existing: pd.DataFrame, classifier_name: str) -> set[tuple[str, str, int]]:
    if existing.empty or not {"classifier_name", "problem_name", "resample", "status"}.issubset(existing.columns):
        return set()
    completed = existing[
        (existing["classifier_name"].astype(str) == str(classifier_name))
        & (existing["status"].astype(str) == "pass")
    ]
    return {
        (str(row["classifier_name"]), str(row["problem_name"]), int(row["resample"]))
        for _, row in completed.iterrows()
    }


def _base_result_row(run: ProblemRun, classifier_name: str) -> dict[str, Any]:
    return {
        "classifier_name": classifier_name,
        "problem_name": run.problem_name,
        "ramanbench_key": run.ramanbench_key,
        "target_idx": run.target_idx,
        "original_dataset_name": run.original_dataset_name,
        "original_target_name": run.original_target_name,
        "resample": run.resample,
        "seed": run.seed,
        "n_train": np.nan,
        "n_test": np.nan,
        "n_timepoints": np.nan,
        "n_classes": np.nan,
        "f1_macro": np.nan,
        "balanced_accuracy": np.nan,
        "accuracy": np.nan,
        "repo_model": "",
        "repo_f1_score": np.nan,
        "repo_balanced_accuracy": np.nan,
        "f1_macro_diff": np.nan,
        "balanced_accuracy_diff": np.nan,
        "repo_match_status": "",
        "train_time_seconds": np.nan,
        "predict_time_seconds": np.nan,
        "train_file": str(run.train_file),
        "test_file": str(run.test_file),
        "status": "",
        "error_message": "",
    }


def _save_results(
    output_path: Path,
    existing: pd.DataFrame,
    result_rows: list[dict[str, Any]],
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(result_rows)
    if overwrite or existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
    columns = [c for c in RESULT_COLUMNS if c in combined.columns]
    if "repo_match_status" in combined.columns:
        insert_at = columns.index("repo_model") if "repo_model" in columns else len(columns)
        columns = columns[:insert_at] + ["repo_match_status"] + columns[insert_at:]
    columns += [c for c in combined.columns if c not in columns]
    combined.to_csv(output_path, index=False, columns=columns)


if __name__ == "__main__":
    raise SystemExit(main())
