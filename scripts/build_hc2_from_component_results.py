"""Build file-based HC2 from RamanBench component results and compare to leaderboard.

Requires the four component folders produced by ``run_hc2_components.py``:
``Arsenal``, ``DrCIF-500``, ``TDE`` and ``STC``. HC2 is built only where every
component has train and test files for that problem and resample. The train
files supply the CAWPE weights; test files supply component probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

COMPONENTS = ("Arsenal", "DrCIF-500", "TDE", "STC")
METRICS = ("accuracy", "f1_score", "balanced_accuracy")


def _default_tsml_eval() -> Path:
    hali_checkout = Path.home() / "Code/tsml-eval"
    windows_checkout = Path(r"C:\Code\tsml-eval")
    return hali_checkout if hali_checkout.exists() else windows_checkout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(r"D:\Results\RamanBench"))
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Data\RamanBench\classification"))
    parser.add_argument("--leaderboard", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/precomputed/classification_metrics.csv")
    parser.add_argument("--tsml-eval", type=Path, default=_default_tsml_eval())
    parser.add_argument("--resamples", default="0,1,2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tsml_path = str(args.tsml_eval.resolve())
    if tsml_path not in sys.path:
        sys.path.insert(0, tsml_path)
    from tsml_eval.estimators.classification.hybrid import build_hivecote_from_results

    root = args.results_root.resolve()
    data_dir = args.data_dir.resolve()
    leaderboard = pd.read_csv(args.leaderboard)
    # The bundled classification leaderboard is the authoritative 21-problem list;
    # the user's extracted data directory may still contain only the original 13.
    problems = sorted(leaderboard["key"].dropna().astype(str).unique())
    resamples = [int(x.strip()) for x in args.resamples.split(",") if x.strip()]
    all_rows = []
    missing_rows = []

    for problem in problems:
        for resample in resamples:
            expected = [
                root / component / "Predictions" / problem / f"{split}Resample{resample}.csv"
                for component in COMPONENTS
                for split in ("train", "test")
            ]
            missing = [str(path) for path in expected if not path.is_file()]
            output = root / "HC2" / "Predictions" / problem / f"testResample{resample}.csv"
            if missing:
                missing_rows.append({"problem": problem, "resample": resample, "missing_files": missing})
                continue

            if output.is_file() and not args.overwrite:
                from tsml_eval.evaluation.storage import ClassifierResults
                hc_result = ClassifierResults().load_from_file(str(output))
                status = "already_present"
            else:
                hc_result, _ = build_hivecote_from_results(
                    paths=[str(root / component) for component in COMPONENTS],
                    dataset=problem,
                    resample_id=resample,
                    output_path=str(root),
                    classifier_name="HC2",
                    write_train_file=False,
                    alpha=4,
                )
                status = "built"

            y_true = np.asarray(hc_result.class_labels)
            y_pred = np.asarray(hc_result.predictions)
            scores = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            }
            board = leaderboard[(leaderboard["key"] == problem) & (leaderboard["seed"] == resample)]
            row = {"problem": problem, "resample": resample, "status": status,
                   "n_test": len(y_true), "hc2_file": str(output)}
            row.update({f"hc2_{k}": v for k, v in scores.items()})
            if board.empty:
                row["leaderboard_status"] = "no matching key/seed"
            else:
                row["leaderboard_status"] = "matched"
                for metric in METRICS:
                    best_idx = board[metric].astype(float).idxmax()
                    best = board.loc[best_idx]
                    value = scores[metric]
                    row[f"leaderboard_best_{metric}"] = float(best[metric])
                    row[f"leaderboard_best_{metric}_model"] = str(best["model"])
                    row[f"hc2_minus_best_{metric}"] = value - float(best[metric])
                    row[f"hc2_rank_{metric}"] = 1 + int((board[metric].astype(float) > value).sum())
                row["n_leaderboard_models"] = int(len(board))
            all_rows.append(row)
            print(f"{status.upper()} {problem} resample {resample}: "
                  f"accuracy={scores['accuracy']:.4f}, "
                  f"balanced_accuracy={scores['balanced_accuracy']:.4f}", flush=True)

    summary_dir = root / "comparisons"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(summary_dir / "HC2_vs_leaderboard.csv", index=False)

    # Rank all four supplied components and HC2 against the leaderboard on
    # exactly the same problem/resample pairs. The leaderboard rows are retained
    # so the rank has a clear reference set (rather than comparing only five models).
    from tsml_eval.evaluation.storage import ClassifierResults

    rank_rows = []
    for result_row in all_rows:
        problem, resample = result_row["problem"], result_row["resample"]
        board = leaderboard[(leaderboard["key"] == problem) & (leaderboard["seed"] == resample)]
        if board.empty:
            continue
        split_results = []
        for _, board_row in board.iterrows():
            entry = {"problem": problem, "resample": resample,
                     "model": str(board_row["model"]), "source": "leaderboard"}
            entry.update({metric: float(board_row[metric]) for metric in METRICS})
            split_results.append(entry)
        result_paths = {
            component: root / component / "Predictions" / problem / f"testResample{resample}.csv"
            for component in COMPONENTS
        }
        result_paths["HC2"] = root / "HC2" / "Predictions" / problem / f"testResample{resample}.csv"
        for model, path in result_paths.items():
            if not path.is_file():
                continue
            result = ClassifierResults().load_from_file(str(path))
            y_true, y_pred = np.asarray(result.class_labels), np.asarray(result.predictions)
            split_results.append({
                "problem": problem, "resample": resample, "model": model, "source": "rerun",
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            })
        split_df = pd.DataFrame(split_results)
        for metric in METRICS:
            split_df[f"rank_{metric}"] = split_df[metric].rank(method="min", ascending=False).astype(int)
        rank_rows.extend(split_df.to_dict(orient="records"))

    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(summary_dir / "component_leaderboard_ranks.csv", index=False)
    if not ranks.empty:
        reruns = ranks[ranks["source"] == "rerun"]
        rank_summary = reruns.groupby("model").agg(
            splits=("problem", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_rank_accuracy=("rank_accuracy", "mean"),
            first_accuracy=("rank_accuracy", lambda x: int((x == 1).sum())),
            mean_f1_macro=("f1_score", "mean"),
            mean_rank_f1_macro=("rank_f1_score", "mean"),
            first_f1_macro=("rank_f1_score", lambda x: int((x == 1).sum())),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_rank_balanced_accuracy=("rank_balanced_accuracy", "mean"),
            first_balanced_accuracy=("rank_balanced_accuracy", lambda x: int((x == 1).sum())),
        ).reset_index().sort_values("mean_rank_accuracy")
        rank_summary.to_csv(summary_dir / "component_leaderboard_rank_summary.csv", index=False)
        print("\nMean ranks on matched splits (rank 1 is best):")
        print(rank_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    (summary_dir / "HC2_build_manifest.json").write_text(
        json.dumps({"built_or_loaded": len(all_rows), "missing_problem_resamples": len(missing_rows),
                    "components": COMPONENTS, "alpha": 4, "resamples": resamples,
                    "missing": missing_rows}, indent=2), encoding="utf-8")

    if all_rows:
        compared = pd.DataFrame(all_rows)
        matched = compared[compared["leaderboard_status"] == "matched"]
        print(f"DONE: {len(all_rows)} HC2 results; {len(missing_rows)} problem/resamples "
              f"skipped for missing components; comparison rows={len(matched)}")
        for metric in METRICS:
            diff_col = f"hc2_minus_best_{metric}"
            if diff_col in matched:
                print(f"Mean HC2 minus leaderboard best {metric}: "
                      f"{matched[diff_col].mean():+.4f}; HC2 ranked first on "
                      f"{int((matched[f'hc2_rank_{metric}'] == 1).sum())}/{len(matched)}")
    else:
        print(f"No HC2 results built. Missing component files for all {len(missing_rows)} problem-resamples.")
    print(f"Outputs: {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
