from pathlib import Path
import pandas as pd

from src.pipeline import run_cv
from src.config import (
    DATASET_SPLIT,
    USE_NON_EMPTY_ONLY,
    MODEL_NAME,
    LOG_FEATURES,
    REDUCED_FEATURES,
    BASELINE_GROUPS,
)
import src.pipeline as pipeline_module
import src.config as config_module


def run_one_experiment(df, exp_name, feature_cols, use_tfidf, tfidf_sources):
    # 临时覆盖全局 config / pipeline 中读取的 TF-IDF 源
    old_sources_cfg = config_module.TFIDF_TEXT_SOURCES
    old_sources_pipe = pipeline_module.TFIDF_TEXT_SOURCES

    config_module.TFIDF_TEXT_SOURCES = tfidf_sources
    pipeline_module.TFIDF_TEXT_SOURCES = tfidf_sources

    try:
        results_df, summary_df, threshold_selection_df, fixed_threshold_summary_df = run_cv(
            df=df,
            feature_cols=feature_cols,
            log_features=LOG_FEATURES,
            model_name=MODEL_NAME,
            use_message_tfidf=use_tfidf,
        )

        summary_df = summary_df.copy()
        summary_df.insert(0, "experiment", exp_name)
        summary_df["feature_cols"] = ",".join(feature_cols) if feature_cols else "__NONE__"
        summary_df["tfidf_sources"] = ",".join(tfidf_sources) if tfidf_sources else "__NONE__"

        results_df = results_df.copy()
        results_df.insert(0, "experiment", exp_name)

        threshold_selection_df = threshold_selection_df.copy()
        threshold_selection_df.insert(0, "experiment", exp_name)

        fixed_threshold_summary_df = fixed_threshold_summary_df.copy()
        fixed_threshold_summary_df.insert(0, "experiment", exp_name)
        fixed_threshold_summary_df["feature_cols"] = ",".join(feature_cols) if feature_cols else "__NONE__"
        fixed_threshold_summary_df["tfidf_sources"] = ",".join(tfidf_sources) if tfidf_sources else "__NONE__"

        return results_df, summary_df, threshold_selection_df, fixed_threshold_summary_df

    finally:
        config_module.TFIDF_TEXT_SOURCES = old_sources_cfg
        pipeline_module.TFIDF_TEXT_SOURCES = old_sources_pipe


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv(results_dir / f"{DATASET_SPLIT}_features_full.csv")

    if USE_NON_EMPTY_ONLY:
        df = df[df["patch_is_empty"] == 0].copy()

    all_results = []
    all_summaries = []
    all_thresholds = []
    all_fixed_threshold_summaries = []

    baseline_features = REDUCED_FEATURES

    experiments = [
        {
            "name": "baseline_only",
            "feature_cols": baseline_features,
            "use_tfidf": False,
            "tfidf_sources": [],
        },
        {
            "name": "assistant_only_tfidf",
            "feature_cols": [],
            "use_tfidf": True,
            "tfidf_sources": ["assistant"],
        },
        {
            "name": "file_path_only_tfidf",
            "feature_cols": [],
            "use_tfidf": True,
            "tfidf_sources": ["file_path"],
        },
        {
            "name": "assistant_file_path_only_tfidf",
            "feature_cols": [],
            "use_tfidf": True,
            "tfidf_sources": ["assistant", "file_path"],
        },
        {
            "name": "baseline_plus_assistant",
            "feature_cols": baseline_features,
            "use_tfidf": True,
            "tfidf_sources": ["assistant"],
        },
        {
            "name": "baseline_plus_assistant_file_path",
            "feature_cols": baseline_features,
            "use_tfidf": True,
            "tfidf_sources": ["assistant", "file_path"],
        },
    ]

    for group_name, drop_cols in BASELINE_GROUPS.items():
        remaining = [c for c in baseline_features if c not in drop_cols]
        experiments.append({
            "name": f"dropgroup_{group_name}",
            "feature_cols": remaining,
            "use_tfidf": True,
            "tfidf_sources": ["assistant"],
        })

    for exp in experiments:
        print(f"\n===== Running: {exp['name']} =====")
        print("feature_cols:", exp["feature_cols"])
        print("use_tfidf:", exp["use_tfidf"])
        print("tfidf_sources:", exp["tfidf_sources"])

        results_df, summary_df, threshold_df, fixed_threshold_summary_df = run_one_experiment(
            df=df,
            exp_name=exp["name"],
            feature_cols=exp["feature_cols"],
            use_tfidf=exp["use_tfidf"],
            tfidf_sources=exp["tfidf_sources"],
        )

        all_results.append(results_df)
        all_summaries.append(summary_df)
        all_thresholds.append(threshold_df)
        all_fixed_threshold_summaries.append(fixed_threshold_summary_df)

    final_results = pd.concat(all_results, ignore_index=True)
    final_summaries = pd.concat(all_summaries, ignore_index=True)
    final_thresholds = pd.concat(all_thresholds, ignore_index=True)
    final_fixed_threshold_summaries = pd.concat(all_fixed_threshold_summaries, ignore_index=True)

    final_results.to_csv(results_dir / "ablation_suite_results.csv", index=False)
    final_summaries.to_csv(results_dir / "ablation_suite_summary.csv", index=False)
    final_thresholds.to_csv(results_dir / "ablation_suite_thresholds.csv", index=False)
    final_fixed_threshold_summaries.to_csv(
        results_dir / "ablation_suite_fixed_threshold_summary.csv",
        index=False
    )

    print("\nSaved:", results_dir / "ablation_suite_results.csv")
    print("Saved:", results_dir / "ablation_suite_summary.csv")
    print("Saved:", results_dir / "ablation_suite_thresholds.csv")
    print("Saved:", results_dir / "ablation_suite_fixed_threshold_summary.csv")

    display_cols = [
        "experiment",
        "chosen_threshold",
        "success_precision",
        "success_recall",
        "success_f1",
        "failure_precision",
        "failure_recall",
        "failure_f1",
        "accuracy",
        "success_auc",
    ]
    print("\n===== Summary =====")
    print(final_summaries[display_cols].sort_values("failure_f1", ascending=False).to_string(index=False))

    fixed_display_cols = [
        "experiment",
        "threshold",
        "failure_precision",
        "failure_recall",
        "failure_f1",
        "accuracy",
        "success_auc",
    ]
    print("\n===== Fixed-threshold Summary =====")
    print(
        final_fixed_threshold_summaries[fixed_display_cols]
        .sort_values(["threshold", "failure_f1"], ascending=[True, False])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()