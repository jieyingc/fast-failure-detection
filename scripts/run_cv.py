from pathlib import Path
import pandas as pd

from src.config import (FEATURE_COLS, LOG_FEATURES, USE_NON_EMPTY_ONLY, DATASET_SPLIT, MODEL_NAME, THRESHOLDS,
                        USE_MESSAGE_TFIDF, CV_TYPE,)
from src.pipeline import run_cv


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv(results_dir / f"{DATASET_SPLIT}_features_full.csv")

    proxy_path = results_dir / f"{DATASET_SPLIT}_proxy_metadata.csv"
    if proxy_path.exists():
        proxy_df = pd.read_csv(proxy_path)

        join_cols = ["instance_id"]
        if all(col in proxy_df.columns for col in ["instance_id", "traj_id", "repo_id"]) and \
                all(col in df.columns for col in ["instance_id", "traj_id", "repo_id"]):
            join_cols = ["instance_id", "traj_id", "repo_id"]
        elif "traj_id" in proxy_df.columns and "traj_id" in df.columns:
            join_cols = ["instance_id", "traj_id"]

        df = df.merge(proxy_df, on=join_cols, how="left")
        print(f"Merged proxy metadata from: {proxy_path}")
        print("After merge:", df.shape)
    else:
        print(f"Proxy metadata not found: {proxy_path}")

    if USE_NON_EMPTY_ONLY:
        df = df[df["patch_is_empty"] == 0].copy()

    results_df, summary_df, threshold_selection_df, threshold_summary_df = run_cv(
        df=df,
        feature_cols=FEATURE_COLS,
        log_features=LOG_FEATURES,
        model_name=MODEL_NAME,
        use_message_tfidf=USE_MESSAGE_TFIDF,
    )

    cv_tag = "gkfold" if CV_TYPE == "group_kfold" else "loro"
    tfidf_tag = "msgtfidf" if USE_MESSAGE_TFIDF else "notfidf"
    threshold_tag = f"{min(THRESHOLDS)}_{max(THRESHOLDS)}"

    results_path = results_dir / f"{cv_tag}_{MODEL_NAME}_{DATASET_SPLIT}_{tfidf_tag}_thrscan_{threshold_tag}_results.csv"
    summary_path = results_dir / f"{cv_tag}_{MODEL_NAME}_{DATASET_SPLIT}_{tfidf_tag}_thrscan_{threshold_tag}_summary.csv"
    thresholds_path = results_dir / f"{cv_tag}_{MODEL_NAME}_{DATASET_SPLIT}_{tfidf_tag}_threshold_selection.csv"
    fixed_thresholds_path = results_dir / f"{cv_tag}_{MODEL_NAME}_{DATASET_SPLIT}_{tfidf_tag}_fixed_threshold_summary.csv"


    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    threshold_selection_df.to_csv(thresholds_path, index=False)
    threshold_summary_df.to_csv(fixed_thresholds_path, index=False)

    print("Saved:", results_path)
    print("Saved:", summary_path)
    print("Saved:", thresholds_path)
    print("Saved:", fixed_thresholds_path)
    print("\nSummary by threshold:")
    print("Model:", MODEL_NAME)
    print("CV_TYPE:", CV_TYPE)
    print("Thresholds:", THRESHOLDS)
    print("Use message TF-IDF:", USE_MESSAGE_TFIDF)
    display_cols = [
        "chosen_threshold",
        "success_precision",
        "success_recall",
        "success_f1",
        "failure_precision",
        "failure_recall",
        "failure_f1",
        "accuracy",
        "success_auc",
        "skipped_test_count",
        "skipped_repo_avg_test_count",
        "discarded_success_step",
        "num_correctly_skipped_failures",
        "num_discarded_successes",
    ]
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
