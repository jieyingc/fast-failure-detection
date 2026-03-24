from pathlib import Path
import pandas as pd

from src.config import (
    FEATURE_COLS,
    LOG_FEATURES,
    USE_NON_EMPTY_ONLY,
    DATASET_SPLIT,
    MODEL_NAME,
    THRESHOLD,
    USE_MESSAGE_TFIDF,
)
from src.pipeline import run_leave_one_repo_out_cv


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv(results_dir / f"{DATASET_SPLIT}_features_full.csv")

    if USE_NON_EMPTY_ONLY:
        df = df[df["patch_is_empty"] == 0].copy()

    tfidf_tag = "msgtfidf" if USE_MESSAGE_TFIDF else "notfidf"

    results_df, summary = run_leave_one_repo_out_cv(
        df=df,
        feature_cols=FEATURE_COLS,
        log_features=LOG_FEATURES,
        model_name=MODEL_NAME,
        threshold=THRESHOLD,
        use_message_tfidf=USE_MESSAGE_TFIDF,
    )

    results_path = results_dir / f"loro_{MODEL_NAME}_{DATASET_SPLIT}_{tfidf_tag}_thr_{THRESHOLD}_results.csv"
    summary_path = results_dir / f"loro_{MODEL_NAME}_{DATASET_SPLIT}_{tfidf_tag}_thr_{THRESHOLD}_summary.csv"

    results_df.to_csv(results_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("Saved:", results_path)
    print("Saved:", summary_path)
    print("\nSummary:")
    print("Model:", MODEL_NAME)
    print("Threshold:", THRESHOLD)
    print("Use message TF-IDF:", USE_MESSAGE_TFIDF)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
