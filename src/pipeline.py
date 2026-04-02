import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .models import get_model
from .evaluate import compute_metrics

from .splits import group_kfold_repo_splits, leave_one_repo_out_splits
from .config import (LABEL_COL, CV_TYPE, N_SPLITS, TFIDF_TEXT_SOURCES, TFIDF_SHOW_TOP_WORDS, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE, TFIDF_STOP_WORDS, TFIDF_SUBLINEAR_TF, THRESHOLDS)


def prepare_features(train_df, test_df, feature_cols, log_features, use_message_tfidf=False):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in log_features:
        if col in train_df.columns and col in test_df.columns:
            train_df[col] = np.log1p(train_df[col].astype(float))
            test_df[col] = np.log1p(test_df[col].astype(float))

    X_train_struct = train_df[feature_cols].astype(float)
    y_train = train_df[LABEL_COL]

    X_test_struct = test_df[feature_cols].astype(float)
    y_test = test_df[LABEL_COL]

    scaler = StandardScaler()
    X_train_struct_scaled = scaler.fit_transform(X_train_struct)
    X_test_struct_scaled = scaler.transform(X_test_struct)

    X_train_final = csr_matrix(X_train_struct_scaled)
    X_test_final = csr_matrix(X_test_struct_scaled)

    if use_message_tfidf:
        text_source_to_col = {
            "message": "message_text_clean",
            "user": "user_text_clean",
            "assistant": "assistant_text_clean",
            "tool": "tool_text_clean",
            "file_path": "file_path_text_clean",
            "patch": "patch_text_clean",
            "tool_observation": "tool_observation_text_clean",
            "function_name": "function_name_text_clean",
            "class_name": "class_name_text_clean",
            "import_name": "import_name_text_clean",
        }

        for src in TFIDF_TEXT_SOURCES:
            if src not in text_source_to_col:
                raise ValueError(f"Unknown TFIDF text source: {src}")

            col = text_source_to_col[src]

            if col not in train_df.columns or col not in test_df.columns:
                raise ValueError(f"Missing text column: {col}")

            train_text = train_df[col].fillna("")
            test_text = test_df[col].fillna("")

            tfidf = TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                stop_words=TFIDF_STOP_WORDS,
                lowercase=True,
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF,
                ngram_range=TFIDF_NGRAM_RANGE,
                sublinear_tf=TFIDF_SUBLINEAR_TF,
            )

            X_train_text = tfidf.fit_transform(train_text)
            X_test_text = tfidf.transform(test_text)

            feature_names = tfidf.get_feature_names_out()
            mean_tfidf = X_train_text.mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[::-1][:TFIDF_SHOW_TOP_WORDS]

            print(f"\n[TF-IDF:{src}] top {len(top_idx)} features by mean TF-IDF:")
            for idx in top_idx:
                print(feature_names[idx], round(mean_tfidf[idx], 6))

            X_train_final = hstack([X_train_final, X_train_text])
            X_test_final = hstack([X_test_final, X_test_text])

    return X_train_final, y_train, X_test_final, y_test


def run_one_fold(
    train_df,
    test_df,
    feature_cols,
    log_features,
    model_name="logreg",
    threshold=0.5,
    use_message_tfidf=False,
):
    X_train, y_train, X_test, y_test = prepare_features(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        log_features=log_features,
        use_message_tfidf=use_message_tfidf,
    )

    model = get_model(model_name)
    model.fit(X_train, y_train)

    # Model is trained on resolved=1 (success)
    success_prob = model.predict_proba(X_test)[:, 1]

    # Failure-focused evaluation
    failure_prob = 1 - success_prob
    y_test_failure = (y_test == 0).astype(int)
    failure_pred = (failure_prob >= threshold).astype(int)

    test_metrics = compute_metrics(y_test_failure, failure_pred, failure_prob)

    return {
        "model_name": model_name,
        "threshold": threshold,
        "test_metrics": test_metrics,
    }


def run_cv(
    df,
    feature_cols,
    log_features,
    model_name="logreg",
    use_message_tfidf=False,
):
    rows = []

    if CV_TYPE == "group_kfold":
        split_iter = group_kfold_repo_splits(
            df,
            repo_col="repo_id",
            n_splits=N_SPLITS,
        )
    elif CV_TYPE == "leave_one_repo_out":
        split_iter = leave_one_repo_out_splits(df, repo_col="repo_id")
    else:
        raise ValueError(f"Unknown CV_TYPE: {CV_TYPE}")

    for fold_id, train_df, test_df in split_iter:
        X_train, y_train, X_test, y_test = prepare_features(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            log_features=log_features,
            use_message_tfidf=use_message_tfidf,
        )

        model = get_model(model_name)
        model.fit(X_train, y_train)

        # model is trained on resolved=1 (success)
        success_prob = model.predict_proba(X_test)[:, 1]

        # evaluate as failure=1
        failure_prob = 1 - success_prob
        y_test_failure = (y_test == 0).astype(int)

        for threshold in THRESHOLDS:
            failure_pred = (failure_prob >= threshold).astype(int)
            test_metrics = compute_metrics(y_test_failure, failure_pred, failure_prob)

            rows.append({
                "fold_id": fold_id,
                "threshold": threshold,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "failure_precision": test_metrics["precision"],
                "failure_recall": test_metrics["recall"],
                "failure_f1": test_metrics["f1"],
                "failure_accuracy": test_metrics["accuracy"],
                "failure_auc": test_metrics["roc_auc"],
            })

    results_df = pd.DataFrame(rows)

    summary = (
        results_df
        .groupby("threshold", as_index=False)[
            [
                "failure_precision",
                "failure_recall",
                "failure_f1",
                "failure_accuracy",
                "failure_auc",
            ]
        ]
        .mean()
    )

    return results_df, summary
