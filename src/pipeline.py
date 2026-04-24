import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .models import get_model
from .evaluate import compute_metrics, compute_dual_metrics, compute_proxy_cost_metrics

from .splits import group_kfold_repo_splits, leave_one_repo_out_splits
from .config import (LABEL_COL, CV_TYPE, N_SPLITS, TFIDF_TEXT_SOURCES, TFIDF_SHOW_TOP_WORDS, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE, TFIDF_STOP_WORDS, TFIDF_SUBLINEAR_TF, THRESHOLDS, MIN_SUCCESS_RECALL)


def prepare_features(train_df, test_df, feature_cols, log_features, use_message_tfidf=False):
    train_df = train_df.copy()
    test_df = test_df.copy()

    feature_cols = feature_cols or []

    for col in log_features:
        if col in train_df.columns and col in test_df.columns:
            train_df[col] = np.log1p(train_df[col].astype(float))
            test_df[col] = np.log1p(test_df[col].astype(float))

    y_train = train_df[LABEL_COL]
    y_test = test_df[LABEL_COL]

    matrices_train = []
    matrices_test = []

    # structured features
    if len(feature_cols) > 0:
        X_train_struct = train_df[feature_cols].astype(float)
        X_test_struct = test_df[feature_cols].astype(float)

        scaler = StandardScaler()
        X_train_struct_scaled = scaler.fit_transform(X_train_struct)
        X_test_struct_scaled = scaler.transform(X_test_struct)

        matrices_train.append(csr_matrix(X_train_struct_scaled))
        matrices_test.append(csr_matrix(X_test_struct_scaled))

    # tf-idf features
    if use_message_tfidf:
        text_source_to_col = {
            "message": "message_text_clean",
            "user": "user_text_clean",
            "assistant": "assistant_text_clean",
            "tool": "tool_text_clean",
            "tool_observation": "tool_observation_text_clean",
            "file_path": "file_path_text_clean",
            "patch": "patch_text_clean",
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

            if TFIDF_SHOW_TOP_WORDS and TFIDF_SHOW_TOP_WORDS > 0:
                feature_names = tfidf.get_feature_names_out()
                mean_tfidf = X_train_text.mean(axis=0).A1
                top_idx = mean_tfidf.argsort()[::-1][:TFIDF_SHOW_TOP_WORDS]

                print(f"\n[TF-IDF:{src}] top {len(top_idx)} features by mean TF-IDF:")
                for idx in top_idx:
                    print(feature_names[idx], round(mean_tfidf[idx], 6))

            matrices_train.append(X_train_text)
            matrices_test.append(X_test_text)

    if len(matrices_train) == 0:
        raise ValueError("No features selected: both feature_cols and TF-IDF are empty.")

    if len(matrices_train) == 1:
        X_train_final = matrices_train[0]
        X_test_final = matrices_test[0]
    else:
        X_train_final = hstack(matrices_train)
        X_test_final = hstack(matrices_test)

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
    min_success_recall=MIN_SUCCESS_RECALL,
):
    rows = []
    threshold_selection_rows = []
    per_fold_threshold_rows = []

    if CV_TYPE == "group_kfold":
        split_iter = group_kfold_repo_splits(df, repo_col="repo_id", n_splits=N_SPLITS,)
        num_folds = N_SPLITS
    elif CV_TYPE == "leave_one_repo_out":
        split_iter = leave_one_repo_out_splits(df, repo_col="repo_id")
        num_folds = df["repo_id"].nunique()
    else:
        raise ValueError(f"Unknown CV_TYPE: {CV_TYPE}")

    for fold_idx, (fold_id, train_df, test_df) in enumerate(split_iter, start=1):
        print(f"[Fold {fold_idx}/{num_folds}] fold_id={fold_id} train={len(train_df)} test={len(test_df)}")

        # 1) inner split for threshold tuning
        inner_train_df, inner_val_df = make_inner_train_val_split(train_df, repo_col="repo_id")

        X_inner_train, y_inner_train, X_inner_val, y_inner_val = prepare_features(
            train_df=inner_train_df,
            test_df=inner_val_df,
            feature_cols=feature_cols,
            log_features=log_features,
            use_message_tfidf=use_message_tfidf,
        )

        inner_model = get_model(model_name)
        inner_model.fit(X_inner_train, y_inner_train)

        success_prob_val = inner_model.predict_proba(X_inner_val)[:, 1]

        chosen_threshold, threshold_df = select_threshold_on_validation(
            y_val_success=y_inner_val.values if hasattr(y_inner_val, "values") else y_inner_val,
            success_prob_val=success_prob_val,
            thresholds=THRESHOLDS,
            min_success_recall=min_success_recall,
        )

        print(f"  chosen_threshold={chosen_threshold:.3f}")

        threshold_df["fold_id"] = fold_id
        threshold_selection_rows.append(threshold_df)

        # 2) retrain on full outer-train
        X_train, y_train, X_test, y_test = prepare_features(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            log_features=log_features,
            use_message_tfidf=use_message_tfidf,
        )

        model = get_model(model_name)
        model.fit(X_train, y_train)

        success_prob_test = model.predict_proba(X_test)[:, 1]
        y_pred_success = (success_prob_test >= chosen_threshold).astype(int)

        metrics = compute_dual_metrics(
            y_true_success=y_test.values if hasattr(y_test, "values") else y_test,
            y_pred_success=y_pred_success,
            y_prob_success=success_prob_test,
        )

        proxy_metrics = compute_proxy_cost_metrics(
            y_true_success=y_test.values if hasattr(y_test, "values") else y_test,
            y_pred_success=y_pred_success,
            gold_test_count=test_df["gold_test_count"].values if "gold_test_count" in test_df.columns else None,
            repo_avg_gold_test_count=test_df[
                "repo_avg_gold_test_count"].values if "repo_avg_gold_test_count" in test_df.columns else None,
            trajectory_step=test_df["trajectory_step"].values if "trajectory_step" in test_df.columns else None,
        )

        for threshold in THRESHOLDS:
            y_pred_success_fixed = (success_prob_test >= threshold).astype(int)

            fixed_metrics = compute_dual_metrics(
                y_true_success=y_test.values if hasattr(y_test, "values") else y_test,
                y_pred_success=y_pred_success_fixed,
                y_prob_success=success_prob_test,
            )

            fixed_proxy_metrics = compute_proxy_cost_metrics(
                y_true_success=y_test.values if hasattr(y_test, "values") else y_test,
                y_pred_success=y_pred_success_fixed,
                gold_test_count=test_df["gold_test_count"].values if "gold_test_count" in test_df.columns else None,
                repo_avg_gold_test_count=test_df[
                    "repo_avg_gold_test_count"].values if "repo_avg_gold_test_count" in test_df.columns else None,
                trajectory_step=test_df["trajectory_step"].values if "trajectory_step" in test_df.columns else None,
            )

            per_fold_threshold_rows.append({
                "fold_id": fold_id,
                "threshold": threshold,

                "success_precision": fixed_metrics["success_precision"],
                "success_recall": fixed_metrics["success_recall"],
                "success_f1": fixed_metrics["success_f1"],

                "failure_precision": fixed_metrics["failure_precision"],
                "failure_recall": fixed_metrics["failure_recall"],
                "failure_f1": fixed_metrics["failure_f1"],

                "accuracy": fixed_metrics["accuracy"],
                "success_auc": fixed_metrics["success_auc"],

                "skipped_test_count": fixed_proxy_metrics.get("skipped_test_count"),
                "skipped_repo_avg_test_count": fixed_proxy_metrics.get("skipped_repo_avg_test_count"),
                "discarded_success_step": fixed_proxy_metrics.get("discarded_success_step"),
                "num_correctly_skipped_failures": fixed_proxy_metrics.get("num_correctly_skipped_failures"),
                "num_discarded_successes": fixed_proxy_metrics.get("num_discarded_successes"),
            })

        rows.append({
            "fold_id": fold_id,
            "chosen_threshold": chosen_threshold,
            "train_size": len(train_df),
            "test_size": len(test_df),

            "success_precision": metrics["success_precision"],
            "success_recall": metrics["success_recall"],
            "success_f1": metrics["success_f1"],

            "failure_precision": metrics["failure_precision"],
            "failure_recall": metrics["failure_recall"],
            "failure_f1": metrics["failure_f1"],

            "accuracy": metrics["accuracy"],
            "success_auc": metrics["success_auc"],

            "skipped_test_count": proxy_metrics.get("skipped_test_count"),
            "skipped_repo_avg_test_count": proxy_metrics.get("skipped_repo_avg_test_count"),
            "discarded_success_step": proxy_metrics.get("discarded_success_step"),
            "num_correctly_skipped_failures": proxy_metrics.get("num_correctly_skipped_failures"),
            "num_discarded_successes": proxy_metrics.get("num_discarded_successes"),
        })

    results_df = pd.DataFrame(rows)
    threshold_selection_df = pd.concat(threshold_selection_rows, ignore_index=True)

    summary = results_df[
        [
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
    ].mean(numeric_only=True)

    summary_df = pd.DataFrame([summary])

    threshold_summary_df = (
        pd.DataFrame(per_fold_threshold_rows)
        .groupby("threshold", as_index=False)
        .mean(numeric_only=True)
    )

    return results_df, summary_df, threshold_selection_df, threshold_summary_df


def make_inner_train_val_split(train_df, repo_col="repo_id"):
    inner_iter = group_kfold_repo_splits(
        train_df,
        repo_col=repo_col,
        n_splits=min(5, train_df[repo_col].nunique())
    )
    _, inner_train_df, inner_val_df = next(inner_iter)
    return inner_train_df, inner_val_df


def select_threshold_on_validation(
    y_val_success,
    success_prob_val,
    thresholds,
    min_success_recall=0.9,
):
    rows = []

    for threshold in thresholds:
        y_pred_success = (success_prob_val >= threshold).astype(int)

        metrics = compute_dual_metrics(
            y_true_success=y_val_success,
            y_pred_success=y_pred_success,
            y_prob_success=success_prob_val,
        )

        rows.append({
            "threshold": threshold,

            "success_precision": metrics["success_precision"],
            "success_recall": metrics["success_recall"],
            "success_f1": metrics["success_f1"],

            "failure_precision": metrics["failure_precision"],
            "failure_recall": metrics["failure_recall"],
            "failure_f1": metrics["failure_f1"],

            "accuracy": metrics["accuracy"],
            "success_auc": metrics["success_auc"],
        })

    threshold_df = pd.DataFrame(rows)

    valid_df = threshold_df[threshold_df["success_recall"] >= min_success_recall].copy()

    if len(valid_df) > 0:
        best_row = valid_df.sort_values(
            by=["failure_precision", "success_recall"],
            ascending=[False, False],
        ).iloc[0]
    else:
        best_row = threshold_df.sort_values(
            by=["success_recall", "failure_precision"],
            ascending=[False, False],
        ).iloc[0]

    return float(best_row["threshold"]), threshold_df
