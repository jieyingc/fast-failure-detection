import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .models import get_model
from .evaluate import compute_metrics
from .splits import leave_one_repo_out_splits
from .config import LABEL_COL


def prepare_features(train_df, test_df, feature_cols, log_features, use_message_tfidf=False):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 1) Structured features
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

    # Turn dense arrays into sparse matrices so they can be concatenated with TF-IDF
    X_train_final = csr_matrix(X_train_struct_scaled)
    X_test_final = csr_matrix(X_test_struct_scaled)

    # 2) Optional message-content TF-IDF
    if use_message_tfidf:
        train_text = train_df["message_text_clean"].fillna("")
        test_text = test_df["message_text_clean"].fillna("")

        tfidf = TfidfVectorizer(
            max_features=100,
            stop_words="english",
            lowercase=True,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 1),
        )

        X_train_text = tfidf.fit_transform(train_text)
        X_test_text = tfidf.transform(test_text)

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


def run_leave_one_repo_out_cv(
    df,
    feature_cols,
    log_features,
    model_name="logreg",
    threshold=0.5,
    use_message_tfidf=False,
):
    rows = []

    for held_out_repo, train_df, test_df in leave_one_repo_out_splits(df):
        out = run_one_fold(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            log_features=log_features,
            model_name=model_name,
            threshold=threshold,
            use_message_tfidf=use_message_tfidf,
        )

        rows.append({
            "held_out_repo": held_out_repo,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "failure_precision": out["test_metrics"]["precision"],
            "failure_recall": out["test_metrics"]["recall"],
            "failure_f1": out["test_metrics"]["f1"],
            "failure_accuracy": out["test_metrics"]["accuracy"],
            "failure_auc": out["test_metrics"]["roc_auc"],
        })

    results_df = pd.DataFrame(rows)

    metric_cols = [
        "failure_precision",
        "failure_recall",
        "failure_f1",
        "failure_accuracy",
        "failure_auc",
    ]
    summary = results_df[metric_cols].mean(numeric_only=True).to_dict()

    return results_df, summary
