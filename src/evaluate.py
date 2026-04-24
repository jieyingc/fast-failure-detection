from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np


def compute_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def compute_dual_metrics(y_true_success, y_pred_success, y_prob_success):
    success_precision = precision_score(y_true_success, y_pred_success, zero_division=0)
    success_recall = recall_score(y_true_success, y_pred_success, zero_division=0)
    success_f1 = f1_score(y_true_success, y_pred_success, zero_division=0)
    accuracy = accuracy_score(y_true_success, y_pred_success)

    try:
        success_auc = roc_auc_score(y_true_success, y_prob_success)
    except ValueError:
        success_auc = float("nan")

    y_true_failure = 1 - y_true_success
    y_pred_failure = 1 - y_pred_success

    failure_precision = precision_score(y_true_failure, y_pred_failure, zero_division=0)
    failure_recall = recall_score(y_true_failure, y_pred_failure, zero_division=0)
    failure_f1 = f1_score(y_true_failure, y_pred_failure, zero_division=0)

    return {
        "success_precision": success_precision,
        "success_recall": success_recall,
        "success_f1": success_f1,
        "accuracy": accuracy,
        "success_auc": success_auc,
        "failure_precision": failure_precision,
        "failure_recall": failure_recall,
        "failure_f1": failure_f1,
    }


def compute_proxy_cost_metrics(
    y_true_success,
    y_pred_success,
    gold_test_count=None,
    repo_avg_gold_test_count=None,
    trajectory_step=None,
):
    y_true_success = np.asarray(y_true_success)
    y_pred_success = np.asarray(y_pred_success)

    predicted_failure_mask = (y_pred_success == 0)
    actual_failure_mask = (y_true_success == 0)
    actual_success_mask = (y_true_success == 1)

    correctly_skipped_failure_mask = predicted_failure_mask & actual_failure_mask
    discarded_success_mask = predicted_failure_mask & actual_success_mask

    out = {}

    if gold_test_count is not None:
        gold_test_count = np.asarray(gold_test_count, dtype=float)
        out["skipped_test_count"] = float(gold_test_count[correctly_skipped_failure_mask].sum())

    if repo_avg_gold_test_count is not None:
        repo_avg_gold_test_count = np.asarray(repo_avg_gold_test_count, dtype=float)
        out["skipped_repo_avg_test_count"] = float(
            repo_avg_gold_test_count[correctly_skipped_failure_mask].sum()
        )

    if trajectory_step is not None:
        trajectory_step = np.asarray(trajectory_step, dtype=float)
        out["discarded_success_step"] = float(trajectory_step[discarded_success_mask].sum())

    out["num_correctly_skipped_failures"] = int(correctly_skipped_failure_mask.sum())
    out["num_discarded_successes"] = int(discarded_success_mask.sum())

    return out