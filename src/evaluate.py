from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

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