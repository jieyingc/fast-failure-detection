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
