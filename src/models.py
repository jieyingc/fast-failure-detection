from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def get_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_name: {model_name}")
