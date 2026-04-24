import json
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

from .data_loading import extract_repo_id


def is_empty_patch(x: Any) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def to_py_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, float) and np.isnan(x):
        return []
    return []


def count_gold_tests(row: pd.Series) -> int:
    ftp_list = to_py_list(row.get("FAIL_TO_PASS"))
    ptp_list = to_py_list(row.get("PASS_TO_PASS"))
    return len(ftp_list) + len(ptp_list)


def parse_messages_for_step(messages_json: str) -> int:
    """
    Return a simple trajectory step proxy.
    Current choice: last message index among parsed messages.
    You can replace this later with a more specific 'step_idx' definition
    if your notebook uses a stricter notion.
    """
    try:
        msgs = json.loads(messages_json)
    except Exception:
        return 0

    if not isinstance(msgs, list) or len(msgs) == 0:
        return 0

    return len(msgs)


def build_proxy_metadata(split_name: str = "tool") -> pd.DataFrame:
    # 1) load datasets
    traj_df = load_dataset("SWE-bench/SWE-smith-trajectories", split=split_name).to_pandas()
    smith_df = load_dataset("SWE-bench/SWE-smith-py", split="train").to_pandas()

    # 2) keep non-empty patch only to stay aligned with your main pipeline
    traj_df["patch_is_empty"] = traj_df["patch"].apply(is_empty_patch)
    traj_df = traj_df.loc[~traj_df["patch_is_empty"]].copy()

    # 3) basic trajectory-side fields
    traj_df["repo_id"] = traj_df["instance_id"].apply(extract_repo_id)
    traj_df["trajectory_step"] = traj_df["messages"].apply(parse_messages_for_step)

    traj_meta = traj_df[
        ["instance_id", "traj_id", "repo_id", "trajectory_step"]
    ].copy()

    # 4) smith-side gold test count
    smith_meta = smith_df.copy()
    smith_meta["gold_test_count"] = smith_meta.apply(count_gold_tests, axis=1)

    # keep one row per instance_id
    smith_meta = smith_meta[["instance_id", "gold_test_count"]].drop_duplicates()

    # 5) merge
    proxy_df = traj_meta.merge(smith_meta, on="instance_id", how="left",)

    proxy_df["gold_test_count"] = pd.to_numeric(proxy_df["gold_test_count"], errors="coerce").fillna(0.0)

    # 6) repo-level average gold test count
    repo_avg_df = (
        proxy_df.groupby("repo_id", as_index=False)["gold_test_count"]
        .mean()
        .rename(columns={"gold_test_count": "repo_avg_gold_test_count"})
    )

    proxy_df = proxy_df.merge(repo_avg_df, on="repo_id", how="left")

    # 7) final columns
    proxy_df = proxy_df[
        [
            "instance_id",
            "traj_id",
            "repo_id",
            "gold_test_count",
            "repo_avg_gold_test_count",
            "trajectory_step",
        ]
    ].copy()

    return proxy_df
