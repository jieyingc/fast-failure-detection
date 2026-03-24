from sklearn.model_selection import GroupShuffleSplit


def leave_one_repo_out_splits(df, repo_col="repo_id"):
    """
    Yields folds:
    - test: one held-out repo
    - train: all remaining repos
    """
    unique_repos = sorted(df[repo_col].unique())

    for held_out_repo in unique_repos:
        test_df = df[df[repo_col] == held_out_repo].copy().reset_index(drop=True)
        train_df = df[df[repo_col] != held_out_repo].copy().reset_index(drop=True)

        yield held_out_repo, train_df, test_df
