from sklearn.model_selection import GroupShuffleSplit, GroupKFold

def leave_one_repo_out_splits(df, repo_col="repo_id", max_folds=None):
    """
    Yields folds:
    - test: one held-out repo
    - train: all remaining repos

    If max_folds is not None, only use the first max_folds repos
    after sorting for a smaller dev run.
    """
    unique_repos = sorted(df[repo_col].unique())

    if max_folds is not None:
        unique_repos = unique_repos[:max_folds]

    for held_out_repo in unique_repos:
        test_df = df[df[repo_col] == held_out_repo].copy().reset_index(drop=True)
        train_df = df[df[repo_col] != held_out_repo].copy().reset_index(drop=True)

        yield held_out_repo, train_df, test_df


def group_kfold_repo_splits(df, repo_col="repo_id", n_splits=10):
    splitter = GroupKFold(n_splits=n_splits)
    groups = df[repo_col]

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df, groups=groups)):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        yield fold_idx, train_df, test_df
