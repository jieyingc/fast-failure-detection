from pathlib import Path

from src.config import DATASET_SPLIT
from src.data_loading import load_hf_split
from src.features import build_feature_df


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    dataset_split = load_hf_split(DATASET_SPLIT)
    df = build_feature_df(dataset_split)

    out_path = out_dir / f"{DATASET_SPLIT}_features_full.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df.shape)


if __name__ == "__main__":
    main()
