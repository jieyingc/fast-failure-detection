from pathlib import Path
from src.config import DATASET_SPLIT
from src.proxy_metadata import build_proxy_metadata


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    df = build_proxy_metadata(DATASET_SPLIT)
    out_path = out_dir / f"{DATASET_SPLIT}_proxy_metadata.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df.shape)


if __name__ == "__main__":
    main()
