from datasets import load_dataset
from .config import DATASET_NAME


def load_hf_split(split_name: str):
    dataset = load_dataset(DATASET_NAME)
    return dataset[split_name]


def extract_repo_id(instance_id: str) -> str:
    return instance_id.split(".", 1)[0]
