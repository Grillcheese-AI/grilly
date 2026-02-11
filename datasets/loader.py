import pickle
from pathlib import Path

from grilly.utils.data import (
    DataLoader,  # pyright: ignore[reportMissingImports]
    Dataset,  # pyright: ignore[reportMissingImports]
    Subset,  # pyright: ignore[reportMissingImports]
    random_split,
)


def load_dataset(file_path: Path, shuffle: bool = True, batch_size: int = 128) -> DataLoader:
    dataset = Dataset(file_path)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return loader


def save_dataset(dataset: Dataset, file_path: Path):
    with Path(file_path).open("wb") as f:
        pickle.dump(dataset, f)


def split_dataset(dataset: Dataset, split_ratio: list[float]) -> list[Subset]:
    return random_split(dataset, split_ratio)
