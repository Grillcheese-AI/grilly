from pathlib import Path
from typing import List
import numpy as np
import pickle
from grilly.utils.data import Dataset  # pyright: ignore[reportMissingImports]
from grilly.utils.data import DataLoader  # pyright: ignore[reportMissingImports]
from grilly.utils.data import random_split
from grilly.utils.data import Subset  # pyright: ignore[reportMissingImports]

def load_dataset(file_path: Path, shuffle: bool = True, batch_size: int = 128) -> DataLoader:
    dataset = Dataset(file_path)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return loader

def save_dataset(dataset: Dataset, file_path: Path):
    with Path(file_path).open('wb') as f:
        pickle.dump(dataset, f)

def split_dataset(dataset: Dataset, split_ratio: List[float]) -> List[Subset]:
    return random_split(dataset, split_ratio)