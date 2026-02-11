"""
Data Loading Utilities for Grilly

Provides Dataset and DataLoader classes similar to PyTorch's torch.utils.data module.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

# ============================================================================
# Dataset Base Classes
# ============================================================================


class Dataset(ABC):
    """
    Abstract base class for datasets.

    All datasets should subclass this class and implement __len__ and __getitem__.

    Example:
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data, labels):
        ...         self.data = data
        ...         self.labels = labels
        ...
        ...     def __len__(self):
        ...         return len(self.data)
        ...
        ...     def __getitem__(self, idx):
        ...         return self.data[idx], self.labels[idx]
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Get a sample by index."""
        raise NotImplementedError


class TensorDataset(Dataset):
    """
    Dataset wrapping numpy arrays.

    Each sample is retrieved by indexing tensors along the first dimension.

    Args:
        *tensors: Numpy arrays with the same first dimension size.

    Example:
        >>> X = np.random.randn(100, 784).astype(np.float32)
        >>> y = np.random.randint(0, 10, 100)
        >>> dataset = TensorDataset(X, y)
        >>> x_sample, y_sample = dataset[0]
    """

    def __init__(self, *tensors: np.ndarray):
        """Initialize the instance."""

        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors), (
            "All tensors must have the same first dimension"
        )
        self.tensors = tensors

    def __len__(self) -> int:
        """Execute len."""

        return self.tensors[0].shape[0]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, ...]:
        """Execute getitem."""

        return tuple(t[idx] for t in self.tensors)


class ArrayDataset(Dataset):
    """
    Dataset from numpy arrays with optional transforms.

    Args:
        data: Input data array
        labels: Labels array (optional)
        transform: Transform function for data
        target_transform: Transform function for labels
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """Initialize the instance."""

        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Execute len."""

        return len(self.data)

    def __getitem__(self, idx: int) -> np.ndarray | tuple[np.ndarray, Any]:
        """Execute getitem."""

        x = self.data[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.labels is None:
            return x

        y = self.labels[idx]
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset: The original dataset
        indices: List of indices to include in subset
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        """Initialize the instance."""

        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        """Execute len."""

        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        """Execute getitem."""

        return self.dataset[self.indices[idx]]


class ConcatDataset(Dataset):
    """
    Concatenation of multiple datasets.

    Args:
        datasets: List of datasets to concatenate
    """

    def __init__(self, datasets: list[Dataset]):
        """Initialize the instance."""

        self.datasets = datasets
        self.cumulative_sizes = []
        cumsum = 0
        for d in datasets:
            cumsum += len(d)
            self.cumulative_sizes.append(cumsum)

    def __len__(self) -> int:
        """Execute len."""

        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Any:
        """Execute getitem."""

        if idx < 0:
            idx = len(self) + idx
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                if i == 0:
                    return self.datasets[i][idx]
                return self.datasets[i][idx - self.cumulative_sizes[i - 1]]
        raise IndexError(f"Index {idx} out of range")


# ============================================================================
# Samplers
# ============================================================================


class RandomSampler:
    """
    Samples elements randomly.

    Args:
        data_source: Dataset to sample from
        replacement: If True, samples with replacement
        num_samples: Number of samples to draw (only with replacement)
    """

    def __init__(
        self, data_source: Dataset, replacement: bool = False, num_samples: int | None = None
    ):
        """Initialize the instance."""

        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not replacement and num_samples is not None:
            raise ValueError("num_samples can only be specified with replacement=True")

    @property
    def num_samples(self) -> int:
        """Execute num samples."""

        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        """Execute iter."""

        n = len(self.data_source)
        if self.replacement:
            for _ in range(self.num_samples):
                yield np.random.randint(0, n)
        else:
            yield from np.random.permutation(n).tolist()

    def __len__(self) -> int:
        """Execute len."""

        return self.num_samples


class SequentialSampler:
    """
    Samples elements sequentially.

    Args:
        data_source: Dataset to sample from
    """

    def __init__(self, data_source: Dataset):
        """Initialize the instance."""

        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        """Execute iter."""

        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        """Execute len."""

        return len(self.data_source)


class BatchSampler:
    """
    Wraps another sampler to yield batches of indices.

    Args:
        sampler: Base sampler or dataset_size (int) for legacy compatibility
        batch_size: Size of mini-batch
        drop_last: If True, drop the last incomplete batch
        shuffle: Only used with legacy int sampler
    """

    def __init__(
        self,
        sampler,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,  # Legacy parameter
    ):
        # Legacy compatibility: if sampler is an int, create a sampler
        """Initialize the instance."""

        if isinstance(sampler, int):
            self.dataset_size = sampler
            if shuffle:
                from grilly.utils.data import RandomSampler as RS

                class DummyDataset:
                    """Minimal dataset shim for sampler compatibility."""

                    def __len__(self):
                        """Return the synthetic dataset length."""
                        return sampler

                self.sampler = RS(DummyDataset())
            else:

                class DummyDataset:
                    """Minimal dataset shim for sequential sampling."""

                    def __len__(self):
                        """Return the synthetic dataset length."""
                        return sampler

                self.sampler = SequentialSampler(DummyDataset())
        else:
            self.sampler = sampler
            self.dataset_size = len(sampler)

        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        """Execute iter."""

        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Execute len."""

        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)


# ============================================================================
# Collate Functions
# ============================================================================


def default_collate(batch: list[tuple]) -> tuple[np.ndarray, ...]:
    """
    Default collate function that stacks samples into batches.

    Args:
        batch: List of samples from dataset

    Returns:
        Tuple of stacked numpy arrays
    """
    # Handle case where batch items are tuples
    if isinstance(batch[0], tuple):
        return tuple(np.stack([item[i] for item in batch]) for i in range(len(batch[0])))
    # Handle single array case
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    # Handle list of arrays (legacy)
    return batch


# ============================================================================
# DataLoader
# ============================================================================


class DataLoader:
    """
    Data loader that wraps a Dataset and provides batching, shuffling, and iteration.

    Args:
        dataset: Dataset to load from (or list for legacy compatibility)
        batch_size: Number of samples per batch
        shuffle: If True, reshuffle data at every epoch
        sampler: Custom sampler (mutually exclusive with shuffle)
        batch_sampler: Custom batch sampler
        num_workers: Not used (included for PyTorch compatibility)
        collate_fn: Function to merge samples into a batch
        drop_last: If True, drop the last incomplete batch

    Example:
        >>> dataset = TensorDataset(X, y)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for X_batch, y_batch in dataloader:
        ...     # Training step
        ...     pass
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler=None,
        batch_sampler=None,
        num_workers: int = 0,  # Ignored, included for compatibility
        collate_fn: Callable | None = None,
        drop_last: bool = False,
    ):
        """Initialize the instance."""

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate

        # Legacy support: if dataset is a list, wrap it
        if isinstance(dataset, list):
            # Use legacy behavior
            self._legacy_mode = True
            self.indices = list(range(len(dataset)))
        else:
            self._legacy_mode = False

            # Set up sampler
            if batch_sampler is not None:
                self.batch_sampler = batch_sampler
            else:
                if sampler is None:
                    if shuffle:
                        sampler = RandomSampler(dataset)
                    else:
                        sampler = SequentialSampler(dataset)
                self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self) -> Iterator:
        """Execute iter."""

        if self._legacy_mode:
            # Legacy behavior for list datasets
            indices = self.indices.copy()
            if self.shuffle:
                np.random.shuffle(indices)

            for i in range(0, len(self.dataset), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    break
                yield [self.dataset[idx] for idx in batch_indices]
        else:
            # New behavior with Dataset objects
            for batch_indices in self.batch_sampler:
                batch = [self.dataset[i] for i in batch_indices]
                yield self.collate_fn(batch)

    def __len__(self) -> int:
        """Execute len."""

        if self._legacy_mode:
            if self.drop_last:
                return len(self.dataset) // self.batch_size
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        return len(self.batch_sampler)


# ============================================================================
# Utility Functions
# ============================================================================


def random_split(
    dataset: Dataset, lengths: Sequence[int], generator: np.random.Generator | None = None
) -> list[Subset]:
    """
    Randomly split a dataset into non-overlapping subsets.

    Args:
        dataset: Dataset to split
        lengths: Lengths of splits to be produced
        generator: Random number generator (optional)

    Returns:
        List of Subset objects

    Example:
        >>> train_set, val_set = random_split(dataset, [80, 20])
    """
    if sum(lengths) != len(dataset):
        raise ValueError(f"Sum of lengths {sum(lengths)} != dataset length {len(dataset)}")

    if generator is None:
        indices = np.random.permutation(len(dataset)).tolist()
    else:
        indices = generator.permutation(len(dataset)).tolist()

    subsets = []
    offset = 0
    for length in lengths:
        subsets.append(Subset(dataset, indices[offset : offset + length]))
        offset += length

    return subsets


# ============================================================================
# Transforms
# ============================================================================


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: list[Callable]):
        """Initialize the instance."""

        self.transforms = transforms

    def __call__(self, x):
        """Invoke the callable instance."""

        for t in self.transforms:
            x = t(x)
        return x


class ToFloat32:
    """Convert to float32 and optionally scale."""

    def __init__(self, scale: float = 1.0):
        """Initialize the instance."""

        self.scale = scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Invoke the callable instance."""

        return x.astype(np.float32) * self.scale


class Normalize:
    """Normalize with mean and std."""

    def __init__(self, mean: float | np.ndarray, std: float | np.ndarray):
        """Initialize the instance."""

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Invoke the callable instance."""

        return (x - self.mean) / (self.std + 1e-8)


class Flatten:
    """Flatten array to 1D (keeping batch dimension if present)."""

    def __init__(self, start_dim: int = 0):
        """Initialize the instance."""

        self.start_dim = start_dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Invoke the callable instance."""

        if self.start_dim == 0:
            return x.flatten()
        # Keep first start_dim dimensions, flatten the rest
        return x.reshape(x.shape[: self.start_dim] + (-1,))


class RandomNoise:
    """Add random Gaussian noise."""

    def __init__(self, std: float = 0.1):
        """Initialize the instance."""

        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Invoke the callable instance."""

        return x + np.random.randn(*x.shape).astype(np.float32) * self.std


class RandomFlip:
    """Random horizontal flip for images."""

    def __init__(self, p: float = 0.5):
        """Initialize the instance."""

        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Invoke the callable instance."""

        if np.random.random() < self.p:
            return np.flip(x, axis=-1).copy()
        return x


class OneHot:
    """Convert integer labels to one-hot encoding."""

    def __init__(self, num_classes: int):
        """Initialize the instance."""

        self.num_classes = num_classes

    def __call__(self, y: int) -> np.ndarray:
        """Invoke the callable instance."""

        one_hot = np.zeros(self.num_classes, dtype=np.float32)
        one_hot[y] = 1.0
        return one_hot


class Lambda:
    """Apply a custom lambda function."""

    def __init__(self, fn: Callable):
        """Initialize the instance."""

        self.fn = fn

    def __call__(self, x):
        """Invoke the callable instance."""

        return self.fn(x)
