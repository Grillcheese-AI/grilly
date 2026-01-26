"""
Data Loading Utilities
"""
import numpy as np
from typing import Iterator, List, Optional, Tuple


class DataLoader:
    """
    Data loader for batching data.
    
    Similar to torch.utils.data.DataLoader
    """
    
    def __init__(
        self,
        dataset: List[np.ndarray],
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False
    ):
        """
        Initialize DataLoader.
        
        Args:
            dataset: List of data samples
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))
    
    def __iter__(self) -> Iterator[List[np.ndarray]]:
        """Iterate over batches"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                break
            yield [self.dataset[idx] for idx in batch_indices]
    
    def __len__(self) -> int:
        """Number of batches"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class BatchSampler:
    """
    Batch sampler for custom batching strategies.
    """
    
    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool = False):
        """
        Initialize BatchSampler.
        
        Args:
            dataset_size: Size of dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batch indices"""
        indices = list(range(self.dataset_size))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.dataset_size, self.batch_size):
            yield indices[i:i + self.batch_size]
    
    def __len__(self) -> int:
        """Number of batches"""
        return (self.dataset_size + self.batch_size - 1) // self.batch_size
