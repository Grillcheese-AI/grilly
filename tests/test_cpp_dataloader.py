"""
Tests for C++ DataLoader with background prefetch.
"""

import numpy as np
import pytest

try:
    from grilly_core import Batch, DataLoader, Tensor

    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ backend not available")


class SimpleDataset:
    """Minimal dataset for testing — returns (data, target) tuples."""

    def __init__(self, size=100, dim=8):
        self.size = size
        self.dim = dim
        np.random.seed(42)
        self.data = np.random.randn(size, dim).astype(np.float32)
        self.targets = np.random.randint(0, 10, size=size).astype(np.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.data[idx], np.array([self.targets[idx]], dtype=np.float32))


class ArrayDataset:
    """Dataset that returns single arrays (no target)."""

    def __init__(self, size=50, dim=4):
        self.size = size
        self.data = np.arange(size * dim, dtype=np.float32).reshape(size, dim)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TestDataLoaderBasic:
    """Basic DataLoader functionality."""

    def test_init(self):
        """DataLoader should initialize from a Python dataset."""
        ds = SimpleDataset(size=20)
        dl = DataLoader(ds, batch_size=4)
        assert dl.dataset_size == 20
        assert dl.num_batches == 5  # 20 / 4

    def test_num_batches_with_remainder(self):
        """Non-divisible size should ceil the batch count."""
        ds = SimpleDataset(size=22)
        dl = DataLoader(ds, batch_size=4)
        assert dl.num_batches == 6  # ceil(22/4) = 6

    def test_num_batches_drop_last(self):
        """drop_last=True should floor the batch count."""
        ds = SimpleDataset(size=22)
        dl = DataLoader(ds, batch_size=4, drop_last=True)
        assert dl.num_batches == 5  # floor(22/4) = 5


class TestDataLoaderIteration:
    """Test iteration protocol."""

    def test_iterate_all_batches(self):
        """Iterating should yield exactly num_batches batches."""
        ds = SimpleDataset(size=20, dim=4)
        dl = DataLoader(ds, batch_size=5)
        dl.iter()

        count = 0
        try:
            while True:
                batch = dl.__next__()
                count += 1
                assert batch.data.valid
        except StopIteration:
            pass

        assert count == 4  # 20 / 5

    def test_batch_data_shape(self):
        """Batch data should have shape [batch_size, dim]."""
        ds = SimpleDataset(size=20, dim=8)
        dl = DataLoader(ds, batch_size=4)
        dl.iter()

        batch = dl.__next__()
        assert batch.data.shape == [4, 8]

    def test_batch_target_shape(self):
        """Batch target should have shape [batch_size, 1]."""
        ds = SimpleDataset(size=20, dim=8)
        dl = DataLoader(ds, batch_size=4)
        dl.iter()

        batch = dl.__next__()
        assert batch.target.shape == [4, 1]

    def test_last_batch_smaller(self):
        """Last batch with remainder should be smaller."""
        ds = SimpleDataset(size=22, dim=4)
        dl = DataLoader(ds, batch_size=5)
        dl.iter()

        batches = []
        try:
            while True:
                batches.append(dl.__next__())
        except StopIteration:
            pass

        # 22 / 5 = 4 full batches + 1 batch of 2
        assert len(batches) == 5
        assert batches[-1].data.shape[0] == 2

    def test_multiple_epochs(self):
        """Calling iter() again should restart iteration."""
        ds = SimpleDataset(size=10, dim=4)
        dl = DataLoader(ds, batch_size=5)

        for epoch in range(3):
            dl.iter()
            count = 0
            try:
                while True:
                    dl.__next__()
                    count += 1
            except StopIteration:
                pass
            assert count == 2

    def test_data_values_correct(self):
        """Batch data should contain actual dataset values."""
        ds = SimpleDataset(size=10, dim=4)
        dl = DataLoader(ds, batch_size=10)  # Single batch = full dataset
        dl.iter()
        batch = dl.__next__()

        batch_data = batch.data.numpy()
        # Each row should come from the dataset
        for i in range(10):
            found = False
            for j in range(10):
                if np.allclose(batch_data[i], ds.data[j]):
                    found = True
                    break
            assert found, f"Row {i} not found in original dataset"


class TestDataLoaderShuffle:
    """Test shuffle functionality."""

    def test_shuffle_changes_order(self):
        """With shuffle=True, order should differ between epochs (usually)."""
        ds = SimpleDataset(size=100, dim=4)
        dl = DataLoader(ds, batch_size=100, shuffle=True)

        # Epoch 1
        dl.iter()
        batch1 = dl.__next__()
        data1 = batch1.data.numpy().copy()

        # Epoch 2
        dl.iter()
        batch2 = dl.__next__()
        data2 = batch2.data.numpy()

        # With 100 items, the probability of identical order is vanishingly small
        assert not np.array_equal(data1, data2), "Shuffle should change order between epochs"

    def test_no_shuffle_deterministic(self):
        """Without shuffle, order should be deterministic."""
        ds = SimpleDataset(size=10, dim=4)
        dl = DataLoader(ds, batch_size=10, shuffle=False)

        dl.iter()
        batch1 = dl.__next__()
        data1 = batch1.data.numpy().copy()

        dl.iter()
        batch2 = dl.__next__()
        data2 = batch2.data.numpy()

        np.testing.assert_array_equal(data1, data2)


class TestDataLoaderSingleArray:
    """Test DataLoader with datasets returning single arrays."""

    def test_single_array_dataset(self):
        """DataLoader should handle datasets that return single arrays."""
        ds = ArrayDataset(size=20, dim=4)
        dl = DataLoader(ds, batch_size=5)
        dl.iter()

        batch = dl.__next__()
        assert batch.data.shape == [5, 4]


class TestDataLoaderPrefetch:
    """Test background prefetch with num_workers > 0."""

    def test_prefetch_single_worker(self):
        """DataLoader with 1 worker should produce correct results."""
        ds = SimpleDataset(size=20, dim=4)
        dl = DataLoader(ds, batch_size=5, num_workers=1)
        dl.iter()

        count = 0
        try:
            while True:
                batch = dl.__next__()
                assert batch.data.valid
                count += 1
        except StopIteration:
            pass

        assert count == 4

    def test_prefetch_multiple_workers(self):
        """DataLoader with 2 workers should produce all batches."""
        ds = SimpleDataset(size=20, dim=4)
        dl = DataLoader(ds, batch_size=5, num_workers=2)
        dl.iter()

        count = 0
        try:
            while True:
                batch = dl.__next__()
                assert batch.data.valid
                count += 1
        except StopIteration:
            pass

        assert count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
