"""Comprehensive CPU-only tests for utils/data.py."""

import numpy as np
import pytest

try:
    from grilly.utils.data import (
        ArrayDataset,
        BatchSampler,
        Compose,
        ConcatDataset,
        DataLoader,
        Flatten,
        Lambda,
        Normalize,
        OneHot,
        RandomFlip,
        RandomNoise,
        RandomSampler,
        SequentialSampler,
        Subset,
        TensorDataset,
        ToFloat32,
        default_collate,
        random_split,
    )
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed numpy RNG for reproducibility in every test."""
    np.random.seed(42)


# ============================================================================
# TensorDataset
# ============================================================================


class TestTensorDataset:
    """Tests for TensorDataset."""

    def test_single_tensor_len(self):
        """Length matches first dimension of the single tensor."""
        data = np.arange(50, dtype=np.float32).reshape(10, 5)
        ds = TensorDataset(data)
        assert len(ds) == 10

    def test_single_tensor_getitem_returns_tuple(self):
        """Indexing a single-tensor dataset returns a 1-tuple."""
        data = np.arange(6, dtype=np.float32).reshape(3, 2)
        ds = TensorDataset(data)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 1
        np.testing.assert_array_equal(item[0], data[0])

    def test_two_tensor_getitem(self):
        """Indexing returns the correct pair of slices."""
        X = np.arange(12, dtype=np.float32).reshape(4, 3)
        y = np.array([0, 1, 2, 3], dtype=np.float32)
        ds = TensorDataset(X, y)
        x_out, y_out = ds[2]
        np.testing.assert_array_equal(x_out, X[2])
        assert y_out == y[2]

    def test_multi_tensor_support(self):
        """Three or more tensors are supported."""
        a = np.ones((5, 2), dtype=np.float32)
        b = np.zeros((5,), dtype=np.float32)
        c = np.full((5, 3), 7.0, dtype=np.float32)
        ds = TensorDataset(a, b, c)
        assert len(ds) == 5
        item = ds[1]
        assert len(item) == 3
        np.testing.assert_array_equal(item[0], a[1])
        np.testing.assert_array_equal(item[2], c[1])

    def test_mismatched_first_dim_raises(self):
        """Tensors with different first-dim lengths raise AssertionError."""
        a = np.zeros((5, 2), dtype=np.float32)
        b = np.zeros((3,), dtype=np.float32)
        with pytest.raises(AssertionError, match="same first dimension"):
            TensorDataset(a, b)

    def test_negative_index(self):
        """Negative indexing works (delegates to numpy)."""
        data = np.arange(20, dtype=np.float32).reshape(5, 4)
        ds = TensorDataset(data)
        np.testing.assert_array_equal(ds[-1][0], data[-1])


# ============================================================================
# ArrayDataset
# ============================================================================


class TestArrayDataset:
    """Tests for ArrayDataset."""

    def test_data_only(self):
        """Dataset with data only returns array, not tuple."""
        data = np.arange(15, dtype=np.float32).reshape(5, 3)
        ds = ArrayDataset(data)
        assert len(ds) == 5
        result = ds[0]
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data[0])

    def test_with_labels(self):
        """Dataset with labels returns (data, label) tuple."""
        data = np.arange(6, dtype=np.float32).reshape(3, 2)
        labels = np.array([10, 20, 30], dtype=np.float32)
        ds = ArrayDataset(data, labels=labels)
        x, y = ds[1]
        np.testing.assert_array_equal(x, data[1])
        assert y == 20.0

    def test_with_transform(self):
        """Transform is applied to data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = ArrayDataset(data, transform=lambda x: x * 10)
        result = ds[0]
        np.testing.assert_array_equal(result, np.array([10.0, 20.0], dtype=np.float32))

    def test_with_target_transform(self):
        """Target transform is applied to labels."""
        data = np.zeros((3, 2), dtype=np.float32)
        labels = np.array([0, 1, 2], dtype=np.float32)
        ds = ArrayDataset(data, labels=labels, target_transform=lambda y: y + 100)
        _, y = ds[2]
        assert y == 102.0

    def test_both_transforms(self):
        """Both data and target transforms are applied together."""
        data = np.ones((4, 2), dtype=np.float32)
        labels = np.array([0, 1, 2, 3], dtype=np.float32)
        ds = ArrayDataset(
            data,
            labels=labels,
            transform=lambda x: x * 2,
            target_transform=lambda y: y * 3,
        )
        x, y = ds[1]
        np.testing.assert_array_equal(x, np.array([2.0, 2.0], dtype=np.float32))
        assert y == 3.0

    def test_no_labels_no_transform(self):
        """Accessing without labels or transforms returns raw data."""
        data = np.eye(3, dtype=np.float32)
        ds = ArrayDataset(data)
        np.testing.assert_array_equal(ds[2], data[2])


# ============================================================================
# Subset
# ============================================================================


class TestSubset:
    """Tests for Subset."""

    def test_len_matches_indices(self):
        """Length equals the number of provided indices."""
        data = np.arange(20, dtype=np.float32).reshape(10, 2)
        full_ds = TensorDataset(data)
        subset = Subset(full_ds, [0, 2, 4, 6])
        assert len(subset) == 4

    def test_getitem_maps_correctly(self):
        """Subset[i] maps to dataset[indices[i]]."""
        data = np.arange(10, dtype=np.float32).reshape(5, 2)
        full_ds = TensorDataset(data)
        indices = [4, 2, 0]
        subset = Subset(full_ds, indices)
        for i, orig_idx in enumerate(indices):
            np.testing.assert_array_equal(subset[i][0], data[orig_idx])

    def test_empty_subset(self):
        """An empty index list gives length 0."""
        data = np.zeros((5, 2), dtype=np.float32)
        full_ds = TensorDataset(data)
        subset = Subset(full_ds, [])
        assert len(subset) == 0

    def test_subset_of_array_dataset(self):
        """Subset works with ArrayDataset."""
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        labels = np.array([10, 20, 30, 40], dtype=np.float32)
        ds = ArrayDataset(data, labels=labels)
        subset = Subset(ds, [1, 3])
        x, y = subset[0]
        np.testing.assert_array_equal(x, data[1])
        assert y == 20.0


# ============================================================================
# ConcatDataset
# ============================================================================


class TestConcatDataset:
    """Tests for ConcatDataset."""

    def test_len_is_sum(self):
        """Total length is the sum of constituent dataset lengths."""
        ds1 = TensorDataset(np.zeros((3, 2), dtype=np.float32))
        ds2 = TensorDataset(np.ones((5, 2), dtype=np.float32))
        cat = ConcatDataset([ds1, ds2])
        assert len(cat) == 8

    def test_getitem_first_dataset(self):
        """Indices in the first dataset range return its items."""
        ds1 = TensorDataset(np.arange(6, dtype=np.float32).reshape(3, 2))
        ds2 = TensorDataset(np.full((4, 2), 99.0, dtype=np.float32))
        cat = ConcatDataset([ds1, ds2])
        np.testing.assert_array_equal(cat[1][0], ds1[1][0])

    def test_getitem_across_boundary(self):
        """Index past the first dataset retrieves from the second."""
        ds1 = TensorDataset(np.zeros((3, 2), dtype=np.float32))
        ds2 = TensorDataset(np.ones((4, 2), dtype=np.float32))
        cat = ConcatDataset([ds1, ds2])
        # Index 3 is the first element of ds2
        np.testing.assert_array_equal(cat[3][0], np.ones(2, dtype=np.float32))

    def test_negative_index(self):
        """Negative indexing wraps around correctly."""
        ds1 = TensorDataset(np.zeros((2, 1), dtype=np.float32))
        ds2 = TensorDataset(np.full((3, 1), 5.0, dtype=np.float32))
        cat = ConcatDataset([ds1, ds2])
        np.testing.assert_array_equal(cat[-1][0], np.array([5.0], dtype=np.float32))

    def test_index_out_of_range_raises(self):
        """Out-of-range index raises IndexError."""
        ds1 = TensorDataset(np.zeros((2, 1), dtype=np.float32))
        cat = ConcatDataset([ds1])
        with pytest.raises(IndexError):
            _ = cat[5]

    def test_three_datasets(self):
        """ConcatDataset with three datasets."""
        ds1 = TensorDataset(np.full((2, 1), 1.0, dtype=np.float32))
        ds2 = TensorDataset(np.full((3, 1), 2.0, dtype=np.float32))
        ds3 = TensorDataset(np.full((4, 1), 3.0, dtype=np.float32))
        cat = ConcatDataset([ds1, ds2, ds3])
        assert len(cat) == 9
        assert cat[0][0].item() == 1.0
        assert cat[2][0].item() == 2.0
        assert cat[5][0].item() == 3.0


# ============================================================================
# RandomSampler
# ============================================================================


class TestRandomSampler:
    """Tests for RandomSampler."""

    def _make_dataset(self, n=20):
        data = np.arange(n, dtype=np.float32).reshape(n, 1)
        return TensorDataset(data)

    def test_no_replacement_returns_all_indices(self):
        """Without replacement every index appears exactly once."""
        ds = self._make_dataset(20)
        sampler = RandomSampler(ds)
        indices = list(sampler)
        assert sorted(indices) == list(range(20))

    def test_no_replacement_is_permuted(self):
        """The order is not sequential (with very high probability)."""
        ds = self._make_dataset(100)
        sampler = RandomSampler(ds)
        indices = list(sampler)
        assert indices != list(range(100))

    def test_len_no_replacement(self):
        """Length equals dataset size when no replacement."""
        ds = self._make_dataset(15)
        sampler = RandomSampler(ds)
        assert len(sampler) == 15

    def test_with_replacement_num_samples(self):
        """With replacement, exactly num_samples indices are yielded."""
        ds = self._make_dataset(10)
        sampler = RandomSampler(ds, replacement=True, num_samples=50)
        indices = list(sampler)
        assert len(indices) == 50
        # All indices should be valid
        assert all(0 <= i < 10 for i in indices)

    def test_len_with_replacement(self):
        """Length equals num_samples when using replacement."""
        ds = self._make_dataset(10)
        sampler = RandomSampler(ds, replacement=True, num_samples=25)
        assert len(sampler) == 25

    def test_num_samples_without_replacement_raises(self):
        """Specifying num_samples without replacement raises ValueError."""
        ds = self._make_dataset(10)
        with pytest.raises(ValueError, match="num_samples"):
            RandomSampler(ds, replacement=False, num_samples=5)


# ============================================================================
# SequentialSampler
# ============================================================================


class TestSequentialSampler:
    """Tests for SequentialSampler."""

    def test_returns_indices_in_order(self):
        """Indices come out 0, 1, 2, ..., n-1."""
        ds = TensorDataset(np.zeros((8, 1), dtype=np.float32))
        sampler = SequentialSampler(ds)
        assert list(sampler) == list(range(8))

    def test_len_matches_dataset(self):
        """Length equals the dataset length."""
        ds = TensorDataset(np.zeros((12, 1), dtype=np.float32))
        sampler = SequentialSampler(ds)
        assert len(sampler) == 12

    def test_multiple_iterations(self):
        """Re-iterating gives the same order."""
        ds = TensorDataset(np.zeros((5, 1), dtype=np.float32))
        sampler = SequentialSampler(ds)
        first = list(sampler)
        second = list(sampler)
        assert first == second == list(range(5))


# ============================================================================
# BatchSampler
# ============================================================================


class TestBatchSampler:
    """Tests for BatchSampler."""

    def _make_sequential_sampler(self, n):
        ds = TensorDataset(np.zeros((n, 1), dtype=np.float32))
        return SequentialSampler(ds)

    def test_batch_sizes_correct(self):
        """All full batches have the requested size."""
        sampler = self._make_sequential_sampler(10)
        bs = BatchSampler(sampler, batch_size=3, drop_last=False)
        batches = list(bs)
        # First 3 batches have size 3
        for b in batches[:-1]:
            assert len(b) == 3
        # Last batch has the remainder
        assert len(batches[-1]) == 1

    def test_drop_last(self):
        """drop_last=True discards the incomplete final batch."""
        sampler = self._make_sequential_sampler(10)
        bs = BatchSampler(sampler, batch_size=3, drop_last=True)
        batches = list(bs)
        assert len(batches) == 3
        for b in batches:
            assert len(b) == 3

    def test_len_no_drop(self):
        """len with drop_last=False uses ceil division."""
        sampler = self._make_sequential_sampler(10)
        bs = BatchSampler(sampler, batch_size=3, drop_last=False)
        assert len(bs) == 4  # ceil(10/3)

    def test_len_with_drop(self):
        """len with drop_last=True uses floor division."""
        sampler = self._make_sequential_sampler(10)
        bs = BatchSampler(sampler, batch_size=3, drop_last=True)
        assert len(bs) == 3  # 10 // 3

    def test_exact_division(self):
        """When dataset size is divisible by batch_size, no remainder."""
        sampler = self._make_sequential_sampler(12)
        bs = BatchSampler(sampler, batch_size=4, drop_last=False)
        batches = list(bs)
        assert len(batches) == 3
        for b in batches:
            assert len(b) == 4

    def test_legacy_int_sampler(self):
        """BatchSampler accepts an int as legacy compatibility."""
        bs = BatchSampler(10, batch_size=3, drop_last=False)
        batches = list(bs)
        total_indices = [idx for batch in batches for idx in batch]
        assert sorted(total_indices) == list(range(10))

    def test_indices_content(self):
        """Sequential batches contain the expected indices."""
        sampler = self._make_sequential_sampler(7)
        bs = BatchSampler(sampler, batch_size=3, drop_last=False)
        batches = list(bs)
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6]


# ============================================================================
# default_collate
# ============================================================================


class TestDefaultCollate:
    """Tests for default_collate."""

    def test_collate_tuple_batch(self):
        """Collating a list of tuples stacks each element."""
        batch = [
            (np.array([1.0, 2.0], dtype=np.float32), np.array([10.0], dtype=np.float32)),
            (np.array([3.0, 4.0], dtype=np.float32), np.array([20.0], dtype=np.float32)),
        ]
        result = default_collate(batch)
        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_array_equal(
            result[0], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            result[1], np.array([[10.0], [20.0]], dtype=np.float32)
        )

    def test_collate_array_batch(self):
        """Collating a list of arrays stacks them."""
        batch = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]
        result = default_collate(batch)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(
            result, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )

    def test_collate_triple_tuple(self):
        """Collating 3-element tuples produces 3 stacked arrays."""
        batch = [
            (
                np.array([1.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
                np.array([3.0], dtype=np.float32),
            ),
            (
                np.array([4.0], dtype=np.float32),
                np.array([5.0], dtype=np.float32),
                np.array([6.0], dtype=np.float32),
            ),
        ]
        result = default_collate(batch)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], np.array([[1.0], [4.0]], dtype=np.float32))

    def test_collate_preserves_dtype(self):
        """Collated output retains float32 dtype."""
        batch = [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]
        result = default_collate(batch)
        assert result.dtype == np.float32


# ============================================================================
# DataLoader
# ============================================================================


class TestDataLoader:
    """Tests for DataLoader with TensorDataset."""

    def _make_dataset(self, n=100):
        X = np.arange(n, dtype=np.float32).reshape(n, 1)
        y = np.arange(n, dtype=np.float32)
        return TensorDataset(X, y)

    def test_basic_iteration_yields_batches(self):
        """Iterating produces the expected number of batches."""
        ds = self._make_dataset(100)
        loader = DataLoader(ds, batch_size=10, shuffle=False)
        batches = list(loader)
        assert len(batches) == 10

    def test_batch_shapes(self):
        """Each batch has the correct shape."""
        ds = self._make_dataset(100)
        loader = DataLoader(ds, batch_size=25, shuffle=False)
        for X_batch, y_batch in loader:
            assert X_batch.shape == (25, 1)
            assert y_batch.shape == (25,)

    def test_sequential_values(self):
        """Without shuffle, first batch contains the first batch_size elements."""
        ds = self._make_dataset(20)
        loader = DataLoader(ds, batch_size=5, shuffle=False)
        first_batch = next(iter(loader))
        X_batch, y_batch = first_batch
        np.testing.assert_array_equal(X_batch.flatten(), np.arange(5, dtype=np.float32))
        np.testing.assert_array_equal(y_batch, np.arange(5, dtype=np.float32))

    def test_shuffle_different_order(self):
        """With shuffle, the data order differs from sequential."""
        ds = self._make_dataset(100)
        loader = DataLoader(ds, batch_size=100, shuffle=True)
        batch = next(iter(loader))
        y_batch = batch[1]
        # It is astronomically unlikely that a shuffle of 100 items is sequential
        assert not np.array_equal(y_batch, np.arange(100, dtype=np.float32))

    def test_shuffle_all_values_present(self):
        """Shuffled loader still contains all original values."""
        ds = self._make_dataset(50)
        loader = DataLoader(ds, batch_size=50, shuffle=True)
        y_batch = next(iter(loader))[1]
        assert set(y_batch.tolist()) == set(range(50))

    def test_drop_last(self):
        """drop_last=True drops incomplete final batch."""
        ds = self._make_dataset(10)
        loader = DataLoader(ds, batch_size=3, shuffle=False, drop_last=True)
        batches = list(loader)
        assert len(batches) == 3
        for X_batch, y_batch in batches:
            assert X_batch.shape[0] == 3

    def test_no_drop_last_includes_remainder(self):
        """drop_last=False keeps the incomplete final batch."""
        ds = self._make_dataset(10)
        loader = DataLoader(ds, batch_size=3, shuffle=False, drop_last=False)
        batches = list(loader)
        assert len(batches) == 4
        # Last batch has 1 element
        assert batches[-1][0].shape[0] == 1

    def test_len(self):
        """__len__ returns the correct batch count."""
        ds = self._make_dataset(100)
        assert len(DataLoader(ds, batch_size=10)) == 10
        assert len(DataLoader(ds, batch_size=30)) == 4  # ceil(100/30)
        assert len(DataLoader(ds, batch_size=30, drop_last=True)) == 3  # 100//30

    def test_batch_size_one(self):
        """batch_size=1 yields one sample per iteration."""
        ds = self._make_dataset(5)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        batches = list(loader)
        assert len(batches) == 5
        assert batches[0][0].shape == (1, 1)

    def test_custom_collate_fn(self):
        """A custom collate_fn is used instead of default_collate."""
        ds = self._make_dataset(6)

        def my_collate(batch):
            # Just return the raw list of tuples
            return batch

        loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=my_collate)
        first_batch = next(iter(loader))
        assert isinstance(first_batch, list)
        assert len(first_batch) == 3


# ============================================================================
# random_split
# ============================================================================


class TestRandomSplit:
    """Tests for random_split."""

    def test_splits_sum_to_original(self):
        """Split lengths sum to original dataset length."""
        ds = TensorDataset(np.zeros((100, 2), dtype=np.float32))
        parts = random_split(ds, [70, 30])
        assert len(parts[0]) + len(parts[1]) == 100

    def test_split_lengths(self):
        """Each split has the requested length."""
        ds = TensorDataset(np.zeros((50, 1), dtype=np.float32))
        parts = random_split(ds, [30, 10, 10])
        assert len(parts[0]) == 30
        assert len(parts[1]) == 10
        assert len(parts[2]) == 10

    def test_no_overlap(self):
        """Splits do not share any indices."""
        ds = TensorDataset(np.arange(20, dtype=np.float32).reshape(20, 1))
        parts = random_split(ds, [12, 8])
        indices_0 = set(parts[0].indices)
        indices_1 = set(parts[1].indices)
        assert indices_0.isdisjoint(indices_1)
        assert indices_0 | indices_1 == set(range(20))

    def test_seed_reproducibility(self):
        """Same seed produces identical splits."""
        ds = TensorDataset(np.arange(30, dtype=np.float32).reshape(30, 1))

        np.random.seed(123)
        parts1 = random_split(ds, [20, 10])
        idx1 = list(parts1[0].indices)

        np.random.seed(123)
        parts2 = random_split(ds, [20, 10])
        idx2 = list(parts2[0].indices)

        assert idx1 == idx2

    def test_generator_reproducibility(self):
        """Using a Generator produces reproducible splits."""
        ds = TensorDataset(np.arange(30, dtype=np.float32).reshape(30, 1))

        gen1 = np.random.default_rng(99)
        parts1 = random_split(ds, [20, 10], generator=gen1)
        idx1 = list(parts1[0].indices)

        gen2 = np.random.default_rng(99)
        parts2 = random_split(ds, [20, 10], generator=gen2)
        idx2 = list(parts2[0].indices)

        assert idx1 == idx2

    def test_lengths_mismatch_raises(self):
        """If sum(lengths) != len(dataset), ValueError is raised."""
        ds = TensorDataset(np.zeros((10, 1), dtype=np.float32))
        with pytest.raises(ValueError, match="Sum of lengths"):
            random_split(ds, [5, 3])

    def test_single_split(self):
        """A single split of full length returns the whole dataset."""
        ds = TensorDataset(np.zeros((10, 1), dtype=np.float32))
        parts = random_split(ds, [10])
        assert len(parts) == 1
        assert len(parts[0]) == 10


# ============================================================================
# Transforms
# ============================================================================


class TestTransforms:
    """Tests for transform classes."""

    # --- Compose ---

    def test_compose_chains(self):
        """Compose applies transforms in order."""
        def add_one(x):
            return x + 1
        def double(x):
            return x * 2
        transform = Compose([add_one, double])
        x = np.array([3.0], dtype=np.float32)
        result = transform(x)
        # (3 + 1) * 2 = 8
        np.testing.assert_array_equal(result, np.array([8.0], dtype=np.float32))

    def test_compose_empty(self):
        """Empty Compose is identity."""
        transform = Compose([])
        x = np.array([5.0], dtype=np.float32)
        np.testing.assert_array_equal(transform(x), x)

    def test_compose_single(self):
        """Compose with one transform works like the transform itself."""
        transform = Compose([lambda x: x + 10])
        x = np.array([1.0], dtype=np.float32)
        np.testing.assert_array_equal(transform(x), np.array([11.0], dtype=np.float32))

    # --- ToFloat32 ---

    def test_tofloat32_converts_dtype(self):
        """ToFloat32 converts int array to float32."""
        x = np.array([1, 2, 3], dtype=np.int32)
        result = ToFloat32()(x)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_tofloat32_with_scale(self):
        """ToFloat32 scales the values."""
        x = np.array([100, 200], dtype=np.uint8)
        result = ToFloat32(scale=1.0 / 255.0)(x)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, np.array([100.0 / 255.0, 200.0 / 255.0]))

    def test_tofloat32_already_float32(self):
        """ToFloat32 on float32 input keeps it float32."""
        x = np.array([1.5, 2.5], dtype=np.float32)
        result = ToFloat32()(x)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, x)

    # --- Normalize ---

    def test_normalize_shifts_mean_std(self):
        """Normalize subtracts mean and divides by std."""
        x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        norm = Normalize(mean=20.0, std=10.0)
        result = norm(x)
        expected = (x - 20.0) / (10.0 + 1e-8)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_normalize_per_channel(self):
        """Normalize with per-channel mean/std arrays."""
        x = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        norm = Normalize(mean=[10.0, 20.0], std=[1.0, 2.0])
        result = norm(x)
        expected_0 = (x[:, 0] - 10.0) / (1.0 + 1e-8)
        expected_1 = (x[:, 1] - 20.0) / (2.0 + 1e-8)
        np.testing.assert_allclose(result[:, 0], expected_0, atol=1e-6)
        np.testing.assert_allclose(result[:, 1], expected_1, atol=1e-6)

    def test_normalize_zero_std_safe(self):
        """Normalize does not produce inf when std is zero (epsilon added)."""
        x = np.array([5.0], dtype=np.float32)
        norm = Normalize(mean=0.0, std=0.0)
        result = norm(x)
        assert np.isfinite(result).all()

    # --- Flatten ---

    def test_flatten_spatial(self):
        """Flatten converts 2D to 1D."""
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = Flatten()(x)
        assert result.shape == (12,)
        np.testing.assert_array_equal(result, np.arange(12, dtype=np.float32))

    def test_flatten_3d(self):
        """Flatten 3D array to 1D."""
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        result = Flatten()(x)
        assert result.shape == (24,)

    def test_flatten_with_start_dim(self):
        """Flatten with start_dim=1 keeps batch dimension."""
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        result = Flatten(start_dim=1)(x)
        assert result.shape == (2, 12)

    # --- RandomNoise ---

    def test_random_noise_adds_noise(self):
        """RandomNoise modifies the input."""
        x = np.zeros((10,), dtype=np.float32)
        noisy = RandomNoise(std=1.0)(x)
        # Very unlikely all zeros after adding N(0,1) noise
        assert not np.allclose(noisy, x)

    def test_random_noise_preserves_shape(self):
        """Output shape equals input shape."""
        x = np.zeros((3, 4), dtype=np.float32)
        result = RandomNoise(std=0.5)(x)
        assert result.shape == x.shape

    def test_random_noise_small_std(self):
        """With small std, output is close to input."""
        x = np.ones((100,), dtype=np.float32) * 50.0
        result = RandomNoise(std=0.001)(x)
        np.testing.assert_allclose(result, x, atol=0.1)

    # --- RandomFlip ---

    def test_random_flip_flips(self):
        """RandomFlip with p=1.0 always flips."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = RandomFlip(p=1.0)(x)
        np.testing.assert_array_equal(result, np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32))

    def test_random_flip_no_flip(self):
        """RandomFlip with p=0.0 never flips."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = RandomFlip(p=0.0)(x)
        np.testing.assert_array_equal(result, x)

    def test_random_flip_2d(self):
        """RandomFlip flips along last axis for 2D input."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = RandomFlip(p=1.0)(x)
        expected = np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    # --- OneHot ---

    def test_onehot_creates_onehot(self):
        """OneHot creates correct one-hot vector."""
        oh = OneHot(num_classes=5)
        result = oh(2)
        expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_onehot_dtype(self):
        """OneHot output is float32."""
        result = OneHot(num_classes=3)(0)
        assert result.dtype == np.float32

    def test_onehot_length(self):
        """OneHot output length equals num_classes."""
        result = OneHot(num_classes=10)(7)
        assert len(result) == 10

    def test_onehot_first_class(self):
        """OneHot for class 0."""
        result = OneHot(num_classes=4)(0)
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_onehot_last_class(self):
        """OneHot for last class."""
        result = OneHot(num_classes=4)(3)
        expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    # --- Lambda ---

    def test_lambda_applies_function(self):
        """Lambda wraps and applies the given function."""
        fn = Lambda(lambda x: x ** 2)
        x = np.array([2.0, 3.0], dtype=np.float32)
        result = fn(x)
        np.testing.assert_array_equal(result, np.array([4.0, 9.0], dtype=np.float32))

    def test_lambda_with_numpy_fn(self):
        """Lambda works with numpy functions."""
        fn = Lambda(np.log)
        x = np.array([1.0, np.e], dtype=np.float32)
        result = fn(x)
        np.testing.assert_allclose(result, np.array([0.0, 1.0], dtype=np.float32), atol=1e-6)

    def test_lambda_identity(self):
        """Lambda with identity function returns input unchanged."""
        fn = Lambda(lambda x: x)
        x = np.array([42.0], dtype=np.float32)
        np.testing.assert_array_equal(fn(x), x)
