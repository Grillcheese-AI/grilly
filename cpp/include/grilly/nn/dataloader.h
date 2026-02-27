#pragma once
/// DataLoader with background prefetch threading.
///
/// Uses pybind11 to call Python Dataset.__getitem__() from worker threads,
/// assembling batches in the background while the GPU processes the current
/// batch. The prefetch queue provides overlap between data loading and
/// GPU computation.
///
/// Thread safety: Python's GIL is acquired when calling __getitem__(), so
/// the parallelism benefit comes from I/O overlap (disk reads, decoding)
/// and batch assembly, not from CPU-parallel Python execution.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "grilly/nn/tensor.h"
#include "grilly/thread_safe_queue.h"

namespace py = pybind11;

namespace grilly {
namespace nn {

/// A single batch of data (input + target).
struct Batch {
    Tensor data;
    Tensor target;
};

/// DataLoader: iterates over a Python Dataset with optional batching,
/// shuffling, and background prefetch.
class DataLoader {
public:
    /// @param dataset  Python object implementing __len__() and __getitem__(idx)
    /// @param batch_size  Number of samples per batch
    /// @param shuffle  Whether to shuffle indices each epoch
    /// @param num_workers  Number of background prefetch threads (0 = sync)
    /// @param drop_last  Drop the last incomplete batch if dataset size
    ///                   is not divisible by batch_size
    DataLoader(py::object dataset, int batch_size = 1, bool shuffle = false,
               int num_workers = 0, bool drop_last = false);

    ~DataLoader();

    // Non-copyable, non-movable (owns threads)
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    /// Total number of batches per epoch.
    int64_t num_batches() const;

    /// Dataset size.
    int64_t dataset_size() const;

    // ── Python iterator protocol ──

    /// Reset for a new epoch (shuffles if enabled, restarts prefetch).
    DataLoader& iter();

    /// Get the next batch. Throws py::stop_iteration when exhausted.
    Batch next();

private:
    py::object dataset_;
    int batch_size_;
    bool shuffle_;
    int num_workers_;
    bool drop_last_;

    int64_t dataset_len_ = 0;
    std::vector<int64_t> indices_;
    int64_t batch_idx_ = 0;
    int64_t total_batches_ = 0;

    // Prefetch
    std::unique_ptr<ThreadSafeQueue<Batch>> prefetch_queue_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_workers_{false};
    std::atomic<int64_t> next_batch_to_load_{0};
    std::mutex load_mutex_;

    void start_workers();
    void stop_and_join_workers();
    void worker_loop();
    Batch load_batch(int64_t batch_start_idx);
};

}  // namespace nn
}  // namespace grilly
