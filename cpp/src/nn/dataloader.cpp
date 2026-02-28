#include "grilly/nn/dataloader.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace grilly {
namespace nn {

DataLoader::DataLoader(py::object dataset, int batch_size, bool shuffle,
                       int num_workers, bool drop_last)
    : dataset_(std::move(dataset)),
      batch_size_(batch_size),
      shuffle_(shuffle),
      num_workers_(num_workers),
      drop_last_(drop_last) {
    // Get dataset length via __len__()
    py::gil_scoped_acquire gil;
    dataset_len_ = py::cast<int64_t>(dataset_.attr("__len__")());

    // Initialize indices
    indices_.resize(dataset_len_);
    std::iota(indices_.begin(), indices_.end(), 0);

    // Calculate total batches
    if (drop_last_) {
        total_batches_ = dataset_len_ / batch_size_;
    } else {
        total_batches_ = (dataset_len_ + batch_size_ - 1) / batch_size_;
    }
}

DataLoader::~DataLoader() {
    stop_and_join_workers();
}

int64_t DataLoader::num_batches() const {
    return total_batches_;
}

int64_t DataLoader::dataset_size() const {
    return dataset_len_;
}

DataLoader& DataLoader::iter() {
    // Stop any running workers
    stop_and_join_workers();

    // Shuffle indices if requested
    if (shuffle_) {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }

    batch_idx_ = 0;
    next_batch_to_load_ = 0;

    // Start prefetch workers if configured
    if (num_workers_ > 0) {
        start_workers();
    }

    return *this;
}

Batch DataLoader::next() {
    if (batch_idx_ >= total_batches_) {
        throw py::stop_iteration();
    }

    Batch batch;

    if (num_workers_ > 0 && prefetch_queue_) {
        // Release GIL while waiting so worker threads can load batches
        bool got_batch;
        {
            py::gil_scoped_release release;
            got_batch = prefetch_queue_->pop(batch);
        }
        if (!got_batch) {
            throw py::stop_iteration();
        }
    } else {
        // Synchronous loading
        int64_t start = batch_idx_ * batch_size_;
        batch = load_batch(start);
    }

    batch_idx_++;
    return batch;
}

void DataLoader::start_workers() {
    stop_workers_ = false;
    prefetch_queue_ = std::make_unique<ThreadSafeQueue<Batch>>(
        static_cast<size_t>(num_workers_ * 2));

    for (int i = 0; i < num_workers_; i++) {
        workers_.emplace_back(&DataLoader::worker_loop, this);
    }
}

void DataLoader::stop_and_join_workers() {
    stop_workers_ = true;
    if (prefetch_queue_) {
        prefetch_queue_->shutdown();
    }
    // Release GIL while joining — workers need GIL to finish load_batch
    {
        py::gil_scoped_release release;
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }
    workers_.clear();
    prefetch_queue_.reset();
}

void DataLoader::worker_loop() {
    while (!stop_workers_) {
        int64_t batch_to_load;
        {
            std::lock_guard<std::mutex> lock(load_mutex_);
            batch_to_load = next_batch_to_load_++;
        }

        if (batch_to_load >= total_batches_) break;

        int64_t start = batch_to_load * batch_size_;
        Batch batch = load_batch(start);

        if (!prefetch_queue_->push(std::move(batch))) break;
    }
}

Batch DataLoader::load_batch(int64_t batch_start_idx) {
    py::gil_scoped_acquire gil;

    int64_t actual_batch_size =
        std::min(static_cast<int64_t>(batch_size_),
                 dataset_len_ - batch_start_idx);

    // Collect samples
    std::vector<py::object> samples;
    samples.reserve(actual_batch_size);

    for (int64_t i = 0; i < actual_batch_size; i++) {
        int64_t idx = indices_[batch_start_idx + i];
        py::object sample = dataset_.attr("__getitem__")(idx);
        samples.push_back(sample);
    }

    // Stack samples into batch tensors
    // Assumes __getitem__ returns a tuple (data, target) of numpy arrays
    Batch batch;

    if (samples.empty()) {
        return batch;
    }

    // Try to interpret as (data, target) tuple
    py::object first = samples[0];
    if (py::isinstance<py::tuple>(first) || py::isinstance<py::list>(first)) {
        py::tuple tup = py::cast<py::tuple>(first);
        if (tup.size() >= 2) {
            // Get shapes from first sample
            py::array_t<float> first_data = py::cast<py::array_t<float>>(tup[0]);
            py::array_t<float> first_target = py::cast<py::array_t<float>>(tup[1]);

            auto data_info = first_data.request();
            auto target_info = first_target.request();

            size_t data_elem = data_info.size;
            size_t target_elem = target_info.size;

            // Build batch data
            std::vector<float> batch_data(actual_batch_size * data_elem);
            std::vector<float> batch_target(actual_batch_size * target_elem);

            for (int64_t i = 0; i < actual_batch_size; i++) {
                py::tuple s = py::cast<py::tuple>(samples[i]);
                py::array_t<float> d = py::cast<py::array_t<float>>(s[0]);
                py::array_t<float> t = py::cast<py::array_t<float>>(s[1]);

                auto d_buf = d.request();
                auto t_buf = t.request();
                std::memcpy(batch_data.data() + i * data_elem,
                            d_buf.ptr, data_elem * sizeof(float));
                std::memcpy(batch_target.data() + i * target_elem,
                            t_buf.ptr, target_elem * sizeof(float));
            }

            // Build shapes: [batch_size, ...sample_shape]
            std::vector<int64_t> data_shape = {actual_batch_size};
            for (int d = 0; d < data_info.ndim; d++) {
                data_shape.push_back(data_info.shape[d]);
            }

            std::vector<int64_t> target_shape = {actual_batch_size};
            for (int d = 0; d < target_info.ndim; d++) {
                target_shape.push_back(target_info.shape[d]);
            }

            batch.data = Tensor(std::move(batch_data), std::move(data_shape));
            batch.target = Tensor(std::move(batch_target),
                                  std::move(target_shape));
        }
    } else if (py::isinstance<py::array>(first)) {
        // Single array — data only, no target
        py::array_t<float> first_arr = py::cast<py::array_t<float>>(first);
        auto info = first_arr.request();
        size_t elem = info.size;

        std::vector<float> batch_data(actual_batch_size * elem);
        for (int64_t i = 0; i < actual_batch_size; i++) {
            py::array_t<float> arr =
                py::cast<py::array_t<float>>(samples[i]);
            auto buf = arr.request();
            std::memcpy(batch_data.data() + i * elem, buf.ptr,
                        elem * sizeof(float));
        }

        std::vector<int64_t> shape = {actual_batch_size};
        for (int d = 0; d < info.ndim; d++) {
            shape.push_back(info.shape[d]);
        }
        batch.data = Tensor(std::move(batch_data), std::move(shape));
    }

    return batch;
}

}  // namespace nn
}  // namespace grilly
