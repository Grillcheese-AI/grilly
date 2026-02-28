#include "grilly/nn/containers.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace grilly {
namespace nn {

// ══════════════════════════════════════════════════════════════════════
// MultiStepContainer
// ══════════════════════════════════════════════════════════════════════

MultiStepContainer::MultiStepContainer(std::shared_ptr<Module> module)
    : module_(std::move(module)) {
    if (module_) {
        register_module("module", module_);
    }
}

Tensor MultiStepContainer::forward(Tensor x_seq) {
    // x_seq shape: [T, N, ...]
    if (x_seq.ndim() < 2) {
        throw std::invalid_argument(
            "MultiStepContainer: input must be at least 2D [T, ...]");
    }

    int64_t T = x_seq.shape(0);
    auto step_shape = std::vector<int64_t>(x_seq.shape().begin() + 1,
                                           x_seq.shape().end());
    int64_t step_numel = std::accumulate(step_shape.begin(), step_shape.end(),
                                         int64_t(1), std::multiplies<int64_t>());

    const float* x_data = x_seq.data();
    step_outputs_.clear();

    for (int64_t t = 0; t < T; t++) {
        std::vector<float> step_data(x_data + t * step_numel,
                                     x_data + (t + 1) * step_numel);
        Tensor x_t(std::move(step_data), step_shape, x_seq.backend());
        step_outputs_.push_back(module_->forward(std::move(x_t)));
    }

    // Stack outputs: determine output step shape from first output
    if (step_outputs_.empty()) {
        return Tensor();
    }

    auto out_step_shape = step_outputs_[0].shape();
    int64_t out_step_numel = step_outputs_[0].numel();

    std::vector<float> all_outputs(static_cast<size_t>(T * out_step_numel));
    for (int64_t t = 0; t < T; t++) {
        const float* src = step_outputs_[t].data();
        std::memcpy(all_outputs.data() + t * out_step_numel, src,
                    out_step_numel * sizeof(float));
    }

    std::vector<int64_t> out_shape = {T};
    out_shape.insert(out_shape.end(), out_step_shape.begin(),
                     out_step_shape.end());

    return Tensor(std::move(all_outputs), std::move(out_shape), x_seq.backend());
}

Tensor MultiStepContainer::backward(const Tensor& grad_output) {
    // Placeholder — full backward requires Module.backward() support
    (void)grad_output;
    throw std::runtime_error(
        "MultiStepContainer::backward not yet implemented");
}

// ══════════════════════════════════════════════════════════════════════
// SeqToANNContainer
// ══════════════════════════════════════════════════════════════════════

SeqToANNContainer::SeqToANNContainer(
    std::vector<std::shared_ptr<Module>> modules)
    : ann_modules_(std::move(modules)) {
    for (size_t i = 0; i < ann_modules_.size(); i++) {
        if (ann_modules_[i]) {
            register_module("module_" + std::to_string(i), ann_modules_[i]);
        }
    }
}

Tensor SeqToANNContainer::forward(Tensor x_seq) {
    // x_seq shape: [T, N, C, ...]
    if (x_seq.ndim() < 2) {
        throw std::invalid_argument(
            "SeqToANNContainer: input must be at least 2D [T, N, ...]");
    }

    T_ = x_seq.shape(0);
    int64_t total = x_seq.numel();

    // Reshape [T, N, ...] → [T*N, ...]
    auto batch_shape = x_seq.shape();
    std::vector<int64_t> merged_shape;
    if (batch_shape.size() >= 2) {
        merged_shape.push_back(batch_shape[0] * batch_shape[1]);
        for (size_t i = 2; i < batch_shape.size(); i++) {
            merged_shape.push_back(batch_shape[i]);
        }
    } else {
        merged_shape.push_back(total);
    }

    Tensor merged = x_seq.reshape(merged_shape);

    // Run through all ANN modules
    for (auto& mod : ann_modules_) {
        if (mod) {
            merged = mod->forward(std::move(merged));
        }
    }

    // Reshape back: [T*N, ...] → [T, N, ...]
    auto out_merged_shape = merged.shape();
    std::vector<int64_t> out_shape;
    out_shape.push_back(T_);
    if (batch_shape.size() >= 2) {
        out_shape.push_back(batch_shape[1]);
    }
    for (size_t i = 1; i < out_merged_shape.size(); i++) {
        out_shape.push_back(out_merged_shape[i]);
    }

    return merged.reshape(out_shape);
}

Tensor SeqToANNContainer::backward(const Tensor& grad_output) {
    (void)grad_output;
    throw std::runtime_error(
        "SeqToANNContainer::backward not yet implemented");
}

// ══════════════════════════════════════════════════════════════════════
// Flatten
// ══════════════════════════════════════════════════════════════════════

Flatten::Flatten(int start_dim, int end_dim)
    : start_dim_(start_dim), end_dim_(end_dim) {}

Tensor Flatten::forward(Tensor input) {
    input_shape_ = input.shape();

    int ndim = static_cast<int>(input.ndim());
    int start = start_dim_ >= 0 ? start_dim_ : ndim + start_dim_;
    int end = end_dim_ >= 0 ? end_dim_ : ndim + end_dim_;

    start = std::max(0, std::min(start, ndim - 1));
    end = std::max(0, std::min(end, ndim - 1));

    if (start > end) {
        return input;  // Nothing to flatten
    }

    // Build new shape
    std::vector<int64_t> new_shape;

    // Dimensions before start_dim
    for (int i = 0; i < start; i++) {
        new_shape.push_back(input_shape_[i]);
    }

    // Flattened dimension
    int64_t flat = 1;
    for (int i = start; i <= end; i++) {
        flat *= input_shape_[i];
    }
    new_shape.push_back(flat);

    // Dimensions after end_dim
    for (int i = end + 1; i < ndim; i++) {
        new_shape.push_back(input_shape_[i]);
    }

    return input.reshape(new_shape);
}

Tensor Flatten::backward(const Tensor& grad_output) {
    if (input_shape_.empty()) {
        throw std::runtime_error("Flatten::backward: no cached input shape");
    }
    return grad_output.reshape(input_shape_);
}

}  // namespace nn
}  // namespace grilly
