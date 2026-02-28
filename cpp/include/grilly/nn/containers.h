#pragma once
/// SNN container modules for temporal sequence processing.
///
/// - MultiStepContainer: wraps any Module, loops T timesteps calling it per step
/// - SeqToANNContainer: reshapes [T,N,...] → [T*N,...], runs a sequence of
///   non-temporal modules, reshapes back. Used to apply standard layers (Linear,
///   Conv2d, BatchNorm) to each timestep independently without explicit looping.
/// - Flatten: reshapes dimensions, used inside SeqToANNContainer pipelines

#include <memory>
#include <vector>

#include "grilly/nn/module.h"
#include "grilly/nn/tensor.h"

namespace grilly {
namespace nn {

/// Wraps a Module for multi-step (temporal) execution.
/// Input shape: [T, N, ...] → calls module.forward() T times.
class MultiStepContainer : public Module {
public:
    explicit MultiStepContainer(std::shared_ptr<Module> module);

    Tensor forward(Tensor x_seq) override;
    Tensor backward(const Tensor& grad_output);

private:
    std::shared_ptr<Module> module_;
    std::vector<Tensor> step_outputs_;  // Cached for backward
};

/// Reshapes temporal input to batch, runs ANN modules, reshapes back.
/// Input: [T, N, C, ...] → reshape to [T*N, C, ...] → run modules → reshape back.
/// This lets standard (non-temporal) layers process all timesteps at once.
class SeqToANNContainer : public Module {
public:
    explicit SeqToANNContainer(std::vector<std::shared_ptr<Module>> modules);

    Tensor forward(Tensor x_seq) override;
    Tensor backward(const Tensor& grad_output);

private:
    std::vector<std::shared_ptr<Module>> ann_modules_;
    int64_t T_ = 0;  // Cached from forward for backward
};

/// Flatten dimensions. Default: flatten all dims except batch (start_dim=1).
class Flatten : public Module {
public:
    Flatten(int start_dim = 1, int end_dim = -1);

    Tensor forward(Tensor input) override;
    Tensor backward(const Tensor& grad_output);

private:
    int start_dim_;
    int end_dim_;
    std::vector<int64_t> input_shape_;  // Cached for backward
};

}  // namespace nn
}  // namespace grilly
