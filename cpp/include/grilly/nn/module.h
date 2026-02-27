#pragma once
/// Neural network Module base class with pybind11 trampoline.
///
/// C++ Module base so Python classes can subclass it via pybind11's trampoline
/// pattern. This is the critical bridge — all 85+ existing Python nn classes
/// inherit from Module. The trampoline (PyModule) intercepts virtual forward()
/// calls and dispatches them to the Python override.
///
/// Design matches PyTorch's nn.Module API:
///   - register_parameter / register_module for named children
///   - parameters() / named_parameters() for optimizer integration
///   - train() / eval() mode switching
///   - state_dict / load_state_dict for serialization
///   - gpu_mode() for transparent GPU tensor returns

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grilly/compute_backend.h"
#include "grilly/nn/parameter.h"
#include "grilly/nn/tensor.h"

namespace py = pybind11;

namespace grilly {
namespace nn {

class Module {
public:
    virtual ~Module() = default;

    /// Forward pass — overridden from Python via trampoline.
    virtual Tensor forward(Tensor input) = 0;

    /// __call__ dispatches to forward.
    Tensor operator()(Tensor input);

    // ── Parameter management ──

    void register_parameter(const std::string& name, Parameter param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

    /// All parameters (including from child modules), non-owning pointers.
    std::vector<Parameter*> parameters();

    /// Named parameters with dotted path (e.g., "encoder.weight").
    std::vector<std::pair<std::string, Parameter*>> named_parameters();

    // ── Training mode ──

    void train(bool mode = true);
    void eval();
    bool is_training() const { return training_; }

    // ── Serialization ──

    py::dict state_dict() const;
    void load_state_dict(py::dict state);

    // ── GPU mode ──

    void gpu_mode(bool enable = true, bool device_local = true);

    // ── Device ──

    void to(const std::string& device);
    const std::string& device() const { return device_; }

    // ── Backend access ──

    ComputeBackend* get_backend();
    void set_backend(ComputeBackend* b) { backend_ = b; }

protected:
    bool training_ = true;
    bool return_gpu_tensor_ = false;
    bool use_device_local_ = false;
    std::string device_ = "vulkan";
    ComputeBackend* backend_ = nullptr;

    std::map<std::string, Parameter> parameters_;
    std::map<std::string, std::shared_ptr<Module>> modules_;

private:
    /// Helper: collect parameters recursively with prefix.
    void collect_parameters(const std::string& prefix,
                            std::vector<std::pair<std::string, Parameter*>>& out);
};

/// Pybind11 trampoline — allows Python classes to subclass Module.
///
/// When Python calls `model(x)`, the C++ operator() calls forward(),
/// which bounces back to the Python override via PYBIND11_OVERRIDE_PURE.
class PyModule : public Module {
public:
    using Module::Module;

    Tensor forward(Tensor input) override {
        PYBIND11_OVERRIDE_PURE(
            Tensor,    // Return type
            Module,    // Parent class
            forward,   // Method name
            input      // Arguments
        );
    }
};

}  // namespace nn
}  // namespace grilly
