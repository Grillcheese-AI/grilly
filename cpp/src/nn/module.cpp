#include "grilly/nn/module.h"

#include <stdexcept>

namespace grilly {
namespace nn {

// ── Forward dispatch ──────────────────────────────────────────────────

Tensor Module::operator()(Tensor input) {
    return forward(std::move(input));
}

// ── Parameter management ──────────────────────────────────────────────

void Module::register_parameter(const std::string& name, Parameter param) {
    // Propagate backend to the parameter
    if (backend_ && !param.backend()) {
        param.set_backend(backend_);
    }
    parameters_[name] = std::move(param);
}

void Module::register_module(const std::string& name,
                             std::shared_ptr<Module> module) {
    // Propagate backend to child module
    if (backend_ && module) {
        module->set_backend(backend_);
    }
    modules_[name] = std::move(module);
}

std::vector<Parameter*> Module::parameters() {
    auto named = named_parameters();
    std::vector<Parameter*> result;
    result.reserve(named.size());
    for (auto& [name, param] : named) {
        result.push_back(param);
    }
    return result;
}

std::vector<std::pair<std::string, Parameter*>> Module::named_parameters() {
    std::vector<std::pair<std::string, Parameter*>> result;
    collect_parameters("", result);
    return result;
}

void Module::collect_parameters(
    const std::string& prefix,
    std::vector<std::pair<std::string, Parameter*>>& out) {
    for (auto& [name, param] : parameters_) {
        std::string full_name = prefix.empty() ? name : prefix + "." + name;
        out.emplace_back(full_name, &param);
    }
    for (auto& [name, mod] : modules_) {
        if (mod) {
            std::string child_prefix = prefix.empty() ? name : prefix + "." + name;
            mod->collect_parameters(child_prefix, out);
        }
    }
}

// ── Training mode ─────────────────────────────────────────────────────

void Module::train(bool mode) {
    training_ = mode;
    for (auto& [name, mod] : modules_) {
        if (mod) mod->train(mode);
    }
}

void Module::eval() {
    train(false);
}

// ── Serialization ─────────────────────────────────────────────────────

py::dict Module::state_dict() const {
    py::dict state;
    for (const auto& [name, param] : parameters_) {
        state[py::cast(name)] = param.numpy();
    }
    for (const auto& [name, mod] : modules_) {
        if (!mod) continue;
        py::dict child_state = mod->state_dict();
        for (auto item : child_state) {
            std::string key = name + "." + py::cast<std::string>(item.first);
            state[py::cast(key)] = item.second;
        }
    }
    return state;
}

void Module::load_state_dict(py::dict state) {
    for (auto item : state) {
        std::string key = py::cast<std::string>(item.first);
        py::array_t<float> arr = py::cast<py::array_t<float>>(item.second);

        // Check if this is a direct parameter
        auto dot_pos = key.find('.');
        if (dot_pos == std::string::npos) {
            // Direct parameter
            auto it = parameters_.find(key);
            if (it != parameters_.end()) {
                it->second = Parameter(
                    Tensor::from_numpy(arr, backend_), true);
            }
        } else {
            // Child module parameter
            std::string mod_name = key.substr(0, dot_pos);
            std::string rest = key.substr(dot_pos + 1);
            auto it = modules_.find(mod_name);
            if (it != modules_.end() && it->second) {
                py::dict sub;
                sub[py::cast(rest)] = arr;
                it->second->load_state_dict(sub);
            }
        }
    }
}

// ── GPU mode ──────────────────────────────────────────────────────────

void Module::gpu_mode(bool enable, bool device_local) {
    return_gpu_tensor_ = enable;
    use_device_local_ = device_local;

    if (enable) {
        // Upload all parameters to GPU
        for (auto& [name, param] : parameters_) {
            if (backend_) param.set_backend(backend_);
            param.ensure_gpu();
        }
    }

    // Propagate to child modules
    for (auto& [name, mod] : modules_) {
        if (mod) mod->gpu_mode(enable, device_local);
    }
}

// ── Device ────────────────────────────────────────────────────────────

void Module::to(const std::string& device) {
    device_ = device;
    if (device == "vulkan" || device == "gpu") {
        gpu_mode(true, true);
    } else if (device == "cpu") {
        // Download all parameters to CPU
        for (auto& [name, param] : parameters_) {
            if (param.on_gpu()) {
                param.release_gpu();
            }
        }
        return_gpu_tensor_ = false;
    }
    for (auto& [name, mod] : modules_) {
        if (mod) mod->to(device);
    }
}

// ── Backend access ────────────────────────────────────────────────────

ComputeBackend* Module::get_backend() {
    if (!backend_) {
        backend_ = createBackend("vulkan").release();
        // Note: this leaks intentionally — the backend lives for the process
        // lifetime. In practice, Python modules share a single backend instance.
    }
    return backend_;
}

}  // namespace nn
}  // namespace grilly
