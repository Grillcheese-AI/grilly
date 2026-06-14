/// bindings_core.cpp — PYBIND11_MODULE entry point and Tensor/NN framework bindings.
///
/// This replaces the monolithic bindings.cpp with a split architecture.
/// Each register_*_ops() function is implemented in its own file for
/// parallel compilation and sub-1000-line files.

#include "bindings_core.h"

#include "grilly/op_graph.h"
#include "grilly/nn/surrogate.h"
#include "grilly/nn/snn.h"
#include "grilly/nn/containers.h"
#include "grilly/nn/optimizer.h"
#include "grilly/nn/dataloader.h"

// Forward declarations for split binding files
void register_siglip_ops(py::module_& m);
void register_mingru_ops(py::module_& m);
void register_bandit_ops(py::module_& m);
void register_eggroll_ops(py::module_& m);

PYBIND11_MODULE(grilly_core, m) {
    m.doc() = "grilly C++ Vulkan backend — eliminates Python->C boundary "
              "crossings for GPU dispatch";

    // ═══════════════════════════════════════════════════════════════════════
    // Device
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<GrillyCoreContext>(m, "Device")
        .def(py::init<>(), "Initialize Vulkan device, buffer pool, and "
                           "pipeline cache")
        .def("load_shaders", &GrillyCoreContext::loadShaders,
             py::arg("shader_dir"),
             "Load all .spv shaders from a directory")
        .def_property_readonly("device_name",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.deviceName();
                               })
        .def_property_readonly("has_cooperative_matrix",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.hasCooperativeMatrix();
                               })
        .def_property_readonly("has_float16",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.hasFloat16();
                               })
        .def("pool_stats",
             [](const GrillyCoreContext& ctx) {
                 auto s = ctx.pool.stats();
                 py::dict d;
                 d["hits"] = s.hits;
                 d["misses"] = s.misses;
                 d["allocations"] = s.allocations;
                 d["total_acquired"] = s.totalAcquired;
                 d["total_released"] = s.totalReleased;
                 return d;
             })
        .def("cache_stats", [](const GrillyCoreContext& ctx) {
            auto s = ctx.cache.cacheStats();
            py::dict d;
            d["hits"] = s.hits;
            d["misses"] = s.misses;
            d["evictions"] = s.evictions;
            d["cached_sets"] = s.cachedSets;
            return d;
        });

    // ═══════════════════════════════════════════════════════════════════════
    // OpGraph (batched execution with operator fusion)
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<grilly::OpGraph>(m, "OpGraph")
        .def(py::init<>())
        .def("size", &grilly::OpGraph::size,
             "Number of ops recorded in the graph")
        .def("clear", &grilly::OpGraph::clear,
             "Clear all recorded ops for reuse")
        .def("optimize",
             [](grilly::OpGraph& graph, GrillyCoreContext& ctx) -> py::dict {
                 grilly::FusionStats stats;
                 {
                     py::gil_scoped_release release;
                     stats = graph.optimize(ctx.cache);
                 }
                 py::dict d;
                 d["ops_fused"] = stats.opsFused;
                 d["barriers_eliminated"] = stats.barriersEliminated;
                 d["original_ops"] = stats.originalOps;
                 d["optimized_ops"] = stats.optimizedOps;
                 return d;
             },
             py::arg("device"),
             "Run fusion optimization pass. Returns fusion statistics.")
        .def("execute",
             [](grilly::OpGraph& graph, GrillyCoreContext& ctx) {
                 {
                     py::gil_scoped_release release;
                     graph.execute(ctx.batch, ctx.cache);
                 }
             },
             py::arg("device"),
             "Execute all recorded ops in a single GPU submission");

    // ═══════════════════════════════════════════════════════════════════════
    // ComputeBackend (abstract interface for multi-backend support)
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<grilly::ComputeBackend, std::unique_ptr<grilly::ComputeBackend>>(
            m, "ComputeBackend")
        .def("name", &grilly::ComputeBackend::name)
        .def("device_name", &grilly::ComputeBackend::deviceName)
        .def("load_shader_dir", &grilly::ComputeBackend::loadShaderDir,
             py::arg("dir"))
        .def("has_shader", &grilly::ComputeBackend::hasShader,
             py::arg("name"))
        .def_property_readonly("has_cooperative_matrix",
             &grilly::ComputeBackend::hasCooperativeMatrix)
        .def_property_readonly("has_float16",
             &grilly::ComputeBackend::hasFloat16);

    m.def("create_backend", &grilly::createBackend,
          py::arg("type") = "vulkan",
          py::return_value_policy::move,
          "Create a GPU compute backend.\n"
          "Supported: 'vulkan'. Coming soon: 'opengl', 'opencl'.");

    // ═══════════════════════════════════════════════════════════════════════
    // NN FRAMEWORK — Tensor, Parameter, Module, SNN, Optimizer, DataLoader
    // ═══════════════════════════════════════════════════════════════════════

    using namespace grilly::nn;

    // ── DType enum ──
    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("Int32", DType::Int32)
        .value("Int64", DType::Int64);

    // ── Tensor ──
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int64_t>, DType, grilly::ComputeBackend*>(),
             py::arg("shape"),
             py::arg("dtype") = DType::Float32,
             py::arg("backend") = nullptr)
        .def_static("from_numpy", &Tensor::from_numpy,
             py::arg("arr"), py::arg("backend") = nullptr,
             "Create tensor from numpy array")
        .def_static("zeros", &Tensor::zeros,
             py::arg("shape"), py::arg("backend") = nullptr,
             "Create zero-filled tensor")
        .def_static("empty", &Tensor::empty,
             py::arg("shape"), py::arg("backend") = nullptr,
             "Create uninitialized tensor")
        .def("numpy", &Tensor::numpy,
             "Download to CPU and return as numpy array")
        .def("gpu_handle", &Tensor::gpu_handle,
             "Upload to GPU and return buffer handle")
        .def("gpu_handle_if_valid", &Tensor::gpu_handle_if_valid,
             "GPU buffer handle if already resident (0 otherwise); does not upload")
        .def("mark_gpu_modified", &Tensor::mark_gpu_modified)
        .def("mark_cpu_modified", &Tensor::mark_cpu_modified)
        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("view", &Tensor::view, py::arg("shape"))
        .def("ensure_gpu", &Tensor::ensure_gpu)
        .def("release_gpu", &Tensor::release_gpu)
        .def_property_readonly("shape",
             [](const Tensor& t) { return t.shape(); })
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("nbytes", &Tensor::nbytes)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("on_gpu", &Tensor::on_gpu)
        .def_property_readonly("on_cpu", &Tensor::on_cpu)
        .def_property_readonly("valid", &Tensor::valid)
        .def_property("requires_grad", &Tensor::requires_grad,
                      &Tensor::set_requires_grad)
        .def("__array__", [](const Tensor& t, py::object /*dtype*/) {
            return t.numpy();
        }, py::arg("dtype") = py::none(),
           "Support np.asarray(tensor) — downloads from GPU if needed")
        .def("__repr__", [](const Tensor& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); i++) {
                if (i > 0) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            s += "], dtype=float32";
            if (t.on_gpu()) s += ", gpu";
            if (t.requires_grad()) s += ", requires_grad=True";
            s += ")";
            return s;
        });

    // ── to_tensor helper ──
    m.def("to_tensor", &to_tensor,
          py::arg("arr"), py::arg("backend") = nullptr,
          "Convert numpy array to Tensor");

    // ── Parameter ──
    py::class_<Parameter, Tensor>(m, "Parameter")
        .def(py::init<Tensor, bool>(),
             py::arg("data"),
             py::arg("requires_grad") = true)
        .def(py::init<std::vector<int64_t>, grilly::ComputeBackend*, bool>(),
             py::arg("shape"),
             py::arg("backend") = nullptr,
             py::arg("requires_grad") = true)
        .def("grad", static_cast<Tensor& (Parameter::*)()>(&Parameter::grad_ref),
             py::return_value_policy::reference)
        .def("set_grad",
             static_cast<void (Parameter::*)(const Tensor&)>(&Parameter::set_grad),
             py::arg("grad"))
        .def("has_grad", &Parameter::has_grad)
        .def("zero_grad", &Parameter::zero_grad);

    // ── Module (with trampoline for Python subclassing) ──
    py::class_<Module, PyModule, std::shared_ptr<Module>>(m, "Module")
        .def(py::init<>())
        .def("forward", &Module::forward, py::arg("input"))
        .def("__call__", &Module::operator(), py::arg("input"))
        .def("register_parameter", &Module::register_parameter,
             py::arg("name"), py::arg("param"))
        .def("register_module", &Module::register_module,
             py::arg("name"), py::arg("module"))
        .def("parameters", &Module::parameters,
             py::return_value_policy::reference)
        .def("named_parameters", &Module::named_parameters,
             py::return_value_policy::reference)
        .def("train", &Module::train, py::arg("mode") = true)
        .def("eval", &Module::eval)
        .def_property_readonly("training", &Module::is_training)
        .def("state_dict", &Module::state_dict)
        .def("load_state_dict", &Module::load_state_dict,
             py::arg("state"))
        .def("gpu_mode", &Module::gpu_mode,
             py::arg("enable") = true, py::arg("device_local") = true)
        .def("to", &Module::to, py::arg("device"));

    // ── SurrogateFunction ──
    py::enum_<SurrogateType>(m, "SurrogateType")
        .value("ATan", SurrogateType::ATan)
        .value("Sigmoid", SurrogateType::Sigmoid)
        .value("FastSigmoid", SurrogateType::FastSigmoid);

    py::class_<SurrogateFunction>(m, "SurrogateFunction")
        .def(py::init<SurrogateType, float>(),
             py::arg("type") = SurrogateType::ATan,
             py::arg("alpha") = 2.0f)
        .def("forward", &SurrogateFunction::forward, py::arg("x"))
        .def("gradient", &SurrogateFunction::gradient, py::arg("x"))
        .def_readwrite("alpha", &SurrogateFunction::alpha)
        .def_readwrite("type", &SurrogateFunction::type);

    // ── MemoryModule ──
    py::class_<MemoryModule, Module, std::shared_ptr<MemoryModule>>(
            m, "MemoryModule")
        .def("register_memory", &MemoryModule::register_memory,
             py::arg("name"), py::arg("default_value"))
        .def("reset", &MemoryModule::reset)
        .def("detach", &MemoryModule::detach);

    // ── BaseNode ──
    py::class_<BaseNode, MemoryModule, std::shared_ptr<BaseNode>>(
            m, "BaseNode")
        .def("single_step_forward", &BaseNode::single_step_forward,
             py::arg("x"))
        .def("multi_step_forward", &BaseNode::multi_step_forward,
             py::arg("x_seq"))
        .def("backward", &BaseNode::backward, py::arg("grad_output"))
        .def_property_readonly("v_threshold", &BaseNode::v_threshold)
        .def_property_readonly("v_reset", &BaseNode::v_reset)
        .def_property("step_mode", &BaseNode::step_mode,
                      &BaseNode::set_step_mode);

    // ── IFNode ──
    py::class_<IFNode, BaseNode, std::shared_ptr<IFNode>>(m, "IFNode")
        .def(py::init<float, float, SurrogateFunction, bool, std::string,
                      bool>(),
             py::arg("v_threshold") = 1.0f,
             py::arg("v_reset") = 0.0f,
             py::arg("surrogate") = ATan(),
             py::arg("detach_reset") = false,
             py::arg("step_mode") = "s",
             py::arg("store_v_seq") = false);

    // ── LIFNode ──
    py::class_<LIFNode, BaseNode, std::shared_ptr<LIFNode>>(m, "LIFNode")
        .def(py::init<float, bool, float, float, SurrogateFunction,
                      bool, std::string, bool>(),
             py::arg("tau") = 2.0f,
             py::arg("decay_input") = false,
             py::arg("v_threshold") = 1.0f,
             py::arg("v_reset") = 0.0f,
             py::arg("surrogate") = ATan(),
             py::arg("detach_reset") = false,
             py::arg("step_mode") = "s",
             py::arg("store_v_seq") = false)
        .def_property_readonly("tau", &LIFNode::tau);

    // ── ParametricLIFNode ──
    py::class_<ParametricLIFNode, BaseNode,
               std::shared_ptr<ParametricLIFNode>>(m, "ParametricLIFNode")
        .def(py::init<float, bool, float, float, SurrogateFunction,
                      bool, std::string, bool>(),
             py::arg("init_tau") = 2.0f,
             py::arg("decay_input") = false,
             py::arg("v_threshold") = 1.0f,
             py::arg("v_reset") = 0.0f,
             py::arg("surrogate") = ATan(),
             py::arg("detach_reset") = false,
             py::arg("step_mode") = "s",
             py::arg("store_v_seq") = false);

    // ── Containers ──
    py::class_<MultiStepContainer, Module,
               std::shared_ptr<MultiStepContainer>>(m, "MultiStepContainer")
        .def(py::init<std::shared_ptr<Module>>(), py::arg("module"),
             py::keep_alive<1, 2>())
        .def("backward", &MultiStepContainer::backward,
             py::arg("grad_output"));

    py::class_<SeqToANNContainer, Module,
               std::shared_ptr<SeqToANNContainer>>(m, "SeqToANNContainer")
        .def(py::init<std::vector<std::shared_ptr<Module>>>(),
             py::arg("modules"),
             py::keep_alive<1, 2>())
        .def("backward", &SeqToANNContainer::backward,
             py::arg("grad_output"));

    py::class_<Flatten, Module, std::shared_ptr<Flatten>>(m, "Flatten")
        .def(py::init<int, int>(),
             py::arg("start_dim") = 1, py::arg("end_dim") = -1)
        .def("backward", &Flatten::backward, py::arg("grad_output"));

    // ── Optimizers ──
    py::class_<Optimizer>(m, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad)
        .def("state_dict", &Optimizer::state_dict)
        .def("load_state_dict", &Optimizer::load_state_dict,
             py::arg("state"))
        .def_property_readonly("param_count", &Optimizer::param_count);

    py::class_<Adam, Optimizer>(m, "Adam")
        .def(py::init([](py::list params, float lr,
                         std::pair<float, float> betas, float eps,
                         float weight_decay, bool amsgrad) {
                 std::vector<Parameter*> ptrs;
                 for (auto& item : params) {
                     ptrs.push_back(item.cast<Parameter*>());
                 }
                 auto opt = new Adam(std::move(ptrs), lr, betas, eps,
                                     weight_decay, amsgrad);
                 opt->py_params_ = params;
                 return opt;
             }),
             py::arg("params"),
             py::arg("lr") = 1e-3f,
             py::arg("betas") = std::make_pair(0.9f, 0.999f),
             py::arg("eps") = 1e-8f,
             py::arg("weight_decay") = 0.0f,
             py::arg("amsgrad") = false)
        .def_property("lr", &Adam::lr, &Adam::set_lr);

    py::class_<AdamW, Optimizer>(m, "AdamW")
        .def(py::init([](py::list params, float lr,
                         std::pair<float, float> betas, float eps,
                         float weight_decay, bool amsgrad) {
                 std::vector<Parameter*> ptrs;
                 for (auto& item : params) {
                     ptrs.push_back(item.cast<Parameter*>());
                 }
                 auto opt = new AdamW(std::move(ptrs), lr, betas, eps,
                                      weight_decay, amsgrad);
                 opt->py_params_ = params;
                 return opt;
             }),
             py::arg("params"),
             py::arg("lr") = 1e-3f,
             py::arg("betas") = std::make_pair(0.9f, 0.999f),
             py::arg("eps") = 1e-8f,
             py::arg("weight_decay") = 0.01f,
             py::arg("amsgrad") = false)
        .def_property("lr", &AdamW::lr, &AdamW::set_lr);

    py::class_<SGD, Optimizer>(m, "SGD")
        .def(py::init([](py::list params, float lr, float momentum,
                         float weight_decay, bool nesterov) {
                 std::vector<Parameter*> ptrs;
                 for (auto& item : params) {
                     ptrs.push_back(item.cast<Parameter*>());
                 }
                 auto opt = new SGD(std::move(ptrs), lr, momentum,
                                    weight_decay, nesterov);
                 opt->py_params_ = params;
                 return opt;
             }),
             py::arg("params"),
             py::arg("lr") = 0.01f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("nesterov") = false)
        .def_property("lr", &SGD::lr, &SGD::set_lr);

    // ── DataLoader ──
    py::class_<Batch>(m, "Batch")
        .def_readwrite("data", &Batch::data)
        .def_readwrite("target", &Batch::target);

    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<py::object, int, bool, int, bool>(),
             py::arg("dataset"),
             py::arg("batch_size") = 1,
             py::arg("shuffle") = false,
             py::arg("num_workers") = 0,
             py::arg("drop_last") = false)
        .def_property_readonly("num_batches", &DataLoader::num_batches)
        .def_property_readonly("dataset_size", &DataLoader::dataset_size)
        .def("__iter__", &DataLoader::iter, py::return_value_policy::reference)
        .def("iter", &DataLoader::iter, py::return_value_policy::reference)
        .def("__next__", &DataLoader::next);

    // ═══════════════════════════════════════════════════════════════════════
    // Register ops from split files
    // ═══════════════════════════════════════════════════════════════════════

    register_linear_ops(m);
    register_activations_ops(m);
    register_conv_ops(m);
    register_attention_ops(m);
    register_normalization_ops(m);
    register_optim_ops(m);
    register_loss_ops(m);
    register_snn_ops(m);
    register_pooling_ops(m);
    register_siglip_ops(m);
    register_perceiver_ops(m);
    register_moqe_train_ops(m);
    register_moe_ops(m);
    register_fusion_ops(m);
    register_vsa_lm_ops(m);
    register_grl_ops(m);
    register_misc_ops(m);
    register_prefix_scan_ops(m);
    register_mingru_ops(m);
    register_bandit_ops(m);
    register_eggroll_ops(m);
    register_autograd_ops(m);
}
