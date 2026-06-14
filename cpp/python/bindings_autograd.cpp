/// Python bindings for the C++ autograd subsystem (TapeContext / Wengert
/// list backward engine). Split out of the legacy monolithic bindings.cpp
/// so it is actually part of the compiled grilly_core module.

#include "bindings_core.h"

#include "grilly/autograd/autograd.h"

void register_autograd_ops(py::module_& m) {
    namespace ag = grilly::autograd;

    py::enum_<ag::OpType>(m, "OpType")
        .value("Add", ag::OpType::Add)
        .value("Sub", ag::OpType::Sub)
        .value("Mul", ag::OpType::Mul)
        .value("Div", ag::OpType::Div)
        .value("Neg", ag::OpType::Neg)
        .value("Pow", ag::OpType::Pow)
        .value("MatMul", ag::OpType::MatMul)
        .value("Linear", ag::OpType::Linear)
        .value("ReLU", ag::OpType::ReLU)
        .value("GELU", ag::OpType::GELU)
        .value("SiLU", ag::OpType::SiLU)
        .value("Tanh", ag::OpType::Tanh)
        .value("Sigmoid", ag::OpType::Sigmoid)
        .value("Softmax", ag::OpType::Softmax)
        .value("LayerNorm", ag::OpType::LayerNorm)
        .value("RMSNorm", ag::OpType::RMSNorm)
        .value("FlashAttention2", ag::OpType::FlashAttention2)
        .value("Conv2d", ag::OpType::Conv2d)
        .value("Conv1d", ag::OpType::Conv1d)
        .value("Sum", ag::OpType::Sum)
        .value("Mean", ag::OpType::Mean)
        .value("Max", ag::OpType::Max)
        .value("Min", ag::OpType::Min)
        .value("Reshape", ag::OpType::Reshape)
        .value("Transpose", ag::OpType::Transpose)
        .value("Slice", ag::OpType::Slice)
        .value("CrossEntropy", ag::OpType::CrossEntropy)
        .value("MSELoss", ag::OpType::MSELoss)
        .value("CubeMindSurprise", ag::OpType::CubeMindSurprise)
        .value("TemporalSurprise", ag::OpType::TemporalSurprise)
        .value("MinGRU", ag::OpType::MinGRU)
        .value("SwiGLU", ag::OpType::SwiGLU)
        .export_values();

    py::class_<ag::TensorRef>(m, "TensorRef")
        .def(py::init<>())
        .def_readwrite("buffer_id", &ag::TensorRef::buffer_id)
        .def_readwrite("ndim", &ag::TensorRef::ndim)
        .def_readwrite("dtype", &ag::TensorRef::dtype)
        .def_readwrite("requires_grad", &ag::TensorRef::requires_grad)
        .def("numel", &ag::TensorRef::numel)
        .def("size_bytes", &ag::TensorRef::size_bytes)
        .def("valid", &ag::TensorRef::valid)
        .def_static("none", &ag::TensorRef::none)
        .def("set_shape",
             [](ag::TensorRef& ref, const std::vector<uint32_t>& shape) {
                 ref.ndim = static_cast<uint32_t>(
                     std::min(shape.size(), size_t(8)));
                 for (uint32_t i = 0; i < ref.ndim; ++i) ref.shape[i] = shape[i];
             },
             py::arg("shape"))
        .def("get_shape",
             [](const ag::TensorRef& ref) -> std::vector<uint32_t> {
                 return std::vector<uint32_t>(ref.shape, ref.shape + ref.ndim);
             });

    py::class_<ag::TapeContext>(m, "TapeContext")
        .def(py::init(
                 [](GrillyCoreContext& ctx, size_t capacity) {
                     return new ag::TapeContext(
                         ctx.pool, ctx.batch, ctx.cache, capacity);
                 }),
             py::arg("device"),
             py::arg("arena_capacity") = ag::TapeArena::kDefaultCapacity,
             py::keep_alive<1, 2>())
        .def("begin", &ag::TapeContext::begin)
        .def("record_op",
             [](ag::TapeContext& tape, ag::OpType op,
                const std::vector<ag::TensorRef>& inputs,
                const std::vector<ag::TensorRef>& outputs) -> ag::Node* {
                 return tape.record_op(
                     op, inputs.data(),
                     static_cast<uint32_t>(inputs.size()),
                     outputs.data(),
                     static_cast<uint32_t>(outputs.size()));
             },
             py::arg("op"), py::arg("inputs"), py::arg("outputs"),
             py::return_value_policy::reference)
        .def("save_for_backward",
             [](ag::TapeContext& tape, ag::Node* node,
                const std::vector<uint32_t>& buffer_ids) {
                 tape.save_for_backward(
                     node, buffer_ids.data(),
                     static_cast<uint32_t>(buffer_ids.size()));
             },
             py::arg("node"), py::arg("buffer_ids"))
        .def("backward",
             [](ag::TapeContext& tape, ag::Node* loss_node,
                uint32_t grad_output_buffer) {
                 tape.backward(loss_node, grad_output_buffer);
             },
             py::arg("loss_node"), py::arg("grad_output_buffer"))
        .def("get_grad_buffer", &ag::TapeContext::get_grad_buffer,
             py::arg("input_buffer_id"))
        .def("register_input",
             [](ag::TapeContext& tape,
                py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                bool requires_grad) -> uint32_t {
                 auto info = arr.request();
                 size_t bytes = static_cast<size_t>(info.size) * sizeof(float);
                 uint32_t id = tape.registry().alloc(bytes);
                 tape.registry().upload(id, info.ptr, bytes);
                 (void)requires_grad;
                 return id;
             },
             py::arg("array"), py::arg("requires_grad") = true,
             "Allocate a resident buffer, upload a numpy array, return its id.")
        .def("register_input_u32",
             [](ag::TapeContext& tape,
                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> arr)
                 -> uint32_t {
                 auto info = arr.request();
                 size_t bytes = static_cast<size_t>(info.size) * sizeof(uint32_t);
                 uint32_t id = tape.registry().alloc(bytes);
                 tape.registry().upload(id, info.ptr, bytes);
                 return id;
             },
             py::arg("array"),
             "Allocate a resident buffer, upload a uint32 numpy array (e.g. "
             "class-index targets), return its id.")
        .def("register_weight",
             [](ag::TapeContext& tape,
                py::array_t<float, py::array::c_style | py::array::forcecast> arr)
                 -> uint32_t {
                 auto info = arr.request();
                 size_t bytes = static_cast<size_t>(info.size) * sizeof(float);
                 uint32_t id = tape.registry().alloc_persistent(bytes);
                 tape.registry().upload(id, info.ptr, bytes);
                 return id;
             },
             py::arg("array"),
             "Allocate a PERSISTENT resident buffer (weight / Adam moment) that "
             "survives begin()/clear() with a stable id; upload a float32 array. "
             "Register all such buffers BEFORE any step-scoped register_input.")
        .def("forward_begin", &ag::TapeContext::forward_begin,
             "Open the resident forward command batch.")
        .def("forward_submit", &ag::TapeContext::forward_submit,
             "Submit the resident forward command batch and wait.")
        .def("forward_linear", &ag::TapeContext::forward_linear,
             py::arg("in_id"), py::arg("weight_id"), py::arg("bias_id"),
             py::arg("M"), py::arg("K"), py::arg("N"),
             "Resident forward Linear: out = in @ weight^T (+bias). "
             "weight (N,K), in (M,K) -> out (M,N). bias_id=0 for none. "
             "Returns the resident output buffer id.")
        .def("forward_rmsnorm", &ag::TapeContext::forward_rmsnorm,
             py::arg("in_id"), py::arg("weight_id"),
             py::arg("positions"), py::arg("features"),
             "Resident forward RMSNorm -> resident output id.")
        .def("forward_swiglu", &ag::TapeContext::forward_swiglu,
             py::arg("in_id"), py::arg("rows"), py::arg("hidden"),
             "Resident forward SwiGLU: [x1|x2](rows,2*hidden)->out(rows,hidden).")
        .def("forward_mingru", &ag::TapeContext::forward_mingru,
             py::arg("g_id"), py::arg("v_id"), py::arg("d_id"),
             py::arg("batch"), py::arg("seqLen"), py::arg("hidden"),
             "Resident forward MinGRU: H=scan(G,V,D), all (batch,seqLen,hidden).")
        .def("forward_embedding", &ag::TapeContext::forward_embedding,
             py::arg("ids_id"), py::arg("table_id"),
             py::arg("batch"), py::arg("seqLen"), py::arg("vocab"), py::arg("dim"),
             "Resident forward embedding lookup -> resident (batch*seq,dim) id.")
        .def("forward_add", &ag::TapeContext::forward_add,
             py::arg("a_id"), py::arg("b_id"), py::arg("total_elements"),
             "Resident forward elementwise add (residual): out=a+b, "
             "non-destructive -> resident output id.")
        .def("adamw_update", &ag::TapeContext::adamw_update,
             py::arg("w_id"), py::arg("grad_id"), py::arg("m_id"), py::arg("v_id"),
             py::arg("numel"), py::arg("lr"), py::arg("beta1"), py::arg("beta2"),
             py::arg("eps"), py::arg("weight_decay"), py::arg("beta1_t"),
             py::arg("beta2_t"), py::arg("clear_grad"),
             "Resident AdamW update on persistent W/m/v from grad (in place). "
             "Wrap calls in forward_begin()/forward_submit() for one batch.")
        .def("read_buffer",
             [](ag::TapeContext& tape, uint32_t buffer_id,
                const std::vector<uint32_t>& shape) -> py::array_t<float> {
                 size_t numel = 1;
                 for (uint32_t s : shape) numel *= s;
                 size_t bytes = numel * sizeof(float);
                 py::array_t<float> out(static_cast<py::ssize_t>(numel));
                 auto info = out.request();
                 tape.registry().download(buffer_id, info.ptr, bytes);
                 out.resize(std::vector<py::ssize_t>(shape.begin(), shape.end()));
                 return out;
             },
             py::arg("buffer_id"), py::arg("shape"),
             "Download a resident buffer by id into a numpy array.")
        .def("end", &ag::TapeContext::end)
        .def("is_recording", &ag::TapeContext::is_recording)
        .def("arena_bytes_used", &ag::TapeContext::arena_bytes_used)
        .def("arena_utilization", &ag::TapeContext::arena_utilization)
        .def("last_backward_stats",
             [](const ag::TapeContext& tape) -> py::dict {
                 auto s = tape.last_backward_stats();
                 py::dict d;
                 d["nodes_visited"] = s.nodes_visited;
                 d["nodes_with_grad"] = s.nodes_with_grad;
                 d["shaders_dispatched"] = s.shaders_dispatched;
                 d["cpu_fallbacks"] = s.cpu_fallbacks;
                 return d;
             });

    py::class_<ag::Node>(m, "AutogradNode")
        .def_readonly("op", &ag::Node::op)
        .def_readonly("seq", &ag::Node::seq)
        .def_readonly("num_inputs", &ag::Node::num_inputs)
        .def_readonly("num_outputs", &ag::Node::num_outputs)
        .def_readonly("num_saved", &ag::Node::num_saved)
        .def_readonly("grad_output_buffer", &ag::Node::grad_output_buffer)
        .def("get_grad_input_buffer",
             [](const ag::Node& node, uint32_t idx) -> uint32_t {
                 if (idx >= ag::kMaxNodeIO) return 0;
                 return node.grad_input_buffers[idx];
             },
             py::arg("index"));
}
