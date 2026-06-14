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
        .export_values();
