/**
 * bindings_channels.cpp — pybind11 bindings for grilly channels.
 *
 * Exposes InProcessChannel to Python with numpy-friendly message passing.
 * When pybind11_protobuf is available, uses native proto casters.
 * Otherwise, falls back to bytes-based serialization.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "grilly/channels/channel.h"

namespace py = pybind11;

void bind_channels(py::module_& m) {
    using namespace grilly::channels;

    // MessageType enum
    py::enum_<MessageType>(m, "MessageType")
        .value("TENSOR_DATA", MessageType::TENSOR_DATA)
        .value("SPIKE_TRAIN", MessageType::SPIKE_TRAIN)
        .value("EXPERT_WEIGHTS", MessageType::EXPERT_WEIGHTS)
        .value("EXPERT_UPDATE", MessageType::EXPERT_UPDATE)
        .value("ROUTE_REQUEST", MessageType::ROUTE_REQUEST)
        .value("ROUTE_RESPONSE", MessageType::ROUTE_RESPONSE)
        .value("MEMORY_CAPSULE", MessageType::MEMORY_CAPSULE)
        .value("MEMORY_QUERY", MessageType::MEMORY_QUERY)
        .value("MEMORY_RESULT", MessageType::MEMORY_RESULT)
        .value("TELEMETRY_EVENT", MessageType::TELEMETRY_EVENT)
        .value("NEUROCHEM_STATE", MessageType::NEUROCHEM_STATE)
        .value("TRAIN_STEP_REQUEST", MessageType::TRAIN_STEP_REQUEST)
        .value("TRAIN_STEP_RESULT", MessageType::TRAIN_STEP_RESULT);

    // MessageEnvelope
    py::class_<MessageEnvelope>(m, "MessageEnvelope")
        .def(py::init<>())
        .def_readwrite("type", &MessageEnvelope::type)
        .def_readwrite("timestamp_ns", &MessageEnvelope::timestamp_ns)
        .def_readwrite("sender_id", &MessageEnvelope::sender_id)
        .def_property("payload",
            [](const MessageEnvelope& e) {
                return py::bytes(reinterpret_cast<const char*>(e.payload.data()),
                                 e.payload.size());
            },
            [](MessageEnvelope& e, py::bytes data) {
                std::string s = data;
                e.payload.assign(s.begin(), s.end());
            })
        .def_property_readonly("size", &MessageEnvelope::size);

    // InProcessChannel
    py::class_<InProcessChannel>(m, "InProcessChannel")
        .def(py::init<const std::string&, size_t>(),
             py::arg("name") = "default",
             py::arg("max_queue_size") = 10000)
        .def("send", &InProcessChannel::send)
        .def("receive", &InProcessChannel::receive)
        .def("has_messages", &InProcessChannel::has_messages)
        .def("queue_size", &InProcessChannel::queue_size)
        .def("clear", &InProcessChannel::clear)
        .def("name", &InProcessChannel::name)

        // Convenience: send numpy array as TensorData
        .def("send_tensor", [](InProcessChannel& ch,
                                py::array_t<float> arr,
                                const std::string& sender_id) {
            auto info = arr.request();
            MessageEnvelope env;
            env.type = MessageType::TENSOR_DATA;
            env.sender_id = sender_id;
            env.payload.assign(
                reinterpret_cast<uint8_t*>(info.ptr),
                reinterpret_cast<uint8_t*>(info.ptr) + info.size * sizeof(float));
            ch.send(std::move(env));
        }, py::arg("array"), py::arg("sender_id") = "python")

        // Convenience: receive as numpy array
        .def("receive_tensor", [](InProcessChannel& ch) -> py::object {
            auto msg = ch.receive();
            if (msg.payload.empty()) return py::none();
            size_t n_floats = msg.payload.size() / sizeof(float);
            auto result = py::array_t<float>(n_floats);
            auto buf = result.request();
            std::memcpy(buf.ptr, msg.payload.data(), msg.payload.size());
            return result;
        })

        // Convenience: send spike train (timesteps × neurons flattened)
        .def("send_spikes", [](InProcessChannel& ch,
                                py::array_t<float> spikes,
                                uint32_t n_neurons,
                                uint32_t n_timesteps,
                                const std::string& sender_id) {
            auto info = spikes.request();
            MessageEnvelope env;
            env.type = MessageType::SPIKE_TRAIN;
            env.sender_id = sender_id;
            // Pack header: n_neurons (4 bytes) + n_timesteps (4 bytes) + data
            env.payload.resize(8 + info.size * sizeof(float));
            std::memcpy(env.payload.data(), &n_neurons, 4);
            std::memcpy(env.payload.data() + 4, &n_timesteps, 4);
            std::memcpy(env.payload.data() + 8, info.ptr, info.size * sizeof(float));
            ch.send(std::move(env));
        }, py::arg("spikes"), py::arg("n_neurons"), py::arg("n_timesteps"),
           py::arg("sender_id") = "python");
}
