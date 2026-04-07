/// Pybind11 bindings for GRL (.grl) checkpoint I/O — implemented in C++ for
/// performance and a single canonical binary encoder/decoder.

#include "bindings_core.h"
#include "grilly/io/grl_checkpoint.h"

#include <string>
#include <vector>

void register_grl_ops(py::module_& m) {
    m.def(
        "grl_write_file",
        [](const std::string& path, const std::string& metadata_json,
           const std::string& index_json, py::bytes payload_bytes) {
            std::string pb = payload_bytes;
            std::vector<uint8_t> payload(pb.begin(), pb.end());
            if (!grilly::io::grl_write_file(path, metadata_json, index_json,
                                            payload)) {
                throw std::runtime_error("grl_write_file failed: " + path);
            }
        },
        py::arg("path"), py::arg("metadata_json"), py::arg("index_json"),
        py::arg("payload"),
        "Write a GRL v1 checkpoint (header + metadata JSON + tensor index JSON "
        "+ raw payload bytes). Matches Python utils/grl_checkpoint layout.");

    m.def(
        "grl_read_file",
        [](const std::string& path) {
            std::string metadata_json;
            std::string index_json;
            std::vector<uint8_t> payload;
            if (!grilly::io::grl_read_file(path, metadata_json, index_json,
                                           payload)) {
                throw std::runtime_error("grl_read_file failed: " + path);
            }
            return py::make_tuple(metadata_json, index_json,
                                  py::bytes(reinterpret_cast<const char*>(
                                                payload.data()),
                                            payload.size()));
        },
        py::arg("path"),
        "Read a GRL v1 file. Returns (metadata_json, index_json, payload_bytes).");
}
