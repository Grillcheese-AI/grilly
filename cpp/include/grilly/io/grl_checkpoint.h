#pragma once
/// GRL v1 checkpoint file layout (shared with Python ``utils/grl_checkpoint.py``).

#include <cstdint>
#include <string>
#include <vector>

namespace grilly::io {

inline constexpr uint32_t kGrlHeaderSize = 64;
inline constexpr uint16_t kGrlFormatVersion = 1;

/// Write a GRL v1 file: header | metadata UTF-8 | index JSON UTF-8 | payload bytes.
bool grl_write_file(const std::string& path,
                    const std::string& metadata_json,
                    const std::string& index_json,
                    const std::vector<uint8_t>& payload);

/// Read a GRL v1 file; on success returns true and fills out-arguments.
bool grl_read_file(const std::string& path,
                   std::string& metadata_json,
                   std::string& index_json,
                   std::vector<uint8_t>& payload);

}  // namespace grilly::io
