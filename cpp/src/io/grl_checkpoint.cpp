#include "grilly/io/grl_checkpoint.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace grilly::io {

static void pack_u64(uint8_t* dst, uint64_t v) {
    std::memcpy(dst, &v, 8);
}

static uint64_t unpack_u64(const uint8_t* src) {
    uint64_t v;
    std::memcpy(&v, src, 8);
    return v;
}

bool grl_write_file(const std::string& path,
                    const std::string& metadata_json,
                    const std::string& index_json,
                    const std::vector<uint8_t>& payload) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs)
        return false;

    const uint64_t meta_off = kGrlHeaderSize;
    const uint64_t meta_len = metadata_json.size();
    const uint64_t idx_off = meta_off + meta_len;
    const uint64_t idx_len = index_json.size();
    const uint64_t pay_off = idx_off + idx_len;
    const uint64_t pay_len = payload.size();

    uint8_t header[kGrlHeaderSize] = {};
    header[0] = 'G';
    header[1] = 'R';
    header[2] = 'L';
    header[3] = 'Y';
    uint16_t ver = kGrlFormatVersion;
    uint16_t flags = 0;
    uint32_t reserved = 0;
    std::memcpy(header + 4, &ver, 2);
    std::memcpy(header + 6, &flags, 2);
    std::memcpy(header + 8, &reserved, 4);
    pack_u64(header + 12, meta_off);
    pack_u64(header + 20, meta_len);
    pack_u64(header + 28, idx_off);
    pack_u64(header + 36, idx_len);
    pack_u64(header + 44, pay_off);
    pack_u64(header + 52, pay_len);

    ofs.write(reinterpret_cast<const char*>(header), kGrlHeaderSize);
    ofs.write(metadata_json.data(), static_cast<std::streamsize>(metadata_json.size()));
    ofs.write(index_json.data(), static_cast<std::streamsize>(index_json.size()));
    if (!payload.empty())
        ofs.write(reinterpret_cast<const char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(ofs);
}

bool grl_read_file(const std::string& path,
                   std::string& metadata_json,
                   std::string& index_json,
                   std::vector<uint8_t>& payload) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs)
        return false;
    const auto end = ifs.tellg();
    ifs.seekg(0);
    if (end < static_cast<std::streamoff>(kGrlHeaderSize))
        return false;

    uint8_t header[kGrlHeaderSize];
    ifs.read(reinterpret_cast<char*>(header), kGrlHeaderSize);
    if (ifs.gcount() != static_cast<std::streamsize>(kGrlHeaderSize))
        return false;
    if (header[0] != 'G' || header[1] != 'R' || header[2] != 'L' || header[3] != 'Y')
        return false;
    uint16_t ver = 0;
    std::memcpy(&ver, header + 4, 2);
    if (ver != kGrlFormatVersion)
        throw std::runtime_error("GRL: unsupported format version");

    const uint64_t meta_off = unpack_u64(header + 12);
    const uint64_t meta_len = unpack_u64(header + 20);
    const uint64_t idx_off = unpack_u64(header + 28);
    const uint64_t idx_len = unpack_u64(header + 36);
    const uint64_t pay_off = unpack_u64(header + 44);
    const uint64_t pay_len = unpack_u64(header + 52);

    if (meta_off + meta_len != idx_off || idx_off + idx_len != pay_off)
        return false;
    if (end < static_cast<std::streamoff>(pay_off + pay_len))
        return false;

    metadata_json.resize(static_cast<size_t>(meta_len));
    index_json.resize(static_cast<size_t>(idx_len));
    payload.resize(static_cast<size_t>(pay_len));

    ifs.seekg(static_cast<std::streamoff>(meta_off));
    ifs.read(metadata_json.data(), static_cast<std::streamsize>(meta_len));
    ifs.read(index_json.data(), static_cast<std::streamsize>(idx_len));
    if (pay_len > 0)
        ifs.read(reinterpret_cast<char*>(payload.data()),
                 static_cast<std::streamsize>(pay_len));

    return static_cast<bool>(ifs) || (pay_len == 0 && meta_len == 0 && idx_len == 0);
}

}  // namespace grilly::io
