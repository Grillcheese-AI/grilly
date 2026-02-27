#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace grilly {

// ── SystemProfile: Hardware-Aware Configuration ──────────────────────────
//
// Decouples hardware specs from code so the same binary runs on AMD (local)
// and Nvidia (A40/A100) with different tuning.  Loaded from profiles.json
// at runtime — no recompile needed when switching hardware.
//
// Usage:
//   auto profile = SystemProfile::load("profiles.json", "A40_MASSIVE");
//   VSACacheConfig cache_cfg;
//   cache_cfg.capacity = profile.maxCacheCapacity;
//   cache_cfg.dim      = profile.vsaDim;

struct SystemProfile {
    // Hardware identity
    std::string deviceName;
    uint32_t subgroupSize;    // 64 for AMD RDNA2, 32 for Nvidia Ampere

    // Memory arena
    size_t arenaSizeBytes;    // Stored in bytes (converted from GB in JSON)

    // VSA parameters
    uint32_t vsaDim;
    uint32_t maxCacheCapacity;
    uint32_t maxConstraintCapacity;
    float surpriseThreshold;
    float coherenceThreshold;

    // Reasoning
    uint32_t thinkingSteps;
    uint32_t batchSize;
    uint32_t workgroupSize;

    // Derived: entries_per_workgroup = workgroupSize / subgroupSize
    uint32_t entriesPerWG() const { return workgroupSize / subgroupSize; }

    // ── Minimal JSON loader (no external dependency) ─────────────────
    //
    // Parses the flat object structure of profiles.json without pulling
    // in nlohmann/json. Only handles the exact schema we need: a top-level
    // object mapping profile names to flat objects with string/number values.
    //
    // For anything more complex, use Python's json module and pass values
    // through pybind11 constructor arguments instead.

    static SystemProfile load(const std::string& path,
                              const std::string& profile_name) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error(
                "SystemProfile: cannot open " + path);
        }

        std::stringstream buf;
        buf << file.rdbuf();
        std::string json = buf.str();

        // Find the profile block
        auto profileStart = json.find("\"" + profile_name + "\"");
        if (profileStart == std::string::npos) {
            throw std::runtime_error(
                "SystemProfile: profile '" + profile_name + "' not found in " + path);
        }

        // Find the opening brace of this profile's object
        auto blockStart = json.find('{', profileStart);
        if (blockStart == std::string::npos) {
            throw std::runtime_error(
                "SystemProfile: malformed JSON for profile '" + profile_name + "'");
        }

        // Find matching closing brace
        int depth = 1;
        size_t blockEnd = blockStart + 1;
        while (blockEnd < json.size() && depth > 0) {
            if (json[blockEnd] == '{') ++depth;
            if (json[blockEnd] == '}') --depth;
            ++blockEnd;
        }

        std::string block = json.substr(blockStart, blockEnd - blockStart);

        SystemProfile p;
        p.deviceName            = extractString(block, "device_name");
        p.subgroupSize          = extractUint(block, "subgroup_size");
        p.arenaSizeBytes        = extractUint(block, "arena_size_gb")
                                  * size_t(1024) * 1024 * 1024;
        p.vsaDim                = extractUint(block, "vsa_dim");
        p.maxCacheCapacity      = extractUint(block, "max_cache_capacity");
        p.maxConstraintCapacity = extractUint(block, "max_constraint_capacity");
        p.surpriseThreshold     = extractFloat(block, "surprise_threshold");
        p.coherenceThreshold    = extractFloat(block, "coherence_threshold");
        p.thinkingSteps         = extractUint(block, "thinking_steps");
        p.batchSize             = extractUint(block, "batch_size");
        p.workgroupSize         = extractUint(block, "workgroup_size");

        return p;
    }

private:
    static std::string extractString(const std::string& block,
                                     const std::string& key) {
        auto pos = block.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        auto colon = block.find(':', pos);
        auto qStart = block.find('"', colon + 1);
        auto qEnd = block.find('"', qStart + 1);
        return block.substr(qStart + 1, qEnd - qStart - 1);
    }

    static uint32_t extractUint(const std::string& block,
                                const std::string& key) {
        auto pos = block.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        auto colon = block.find(':', pos);
        // Skip whitespace after colon to find the number
        size_t start = colon + 1;
        while (start < block.size() &&
               (block[start] == ' ' || block[start] == '\t')) {
            ++start;
        }
        return static_cast<uint32_t>(std::stoul(block.substr(start)));
    }

    static float extractFloat(const std::string& block,
                              const std::string& key) {
        auto pos = block.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0f;
        auto colon = block.find(':', pos);
        size_t start = colon + 1;
        while (start < block.size() &&
               (block[start] == ' ' || block[start] == '\t')) {
            ++start;
        }
        return std::stof(block.substr(start));
    }
};

}  // namespace grilly
