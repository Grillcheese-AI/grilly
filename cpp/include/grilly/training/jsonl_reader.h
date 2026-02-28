#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "grilly/training/pipeline.h"

namespace grilly {
namespace training {

/// Streaming JSONL reader for pretraining data.
///
/// Reads one line at a time from a JSONL file and extracts the fields
/// needed for VSA encoding: "lemmas" (tokens) and "deps" (dependency roles).
///
/// This avoids creating 491K Python ParsedDocument objects (~2GB heap)
/// by parsing directly in the C++ producer thread.
///
/// Expected JSONL format per line:
///   {"lemmas": ["give", "three", "tip"], "deps": ["ROOT", "nummod", "dobj"], ...}
///
class JsonlReader {
public:
    /// Open a JSONL file for streaming.
    /// @param path  Path to the .jsonl file
    /// @return true if the file was opened successfully
    bool open(const std::string& path) {
        file_.open(path, std::ios::in);
        path_ = path;
        line_number_ = 0;
        return file_.is_open();
    }

    /// Read the next line and parse it into a ParsedDocument.
    /// @param doc  Output document (tokens, dependency_roles, positions)
    /// @return true if a document was read, false at EOF or error
    bool next(ParsedDocument& doc) {
        std::string line;
        while (std::getline(file_, line)) {
            ++line_number_;

            // Skip empty lines
            if (line.empty() || line[0] == '\n') continue;

            try {
                auto j = nlohmann::json::parse(line);

                // Extract lemmas → tokens
                doc.tokens.clear();
                if (j.contains("lemmas") && j["lemmas"].is_array()) {
                    for (const auto& tok : j["lemmas"]) {
                        doc.tokens.push_back(tok.get<std::string>());
                    }
                }

                // Extract deps → dependency_roles
                doc.dependency_roles.clear();
                if (j.contains("deps") && j["deps"].is_array()) {
                    for (const auto& dep : j["deps"]) {
                        doc.dependency_roles.push_back(dep.get<std::string>());
                    }
                }

                // Generate positions (0, 1, 2, ...)
                doc.positions.clear();
                for (uint32_t i = 0; i < doc.tokens.size(); ++i) {
                    doc.positions.push_back(i);
                }

                // Extract SVC triple: {"svc": {"s": "...", "v": "...", "c": "..."}}
                doc.svc_subject.clear();
                doc.svc_verb.clear();
                doc.svc_complement.clear();
                if (j.contains("svc") && j["svc"].is_object()) {
                    const auto& svc = j["svc"];
                    if (svc.contains("s") && svc["s"].is_string())
                        doc.svc_subject = svc["s"].get<std::string>();
                    if (svc.contains("v") && svc["v"].is_string())
                        doc.svc_verb = svc["v"].get<std::string>();
                    if (svc.contains("c") && svc["c"].is_string())
                        doc.svc_complement = svc["c"].get<std::string>();
                }

                // Clear LLM token IDs (Phase 2 will populate these)
                doc.llm_token_ids.clear();

                return true;

            } catch (const nlohmann::json::parse_error&) {
                // Skip malformed lines
                ++parse_errors_;
                continue;
            }
        }

        return false;  // EOF
    }

    /// Close the file.
    void close() {
        if (file_.is_open()) file_.close();
    }

    /// Number of lines read so far.
    uint64_t lines_read() const { return line_number_; }

    /// Number of parse errors encountered.
    uint64_t parse_errors() const { return parse_errors_; }

    /// Path of the currently open file.
    const std::string& path() const { return path_; }

private:
    std::ifstream file_;
    std::string path_;
    uint64_t line_number_ = 0;
    uint64_t parse_errors_ = 0;
};

}  // namespace training
}  // namespace grilly
