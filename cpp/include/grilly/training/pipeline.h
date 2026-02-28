#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "grilly/cubemind/semantic_assigner.h"
#include "grilly/cubemind/text_encoder.h"
#include "grilly/cubemind/types.h"
#include "grilly/thread_safe_queue.h"

namespace grilly {
namespace training {

/// A single document with pre-parsed linguistic structure.
///
/// In production, this is populated by spaCy (Python) or a C++ NLP library.
/// The DataLoader accepts these and encodes them into TrainingPayloads
/// on a background thread.
struct ParsedDocument {
    std::vector<std::string> tokens;              // Lemmatized words
    std::vector<std::string> dependency_roles;    // spaCy dep labels
    std::vector<uint32_t> positions;              // Token positions
    std::vector<uint32_t> llm_token_ids;          // Token IDs for LLM embedding

    // SVC triple from pre-parsed linguistic analysis.
    // Extracted from JSONL "svc": {"s": "dog", "v": "be", "c": "animal"}
    std::string svc_subject;
    std::string svc_verb;
    std::string svc_complement;
};

/// Pipeline statistics for monitoring throughput.
struct PipelineStats {
    uint64_t documents_encoded;     // Total documents processed by producer
    uint64_t payloads_consumed;     // Total payloads consumed by training loop
    uint64_t queue_current_size;    // Current items waiting in queue
    double encoding_docs_per_sec;   // Producer throughput
    double elapsed_seconds;         // Wall-clock time since start
    double producer_busy_pct;       // Fraction of time producer was encoding (vs blocked)
};

/// Background data loader (Producer).
///
/// Runs on a dedicated std::thread. For each ParsedDocument:
///   1. Uses TextEncoder to produce BitpackedVec (BLAKE3 roles + vsaBind + vsaBundle)
///   2. Pushes TrainingPayload to the ThreadSafeQueue
///
/// Because the TextEncoder internally uses SemanticAssigner's memoized fillers,
/// the encoding cost rapidly drops to near-zero as the vocabulary cache warms.
///
/// Usage:
///   DataLoader loader(encoder, queue, /*num_workers=*/1);
///   loader.submit(documents);
///   loader.start();
///   // ... training loop pops from queue ...
///   loader.join();
///
class DataLoader {
public:
    /// @param encoder  TextEncoder with pre-loaded semantic fillers
    /// @param queue    Thread-safe queue shared with the training loop
    DataLoader(cubemind::TextEncoder& encoder,
               ThreadSafeQueue<TrainingPayload>& queue);

    /// Submit a batch of pre-parsed documents for encoding.
    /// Can be called before or after start().
    void submit(const std::vector<ParsedDocument>& documents);

    /// Submit JSONL file paths for streaming.
    /// The worker thread reads and parses files directly in C++,
    /// avoiding 491K Python ParsedDocument allocations (~2GB heap).
    void submit_files(const std::vector<std::string>& paths);

    /// Start the background encoding thread.
    void start();

    /// Signal the loader to stop after processing all submitted documents.
    /// Blocks until the background thread finishes.
    void join();

    /// Signal immediate shutdown (may discard remaining documents).
    void stop();

    /// Check if the background thread is still running.
    bool running() const { return running_.load(std::memory_order_relaxed); }

    /// Get current pipeline statistics.
    PipelineStats stats() const;

private:
    cubemind::TextEncoder& encoder_;
    ThreadSafeQueue<TrainingPayload>& queue_;

    // Document buffer (producer reads from this)
    std::vector<ParsedDocument> documents_;
    std::vector<std::string> file_paths_;   // JSONL file paths for streaming mode
    bool stream_mode_ = false;              // true = read from files, false = read from documents_
    std::mutex doc_mutex_;

    // Background thread
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    // Stats
    std::atomic<uint64_t> docs_encoded_{0};
    std::atomic<uint64_t> sequence_counter_{0};
    double start_time_ = 0.0;
    std::atomic<double> busy_time_{0.0};

    // Worker entry points
    void worker_loop();           // Process pre-loaded documents
    void worker_loop_files();     // Stream from JSONL files
};

/// TrainingPipeline: Orchestrates the full producer-consumer loop.
///
/// This is the top-level entry point that owns:
///   - TextEncoder (with SemanticAssigner fillers)
///   - ThreadSafeQueue (bounded, backpressure-aware)
///   - DataLoader (background encoding thread)
///
/// The training loop (consumer) calls pop() to get the next
/// TrainingPayload, which contains both the VSA state for
/// surprise-momentum and the LLM token IDs for the forward pass.
///
class TrainingPipeline {
public:
    /// @param dim          VSA dimension (default 10240)
    /// @param ft_dim       Dense embedding dimension (default 300)
    /// @param queue_depth  Maximum payloads buffered (backpressure threshold)
    TrainingPipeline(uint32_t dim = 10240, uint32_t ft_dim = 300,
                     size_t queue_depth = 1024);

    ~TrainingPipeline();

    /// Access the TextEncoder for loading fillers, etc.
    cubemind::TextEncoder& encoder() { return encoder_; }

    /// Access the SemanticAssigner for registering float embeddings.
    cubemind::SemanticAssigner& assigner() { return assigner_; }

    /// Submit documents and start background encoding.
    void start(const std::vector<ParsedDocument>& documents);

    /// Start from JSONL file paths (zero Python allocation mode).
    /// The C++ producer thread reads and parses files directly.
    void start_with_files(const std::vector<std::string>& paths);

    /// Pop the next TrainingPayload (blocks until available).
    /// Returns false if pipeline has been shut down and queue is drained.
    bool pop(TrainingPayload& out);

    /// Non-blocking pop.
    bool try_pop(TrainingPayload& out);

    /// Shut down the pipeline gracefully.
    void stop();

    /// Wait for the data loader to finish all documents.
    void join();

    /// Pipeline statistics.
    PipelineStats stats() const;

    /// Queue depth (current items waiting).
    size_t queue_size() const { return queue_.size(); }

private:
    cubemind::TextEncoder encoder_;
    cubemind::SemanticAssigner assigner_;
    ThreadSafeQueue<TrainingPayload> queue_;
    DataLoader loader_;
};

}  // namespace training
}  // namespace grilly
