#include "grilly/training/pipeline.h"
#include "grilly/training/jsonl_reader.h"

#include <algorithm>
#include <chrono>

namespace grilly {
namespace training {

// ── Portable high-resolution clock ──────────────────────────────────────

static double now_seconds() {
    using Clock = std::chrono::high_resolution_clock;
    auto tp = Clock::now().time_since_epoch();
    return std::chrono::duration<double>(tp).count();
}

// ── DataLoader ──────────────────────────────────────────────────────────

DataLoader::DataLoader(cubemind::TextEncoder& encoder,
                       ThreadSafeQueue<TrainingPayload>& queue)
    : encoder_(encoder), queue_(queue) {}

void DataLoader::submit(const std::vector<ParsedDocument>& documents) {
    std::lock_guard<std::mutex> lock(doc_mutex_);
    documents_.insert(documents_.end(), documents.begin(), documents.end());
    stream_mode_ = false;
}

void DataLoader::submit_files(const std::vector<std::string>& paths) {
    std::lock_guard<std::mutex> lock(doc_mutex_);
    file_paths_ = paths;
    stream_mode_ = true;
}

void DataLoader::start() {
    if (running_.load()) return;  // Already running

    running_.store(true);
    stop_requested_.store(false);
    start_time_ = now_seconds();
    docs_encoded_.store(0);
    busy_time_.store(0.0);

    if (stream_mode_) {
        worker_ = std::thread(&DataLoader::worker_loop_files, this);
    } else {
        worker_ = std::thread(&DataLoader::worker_loop, this);
    }
}

void DataLoader::join() {
    if (worker_.joinable()) {
        worker_.join();
    }
}

void DataLoader::stop() {
    stop_requested_.store(true);
    queue_.shutdown();
    join();
}

PipelineStats DataLoader::stats() const {
    double elapsed = now_seconds() - start_time_;
    uint64_t encoded = docs_encoded_.load(std::memory_order_relaxed);

    PipelineStats s;
    s.documents_encoded = encoded;
    s.payloads_consumed = queue_.total_popped();
    s.queue_current_size = queue_.size();
    s.encoding_docs_per_sec = (elapsed > 0.0) ? encoded / elapsed : 0.0;
    s.elapsed_seconds = elapsed;
    double bt = busy_time_.load(std::memory_order_relaxed);
    s.producer_busy_pct = (elapsed > 0.0) ? (bt / elapsed) * 100.0 : 0.0;
    return s;
}

void DataLoader::worker_loop() {
    // Snapshot the documents to process
    std::vector<ParsedDocument> local_docs;
    {
        std::lock_guard<std::mutex> lock(doc_mutex_);
        local_docs = std::move(documents_);
    }

    for (size_t i = 0; i < local_docs.size(); ++i) {
        if (stop_requested_.load(std::memory_order_relaxed)) break;

        const auto& doc = local_docs[i];

        // ── Encode: TextEncoder (BLAKE3 roles + vsaBind + vsaBundle) ────
        //
        // This is where the SemanticAssigner's memoization cache pays off.
        // For the first few thousand documents, cache misses trigger the
        // 3M FLOP LSH projection. After that, the Zipfian distribution
        // ensures >92% of lookups are O(1) hash table hits.
        double t0 = now_seconds();

        cubemind::BitpackedVec vsa_state = encoder_.encode_sentence(
            doc.tokens, doc.dependency_roles, doc.positions);

        double t1 = now_seconds();
        busy_time_.store(busy_time_.load(std::memory_order_relaxed) + (t1 - t0),
                         std::memory_order_relaxed);

        // ── Build the TrainingPayload ───────────────────────────────────
        TrainingPayload payload;
        payload.vsa_state = std::move(vsa_state);
        payload.llm_input_tokens = doc.llm_token_ids;
        payload.emotion = {0.0f, 0.0f};  // Surprise computed by consumer
        payload.sequence_id = sequence_counter_.fetch_add(
            1, std::memory_order_relaxed);
        payload.svc_subject = doc.svc_subject;
        payload.svc_verb = doc.svc_verb;
        payload.svc_complement = doc.svc_complement;

        // ── Push to queue (may block if at capacity = backpressure) ─────
        if (!queue_.push(std::move(payload))) {
            break;  // Queue shut down
        }

        docs_encoded_.fetch_add(1, std::memory_order_relaxed);
    }

    // Signal to consumer: no more data coming
    queue_.shutdown();
    running_.store(false);
}

void DataLoader::worker_loop_files() {
    // Snapshot the file paths to process
    std::vector<std::string> local_paths;
    {
        std::lock_guard<std::mutex> lock(doc_mutex_);
        local_paths = std::move(file_paths_);
    }

    JsonlReader reader;
    ParsedDocument doc;

    for (const auto& path : local_paths) {
        if (stop_requested_.load(std::memory_order_relaxed)) break;

        if (!reader.open(path)) {
            continue;  // Skip files that can't be opened
        }

        while (reader.next(doc)) {
            if (stop_requested_.load(std::memory_order_relaxed)) break;

            // Skip very short sentences (< 3 tokens)
            if (doc.tokens.size() < 3) continue;

            // ── Encode: TextEncoder with lazy BLAKE3 caching ─────────
            double t0 = now_seconds();

            cubemind::BitpackedVec vsa_state = encoder_.encode_sentence(
                doc.tokens, doc.dependency_roles, doc.positions);

            double t1 = now_seconds();
            busy_time_.store(
                busy_time_.load(std::memory_order_relaxed) + (t1 - t0),
                std::memory_order_relaxed);

            // ── Build the TrainingPayload ────────────────────────────
            TrainingPayload payload;
            payload.vsa_state = std::move(vsa_state);
            payload.llm_input_tokens = doc.llm_token_ids;
            payload.emotion = {0.0f, 0.0f};
            payload.sequence_id = sequence_counter_.fetch_add(
                1, std::memory_order_relaxed);
            payload.svc_subject = std::move(doc.svc_subject);
            payload.svc_verb = std::move(doc.svc_verb);
            payload.svc_complement = std::move(doc.svc_complement);

            // ── Push to queue (backpressure if at capacity) ──────────
            if (!queue_.push(std::move(payload))) {
                break;  // Queue shut down
            }

            docs_encoded_.fetch_add(1, std::memory_order_relaxed);
        }

        reader.close();
    }

    // Signal to consumer: no more data coming
    queue_.shutdown();
    running_.store(false);
}

// ── TrainingPipeline ────────────────────────────────────────────────────

TrainingPipeline::TrainingPipeline(uint32_t dim, uint32_t ft_dim,
                                   size_t queue_depth)
    : encoder_(dim, ft_dim),
      assigner_(dim, ft_dim),
      queue_(queue_depth),
      loader_(encoder_, queue_) {}

TrainingPipeline::~TrainingPipeline() {
    stop();
}

void TrainingPipeline::start(const std::vector<ParsedDocument>& documents) {
    loader_.submit(documents);
    loader_.start();
}

void TrainingPipeline::start_with_files(const std::vector<std::string>& paths) {
    loader_.submit_files(paths);
    loader_.start();
}

bool TrainingPipeline::pop(TrainingPayload& out) {
    return queue_.pop(out);
}

bool TrainingPipeline::try_pop(TrainingPayload& out) {
    return queue_.try_pop(out);
}

void TrainingPipeline::stop() {
    loader_.stop();
}

void TrainingPipeline::join() {
    loader_.join();
}

PipelineStats TrainingPipeline::stats() const {
    return loader_.stats();
}

}  // namespace training
}  // namespace grilly
