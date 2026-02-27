#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#include "grilly/cubemind/types.h"

namespace grilly {

/// Payload passed from the CPU text parser to the LLM training thread.
///
/// Contains both the geometric VSA representation (for surprise-momentum)
/// and the raw token IDs (for the standard transformer embedding layer).
struct TrainingPayload {
    cubemind::BitpackedVec vsa_state;           // GPU-ready bipolar encoding
    std::vector<uint32_t> llm_input_tokens;     // Token IDs for LLM embedding
    cubemind::EmotionState emotion;             // Cached for pre-computed surprise
    uint64_t sequence_id;                       // Monotonic ordering for debugging

    // SVC triple for WorldModel coherence checks.
    // Carried from ParsedDocument through the queue to the consumer.
    std::string svc_subject;
    std::string svc_verb;
    std::string svc_complement;
};

/// Bounded, thread-safe MPSC (Multi-Producer, Single-Consumer) queue.
///
/// Uses std::mutex + std::condition_variable for safe handoff between
/// the background data loader thread(s) and the main GPU training thread.
///
/// Bounded capacity provides backpressure: if the data loader runs faster
/// than the GPU can consume, push() blocks instead of eating unbounded RAM.
///
/// Memory layout:
///   - Each TrainingPayload is ~1.3 KB (1280B bitpacked + 40B metadata)
///   - At capacity=1024: ~1.3 MB queue memory
///   - At capacity=4096: ~5.2 MB queue memory
///
template <typename T = TrainingPayload>
class ThreadSafeQueue {
public:
    /// @param max_capacity Maximum items before push() blocks (0 = unbounded)
    explicit ThreadSafeQueue(size_t max_capacity = 1024)
        : max_capacity_(max_capacity), shutdown_(false) {}

    /// Push a payload onto the queue. Blocks if queue is at capacity.
    /// Returns false if the queue has been shut down.
    bool push(T payload) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Backpressure: wait until there's room or shutdown
        if (max_capacity_ > 0) {
            full_cond_.wait(lock, [this]() {
                return queue_.size() < max_capacity_ || shutdown_;
            });
        }

        if (shutdown_) return false;

        queue_.push(std::move(payload));
        total_pushed_++;
        empty_cond_.notify_one();  // Wake up sleeping consumer
        return true;
    }

    /// Pop the next payload. Blocks until data is available.
    /// Returns false if the queue has been shut down AND is empty.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Sleep efficiently until the data loader provides a payload
        empty_cond_.wait(lock, [this]() {
            return !queue_.empty() || shutdown_;
        });

        if (queue_.empty() && shutdown_) return false;

        out = std::move(queue_.front());
        queue_.pop();
        total_popped_++;

        // Release backpressure: wake up blocked producer
        if (max_capacity_ > 0) {
            full_cond_.notify_one();
        }

        return true;
    }

    /// Non-blocking try_pop. Returns false if queue is empty.
    bool try_pop(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;

        out = std::move(queue_.front());
        queue_.pop();
        total_popped_++;

        if (max_capacity_ > 0) {
            full_cond_.notify_one();
        }
        return true;
    }

    /// Signal shutdown. Unblocks all waiting threads.
    /// After shutdown, push() returns false and pop() drains remaining items.
    void shutdown() {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
        empty_cond_.notify_all();
        full_cond_.notify_all();
    }

    /// Current number of items in the queue.
    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t capacity() const { return max_capacity_; }
    uint64_t total_pushed() const { return total_pushed_; }
    uint64_t total_popped() const { return total_popped_; }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable empty_cond_;   // Signaled when item added
    std::condition_variable full_cond_;    // Signaled when item removed
    size_t max_capacity_;
    bool shutdown_;
    uint64_t total_pushed_ = 0;
    uint64_t total_popped_ = 0;
};

}  // namespace grilly
