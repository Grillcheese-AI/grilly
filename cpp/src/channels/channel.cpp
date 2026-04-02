/**
 * channel.cpp — InProcessChannel implementation.
 */

#include "grilly/channels/channel.h"

#include <chrono>
#include <stdexcept>

namespace grilly {
namespace channels {

InProcessChannel::InProcessChannel(const std::string& name, size_t max_queue_size)
    : name_(name), max_queue_size_(max_queue_size) {}

void InProcessChannel::send(MessageEnvelope envelope) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Set timestamp if not set
    if (envelope.timestamp_ns == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        envelope.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }

    // Notify listeners first (synchronous)
    auto it = listeners_.find(static_cast<uint8_t>(envelope.type));
    if (it != listeners_.end()) {
        for (auto& listener : it->second) {
            listener(envelope);
        }
    }

    // Queue the message (drop oldest if full)
    if (queue_.size() >= max_queue_size_) {
        queue_.pop();
    }
    queue_.push(std::move(envelope));
}

MessageEnvelope InProcessChannel::receive() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return MessageEnvelope{MessageType::TENSOR_DATA, 0, "", {}};
    }
    auto msg = std::move(queue_.front());
    queue_.pop();
    return msg;
}

bool InProcessChannel::has_messages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !queue_.empty();
}

void InProcessChannel::subscribe(MessageType type, ChannelListener listener) {
    std::lock_guard<std::mutex> lock(mutex_);
    listeners_[static_cast<uint8_t>(type)].push_back(std::move(listener));
}

size_t InProcessChannel::queue_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void InProcessChannel::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) queue_.pop();
}

}  // namespace channels
}  // namespace grilly
