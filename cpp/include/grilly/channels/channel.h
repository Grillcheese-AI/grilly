/**
 * grilly/channels/channel.h — Protobuf-based C++/Python channel interface.
 *
 * Provides zero-copy message passing between Vulkan compute (C++)
 * and CubeMind brain modules (Python) via protobuf serialization.
 *
 * Two modes:
 *   1. In-process: direct pybind11 calls with native proto casters
 *   2. IPC: shared memory or Unix domain sockets for multi-process
 *
 * Usage (C++ side):
 *   auto channel = grilly::InProcessChannel();
 *   channel.send<SpikeTrain>(spike_data);
 *   auto result = channel.receive<TensorData>();
 *
 * Usage (Python side, via pybind11):
 *   channel = grilly_core.InProcessChannel()
 *   channel.send_spike_train(spike_data)
 *   result = channel.receive_tensor()
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace grilly {
namespace channels {

/**
 * MessageType — identifies the protobuf message type without parsing.
 */
enum class MessageType : uint8_t {
    TENSOR_DATA = 0,
    SPIKE_TRAIN = 1,
    EXPERT_WEIGHTS = 2,
    EXPERT_UPDATE = 3,
    ROUTE_REQUEST = 4,
    ROUTE_RESPONSE = 5,
    MEMORY_CAPSULE = 6,
    MEMORY_QUERY = 7,
    MEMORY_RESULT = 8,
    TELEMETRY_EVENT = 9,
    NEUROCHEM_STATE = 10,
    TRAIN_STEP_REQUEST = 11,
    TRAIN_STEP_RESULT = 12,
};

/**
 * MessageEnvelope — header + serialized payload.
 */
struct MessageEnvelope {
    MessageType type;
    uint64_t timestamp_ns;
    std::string sender_id;
    std::vector<uint8_t> payload;  // serialized protobuf bytes

    size_t size() const { return payload.size(); }
};

/**
 * ChannelListener — callback interface for async message handling.
 */
using ChannelListener = std::function<void(const MessageEnvelope&)>;

/**
 * BaseChannel — abstract channel interface.
 */
class BaseChannel {
public:
    virtual ~BaseChannel() = default;

    /// Send a serialized message.
    virtual void send(MessageEnvelope envelope) = 0;

    /// Receive next message (blocking). Returns empty if no messages.
    virtual MessageEnvelope receive() = 0;

    /// Check if messages are available.
    virtual bool has_messages() const = 0;

    /// Register an async listener for a message type.
    virtual void subscribe(MessageType type, ChannelListener listener) = 0;

    /// Channel name for debugging.
    virtual std::string name() const = 0;
};

/**
 * InProcessChannel — thread-safe in-process message queue.
 *
 * For single-process use: C++ compute threads push messages,
 * Python pybind11 threads pull them. Zero serialization overhead
 * when combined with pybind11_protobuf native casters.
 */
class InProcessChannel : public BaseChannel {
public:
    explicit InProcessChannel(const std::string& name = "default",
                               size_t max_queue_size = 10000);

    void send(MessageEnvelope envelope) override;
    MessageEnvelope receive() override;
    bool has_messages() const override;
    void subscribe(MessageType type, ChannelListener listener) override;
    std::string name() const override { return name_; }

    /// Queue size for monitoring.
    size_t queue_size() const;

    /// Clear all queued messages.
    void clear();

private:
    std::string name_;
    size_t max_queue_size_;
    mutable std::mutex mutex_;
    std::queue<MessageEnvelope> queue_;
    std::unordered_map<uint8_t, std::vector<ChannelListener>> listeners_;
};

}  // namespace channels
}  // namespace grilly
