#pragma once
/// BufferRegistry — maps opaque uint32_t buffer_ids to GrillyBuffers.
///
/// The autograd graph (TensorRef, Node) addresses GPU memory by uint32_t
/// buffer_id. BufferPool, however, deals in GrillyBuffer value types and has
/// no id concept. This registry is the missing bridge: it assigns a stable
/// nonzero id to each GrillyBuffer and resolves id -> GrillyBuffer& so the
/// BackwardEngine can build descriptor sets for dispatch.
///
/// Two ownership classes:
///   - external: buffers owned elsewhere (forward activations/weights).
///       Registered for resolution only; never released by the registry.
///   - owned:    buffers the registry allocated from BufferPool (gradient
///       and temporary buffers). Released back to the pool on clear().
///   - persistent: buffers (weights + optimizer moments) that must survive
///       clear() with a STABLE id across training steps. Registered ONCE,
///       before any step-scoped buffer. Released only at destruction.
///
/// Lifecycle: step-scoped ids are step-scoped. clear() releases owned
/// step-scoped buffers and truncates the table back to the persistent prefix;
/// call it from TapeContext::begin() alongside arena_.reset(). Persistent
/// entries (ids 1..persistent_watermark_-1) keep their buffers and ids.
///
/// id 0 is reserved to mean "none" (matches TensorRef::none() and the
/// `== 0` invalid checks throughout the autograd code).

#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "grilly/buffer_pool.h"

namespace grilly {
namespace autograd {

class BufferRegistry {
public:
    /// How an allocated buffer should be backed when calling alloc().
    enum class Kind {
        Compute,   ///< DEVICE_LOCAL VRAM (fast GPU access) — default for grads.
        Readback,  ///< HOST_CACHED (fast CPU read-after-GPU-write).
        HostVisible ///< WC host-visible (CPU sequential write).
    };

    explicit BufferRegistry(BufferPool& pool) : pool_(pool) {
        // entries_[0] is the reserved "none" slot; never handed out.
        entries_.push_back(Entry{});
    }

    ~BufferRegistry() {
        clear();
        // release persistent owned buffers too (object is dying).
        for (size_t i = 1; i < persistent_watermark_; ++i) {
            if (entries_[i].owned) pool_.release(entries_[i].buf);
        }
    }

    BufferRegistry(const BufferRegistry&) = delete;
    BufferRegistry& operator=(const BufferRegistry&) = delete;

    /// Register a buffer owned elsewhere (forward activation/weight).
    /// Returns a fresh nonzero id. The registry will NOT release this buffer.
    uint32_t register_external(const GrillyBuffer& buf) {
        entries_.push_back(Entry{buf, /*owned=*/false});
        return static_cast<uint32_t>(entries_.size() - 1);
    }

    /// Allocate a new buffer from the pool and register it as owned.
    /// Released back to the pool on clear(). Returns a fresh nonzero id.
    uint32_t alloc(size_t bytes, Kind kind = Kind::Compute) {
        GrillyBuffer buf;
        switch (kind) {
            case Kind::Readback:    buf = pool_.acquireReadback(bytes); break;
            case Kind::HostVisible: buf = pool_.acquire(bytes); break;
            case Kind::Compute:
            default:                buf = pool_.acquireDeviceLocal(bytes); break;
        }
        entries_.push_back(Entry{buf, /*owned=*/true});
        return static_cast<uint32_t>(entries_.size() - 1);
    }

    /// Allocate a PERSISTENT buffer (weight / optimizer moment) that survives
    /// clear() with a stable id. Must be called before any step-scoped buffer
    /// exists (else the watermark would swallow them). Released at destruction.
    uint32_t alloc_persistent(size_t bytes, Kind kind = Kind::Compute) {
        if (entries_.size() != persistent_watermark_) {
            throw std::runtime_error(
                "BufferRegistry: alloc_persistent must precede step-scoped buffers");
        }
        GrillyBuffer buf;
        switch (kind) {
            case Kind::Readback:    buf = pool_.acquireReadback(bytes); break;
            case Kind::HostVisible: buf = pool_.acquire(bytes); break;
            case Kind::Compute:
            default:                buf = pool_.acquireDeviceLocal(bytes); break;
        }
        entries_.push_back(Entry{buf, /*owned=*/true, /*persistent=*/true});
        persistent_watermark_ = entries_.size();
        return static_cast<uint32_t>(entries_.size() - 1);
    }

    /// Resolve an id to its GrillyBuffer. Throws on invalid id (0 or OOB).
    GrillyBuffer& resolve(uint32_t id) {
        if (id == 0 || id >= entries_.size()) {
            throw std::runtime_error(
                "BufferRegistry: invalid buffer_id " + std::to_string(id));
        }
        return entries_[id].buf;
    }

    const GrillyBuffer& resolve(uint32_t id) const {
        if (id == 0 || id >= entries_.size()) {
            throw std::runtime_error(
                "BufferRegistry: invalid buffer_id " + std::to_string(id));
        }
        return entries_[id].buf;
    }

    /// True if id is valid and the registry owns (will release) its buffer.
    bool owns(uint32_t id) const {
        return id != 0 && id < entries_.size() && entries_[id].owned;
    }

    /// True if id refers to a live buffer.
    bool valid(uint32_t id) const {
        return id != 0 && id < entries_.size();
    }

    /// Number of registered buffers (excluding the reserved id-0 slot).
    size_t size() const { return entries_.size() - 1; }

    /// Upload host data into a registered buffer (staged for DEVICE_LOCAL).
    void upload(uint32_t id, const void* data, size_t bytes) {
        GrillyBuffer& buf = resolve(id);
        if (buf.mappedPtr) {
            pool_.upload(buf, static_cast<const float*>(data), bytes);
        } else {
            pool_.uploadStaged(buf, data, bytes);
        }
    }

    /// Download a registered buffer's contents into host memory, starting at
    /// srcOffset bytes (default 0 = whole-buffer read, the legacy behavior).
    void download(uint32_t id, void* out, size_t bytes, size_t srcOffset = 0) {
        GrillyBuffer& buf = resolve(id);
        if (buf.mappedPtr) {
            pool_.download(buf, static_cast<float*>(out), bytes, srcOffset);
        } else {
            pool_.downloadStaged(buf, out, bytes, srcOffset);
        }
    }

    /// Release step-scoped owned buffers back to the pool and truncate the
    /// table to the persistent prefix. Persistent entries (and their ids) and
    /// external step buffers' owners are untouched. Erasing the suffix of a
    /// deque keeps references to the surviving persistent prefix valid.
    void clear() {
        for (size_t i = persistent_watermark_; i < entries_.size(); ++i) {
            if (entries_[i].owned) {
                pool_.release(entries_[i].buf);
            }
        }
        entries_.erase(entries_.begin() + persistent_watermark_, entries_.end());
    }

private:
    struct Entry {
        GrillyBuffer buf{};
        bool owned = false;
        bool persistent = false;
    };

    BufferPool& pool_;
    // First index past the persistent prefix (== entries_.size() when only the
    // persistent + reserved id-0 slots exist). clear() truncates back to here.
    size_t persistent_watermark_ = 1;
    // deque, NOT vector: resolve() hands out GrillyBuffer& into this container and
    // callers hold those references across subsequent alloc() calls. deque::push_back
    // keeps existing element references valid; vector::push_back would reallocate and
    // dangle them (root cause of the resident-forward 0xC0000005 / invalid-VkBuffer crash).
    std::deque<Entry> entries_;
};

}  // namespace autograd
}  // namespace grilly
