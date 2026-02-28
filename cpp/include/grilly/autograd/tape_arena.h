#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

namespace grilly {
namespace autograd {

// Forward declaration — Node is defined in autograd.h but the arena
// needs to thread the Wengert list through it.
struct Node;

// ── TapeArena: Bump Allocator + Wengert List ────────────────────────────
//
// Production autograd engines (PyTorch's c10, JAX's jaxpr) avoid per-node
// heap allocation because:
//   1. Standard allocators (malloc/new) take OS-level locks — serializing
//      what should be a lock-free forward pass
//   2. Heap fragmentation from millions of small nodes (~64-128 bytes each)
//      destroys L1/L2 cache locality
//   3. Freeing nodes one-by-one after backward() costs O(N) destructor calls
//
// The TapeArena solves all three:
//   - Allocation is a pointer bump: ~1 ns, no locks, no syscalls
//   - All nodes live in one contiguous block: perfect cache locality
//   - reset() snaps the pointer to zero: O(1) graph clearing
//
// ── The Wengert List Insight ────────────────────────────────────────────
//
// Because operations are recorded sequentially during the forward pass,
// the allocation order IS a valid topological order. You can't allocate
// a MatMul node before its Linear input exists. This means:
//   - No DFS/BFS topological sort needed
//   - No edge list, no child-count hash maps
//   - Backward = walk the intrusive linked list in reverse
//
// The arena tracks `tail` — the most recently allocated node. Each node
// stores `prev_in_tape` pointing to the node allocated before it.
// backward() simply follows: tail → prev → prev → ... → nullptr
//
// ── Memory Alignment ────────────────────────────────────────────────────
//
// Autograd nodes interact with:
//   - Vulkan buffer handles (VkBuffer = uint64_t, needs 8-byte alignment)
//   - SIMD/AVX bitpacked operations (need 16 or 32-byte alignment)
//   - GPU push constants (Vulkan spec requires 4-byte alignment minimum)
//
// allocate<T>() respects alignof(T) automatically.

/// Align a pointer address forward to the given power-of-2 alignment.
/// e.g., align_forward(0x1003, 16) → 0x1010
inline uintptr_t align_forward(uintptr_t ptr, size_t alignment) {
    uintptr_t a = static_cast<uintptr_t>(alignment);
    uintptr_t mask = a - 1;
    uintptr_t misalign = ptr & mask;
    if (misalign != 0) {
        ptr += a - misalign;
    }
    return ptr;
}

class TapeArena {
public:
    /// Default 64 MB — enough for ~500K nodes at 128 bytes each.
    /// A 12-layer transformer forward pass produces ~200-500 nodes,
    /// so 64 MB handles thousands of forward passes before overflow.
    static constexpr size_t kDefaultCapacity = 64 * 1024 * 1024;

    explicit TapeArena(size_t capacity = kDefaultCapacity)
        : capacity_(capacity), offset_(0), tail_(nullptr) {
        buffer_ = new (std::nothrow) uint8_t[capacity_];
        if (!buffer_) {
            throw std::bad_alloc();
        }
    }

    ~TapeArena() {
        delete[] buffer_;
    }

    TapeArena(const TapeArena&) = delete;
    TapeArena& operator=(const TapeArena&) = delete;
    TapeArena(TapeArena&&) = delete;
    TapeArena& operator=(TapeArena&&) = delete;

    /// O(1) allocation via pointer bumping + placement new.
    /// Does NOT thread the Wengert list — use allocate_node() for that.
    ///
    /// IMPORTANT: T must be trivially destructible or must not own
    /// dynamic host memory (no std::vector, no std::shared_ptr).
    template <typename T, typename... Args>
    T* allocate(Args&&... args) {
        uintptr_t current = reinterpret_cast<uintptr_t>(buffer_ + offset_);
        uintptr_t aligned = align_forward(current, alignof(T));
        size_t padding = aligned - current;

        size_t needed = padding + sizeof(T);
        if (offset_ + needed > capacity_) {
            throw std::bad_alloc();
        }

        offset_ += padding;
        void* ptr = buffer_ + offset_;
        offset_ += sizeof(T);

        return new (ptr) T(std::forward<Args>(args)...);
    }

    /// Allocate a Node subclass AND thread it into the Wengert list.
    /// This is the primary allocation method during the forward pass.
    /// Sets node->prev_in_tape = current tail, then updates tail.
    template <typename T, typename... Args>
    T* allocate_node(Args&&... args);
    // Defined after Node is complete (in autograd.h) to avoid circular dependency.

    /// Allocate raw bytes (aligned). Returns nullptr on OOM.
    void* allocate_raw(size_t bytes, size_t alignment = 8) {
        uintptr_t current = reinterpret_cast<uintptr_t>(buffer_ + offset_);
        uintptr_t aligned = align_forward(current, alignment);
        size_t padding = aligned - current;

        size_t needed = padding + bytes;
        if (offset_ + needed > capacity_) {
            return nullptr;
        }

        offset_ += padding;
        void* ptr = buffer_ + offset_;
        offset_ += bytes;
        return ptr;
    }

    /// O(1) graph clearing — resets bump pointer AND Wengert list head.
    ///
    /// WARNING: Destructors of allocated objects are NOT called.
    /// This is intentional: arena objects must not own heap memory.
    /// GPU buffers are managed by BufferPool, not by nodes.
    void reset() {
        offset_ = 0;
        tail_ = nullptr;
    }

    /// Save/restore for nested scopes (gradient checkpointing).
    size_t save_offset() const { return offset_; }
    Node* save_tail() const { return tail_; }
    void restore(size_t saved_offset, Node* saved_tail) {
        if (saved_offset <= offset_) {
            offset_ = saved_offset;
            tail_ = saved_tail;
        }
    }

    /// The most recently allocated node — start here for backward().
    Node* tail() const { return tail_; }
    void set_tail(Node* t) { tail_ = t; }

    size_t bytes_used() const { return offset_; }
    size_t capacity() const { return capacity_; }
    float utilization() const {
        return capacity_ > 0 ? static_cast<float>(offset_) / capacity_ : 0.0f;
    }

private:
    uint8_t* buffer_;
    size_t capacity_;
    size_t offset_;
    Node* tail_;    // Head of the Wengert list (most recent node)
};

}  // namespace autograd
}  // namespace grilly
