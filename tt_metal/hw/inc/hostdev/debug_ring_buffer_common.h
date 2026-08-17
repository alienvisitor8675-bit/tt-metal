// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

constexpr int16_t DEBUG_RING_BUFFER_STARTING_INDEX = -1;
constexpr int DEBUG_RING_BUFFER_SPSC_ELEMENTS = 32;

struct debug_spsc_ring_buf_msg_t {
    int16_t current_ptr;
    uint16_t wrapped;
    uint32_t data[DEBUG_RING_BUFFER_SPSC_ELEMENTS];
};

// head is shared by every writer, so entries evict in push order, not per-writer: capacity must
// cover all concurrent writers (Quasar 22, Blackhole 5) or a writer's entries are lost.
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR = 128;
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE = 32;

struct debug_mpsc_ring_buf_slot_t {
    uint32_t data;
    uint32_t write_id;  // thread_idx + 1; 0 means never written
};

// Quasar's head lives here rather than L1: DM and TRISC atomics run in different coherence
// domains. Distinct register file from the Sync Unit semaphores LLK drives, so indices don't clash.
constexpr uint32_t WATCHER_RING_BUF_SEMAPHORE = 31;

template <int Capacity>
struct debug_mpsc_ring_buf_msg_tmpl_t {
    uint32_t head;
    uint8_t _pad[60];  // Pad to 64-byte cache line
    debug_mpsc_ring_buf_slot_t slots[Capacity];
};

using debug_mpsc_ring_buf_msg_quasar_t = debug_mpsc_ring_buf_msg_tmpl_t<DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR>;
using debug_mpsc_ring_buf_msg_blackhole_t = debug_mpsc_ring_buf_msg_tmpl_t<DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE>;

// Host doesn't know the target arch at compile time; it reads through the largest variant.
static_assert(
    DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE <= DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR,
    "host view must alias the largest MPSC variant");
using debug_mpsc_ring_buf_view_t = debug_mpsc_ring_buf_msg_quasar_t;

// Device-side constants (debug_ring_buf_size is in core_config.h for codegen)
#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#if defined(ARCH_QUASAR)
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR;
using debug_mpsc_ring_buf_msg_t = debug_mpsc_ring_buf_msg_quasar_t;
#elif defined(ARCH_BLACKHOLE)
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE;
using debug_mpsc_ring_buf_msg_t = debug_mpsc_ring_buf_msg_blackhole_t;
#endif

#if defined(ARCH_QUASAR) || defined(ARCH_BLACKHOLE)
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS;
constexpr uint32_t DEBUG_RING_BUFFER_MASK = DEBUG_RING_BUFFER_MPSC_ELEMENTS - 1;
#else
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_SPSC_ELEMENTS;
#endif

#endif  // KERNEL_BUILD || FW_BUILD
