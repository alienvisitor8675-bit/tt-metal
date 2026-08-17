// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#include "hostdev/dev_msgs.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_RING_BUFFER) && !defined(FORCE_WATCHER_OFF)

// Ring buffer modes:
// - Quasar:    MPSC, slot claimed via a NEO cluster semaphore (DM and TRISC caches are not
//              coherent, so a RISC-V atomic on an L1 word would not serialize between them)
// - Blackhole: MPSC, slot claimed via a 32-bit RISC-V atomic on the in-mailbox head
// - Wormhole:  SPSC, no synchronization between RISCs

#if defined(ARCH_QUASAR) || defined(ARCH_BLACKHOLE)
#include "internal/hw_thread.h"
#include "risc_common.h"
#if defined(ARCH_QUASAR)
#include "tt_t6_trisc_map.h"
#endif

#if defined(ARCH_QUASAR)
constexpr uintptr_t watcher_ring_buf_sem =
    GLOBAL_REGS_BASE + SEMAPHORE_REGS_BASE + SEMAPHORE_REGS_STRIDE * WATCHER_RING_BUF_SEMAPHORE;
#endif

// Must be inline - DM stack is only 1KB, can't afford function call overhead
inline __attribute__((always_inline)) void push_to_ring_buffer(uint32_t val) {
    auto* wrapper = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    auto* buf = reinterpret_cast<debug_mpsc_ring_buf_msg_t*>(wrapper->data);

#if defined(ARCH_QUASAR)
    // A read at +4*(inc+8) posts `inc` and returns the pre-increment value.
    uint32_t pos = *reinterpret_cast<volatile uint32_t*>(watcher_ring_buf_sem + 4 * (1 + 8));
#else
    uint32_t pos = __atomic_fetch_add(&buf->head, 1, __ATOMIC_RELAXED);
#endif
    uint32_t idx = pos & DEBUG_RING_BUFFER_MASK;

    buf->slots[idx].data = val;

    uint32_t thread_idx = internal_::get_hw_thread_idx();
    __atomic_store_n(&buf->slots[idx].write_id, thread_idx + 1, __ATOMIC_RELEASE);
}

#else  // WH: SPSC ring buffer

inline __attribute__((always_inline)) void push_to_ring_buffer(uint32_t val) {
    auto* wrapper = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    auto* buf = reinterpret_cast<debug_spsc_ring_buf_msg_t tt_l1_ptr*>(wrapper->data);
    volatile tt_l1_ptr int16_t* curr_ptr = &buf->current_ptr;
    volatile tt_l1_ptr uint16_t* wrapped = &buf->wrapped;
    uint32_t* data = buf->data;

    // Bounds check, set to -1 to wrap since we increment before using
    if (*curr_ptr >= DEBUG_RING_BUFFER_ELEMENTS - 1) {
        *curr_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
        *wrapped = 1;
    }
    data[++(*curr_ptr)] = val;
}

#endif  // ARCH_QUASAR || ARCH_BLACKHOLE

// Quasar: hardware raises GLOBAL_SEMAPHORES/POST_ON_UNINITIALIZED if a semaphore is posted before
// it is initialized, and the register is not reachable from host, so DM0 firmware seeds it.
inline __attribute__((always_inline)) void init_ring_buffer() {
#if defined(ARCH_QUASAR)
    *reinterpret_cast<volatile uint32_t*>(watcher_ring_buf_sem) = 0;
#endif
}

#define WATCHER_RING_BUFFER_PUSH(x) push_to_ring_buffer(x)
#define WATCHER_RING_BUFFER_INIT() init_ring_buffer()
#else  // !defined(WATCHER_ENABLED)
#define WATCHER_RING_BUFFER_PUSH(x)
#define WATCHER_RING_BUFFER_INIT()
#endif  // defined(WATCHER_ENABLED)

#endif  // KERNEL_BUILD || FW_BUILD
