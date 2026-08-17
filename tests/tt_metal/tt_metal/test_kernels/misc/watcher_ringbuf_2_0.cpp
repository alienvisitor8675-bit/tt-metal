// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TENSIX cores only; ethernet and DRAM use watcher_ringbuf.cpp.

#include <cstdint>
#include "api/debug/ring_buffer.h"
#include "experimental/kernel_args.h"
#if defined(ARCH_QUASAR) || defined(ARCH_BLACKHOLE)
// get_hw_thread_idx(), not get_my_thread_id(), must match ring_buffer.h's MPSC write_id encoding.
#include "internal/hw_thread.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#endif

void kernel_main() {
    constexpr uint32_t num_pushes = get_arg(args::num_pushes);

#if defined(COMPILE_FOR_DM) || (defined(COMPILE_FOR_TRISC) && defined(ARCH_QUASAR) && defined(MULTI_DM_TEST))
    // Quasar only: WH/BH DM cores use COMPILE_FOR_BRISC/COMPILE_FOR_NCRISC instead, and WH/BH
    // TRISCs are handled below. Multi-writer MPSC test: every launched DM/TRISC pushes with no
    // per-thread filter (the point is many concurrent writers); the single-DM test filters down
    // to the one specified DM.
    uint32_t thread_idx = internal_::get_hw_thread_idx();
#if defined(COMPILE_FOR_DM) && !defined(MULTI_DM_TEST)
    // Single-DM test: only the specified DM runs.
    constexpr uint32_t dm_id = get_arg(args::dm_id);
    if (dm_id != thread_idx) {
        return;
    }
#endif
    for (uint32_t seq = 0; seq < num_pushes; seq++) {
        WATCHER_RING_BUFFER_PUSH((thread_idx << 16) | seq);
    }
    return;
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) ||      \
    (defined(UCK_CHLKC_UNPACK) && defined(WATCHER_RINGBUF_TRISC0)) || \
    (defined(UCK_CHLKC_MATH) && defined(WATCHER_RINGBUF_TRISC1)) ||   \
    (defined(UCK_CHLKC_PACK) && defined(WATCHER_RINGBUF_TRISC2))
#if defined(ARCH_QUASAR) || defined(ARCH_BLACKHOLE)
    // Quasar/BH MPSC: use HAL thread_idx.
    uint32_t thread_idx = internal_::get_hw_thread_idx();
    for (uint32_t seq = 0; seq < num_pushes; seq++) {
        WATCHER_RING_BUFFER_PUSH((thread_idx << 16) | seq);
    }
#else
    // WH SPSC: use idx pattern, compile-time filter ensures only matching TRISC runs.
    for (uint32_t idx = 0; idx < num_pushes; idx++) {
        WATCHER_RING_BUFFER_PUSH((idx + 1) + (idx << 16));
    }
#endif
#endif
}
