// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// L1MetaData id-free datacopy verification: the two ops (unpack+math copy, and pack) take an L1MetaData
// (data format + tile geometry as NTTPs, absolute L1 address as the only runtime state) -- NO CB id
// on the op surface. The register format is derived on-device from the L1 format. Legacy inits are
// kept so this run isolates the id-free OP path + address seam + infer fn. Identity bf16->bf16 output
// must be bit-identical to the legacy path.
#include "api/compute/experimental/tile_move_copy_spec.h"
#include "api/compute/experimental/pack_spec.h"

void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    // Named source accessor objects (CB id in the type -> descriptor folds). to_l1_descriptor(accessor)
    // is the source-decoupling seam: pass the object, not a CB id. Format + geometry fold/DCE; the L1
    // address is resolved per-iteration from the same accessor.
    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto in_desc = experimental::to_l1_descriptor(in_cb);
    constexpr auto out_desc = experimental::to_l1_descriptor(out_cb);
    using InInfo = experimental::L1MetaData<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    using OutInfo = experimental::L1MetaData<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        cb0.wait_front(1);
        cb16.reserve_back(1);
        experimental::copy_tile(InInfo(in_cb.read_address()), 0);

        tile_regs_commit();
        tile_regs_wait();

        experimental::pack_tile(OutInfo(out_cb.write_address()), 0);
        cb0.pop_front(1);
        cb16.push_back(1);

        tile_regs_release();
    }
}
