#pragma once
#include "vucol/Buffer.hpp"
#include "vucol/Context.hpp"
#include <vublas/ops/Gemv.hpp>

namespace vublas{
    class GemvNaiveFP32: public Gemv{
        public:
        GemvNaiveFP32(
            vucol::Context& ctx,
            uint32_t subgroup_tile_m,
            uint32_t subgroup_tile_n,
            uint32_t subgroup_tile_p,
            uint32_t subgroup_tile_cnt_m,
            uint32_t subgroup_tile_cnt_p,
            uint32_t shared_tile_n_multiplier,
            uint32_t reg_tile_m,
            uint32_t k_unroll,
            uint32_t reg_tile_p
        );
    };
}
