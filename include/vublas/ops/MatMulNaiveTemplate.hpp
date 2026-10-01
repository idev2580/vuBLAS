#pragma once
#include <string_view>
#include <vector>
#include <vublas/ops/MatMul.hpp>

namespace vublas{
    class MatMulNaiveTemplateFP32: public MatMul{
        private:
        MatMulNaiveTemplateFP32(
            vucol::Context& ctx,
            std::vector<uint32_t> spirv,
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

        public:
        MatMulNaiveTemplateFP32(
            vucol::Context& ctx,
            std::string_view epilogueSource,
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
