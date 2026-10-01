#pragma once
#include <vublas/ops/GemmOutPlaceNaiveTemplate.hpp>
#include <vublas/ops/GemvOutPlace.hpp>

namespace vublas{
    class GemvOutPlaceNaiveTemplateFP32: public GemvOutPlace{
        public:
        GemvOutPlaceNaiveTemplateFP32(
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
