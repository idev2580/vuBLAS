#include <vublas/ops/GemvOutPlaceNaiveTemplate.hpp>

namespace vublas{
    GemvOutPlaceNaiveTemplateFP32::GemvOutPlaceNaiveTemplateFP32(
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
    ):GemvOutPlace(ctx){
        gemm = std::make_unique<GemmOutPlaceNaiveTemplateFP32>(
            ctx, epilogueSource,
            subgroup_tile_m, subgroup_tile_n, subgroup_tile_p,
            subgroup_tile_cnt_m, subgroup_tile_cnt_p,
            shared_tile_n_multiplier, reg_tile_m, k_unroll, reg_tile_p
        );
    }
}
