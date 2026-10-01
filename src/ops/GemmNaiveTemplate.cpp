#include "ShaderTemplate.hpp"
#include "ShaderTemplates.hpp"
#include <vublas/ops/GemmNaiveTemplate.hpp>

namespace vublas{
    GemmNaiveTemplateFP32::GemmNaiveTemplateFP32(
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
    ):GemmNaiveTemplateFP32(
        ctx,
        detail::compileShaderTemplate(
            detail::gemmNaiveShaderTemplate(),
            epilogueSource,
            "GemmNaiveTemplateFP32.comp"
        ),
        subgroup_tile_m,
        subgroup_tile_n,
        subgroup_tile_p,
        subgroup_tile_cnt_m,
        subgroup_tile_cnt_p,
        shared_tile_n_multiplier,
        reg_tile_m,
        k_unroll,
        reg_tile_p
    ){}

    GemmNaiveTemplateFP32::GemmNaiveTemplateFP32(
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
    ):Gemm(
        ctx, spirv,
        subgroup_tile_m, subgroup_tile_n, subgroup_tile_p,
        subgroup_tile_cnt_m, subgroup_tile_cnt_p,
        shared_tile_n_multiplier, reg_tile_m, k_unroll, reg_tile_p
    ){}
}
