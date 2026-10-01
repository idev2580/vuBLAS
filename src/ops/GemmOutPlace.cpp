#include "vucol/ShaderPipeline.hpp"
#include <vublas/ops/GemmOutPlace.hpp>

namespace vublas{
    GemmOutPlace::GemmOutPlace(
        vucol::Context& ctx,
        std::span<const uint32_t> shaderBytecodes,
        uint32_t subgroup_tile_m,
        uint32_t subgroup_tile_n,
        uint32_t subgroup_tile_p,
        uint32_t subgroup_tile_cnt_m,
        uint32_t subgroup_tile_cnt_p,
        uint32_t shared_tile_n_multiplier,
        uint32_t reg_tile_m,
        uint32_t inner_tile_n,
        uint32_t reg_tile_p
    ):
        tile_m(subgroup_tile_cnt_m * subgroup_tile_m * reg_tile_m),
        tile_n(shared_tile_n_multiplier * subgroup_tile_n * inner_tile_n),
        tile_p(subgroup_tile_cnt_p * subgroup_tile_p * reg_tile_p){
        const uint32_t threadgroup_tile_m =
            subgroup_tile_cnt_m * subgroup_tile_m;
        const uint32_t threadgroup_tile_p =
            subgroup_tile_cnt_p * subgroup_tile_p;

        this->pipeline = ctx.createShaderPipeline({
            .spirv = shaderBytecodes,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
                {2, vucol::DescriptorType::StorageBuffer},
                {3, vucol::DescriptorType::StorageBuffer}
            },
            .pushConstantSize = sizeof(GemmOutPlaceArguments),
            .specConstants = {
                {0, vucol::specConstant(shared_tile_n_multiplier)},
                {1, vucol::specConstant(subgroup_tile_cnt_m)},
                {2, vucol::specConstant(subgroup_tile_cnt_p)},
                {3, vucol::specConstant(subgroup_tile_m)},
                {4, vucol::specConstant(subgroup_tile_n)},
                {5, vucol::specConstant(subgroup_tile_p)},
                {6, vucol::specConstant(threadgroup_tile_m)},
                {7, vucol::specConstant(threadgroup_tile_p)},
                {8, vucol::specConstant(reg_tile_m)},
                {9, vucol::specConstant(inner_tile_n)},
                {10, vucol::specConstant(reg_tile_p)}
            },
            .requiredSubgroupSize = 32,
        });
    }
    DispatchPlan GemmOutPlace::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* gemmArgs = static_cast<const GemmOutPlaceArguments*>(args);
        const uint32_t tile_r_size = tile_m;
        const uint32_t tile_c_size = tile_p;
        const uint32_t tiled_m = (gemmArgs->m / tile_r_size) + (gemmArgs->m % tile_r_size != 0);
        const uint32_t tiled_p = (gemmArgs->p / tile_c_size) + (gemmArgs->p % tile_c_size != 0);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, inputs[1], vucol::BufferAccess::Read},
                {2, inputs[2], vucol::BufferAccess::Read},
                {3, outputs[0], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = gemmArgs->b,
            .dispatchY = tiled_m,
            .dispatchZ = tiled_p,
        };
    }
    DispatchPlan GemmOutPlace::operator()(
        BufferView A,
        BufferView B,
        BufferView C,
        BufferView outC,
        const GemmOutPlaceArguments& args
    ){
        std::vector<BufferView> inputs = {A, B, C};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {outC};
        return this->execute(inputs, inouts, outputs, &args, sizeof(GemmOutPlaceArguments));
    }

    DispatchPlan GemmOutPlace::operator()(
        BufferView A,
        BufferView B,
        BufferView C,
        BufferView outC,
        const GemmArguments& args
    ){
        const GemmOutPlaceArguments outPlaceArgs = GemmOutPlaceArguments::sameOutputLayout(args);
        return (*this)(A, B, C, outC, outPlaceArgs);
    }
}
