#include "vucol/ShaderPipeline.hpp"
#include <vublas/ops/Axpy.hpp>
#include <AxpyFP32_SPIRV>

namespace vublas{
    AxpyFP32::AxpyFP32(
        vucol::Context& ctx,
        uint32_t thread_num
    ):thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = AxpyFP32_SPIRV,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(AxpyArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
            }
        });
    }
    DispatchPlan AxpyFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* axpyArgs = static_cast<const AxpyArguments*>(args);
        const uint32_t group_cnt = axpyArgs->n / thread_num + (axpyArgs->n % thread_num == 0 ? 0 : 1);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, inouts[0], vucol::BufferAccess::ReadWrite},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = axpyArgs->b,
            .dispatchY = group_cnt,
            .dispatchZ = 1,
        };
    }
    DispatchPlan AxpyFP32::operator()(
        BufferView A,
        BufferView B,
        const AxpyArguments& args
    ){
        std::vector<BufferView> inputs = {A};
        std::vector<BufferView> inouts = {B};
        std::vector<BufferView> outputs = {};
        return this->execute(inputs, inouts, outputs, &args, sizeof(AxpyArguments));
    }
}
