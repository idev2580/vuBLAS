#include "vucol/ShaderPipeline.hpp"
#include <vublas/ops/DotProductNaive.hpp>
#include <DotProductNaiveFP32_SPIRV>

namespace vublas{
    DotProductNaiveFP32::DotProductNaiveFP32(
        vucol::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = DotProductNaiveFP32_SPIRV,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
                {2, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(BinaryReductionArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
                {1, vucol::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan DotProductNaiveFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const BinaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, inputs[1], vucol::BufferAccess::Read},
                {2, outputs[0], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan DotProductNaiveFP32::operator()(
        BufferView a,
        BufferView b,
        BufferView out,
        const BinaryReductionArguments& args
    ){
        std::vector<BufferView> inputs = {a, b};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {out};
        return this->execute(inputs, inouts, outputs, &args, sizeof(BinaryReductionArguments));
    }
}
