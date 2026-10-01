#include "vucol/ShaderPipeline.hpp"
#include <vublas/ops/AxpyOutPlace.hpp>
#include <AxpyOutPlaceFP32_SPIRV>

namespace vublas{
    AxpyOutPlaceArguments AxpyOutPlaceArguments::sameOutputLayout(const AxpyArguments& args){
        return {
            .b = args.b,
            .n = args.n,
            .alpha = args.alpha,
            .a_b_stride = args.a_b_stride,
            .a_n_stride = args.a_n_stride,
            .b_b_stride = args.b_b_stride,
            .b_n_stride = args.b_n_stride,
            .out_b_b_stride = args.b_b_stride,
            .out_b_n_stride = args.b_n_stride
        };
    }

    void AxpyOutPlaceArguments::fromInPlace(const AxpyArguments& args){
        *this = sameOutputLayout(args);
    }

    AxpyOutPlaceFP32::AxpyOutPlaceFP32(
        vucol::Context& ctx,
        uint32_t thread_num
    ):thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = AxpyOutPlaceFP32_SPIRV,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
                {2, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(AxpyOutPlaceArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
            }
        });
    }

    DispatchPlan AxpyOutPlaceFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* axpyArgs = static_cast<const AxpyOutPlaceArguments*>(args);
        const uint32_t group_cnt =
            axpyArgs->n / thread_num + (axpyArgs->n % thread_num == 0 ? 0 : 1);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, inputs[1], vucol::BufferAccess::Read},
                {2, outputs[0], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = axpyArgs->b,
            .dispatchY = group_cnt,
            .dispatchZ = 1,
        };
    }

    DispatchPlan AxpyOutPlaceFP32::operator()(
        BufferView A,
        BufferView B,
        BufferView outB,
        const AxpyOutPlaceArguments& args
    ){
        std::vector<BufferView> inputs = {A, B};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {outB};
        return this->execute(inputs, inouts, outputs, &args, sizeof(AxpyOutPlaceArguments));
    }

    DispatchPlan AxpyOutPlaceFP32::operator()(
        BufferView A,
        BufferView B,
        BufferView outB,
        const AxpyArguments& args
    ){
        const AxpyOutPlaceArguments outPlaceArgs =
            AxpyOutPlaceArguments::sameOutputLayout(args);
        return (*this)(A, B, outB, outPlaceArgs);
    }
}
