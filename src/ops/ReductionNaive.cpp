#include "vucol/ShaderPipeline.hpp"
#include <vublas/ops/ReductionNaive.hpp>
#include <AvgNaiveFP32_SPIRV>
#include <MaxNaiveFP32_SPIRV>
#include <MinNaiveFP32_SPIRV>
#include <SumNaiveFP32_SPIRV>

namespace vublas{
    MaxNaiveFP32::MaxNaiveFP32(
        vucol::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = MaxNaiveFP32_SPIRV,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
                {2, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(IndexedUnaryReductionArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
                {1, vucol::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan MaxNaiveFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const IndexedUnaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, outputs[0], vucol::BufferAccess::Write},
                {2, outputs[1], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan MaxNaiveFP32::operator()(
        BufferView a,
        BufferView outValue,
        BufferView outIndex,
        const IndexedUnaryReductionArguments& args
    ){
        std::vector<BufferView> inputs = {a};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {outValue, outIndex};
        return this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(IndexedUnaryReductionArguments)
        );
    }

    MinNaiveFP32::MinNaiveFP32(
        vucol::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = MinNaiveFP32_SPIRV,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
                {2, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(IndexedUnaryReductionArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
                {1, vucol::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan MinNaiveFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const IndexedUnaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, outputs[0], vucol::BufferAccess::Write},
                {2, outputs[1], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan MinNaiveFP32::operator()(
        BufferView a,
        BufferView outValue,
        BufferView outIndex,
        const IndexedUnaryReductionArguments& args
    ){
        std::vector<BufferView> inputs = {a};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {outValue, outIndex};
        return this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(IndexedUnaryReductionArguments)
        );
    }

    AvgNaiveFP32::AvgNaiveFP32(
        vucol::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = AvgNaiveFP32_SPIRV,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(UnaryReductionArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
                {1, vucol::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan AvgNaiveFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const UnaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, outputs[0], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan AvgNaiveFP32::operator()(
        BufferView a,
        BufferView out,
        const UnaryReductionArguments& args
    ){
        std::vector<BufferView> inputs = {a};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {out};
        return this->execute(inputs, inouts, outputs, &args, sizeof(UnaryReductionArguments));
    }

    SumNaiveFP32::SumNaiveFP32(
        vucol::Context& ctx,
        uint32_t thread_num,
        uint32_t values_per_thread
    ):thread_num(thread_num), values_per_thread(values_per_thread){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = SumNaiveFP32_SPIRV,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(UnaryReductionArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
                {1, vucol::specConstant(std::uint32_t{values_per_thread})},
            }
        });
    }

    DispatchPlan SumNaiveFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* reductionArgs =
            static_cast<const UnaryReductionArguments*>(args);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, outputs[0], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = reductionArgs->b,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan SumNaiveFP32::operator()(
        BufferView a,
        BufferView out,
        const UnaryReductionArguments& args
    ){
        std::vector<BufferView> inputs = {a};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {out};
        return this->execute(inputs, inouts, outputs, &args, sizeof(UnaryReductionArguments));
    }
}
