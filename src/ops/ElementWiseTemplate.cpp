#include "ShaderTemplate.hpp"
#include "ShaderTemplates.hpp"
#include <vublas/ops/ElementWiseTemplate.hpp>

namespace vublas{
    namespace{
        constexpr std::string_view operationMarker =
            "/*__VUBLAS_OPERATION__*/";
    }

    UnaryElementwiseTemplateFP32::UnaryElementwiseTemplateFP32(
        vucol::Context& ctx,
        std::string_view operationSource,
        uint32_t thread_num
    ):UnaryElementwiseTemplateFP32(
        ctx,
        detail::compileShaderTemplate(
            detail::unaryElementwiseShaderTemplate(),
            operationSource,
            "UnaryElementwiseTemplateFP32.comp",
            operationMarker
        ),
        thread_num
    ){}

    UnaryElementwiseTemplateFP32::UnaryElementwiseTemplateFP32(
        vucol::Context& ctx,
        std::vector<uint32_t> spirv,
        uint32_t thread_num
    ):thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = spirv,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(UnaryElementwiseArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
            }
        });
    }

    DispatchPlan UnaryElementwiseTemplateFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* elementwiseArgs =
            static_cast<const UnaryElementwiseArguments*>(args);
        const uint32_t groupCount =
            elementwiseArgs->size / thread_num +
            (elementwiseArgs->size % thread_num == 0 ? 0 : 1);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, outputs[0], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = groupCount,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan UnaryElementwiseTemplateFP32::operator()(
        BufferView a,
        BufferView out,
        const UnaryElementwiseArguments& args
    ){
        std::vector<BufferView> inputs = {a};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {out};
        return this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(UnaryElementwiseArguments)
        );
    }

    BinaryElementwiseTemplateFP32::BinaryElementwiseTemplateFP32(
        vucol::Context& ctx,
        std::string_view operationSource,
        uint32_t thread_num
    ):BinaryElementwiseTemplateFP32(
        ctx,
        detail::compileShaderTemplate(
            detail::binaryElementwiseShaderTemplate(),
            operationSource,
            "BinaryElementwiseTemplateFP32.comp",
            operationMarker
        ),
        thread_num
    ){}

    BinaryElementwiseTemplateFP32::BinaryElementwiseTemplateFP32(
        vucol::Context& ctx,
        std::vector<uint32_t> spirv,
        uint32_t thread_num
    ):thread_num(thread_num){
        this->pipeline = ctx.createShaderPipeline({
            .spirv = spirv,
            .bindings = {
                {0, vucol::DescriptorType::StorageBuffer},
                {1, vucol::DescriptorType::StorageBuffer},
                {2, vucol::DescriptorType::StorageBuffer},
            },
            .pushConstantSize = sizeof(BinaryElementwiseArguments),
            .specConstants = {
                {0, vucol::specConstant(std::uint32_t{thread_num})},
            }
        });
    }

    DispatchPlan BinaryElementwiseTemplateFP32::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        const auto* elementwiseArgs =
            static_cast<const BinaryElementwiseArguments*>(args);
        const uint32_t groupCount =
            elementwiseArgs->size / thread_num +
            (elementwiseArgs->size % thread_num == 0 ? 0 : 1);
        return {
            .pipeline = pipeline,
            .bindings = {
                {0, inputs[0], vucol::BufferAccess::Read},
                {1, inputs[1], vucol::BufferAccess::Read},
                {2, outputs[0], vucol::BufferAccess::Write},
            },
            .pushConstants = copyPushConstants(args, argsSize),
            .dispatchX = groupCount,
            .dispatchY = 1,
            .dispatchZ = 1,
        };
    }

    DispatchPlan BinaryElementwiseTemplateFP32::operator()(
        BufferView a,
        BufferView b,
        BufferView out,
        const BinaryElementwiseArguments& args
    ){
        std::vector<BufferView> inputs = {a, b};
        std::vector<BufferView> inouts = {};
        std::vector<BufferView> outputs = {out};
        return this->execute(
            inputs,
            inouts,
            outputs,
            &args,
            sizeof(BinaryElementwiseArguments)
        );
    }
}
