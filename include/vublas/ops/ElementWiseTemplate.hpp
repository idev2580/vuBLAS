#pragma once
#include <vublas/ops/ElementWise.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace vublas{
    class UnaryElementwiseTemplateFP32: public UnaryElementwise{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;

        UnaryElementwiseTemplateFP32(
            vucol::Context& ctx,
            std::vector<uint32_t> spirv,
            uint32_t thread_num
        );

        public:
        UnaryElementwiseTemplateFP32(
            vucol::Context& ctx,
            std::string_view operationSource,
            uint32_t thread_num = 64
        );

        virtual DispatchPlan execute(
            std::span<const BufferView> inputs,
            std::span<const BufferView> inouts,
            std::span<const BufferView> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual DispatchPlan operator()(
            BufferView a,
            BufferView out,
            const UnaryElementwiseArguments& args
        ) override;
    };

    class BinaryElementwiseTemplateFP32: public BinaryElementwise{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;

        BinaryElementwiseTemplateFP32(
            vucol::Context& ctx,
            std::vector<uint32_t> spirv,
            uint32_t thread_num
        );

        public:
        BinaryElementwiseTemplateFP32(
            vucol::Context& ctx,
            std::string_view operationSource,
            uint32_t thread_num = 64
        );

        virtual DispatchPlan execute(
            std::span<const BufferView> inputs,
            std::span<const BufferView> inouts,
            std::span<const BufferView> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual DispatchPlan operator()(
            BufferView a,
            BufferView b,
            BufferView out,
            const BinaryElementwiseArguments& args
        ) override;
    };
}
