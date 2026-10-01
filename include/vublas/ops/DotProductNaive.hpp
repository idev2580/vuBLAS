#pragma once
#include "vucol/Buffer.hpp"
#include "vucol/Context.hpp"
#include <vublas/ops/Reduction.hpp>

namespace vublas{
    class DotProductNaiveFP32:public BinaryReduction{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        DotProductNaiveFP32(
            vucol::Context& ctx,
            uint32_t thread_num = 32,
            uint32_t values_per_thread = 64
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
            const BinaryReductionArguments& args
        ) override;
    };
}
