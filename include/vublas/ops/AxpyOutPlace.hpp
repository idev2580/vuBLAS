#pragma once
#include "vucol/Buffer.hpp"
#include <cstdint>
#include <vucol/Context.hpp>
#include <vublas/ops/Axpy.hpp>
#include <vublas/ops/Operator.hpp>

namespace vublas{
    class AxpyOutPlaceFP32: public Operator{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;

        public:
        AxpyOutPlaceFP32(
            vucol::Context& ctx,
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
            BufferView A,
            BufferView B,
            BufferView outB,
            const AxpyOutPlaceArguments& args
        );
        virtual DispatchPlan operator()(
            BufferView A,
            BufferView B,
            BufferView outB,
            const AxpyArguments& args
        );
    };
}
