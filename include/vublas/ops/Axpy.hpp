#pragma once
#include "vucol/Buffer.hpp"
#include <cstdint>
#include <vucol/Context.hpp>
#include <vublas/ops/Operator.hpp>

namespace vublas{
    // TODO: For broadcasted operations, elementwise operators should also need to support strided operation.
    struct AxpyArguments{
        uint32_t b;
        uint32_t n;
        float alpha;
        uint32_t a_b_stride;
        uint32_t a_n_stride;
        uint32_t b_b_stride;
        uint32_t b_n_stride;
    };
    struct AxpyOutPlaceArguments{
        uint32_t b;
        uint32_t n;
        float alpha;
        uint32_t a_b_stride;
        uint32_t a_n_stride;
        uint32_t b_b_stride;
        uint32_t b_n_stride;
        uint32_t out_b_b_stride;
        uint32_t out_b_n_stride;

        static AxpyOutPlaceArguments sameOutputLayout(const AxpyArguments& args);
        void fromInPlace(const AxpyArguments& args);
    };
    class AxpyFP32: public Operator{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;

        public:
        AxpyFP32(
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
            const AxpyArguments& args
        );
    };
}
