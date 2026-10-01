#pragma once
#include "vucol/Buffer.hpp"
#include "vucol/Context.hpp"
#include <vublas/ops/Reduction.hpp>

namespace vublas{
    class MaxNaiveFP32:public IndexedUnaryReduction{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        MaxNaiveFP32(
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
            BufferView outValue,
            BufferView outIndex,
            const IndexedUnaryReductionArguments& args
        ) override;
    };
    class MinNaiveFP32:public IndexedUnaryReduction{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        MinNaiveFP32(
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
            BufferView outValue,
            BufferView outIndex,
            const IndexedUnaryReductionArguments& args
        ) override;
    };
    class AvgNaiveFP32:public UnaryReduction{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        AvgNaiveFP32(
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
            BufferView out,
            const UnaryReductionArguments& args
        ) override;
    };
    class SumNaiveFP32:public UnaryReduction{
        private:
        vucol::ShaderPipeline pipeline;
        uint32_t thread_num;
        uint32_t values_per_thread;

        public:
        SumNaiveFP32(
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
            BufferView out,
            const UnaryReductionArguments& args
        ) override;
    };
}
