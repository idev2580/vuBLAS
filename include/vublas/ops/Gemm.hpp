#pragma once
#include "vucol/Buffer.hpp"
#include <cstdint>
#include <vucol/Context.hpp>
#include <vublas/ops/Operator.hpp>
#include <vublas/ops/GemmArguments.hpp>

namespace vublas{
    class Gemm: public Operator{
        private:
        vucol::ShaderPipeline pipeline;

        uint32_t tile_m;
        uint32_t tile_n;
        uint32_t tile_p;

        protected:
        Gemm(
            vucol::Context& ctx
        );

        public:
        Gemm(
            vucol::Context& ctx,
            std::span<const uint32_t> shaderBytecodes,
            uint32_t subgroup_tile_m,
            uint32_t subgroup_tile_n,
            uint32_t subgroup_tile_p,
            uint32_t subgroup_tile_cnt_m,
            uint32_t subgroup_tile_cnt_p,
            uint32_t shared_tile_n_multiplier,
            uint32_t reg_tile_m,
            uint32_t inner_tile_n,
            uint32_t reg_tile_p
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
            BufferView C,
            const GemmArguments& args
        );
    };
}
