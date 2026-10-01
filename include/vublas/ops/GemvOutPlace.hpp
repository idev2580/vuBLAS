#pragma once
#include "vublas/ops/Operator.hpp"
#include "vublas/ops/GemmOutPlace.hpp"
#include <vublas/ops/GemvArguments.hpp>
#include <cstdint>
#include <memory>

namespace vublas{
    class GemvOutPlace: public Operator{
        protected:
        std::unique_ptr<GemmOutPlace> gemm;
        explicit GemvOutPlace(vucol::Context& ctx);

        public:
        virtual DispatchPlan execute(
            std::span<const BufferView> inputs,
            std::span<const BufferView> inouts,
            std::span<const BufferView> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual DispatchPlan operator()(
            BufferView A,
            BufferView X,
            BufferView Y,
            BufferView outY,
            const GemvOutPlaceArguments& args
        );
        virtual DispatchPlan operator()(
            BufferView A,
            BufferView X,
            BufferView Y,
            BufferView outY,
            const GemvArguments& args
        );
    };
}
