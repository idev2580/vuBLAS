#pragma once
#include "vucol/Buffer.hpp"
#include "vublas/ops/Operator.hpp"
#include <cstdint>

namespace vublas{
    struct UnaryElementwiseArguments{
        uint32_t size;
    };

    struct BinaryElementwiseArguments{
        uint32_t size;
    };

    class UnaryElementwise: public Operator{
        public:
        virtual DispatchPlan operator()(
            BufferView a,
            BufferView out,
            const UnaryElementwiseArguments& args
        ) = 0;
    };
    class BinaryElementwise: public Operator{
        public:
        virtual DispatchPlan operator()(
            BufferView a,
            BufferView b,
            BufferView out,
            const BinaryElementwiseArguments& args
        ) = 0;
    };
}
