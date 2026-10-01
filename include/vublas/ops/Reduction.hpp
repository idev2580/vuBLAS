#pragma once
#include "vucol/Buffer.hpp"
#include "vublas/ops/Operator.hpp"

#include <cstdint>
namespace vublas{
    struct UnaryReductionArguments{
        uint32_t b;
        uint32_t n;
        uint32_t a_b_stride;
        uint32_t a_n_stride;
        uint32_t out_b_stride;
        uint32_t out_n_stride;
    };

    struct BinaryReductionArguments{
        uint32_t b;
        uint32_t n;
        uint32_t a_b_stride;
        uint32_t a_n_stride;
        uint32_t b_b_stride;
        uint32_t b_n_stride;
        uint32_t out_b_stride;
        uint32_t out_n_stride;
    };

    struct IndexedUnaryReductionArguments{
        uint32_t b;
        uint32_t n;
        uint32_t a_b_stride;
        uint32_t a_n_stride;
        uint32_t out_value_b_stride;
        uint32_t out_value_n_stride;
        uint32_t out_index_b_stride;
        uint32_t out_index_n_stride;
    };

    class UnaryReduction: public Operator{
        public:
        virtual DispatchPlan operator()(
            BufferView a,
            BufferView out,
            const UnaryReductionArguments& args
        ) = 0;
    };

    class BinaryReduction: public Operator{
        public:
        virtual DispatchPlan operator()(
            BufferView a,
            BufferView b,
            BufferView out,
            const BinaryReductionArguments& args
        ) = 0;
    };

    class IndexedUnaryReduction: public Operator{
        public:
        virtual DispatchPlan operator()(
            BufferView a,
            BufferView outValue,
            BufferView outIndex,
            const IndexedUnaryReductionArguments& args
        ) = 0;
    };
}
