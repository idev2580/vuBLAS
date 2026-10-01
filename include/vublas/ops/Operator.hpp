#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <vublas/BufferView.hpp>
#include <vublas/ExecutionPlan.hpp>

namespace vublas{
    class Operator{
        protected:
        static std::vector<std::byte> copyPushConstants(
            const void* args,
            std::size_t argsSize
        );

        public:
        virtual ~Operator() = default;

        virtual DispatchPlan execute(
            std::span<const BufferView> inputs,
            std::span<const BufferView> inouts,
            std::span<const BufferView> outputs,
            const void* args,
            std::size_t argsSize
        ) = 0;
    };
}
