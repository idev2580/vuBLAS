#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <vucol/Context.hpp>
#include <vucol/DescriptorSet.hpp>
#include <vucol/ShaderPipeline.hpp>
#include <vublas/BufferView.hpp>

namespace vublas{
    struct BufferBinding{
        std::uint32_t binding;
        BufferView view;
        vucol::BufferAccess access;
    };

    struct DispatchPlanResource{
        vucol::DescriptorSet descriptorSet;
        vucol::Context* context = nullptr;
    };

    struct DispatchPlan{
        vucol::ShaderPipeline pipeline;
        std::vector<BufferBinding> bindings;
        std::vector<std::byte> pushConstants;
        std::uint32_t dispatchX;
        std::uint32_t dispatchY;
        std::uint32_t dispatchZ;
        mutable std::optional<DispatchPlanResource> resource;

        void allocate(vucol::Context& ctx) const;
        void record(vucol::Context& ctx) const;
    };

    class ExecutionPlan{
        private:
        std::vector<DispatchPlan> plans;

        public:
        void append(DispatchPlan plan);
        void record(vucol::Context& ctx) const;
        [[nodiscard]] vucol::DispatchToken execute(vucol::Context& ctx, bool recordGpuTimestamp = false) const;
    };
}
