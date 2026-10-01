#include <vublas/ExecutionPlan.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vublas{
    void DispatchPlan::allocate(vucol::Context& ctx) const{
        if(resource){
            if(resource->context != &ctx){
                throw std::runtime_error(
                    "DispatchPlanResource belongs to a different Context."
                );
            }
            return;
        }

        if(!pipeline){
            throw std::runtime_error(
                "Cannot allocate resources for an empty ShaderPipeline."
            );
        }
        if(pushConstants.size() != pipeline.pushConstantSize()){
            throw std::runtime_error(
                "DispatchPlan push-constant size does not match its pipeline."
            );
        }

        const auto layoutBindings = pipeline.bindings();
        if(bindings.size() != layoutBindings.size()){
            throw std::runtime_error(
                "DispatchPlan buffer-binding count does not match its pipeline."
            );
        }
        for(const auto& layoutBinding : layoutBindings){
            const auto count = std::count_if(
                bindings.begin(),
                bindings.end(),
                [&](const BufferBinding& binding){
                    return binding.binding == layoutBinding.binding;
                }
            );
            if(count != 1){
                throw std::runtime_error(
                    "DispatchPlan must bind every pipeline binding exactly once."
                );
            }
        }

        DispatchPlanResource allocated;
        allocated.context = &ctx;
        if(!layoutBindings.empty()){
            allocated.descriptorSet = ctx.createDescriptorSet(pipeline);
            for(const auto& binding : bindings){
                allocated.descriptorSet.bindBuffer(
                    binding.binding,
                    binding.view.buffer,
                    binding.view.offset,
                    binding.view.size,
                    binding.access
                );
            }
        }
        resource = std::move(allocated);
    }

    void DispatchPlan::record(vucol::Context& ctx) const{
        allocate(ctx);
        if(pushConstants.size() != pipeline.pushConstantSize()){
            throw std::runtime_error(
                "DispatchPlan push-constant size does not match its pipeline."
            );
        }

        ctx.use(pipeline);

        if(!pipeline.bindings().empty()){
            if(!resource->descriptorSet){
                throw std::runtime_error(
                    "DispatchPlanResource has no required DescriptorSet."
                );
            }
            ctx.bind(resource->descriptorSet);
        }

        if(!pushConstants.empty()){
            ctx.push(pushConstants.data(), pushConstants.size());
        }

        ctx.dispatch(dispatchX, dispatchY, dispatchZ);
    }

    void ExecutionPlan::append(DispatchPlan plan){
        plans.push_back(std::move(plan));
    }

    void ExecutionPlan::record(vucol::Context& ctx) const{
        for(const auto& plan : plans){
            plan.record(ctx);
        }
    }

    vucol::DispatchToken ExecutionPlan::execute(vucol::Context& ctx, bool recordGpuTimestamp) const{
        recordGpuTimestamp? ctx.beginTimed(): ctx.begin();
        record(ctx);
        return ctx.submitAsync();
    }
}
