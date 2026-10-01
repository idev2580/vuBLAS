#include "vublas/ops/GemvArguments.hpp"
#include <vublas/ops/GemvOutPlace.hpp>
#include <vublas/ops/GemmOutPlace.hpp>

namespace vublas{
    GemvOutPlace::GemvOutPlace(vucol::Context&):gemm(nullptr){}
    DispatchPlan GemvOutPlace::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        GemmOutPlaceArguments m_args = convertGemvToGemm(*(GemvOutPlaceArguments*)args);
        return gemm->execute(
            inputs,
            inouts,
            outputs,
            &m_args,
            sizeof(GemmOutPlaceArguments)
        );
    }

    DispatchPlan GemvOutPlace::operator()(
        BufferView A,
        BufferView B,
        BufferView C,
        BufferView outC,
        const GemvOutPlaceArguments& args
    ){
        GemmOutPlaceArguments m_args = convertGemvToGemm(args);
        return (*gemm)(A,B,C,outC,m_args);
    }

    DispatchPlan GemvOutPlace::operator()(
        BufferView A,
        BufferView B,
        BufferView C,
        BufferView outC,
        const GemvArguments& args
    ){
        const GemvOutPlaceArguments outPlaceArgs = GemvOutPlaceArguments::sameOutputLayout(args);
        return (*this)(A, B, C, outC, outPlaceArgs);
    }
}
