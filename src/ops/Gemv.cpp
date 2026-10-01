#include "vublas/ops/GemvArguments.hpp"
#include <vublas/ops/Gemv.hpp>
#include <vublas/ops/Gemm.hpp>

namespace vublas{
    Gemv::Gemv(vucol::Context&):gemm(nullptr){}
    DispatchPlan Gemv::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        GemmArguments m_args = convertGemvToGemm(*(GemvArguments*)args);
        return gemm->execute(inputs, inouts, outputs, &m_args, sizeof(GemmArguments));
    }

    DispatchPlan Gemv::operator()(
        BufferView A,
        BufferView B,
        BufferView C,
        const GemvArguments& args
    ){
        GemmArguments m_args = convertGemvToGemm(args);
        return (*gemm)(A,B,C,m_args);
    }
}
