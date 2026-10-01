#include <vublas/ops/GemmArguments.hpp>

namespace vublas{
    GemmOutPlaceArguments GemmOutPlaceArguments::sameOutputLayout(const GemmArguments& args){
        return {
            .b = args.b,
            .m = args.m,
            .n = args.n,
            .p = args.p,
            .alpha = args.alpha,
            .beta = args.beta,
            .a_stride = args.a_stride,
            .b_stride = args.b_stride,
            .c_stride = args.c_stride,
            .a_m_stride = args.a_m_stride,
            .a_n_stride = args.a_n_stride,
            .b_n_stride = args.b_n_stride,
            .b_p_stride = args.b_p_stride,
            .c_m_stride = args.c_m_stride,
            .c_p_stride = args.c_p_stride,
            .out_c_stride = args.c_stride,
            .out_c_m_stride = args.c_m_stride,
            .out_c_p_stride = args.c_p_stride
        };
    }

    void GemmOutPlaceArguments::fromInPlace(const GemmArguments& args){
        *this = sameOutputLayout(args);
    }
}
