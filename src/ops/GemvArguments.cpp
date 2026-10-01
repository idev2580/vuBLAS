#include "vublas/ops/GemmArguments.hpp"
#include <vublas/ops/GemvArguments.hpp>

namespace vublas{
    GemmArguments convertGemvToGemm(const GemvArguments& args){
        return {
            .b = 1,
            .m = args.m,
            .n = args.n,
            .p = args.b,
            .alpha = args.alpha,
            .beta = args.beta,
            .a_stride = 0,
            .b_stride = 0,
            .c_stride = 0,
            .a_m_stride = args.a_m_stride,
            .a_n_stride = args.a_n_stride,
            .b_n_stride = args.x_n_stride,
            .b_p_stride = args.x_b_stride,
            .c_m_stride = args.y_m_stride,
            .c_p_stride = args.y_b_stride
        };
    }

    GemvOutPlaceArguments GemvOutPlaceArguments::sameOutputLayout(const GemvArguments& args){
        return {
            .b = args.b,
            .m = args.m,
            .n = args.n,
            .alpha = args.alpha,
            .beta = args.beta,
            .a_m_stride = args.a_m_stride,
            .a_n_stride = args.a_n_stride,
            .x_n_stride = args.x_n_stride,
            .x_b_stride = args.x_b_stride,
            .y_m_stride = args.y_m_stride,
            .y_b_stride = args.y_b_stride,
            .out_y_m_stride = args.y_m_stride,
            .out_y_b_stride = args.y_b_stride
        };
    }

    void GemvOutPlaceArguments::fromInPlace(const GemvArguments& args){
        *this = sameOutputLayout(args);
    }

    GemmOutPlaceArguments convertGemvToGemm(const GemvOutPlaceArguments& args){
        return {
            .b = 1,
            .m = args.m,
            .n = args.n,
            .p = args.b,
            .alpha = args.alpha,
            .beta = args.beta,
            .a_stride = 0,
            .b_stride = 0,
            .c_stride = 0,
            .a_m_stride = args.a_m_stride,
            .a_n_stride = args.a_n_stride,
            .b_n_stride = args.x_n_stride,
            .b_p_stride = args.x_b_stride,
            .c_m_stride = args.y_m_stride,
            .c_p_stride = args.y_b_stride,
            .out_c_stride = 0,
            .out_c_m_stride = args.out_y_m_stride,
            .out_c_p_stride = args.out_y_b_stride
        };
    }
}
