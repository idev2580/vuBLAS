#pragma once
#include <cstdint>

namespace vublas{
    struct GemmArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        uint32_t p;
        float alpha;
        float beta;
        uint32_t a_stride;
        uint32_t b_stride;
        uint32_t c_stride;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t b_n_stride;
        uint32_t b_p_stride;
        uint32_t c_m_stride;
        uint32_t c_p_stride;
    };
    struct GemmOutPlaceArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        uint32_t p;
        float alpha;
        float beta;
        uint32_t a_stride;
        uint32_t b_stride;
        uint32_t c_stride;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t b_n_stride;
        uint32_t b_p_stride;
        uint32_t c_m_stride;
        uint32_t c_p_stride;
        uint32_t out_c_stride;
        uint32_t out_c_m_stride;
        uint32_t out_c_p_stride;

        static GemmOutPlaceArguments sameOutputLayout(const GemmArguments& args);
        void fromInPlace(const GemmArguments& args);
    };
}
