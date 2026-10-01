#pragma once
#include <cstdint>
#include <vublas/ops/GemmArguments.hpp>

namespace vublas{
    struct GemvArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        float alpha;
        float beta;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t x_n_stride;
        uint32_t x_b_stride;
        uint32_t y_m_stride;
        uint32_t y_b_stride;
    };
    struct GemvOutPlaceArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        float alpha;
        float beta;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t x_n_stride;
        uint32_t x_b_stride;
        uint32_t y_m_stride;
        uint32_t y_b_stride;
        uint32_t out_y_m_stride;
        uint32_t out_y_b_stride;

        static GemvOutPlaceArguments sameOutputLayout(const GemvArguments& args);
        void fromInPlace(const GemvArguments& args);
    };
    GemmArguments convertGemvToGemm(const GemvArguments& args);
    GemmOutPlaceArguments convertGemvToGemm(const GemvOutPlaceArguments& args);
}
