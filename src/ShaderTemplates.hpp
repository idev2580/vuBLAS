#pragma once
#include <string_view>

namespace vublas::detail{
    std::string_view gemmNaiveShaderTemplate();
    std::string_view matmulNaiveShaderTemplate();
    std::string_view gemmOutPlaceNaiveShaderTemplate();
    std::string_view unaryElementwiseShaderTemplate();
    std::string_view binaryElementwiseShaderTemplate();
}
