#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

namespace vublas::detail{
    std::vector<uint32_t> compileShaderTemplate(
        std::string_view shaderTemplate,
        std::string_view injectedSource,
        std::string_view sourceName,
        std::string_view marker = "/*__VUBLAS_EPILOGUE__*/"
    );
}
