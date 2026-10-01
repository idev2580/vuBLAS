#include "ShaderTemplate.hpp"
#include <vucol/ShaderCompiler.hpp>
#include <stdexcept>
#include <string>

namespace vublas::detail{
    std::vector<uint32_t> compileShaderTemplate(
        std::string_view shaderTemplate,
        std::string_view injectedSource,
        std::string_view sourceName,
        std::string_view marker
    ){
        if(injectedSource.empty()){
            throw std::invalid_argument("injected GLSL source must not be empty");
        }

        const std::size_t markerOffset = shaderTemplate.find(marker);
        if(markerOffset == std::string_view::npos ||
           shaderTemplate.find(marker, markerOffset + marker.size()) != std::string_view::npos){
            throw std::runtime_error("shader template must contain exactly one epilogue marker");
        }

        std::string source(shaderTemplate);
        source.replace(markerOffset, marker.size(), injectedSource);
        const vucol::ShaderCompileOptions selected{
            .vulkanVersion = vucol::VulkanVersion::Vulkan13,
            .spirvVersion = vucol::SpirvVersion::Spirv16,
        };
        return vucol::compileGlslToSpirv(source, selected, sourceName);
    }
}
