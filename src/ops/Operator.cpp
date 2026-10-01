#include <vublas/ops/Operator.hpp>

#include <cstring>

namespace vublas{
    std::vector<std::byte> Operator::copyPushConstants(
        const void* args,
        std::size_t argsSize
    ){
        std::vector<std::byte> bytes(argsSize);
        if(argsSize != 0){
            std::memcpy(bytes.data(), args, argsSize);
        }
        return bytes;
    }
}
