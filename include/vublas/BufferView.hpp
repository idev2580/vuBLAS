#pragma once

#include <cstddef>
#include <utility>

#include <vucol/Buffer.hpp>

namespace vublas{
    struct BufferView{
        vucol::Buffer buffer;
        std::size_t offset;
        std::size_t size;

        BufferView(vucol::Buffer buffer)
            : buffer(std::move(buffer)),
              offset(0),
              size(this->buffer.size()){
        }

        BufferView(vucol::Buffer buffer, std::size_t offset, std::size_t size)
            : buffer(std::move(buffer)), offset(offset), size(size){
        }
    };
}
