#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
    struct BenchmarkConfig {
        uint32_t batch = 64;
        uint32_t m = 4096;
        uint32_t n = 1024;
        uint32_t p = 1024;
        uint32_t tile_m = 128;
        uint32_t tile_n = 32;
        uint32_t tile_p = 128;
        uint32_t thread_tile_m = 8;
        uint32_t thread_tile_p = 8;
        uint32_t iterations = 50;
        uint32_t warmup_iterations = 5;
    };

    const char* cublas_status_name(cublasStatus_t status) {
        switch(status) {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
        }
        return "CUBLAS_STATUS_UNKNOWN";
    }

    void check_cuda(cudaError_t status, const char* operation) {
        if(status != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + " failed: " + cudaGetErrorString(status)
            );
        }
    }

    void check_cublas(cublasStatus_t status, const char* operation) {
        if(status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                std::string(operation) + " failed: " + cublas_status_name(status)
            );
        }
    }

    uint32_t parse_u32(
        int argc,
        char** argv,
        int index,
        uint32_t default_value,
        const char* name
    ) {
        if(index >= argc) {
            return default_value;
        }

        const unsigned long long value = std::stoull(argv[index]);
        if(value == 0 || value > std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range(std::string(name) + " must fit in uint32_t and be greater than zero");
        }
        return static_cast<uint32_t>(value);
    }

    BenchmarkConfig parse_config(int argc, char** argv) {
        BenchmarkConfig config;
        config.batch = parse_u32(argc, argv, 1, config.batch, "batch");
        config.m = parse_u32(argc, argv, 2, config.m, "m");
        config.n = parse_u32(argc, argv, 3, config.n, "n");
        config.p = parse_u32(argc, argv, 4, config.p, "p");
        config.tile_m = parse_u32(argc, argv, 5, config.tile_m, "tile_m");
        config.tile_n = parse_u32(argc, argv, 6, config.tile_n, "tile_n");
        config.tile_p = parse_u32(argc, argv, 7, config.tile_p, "tile_p");
        config.thread_tile_m = parse_u32(
            argc,
            argv,
            8,
            config.thread_tile_m,
            "thread_tile_m"
        );
        config.thread_tile_p = parse_u32(
            argc,
            argv,
            9,
            config.thread_tile_p,
            "thread_tile_p"
        );
        config.iterations = parse_u32(
            argc,
            argv,
            10,
            config.iterations,
            "iterations"
        );
        return config;
    }

    int checked_int(uint32_t value, const char* name) {
        if(value > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(std::string(name) + " exceeds the cuBLAS int range");
        }
        return static_cast<int>(value);
    }

    int64_t checked_elements(uint32_t lhs, uint32_t rhs, const char* name) {
        const uint64_t value = uint64_t(lhs) * uint64_t(rhs);
        if(value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::overflow_error(std::string(name) + " exceeds int64_t");
        }
        return static_cast<int64_t>(value);
    }

    size_t checked_bytes(uint32_t batch, int64_t stride, const char* name) {
        const uint64_t elements = uint64_t(batch) * uint64_t(stride);
        if(elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::overflow_error(std::string(name) + " allocation exceeds size_t");
        }
        return static_cast<size_t>(elements) * sizeof(float);
    }

    class DeviceBuffer {
        private:
        void* pointer = nullptr;

        public:
        explicit DeviceBuffer(size_t bytes) {
            check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
        }

        ~DeviceBuffer() {
            if(pointer != nullptr) {
                cudaFree(pointer);
            }
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        float* data() {
            return static_cast<float*>(pointer);
        }
    };

    class CublasHandle {
        private:
        cublasHandle_t handle = nullptr;

        public:
        CublasHandle() {
            check_cublas(cublasCreate(&handle), "cublasCreate");
        }

        ~CublasHandle() {
            if(handle != nullptr) {
                cublasDestroy(handle);
            }
        }

        CublasHandle(const CublasHandle&) = delete;
        CublasHandle& operator=(const CublasHandle&) = delete;

        cublasHandle_t get() const {
            return handle;
        }
    };

    class CudaEvent {
        private:
        cudaEvent_t event = nullptr;

        public:
        CudaEvent() {
            check_cuda(cudaEventCreate(&event), "cudaEventCreate");
        }

        ~CudaEvent() {
            if(event != nullptr) {
                cudaEventDestroy(event);
            }
        }

        CudaEvent(const CudaEvent&) = delete;
        CudaEvent& operator=(const CudaEvent&) = delete;

        cudaEvent_t get() const {
            return event;
        }
    };

    struct GemmCall {
        int batch;
        int m;
        int n;
        int p;
        int64_t a_stride;
        int64_t b_stride;
        int64_t c_stride;
        float* a;
        float* b;
        float* c;
    };

    cublasStatus_t launch_gemm(
        cublasHandle_t handle,
        const GemmCall& call,
        cublasComputeType_t compute_type
    ) {
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        // Row-major C = A * B is column-major C^T = B^T * A^T.
        return cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            call.p,
            call.m,
            call.n,
            &alpha,
            call.b,
            CUDA_R_32F,
            call.p,
            call.b_stride,
            call.a,
            CUDA_R_32F,
            call.n,
            call.a_stride,
            &beta,
            call.c,
            CUDA_R_32F,
            call.p,
            call.c_stride,
            call.batch,
            compute_type,
            CUBLAS_GEMM_DEFAULT
        );
    }

    void run_mode(
        cublasHandle_t handle,
        const GemmCall& call,
        const BenchmarkConfig& config,
        const char* mode_name,
        cublasComputeType_t compute_type
    ) {
        for(uint32_t i = 0; i < config.warmup_iterations; i++) {
            check_cublas(launch_gemm(handle, call, compute_type), "cuBLAS warm-up GEMM");
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warm-up");

        CudaEvent start;
        CudaEvent end;
        check_cuda(cudaEventRecord(start.get()), "cudaEventRecord(start)");
        for(uint32_t i = 0; i < config.iterations; i++) {
            check_cublas(launch_gemm(handle, call, compute_type), "cuBLAS measured GEMM");
        }
        check_cuda(cudaEventRecord(end.get()), "cudaEventRecord(end)");
        check_cuda(cudaEventSynchronize(end.get()), "cudaEventSynchronize(end)");

        float elapsed_ms = 0.0f;
        check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start.get(), end.get()),
            "cudaEventElapsedTime"
        );

        const double flops_per_iteration =
            2.0 *
            static_cast<double>(config.batch) *
            static_cast<double>(config.m) *
            static_cast<double>(config.n) *
            static_cast<double>(config.p);
        const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
        const double tflops =
            flops_per_iteration * static_cast<double>(config.iterations) /
            seconds /
            1.0e12;

        std::cout
            << "  " << mode_name << "\n"
            << "    elapsed_ms_total=" << elapsed_ms << "\n"
            << "    elapsed_ms_per_iteration="
            << elapsed_ms / static_cast<double>(config.iterations) << "\n"
            << "    gpu_tflops=" << tflops << "\n";
    }

    void print_config(const BenchmarkConfig& config, const cudaDeviceProp& device) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout
            << "cuBLAS strided-batched FP32 GEMM benchmark\n"
            << "  device=" << device.name << "\n"
            << "  batch=" << config.batch
            << ", m=" << config.m
            << ", n=" << config.n
            << ", p=" << config.p << "\n"
            << "  vucol_tile_m=" << config.tile_m
            << ", vucol_tile_n=" << config.tile_n
            << ", vucol_tile_p=" << config.tile_p << "\n"
            << "  vucol_thread_tile_m=" << config.thread_tile_m
            << ", vucol_thread_tile_p=" << config.thread_tile_p << "\n"
            << "  warmup_iterations=" << config.warmup_iterations
            << ", measured_iterations=" << config.iterations << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        const BenchmarkConfig config = parse_config(argc, argv);

        int device_id = 0;
        check_cuda(cudaGetDevice(&device_id), "cudaGetDevice");
        cudaDeviceProp device{};
        check_cuda(cudaGetDeviceProperties(&device, device_id), "cudaGetDeviceProperties");
        print_config(config, device);

        const int batch = checked_int(config.batch, "batch");
        const int m = checked_int(config.m, "m");
        const int n = checked_int(config.n, "n");
        const int p = checked_int(config.p, "p");
        const int64_t a_stride = checked_elements(config.m, config.n, "a_stride");
        const int64_t b_stride = checked_elements(config.n, config.p, "b_stride");
        const int64_t c_stride = checked_elements(config.m, config.p, "c_stride");
        const size_t a_bytes = checked_bytes(config.batch, a_stride, "A");
        const size_t b_bytes = checked_bytes(config.batch, b_stride, "B");
        const size_t c_bytes = checked_bytes(config.batch, c_stride, "C");

        DeviceBuffer a(a_bytes);
        DeviceBuffer b(b_bytes);
        DeviceBuffer c(c_bytes);
        check_cuda(cudaMemset(a.data(), 0, a_bytes), "cudaMemset(A)");
        check_cuda(cudaMemset(b.data(), 0, b_bytes), "cudaMemset(B)");
        check_cuda(cudaMemset(c.data(), 0, c_bytes), "cudaMemset(C)");

        CublasHandle handle;
        const GemmCall call = {
            batch,
            m,
            n,
            p,
            a_stride,
            b_stride,
            c_stride,
            a.data(),
            b.data(),
            c.data()
        };

        run_mode(
            handle.get(),
            call,
            config,
            "strict_fp32",
            CUBLAS_COMPUTE_32F_PEDANTIC
        );
        run_mode(
            handle.get(),
            call,
            config,
            "tf32_tensor_core",
            CUBLAS_COMPUTE_32F_FAST_TF32
        );
    } catch(const std::exception& error) {
        std::cerr << "cuBLAS benchmark failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
