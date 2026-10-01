#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <vucol/Context.hpp>
#include <vublas/ExecutionPlan.hpp>
#include <vublas/ops/GemmNaive.hpp>

constexpr int GPU_IDX = 0;
constexpr uint32_t B = 2;
constexpr int M = 4096;
constexpr int N = 1024;
constexpr int P = 4096;
constexpr size_t DEEP_SELECTION_COUNT = 10;
constexpr uint32_t DEEP_REPEAT_COUNT = 10;

vucol::DispatchToken execute_plan(
    vucol::Context& ctx,
    vublas::DispatchPlan plan
){
    vublas::ExecutionPlan executionPlan;
    executionPlan.append(std::move(plan));
    return executionPlan.execute(ctx);
}

struct SweepConfig {
    uint32_t subgroup_tile_m;
    uint32_t subgroup_tile_n;
    uint32_t subgroup_tile_p;
    uint32_t subgroup_tile_cnt_m;
    uint32_t subgroup_tile_cnt_p;
    uint32_t shared_tile_n_multiplier;
    uint32_t reg_tile_m;
    uint32_t reg_tile_p;
    uint32_t k_unroll;
};

constexpr std::array<SweepConfig, 57> SWEEP_CONFIGS = {{
    {1, 1, 32, 2, 2, 2, 16, 2, 2},
    {1, 1, 32, 2, 2, 3, 16, 2, 1},
    {1, 1, 32, 2, 2, 3, 16, 2, 2},
    {1, 1, 32, 2, 2, 3, 16, 4, 1},
    {1, 1, 32, 2, 2, 4, 16, 2, 1},
    {1, 2, 32, 2, 2, 2, 16, 2, 1},
    {1, 2, 32, 2, 2, 3, 16, 2, 1},
    {2, 1, 16, 2, 2, 2, 16, 4, 4},
    {2, 1, 16, 2, 2, 2, 8, 4, 2},
    {2, 1, 16, 2, 2, 3, 8, 4, 1},
    {2, 1, 16, 2, 2, 3, 8, 4, 2},
    {2, 1, 16, 2, 2, 4, 8, 4, 1},
    {2, 1, 16, 2, 2, 4, 8, 4, 2},
    {2, 2, 16, 2, 2, 2, 8, 4, 1},
    {2, 2, 16, 2, 2, 2, 8, 4, 2},
    {2, 2, 16, 2, 2, 3, 8, 4, 1},
    {2, 2, 16, 2, 2, 4, 8, 4, 1},
    {4, 1, 8, 2, 2, 2, 16, 4, 4},
    {4, 1, 8, 2, 2, 2, 8, 4, 4},
    {4, 1, 8, 2, 2, 2, 8, 8, 2},
    {4, 1, 8, 2, 2, 2, 8, 8, 4},
    {4, 1, 8, 2, 2, 3, 8, 4, 2},
    {4, 1, 8, 2, 2, 3, 8, 4, 4},
    {4, 1, 8, 2, 2, 3, 8, 8, 1},
    {4, 1, 8, 2, 2, 3, 8, 8, 2},
    {4, 1, 8, 2, 2, 4, 8, 4, 2},
    {4, 1, 8, 2, 2, 4, 8, 8, 1},
    {4, 1, 8, 2, 2, 4, 8, 8, 2},
    {4, 2, 8, 2, 2, 2, 8, 4, 2},
    {4, 2, 8, 2, 2, 2, 8, 8, 1},
    {4, 2, 8, 2, 2, 2, 8, 8, 2},
    {4, 2, 8, 2, 2, 3, 8, 4, 1},
    {4, 2, 8, 2, 2, 3, 8, 8, 1},
    {4, 2, 8, 2, 2, 4, 16, 4, 1},
    {4, 2, 8, 2, 2, 4, 8, 4, 1},
    {4, 2, 8, 2, 2, 4, 8, 8, 1},
    {8, 1, 4, 2, 2, 2, 8, 8, 4},
    {8, 1, 4, 2, 2, 3, 8, 8, 2},
    {8, 1, 4, 2, 2, 4, 8, 8, 2},
    {8, 2, 4, 2, 2, 2, 8, 8, 2},
    {8, 2, 4, 2, 2, 3, 8, 8, 1},
    {8, 2, 4, 2, 2, 4, 8, 8, 1},
    {1, 1, 32, 2, 2, 2, 16, 4, 1},
    {1, 1, 32, 2, 2, 2, 16, 4, 2},
    {1, 1, 32, 2, 2, 3, 16, 4, 2},
    {1, 1, 32, 2, 2, 4, 16, 4, 1},
    {1, 1, 32, 2, 2, 4, 16, 4, 2},
    {4, 1, 8, 2, 2, 1, 8, 8, 4},
    {4, 1, 8, 2, 2, 2, 8, 8, 3},
    {4, 1, 8, 2, 2, 2, 8, 8, 5},
    {4, 1, 8, 2, 2, 2, 8, 8, 6},
    {4, 1, 8, 2, 2, 3, 8, 8, 4},
    {8, 1, 4, 2, 2, 1, 8, 8, 4},
    {8, 1, 4, 2, 2, 2, 8, 8, 3},
    {8, 1, 4, 2, 2, 2, 8, 8, 5},
    {8, 1, 4, 2, 2, 2, 8, 8, 6},
    {8, 1, 4, 2, 2, 3, 8, 8, 4}
}};

constexpr std::array<std::array<uint32_t, 2>, 4>
SUBGROUP_TILE_COUNT_VALUES = {{
    {1, 8},
    {2, 4},
    {4, 2},
    {8, 1}
}};

uint64_t matrix_elements(uint32_t batch, uint32_t rows, uint32_t cols) {
    return uint64_t(batch) * uint64_t(rows) * uint64_t(cols);
}

size_t checked_bytes(uint64_t elements) {
    if(elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::overflow_error("matrix allocation size exceeds size_t");
    }
    return static_cast<size_t>(elements) * sizeof(float);
}

size_t checked_total_bytes(size_t a_bytes, size_t b_bytes, size_t c_bytes) {
    if(a_bytes > std::numeric_limits<size_t>::max() - b_bytes) {
        throw std::overflow_error("total matrix allocation size exceeds size_t");
    }
    const size_t ab_bytes = a_bytes + b_bytes;
    if(ab_bytes > std::numeric_limits<size_t>::max() - c_bytes) {
        throw std::overflow_error("total matrix allocation size exceeds size_t");
    }
    return ab_bytes + c_bytes;
}
uint32_t checked_stride(uint32_t lhs, uint32_t rhs, const char* name) {
    const uint64_t value = uint64_t(lhs) * uint64_t(rhs);
    if(value > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error(std::string(name) + " exceeds uint32_t");
    }
    return static_cast<uint32_t>(value);
}

bool is_valid(const SweepConfig& config) {
    const bool has_subgroup_size =
        uint64_t(config.subgroup_tile_m) *
        config.subgroup_tile_p == 32;
    const bool has_enough_values_per_thread =
        uint64_t(config.reg_tile_m) *
        config.reg_tile_p >= 32;

    const uint64_t shared_m =
        uint64_t(config.subgroup_tile_cnt_m) *
        config.subgroup_tile_m *
        config.reg_tile_m;
    const uint64_t shared_n =
        uint64_t(config.shared_tile_n_multiplier) *
        config.subgroup_tile_n *
        config.k_unroll;
    const uint64_t shared_p =
        uint64_t(config.subgroup_tile_cnt_p) *
        config.subgroup_tile_p *
        config.reg_tile_p;
    const uint64_t shared_bytes =
        shared_n * (shared_m + shared_p) * sizeof(float);
    const bool fits_shared_memory =
        shared_bytes <= 64ull * 1024ull;

    return
        has_subgroup_size &&
        has_enough_values_per_thread &&
        fits_shared_memory;
}

void print_result(
    const SweepConfig& config,
    uint32_t repeat,
    double tflops
) {
    printf(
        "(%u %u %u %u %u %u %u %u %u)[%u]:(%.6f)\n",
        config.subgroup_tile_m,
        config.subgroup_tile_n,
        config.subgroup_tile_p,
        config.subgroup_tile_cnt_m,
        config.subgroup_tile_cnt_p,
        config.shared_tile_n_multiplier,
        config.reg_tile_m,
        config.k_unroll,
        config.reg_tile_p,
        repeat,
        tflops
    );
}

void print_scan_result(
    const SweepConfig& config,
    double tflops
) {
    printf(
        "(%u %u %u %u %u %u %u %u %u)[scan]:(%.6f)\n",
        config.subgroup_tile_m,
        config.subgroup_tile_n,
        config.subgroup_tile_p,
        config.subgroup_tile_cnt_m,
        config.subgroup_tile_cnt_p,
        config.shared_tile_n_multiplier,
        config.reg_tile_m,
        config.k_unroll,
        config.reg_tile_p,
        tflops
    );
}

void print_failure(const SweepConfig& config, const char* error) {
    printf(
        "(%u %u %u %u %u %u %u %u %u)[error]:(%s)\n",
        config.subgroup_tile_m,
        config.subgroup_tile_n,
        config.subgroup_tile_p,
        config.subgroup_tile_cnt_m,
        config.subgroup_tile_cnt_p,
        config.shared_tile_n_multiplier,
        config.reg_tile_m,
        config.k_unroll,
        config.reg_tile_p,
        error
    );
}

int main() {
    const uint64_t a_elements =
        matrix_elements(B, M, N);
    const uint64_t b_elements =
        matrix_elements(B, N, P);
    const uint64_t c_elements =
        matrix_elements(B, M, P);

    const size_t a_bytes = checked_bytes(a_elements);
    const size_t b_bytes = checked_bytes(b_elements);
    const size_t c_bytes = checked_bytes(c_elements);
    const size_t total_bytes =
        checked_total_bytes(a_bytes, b_bytes, c_bytes);
    printf("Total VRAM: %llu\n", static_cast<unsigned long long>(total_bytes));

    std::vector<float> a(static_cast<size_t>(a_elements));
    std::vector<float> b(static_cast<size_t>(b_elements));
    std::vector<float> c(static_cast<size_t>(c_elements), 0.0f);

    const uint32_t a_stride =
        checked_stride(M, N, "a_stride");
    const uint32_t b_stride =
        checked_stride(N, P, "b_stride");
    const uint32_t c_stride =
        checked_stride(M, P, "c_stride");

    const vublas::GemmArguments args = {
        .b = B,
        .m = M,
        .n = N,
        .p = P,
        .alpha = 1.0f,
        .beta = 0.0f,
        .a_stride = a_stride,
        .b_stride = b_stride,
        .c_stride = c_stride,
        .a_m_stride = N,
        .a_n_stride = 1,
        .b_n_stride = P,
        .b_p_stride = 1,
        .c_m_stride = P,
        .c_p_stride = 1
    };

    vucol::Context ctx({GPU_IDX});
    auto bufferA = ctx.createBuffer(a_bytes, vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(b_bytes, vucol::BufferType::Auto);
    auto bufferC = ctx.createBuffer(c_bytes, vucol::BufferType::Auto);

    bufferA.write(a.data(), a_bytes);
    bufferB.write(b.data(), b_bytes);
    bufferC.write(c.data(), c_bytes);

    const double flop_count =
        2.0 *
        static_cast<double>(B) *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(P);

    std::vector<std::pair<SweepConfig, double>> scan_results;

    for(const SweepConfig& base_config : SWEEP_CONFIGS) {
        for(const auto& counts : SUBGROUP_TILE_COUNT_VALUES) {
            SweepConfig config = base_config;
            config.subgroup_tile_cnt_m = counts[0];
            config.subgroup_tile_cnt_p = counts[1];

            if(!is_valid(config)) {
                continue;
            }

            try {
                vublas::GemmNaiveFP32 gemm(
                    ctx,
                    config.subgroup_tile_m,
                    config.subgroup_tile_n,
                    config.subgroup_tile_p,
                    config.subgroup_tile_cnt_m,
                    config.subgroup_tile_cnt_p,
                    config.shared_tile_n_multiplier,
                    config.reg_tile_m,
                    config.k_unroll,
                    config.reg_tile_p
                );

                const auto start =
                    std::chrono::steady_clock::now();
                auto token = execute_plan(
                    ctx,
                    gemm(
                        bufferA,
                        bufferB,
                        bufferC,
                        args
                    )
                );
                token.wait();
                const auto end =
                    std::chrono::steady_clock::now();
                const double seconds =
                    std::chrono::duration<double>(
                        end - start
                    ).count();
                const double tflops =
                    flop_count / seconds / 1.0e12;
                print_scan_result(config, tflops);
                fflush(stdout);

                scan_results.emplace_back(config, tflops);
            } catch(const std::exception& error) {
                print_failure(
                    config,
                    error.what()
                );
            }
        }
    }

    std::stable_sort(
        scan_results.begin(),
        scan_results.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.second > rhs.second;
        }
    );
    if(scan_results.size() > DEEP_SELECTION_COUNT) {
        scan_results.resize(DEEP_SELECTION_COUNT);
    }

    printf(
        "Selected configurations: %llu\n",
        static_cast<unsigned long long>(scan_results.size())
    );
    printf(
        "####################################################################################################\n"
        "######################################## DEEP MEASUREMENT ##########################################\n"
        "####################################################################################################\n"
    );
    fflush(stdout);

    for(const auto& scan_result : scan_results) {
        const SweepConfig& config = scan_result.first;
        try {
            vublas::GemmNaiveFP32 gemm(
                ctx,
                config.subgroup_tile_m,
                config.subgroup_tile_n,
                config.subgroup_tile_p,
                config.subgroup_tile_cnt_m,
                config.subgroup_tile_cnt_p,
                config.shared_tile_n_multiplier,
                config.reg_tile_m,
                config.k_unroll,
                config.reg_tile_p
            );

            for(uint32_t repeat = 0;
                repeat < DEEP_REPEAT_COUNT;
                repeat++) {
                const auto start =
                    std::chrono::steady_clock::now();
                auto token = execute_plan(
                    ctx,
                    gemm(
                        bufferA,
                        bufferB,
                        bufferC,
                        args
                    )
                );
                token.wait();
                const auto end =
                    std::chrono::steady_clock::now();
                const double seconds =
                    std::chrono::duration<double>(
                        end - start
                    ).count();
                print_result(
                    config,
                    repeat,
                    flop_count / seconds / 1.0e12
                );
                fflush(stdout);
            }
        } catch(const std::exception& error) {
            print_failure(
                config,
                error.what()
            );
        }
    }

    return 0;
}
