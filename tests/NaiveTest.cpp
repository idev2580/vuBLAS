#include <iostream>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <utility>
#include <vublas/BufferView.hpp>
#include <vublas/ExecutionPlan.hpp>
#include <vublas/ops/AxpyOutPlace.hpp>
#include <vublas/ops/DotProductNaive.hpp>
#include <vublas/ops/ElementWiseTemplate.hpp>
#include <vublas/ops/GemmNaive.hpp>
#include <vublas/ops/GemmNaiveTemplate.hpp>
#include <vublas/ops/GemmOutPlaceNaive.hpp>
#include <vublas/ops/GemmOutPlaceNaiveTemplate.hpp>
#include <vublas/ops/GemvNaive.hpp>
#include <vublas/ops/GemvNaiveTemplate.hpp>
#include <vublas/ops/GemvOutPlaceNaive.hpp>
#include <vublas/ops/GemvOutPlaceNaiveTemplate.hpp>
#include <vublas/ops/MatMulNaive.hpp>
#include <vublas/ops/MatMulNaiveTemplate.hpp>
#include <vublas/ops/ReductionNaive.hpp>
#include "lib/MinCpuBlas.hpp"

constexpr int gemm_test_iter = 10;
constexpr int max_batch = 2;
constexpr int max_m = 1000;
constexpr int max_n = 1000;
constexpr int max_p = 1000;
constexpr int matmul_test_iter = 10;

vucol::DispatchToken execute_plan(
    vucol::Context& ctx,
    vublas::DispatchPlan plan
){
    vublas::ExecutionPlan executionPlan;
    executionPlan.append(std::move(plan));
    return executionPlan.execute(ctx);
}

bool is_equal_tensor(
    const std::vector<float>& y,
    const std::vector<float>& sol_y,
    const float eps = 0.00001 // 10^-5까지는 정상으로 본다(합의 순서가 CPU와 많이 다르다.)
){
    for(int64_t i=0; i<sol_y.size(); i++){
        const float pos_delta = std::abs(sol_y[i] - y[i]);
        const float cpu_abs = std::abs(sol_y[i]);
        const float error_rel = pos_delta / cpu_abs;

        if(error_rel > eps){
            printf("[%d]: sol_y=%.15f, y=%.15f, RelErr = %.15f\n", (int)i, sol_y[i], y[i], error_rel);
            return false;
        }
    }
    return true;
}
void print_vector(
    const std::vector<float>& v
){
    printf("[");
    for(int i=0; i<v.size(); i++){
        if(i == v.size() - 1){
            printf("%f", v[i]);
            break;
        } else {
            printf("%f, ", v[i]);
        }
    }
    printf("]\n");
}

void run_cpu_gemv(
    const vublas::GemvArguments& args,
    const std::vector<float>& a,
    const std::vector<float>& x,
    std::vector<float>& y
){
    for(uint32_t batch_id=0; batch_id < args.b; batch_id++){
        for(uint32_t r=0; r < args.m; r++){
            float acc = 0.0f;
            for(uint32_t col=0; col < args.n; col++){
                const uint64_t a_idx =
                    uint64_t(r) * args.a_m_stride + uint64_t(col) * args.a_n_stride;
                const uint64_t x_idx =
                    uint64_t(col) * args.x_n_stride + uint64_t(batch_id) * args.x_b_stride;
                acc += a[a_idx] * x[x_idx];
            }

            const uint64_t y_idx =
                uint64_t(r) * args.y_m_stride + uint64_t(batch_id) * args.y_b_stride;
            y[y_idx] = args.alpha * acc + args.beta * y[y_idx];
        }
    }
}

TEST(AxpyOutPlaceTest, SupportsBatchAndDistinctStrides){
    vucol::Context ctx;
    vublas::AxpyOutPlaceFP32 axpy(ctx);

    const uint32_t batch = 3;
    const uint32_t n = 5;
    const uint32_t a_n_stride = 2;
    const uint32_t a_b_stride = n * a_n_stride + 3;
    const uint32_t b_n_stride = 3;
    const uint32_t b_b_stride = n * b_n_stride + 2;
    const uint32_t out_b_n_stride = 2;
    const uint32_t out_b_b_stride = n * out_b_n_stride + 4;
    const float alpha = 1.25f;
    const float sentinel = -999.0f;

    const uint64_t a_size =
        uint64_t(batch - 1) * a_b_stride + uint64_t(n - 1) * a_n_stride + 1;
    const uint64_t b_size =
        uint64_t(batch - 1) * b_b_stride + uint64_t(n - 1) * b_n_stride + 1;
    const uint64_t out_b_size =
        uint64_t(batch - 1) * out_b_b_stride + uint64_t(n - 1) * out_b_n_stride + 1;

    std::vector<float> a(a_size, sentinel);
    std::vector<float> b(b_size, sentinel);
    std::vector<float> original_b(b_size, sentinel);
    std::vector<float> out_b(out_b_size, sentinel);
    std::vector<float> expected_out_b(out_b_size, sentinel);

    for(uint32_t batch_id=0; batch_id < batch; batch_id++){
        for(uint32_t i=0; i < n; i++){
            const uint64_t a_idx =
                uint64_t(batch_id) * a_b_stride + uint64_t(i) * a_n_stride;
            const uint64_t b_idx =
                uint64_t(batch_id) * b_b_stride + uint64_t(i) * b_n_stride;
            const uint64_t out_b_idx =
                uint64_t(batch_id) * out_b_b_stride + uint64_t(i) * out_b_n_stride;
            a[a_idx] = float(batch_id * n + i) * 0.2f + 0.1f;
            b[b_idx] = float(batch_id * n + i) * 0.15f + 0.05f;
            expected_out_b[out_b_idx] = alpha * a[a_idx] + b[b_idx];
        }
    }
    original_b = b;

    auto bufferA = ctx.createBuffer(sizeof(float) * a.size(), vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(sizeof(float) * b.size(), vucol::BufferType::Auto);
    auto bufferOutB = ctx.createBuffer(sizeof(float) * out_b.size(), vucol::BufferType::Auto);

    bufferA.write(a.data(), sizeof(float) * a.size());
    bufferB.write(b.data(), sizeof(float) * b.size());
    bufferOutB.write(out_b.data(), sizeof(float) * out_b.size());

    vublas::AxpyOutPlaceArguments args = {
        .b = batch,
        .n = n,
        .alpha = alpha,
        .a_b_stride = a_b_stride,
        .a_n_stride = a_n_stride,
        .b_b_stride = b_b_stride,
        .b_n_stride = b_n_stride,
        .out_b_b_stride = out_b_b_stride,
        .out_b_n_stride = out_b_n_stride
    };

    auto token = execute_plan(ctx, axpy(bufferA, bufferB, bufferOutB, args));
    token.wait();
    bufferB.read(b.data(), sizeof(float) * b.size());
    bufferOutB.read(out_b.data(), sizeof(float) * out_b.size());

    EXPECT_TRUE(is_equal_tensor(out_b, expected_out_b));
    EXPECT_TRUE(is_equal_tensor(b, original_b));
}

TEST(BufferViewTest, AppliesNonZeroOffsetsToInputsAndOutput){
    vucol::Context ctx;
    vublas::AxpyOutPlaceFP32 axpy(ctx);

    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
    const std::vector<float> expected = {12.0f, 24.0f, 36.0f, 48.0f};
    std::vector<float> output(expected.size(), -1.0f);

    const std::size_t rangeSize = sizeof(float) * a.size();
    const std::size_t requiredAlignment =
        ctx.bufferOffsetAlignment(vucol::DescriptorType::StorageBuffer);
    const std::size_t alignment = requiredAlignment == 0 ? 1 : requiredAlignment;
    const std::size_t rangeStride =
        ((rangeSize + alignment - 1) / alignment) * alignment;
    const std::size_t aOffset = rangeStride;
    const std::size_t bOffset = rangeStride * 2;
    const std::size_t outputOffset = rangeStride * 3;
    const std::size_t workspaceSize = outputOffset + rangeSize;

    auto workspace = ctx.createBuffer(workspaceSize, vucol::BufferType::Auto);
    workspace.write(a.data(), rangeSize, aOffset);
    workspace.write(b.data(), rangeSize, bOffset);
    workspace.write(output.data(), rangeSize, outputOffset);

    const vublas::AxpyOutPlaceArguments args = {
        .b = 1,
        .n = static_cast<uint32_t>(a.size()),
        .alpha = 2.0f,
        .a_b_stride = static_cast<uint32_t>(a.size()),
        .a_n_stride = 1,
        .b_b_stride = static_cast<uint32_t>(b.size()),
        .b_n_stride = 1,
        .out_b_b_stride = static_cast<uint32_t>(output.size()),
        .out_b_n_stride = 1,
    };

    auto plan = axpy(
        vublas::BufferView{workspace, aOffset, rangeSize},
        vublas::BufferView{workspace, bOffset, rangeSize},
        vublas::BufferView{workspace, outputOffset, rangeSize},
        args
    );

    ASSERT_EQ(plan.bindings.size(), 3);
    EXPECT_EQ(plan.bindings[0].view.offset, aOffset);
    EXPECT_EQ(plan.bindings[0].view.size, rangeSize);
    EXPECT_EQ(plan.bindings[1].view.offset, bOffset);
    EXPECT_EQ(plan.bindings[1].view.size, rangeSize);
    EXPECT_EQ(plan.bindings[2].view.offset, outputOffset);
    EXPECT_EQ(plan.bindings[2].view.size, rangeSize);

    auto token = execute_plan(ctx, std::move(plan));
    token.wait();
    workspace.read(output.data(), rangeSize, outputOffset);

    EXPECT_TRUE(is_equal_tensor(output, expected));
}

template<typename MatMulOp>
void run_matmul_test(const char* op_name){
    vucol::Context ctx;
    //ctx.printGpuInfo(std::cout);
    /*for(auto tile : ctx.cooperativeMatrixSupportInfo().tiles){
        printf(
            "Tile: m=%d, n=%d, k=%d, aType=%d, bType=%d, cType=%d, resultType=%d, saturatingAccumulation=%d, scope=%d\n",
            tile.m,
            tile.n,
            tile.k,
            (int)tile.aType,
            (int)tile.bType,
            (int)tile.cType,
            (int)tile.resultType,
            (int)tile.saturatingAccumulation,
            (int)tile.scope
        );
    }*/
    MatMulOp matmul(ctx, 1, 1, 32, 4, 2, 4, 16, 1, 4);
    auto bufferA = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_n, vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(sizeof(float) * max_batch * max_n * max_p, vucol::BufferType::Auto);
    auto bufferC = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_p, vucol::BufferType::Auto);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> b_dis(1, max_batch);
    std::uniform_int_distribution<int> m_dis(1, max_m);
    std::uniform_int_distribution<int> n_dis(1, max_n);
    std::uniform_int_distribution<int> p_dis(1, max_p);
    std::uniform_real_distribution<float> f_dis(0.0, 1.0);
    for(int i=0; i < matmul_test_iter; i++){
        uint32_t batch = b_dis(gen);
        uint32_t m = m_dis(gen);
        uint32_t n = n_dis(gen);
        uint32_t p = p_dis(gen);

        std::vector<float> a(batch * m * n, 0.0);
        std::vector<float> b(batch * n * p, 0.0);
        std::vector<float> c(batch * m * p, 0.0);
        std::vector<float> cpu_c(batch * m * p, 0.0);
        std::vector<float> expected_c(batch * m * p, 0.0);

        const bool is_a_trans = f_dis(gen) > 0.5f;
        const bool is_b_trans = f_dis(gen) > 0.5f;
        const bool is_c_trans = false;

        for(uint64_t j=0; j < batch * m * n; j++){
            a[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * n * p; j++){
            b[j] = f_dis(gen);
        }

        SimpleBLAS::gemm(
            is_a_trans,
            is_b_trans,
            1.0f,
            0.0f,
            batch,
            m,
            n,
            p,
            a,
            b,
            cpu_c
        );

        if(!is_c_trans){
            expected_c = cpu_c;
        } else {
            const uint64_t c_size = uint64_t(m) * uint64_t(p);
            for(uint32_t batch_id=0; batch_id < batch; batch_id++){
                for(uint32_t r=0; r < m; r++){
                    for(uint32_t col=0; col < p; col++){
                        expected_c[uint64_t(batch_id) * c_size + uint64_t(col) * m + r] =
                            cpu_c[uint64_t(batch_id) * c_size + uint64_t(r) * p + col];
                    }
                }
            }
        }

        bufferA.write(a.data(), sizeof(float) * batch * m * n);
        bufferB.write(b.data(), sizeof(float) * batch * n * p);
        bufferC.write(c.data(), sizeof(float) * batch * m * p);
        vublas::MatMulArguments matmul_args = {
            .b = batch,
            .m = m,
            .n = n,
            .p = p,
            .a_stride = m * n,
            .b_stride = n * p,
            .c_stride = m * p,
            .a_m_stride = is_a_trans ? 1 : n,
            .a_n_stride = is_a_trans ? m : 1,
            .b_n_stride = is_b_trans ? 1 : p,
            .b_p_stride = is_b_trans ? n : 1,
            .c_m_stride = is_c_trans ? 1 : p,
            .c_p_stride = is_c_trans ? m : 1
        };
        auto token = execute_plan(
            ctx,
            matmul(bufferA, bufferB, bufferC, matmul_args)
        );
        token.wait();
        bufferC.read(c.data(), sizeof(float) * batch * m * p);

        const bool matmul_res = is_equal_tensor(c, expected_c);
        EXPECT_TRUE(matmul_res) << op_name << " result is not equal to CPU GEMM result!";
        if(!matmul_res){
            printf(
                "%s Transposed(A:%d, B:%d, C:%d)\n",
                op_name,
                is_a_trans ? 1 : 0,
                is_b_trans ? 1 : 0,
                is_c_trans ? 1 : 0
            );
            break;
        }
    }
}

TEST(GEMMTest, BasicAssertion){
    vucol::Context ctx;
    ctx.printGpuInfo(std::cout);
    vublas::GemmNaiveFP32 gemm(ctx, 1, 1, 32, 4, 2, 4, 16, 1, 4);
    auto bufferA = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_n, vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(sizeof(float) * max_batch * max_n * max_p, vucol::BufferType::Auto);
    auto bufferC = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_p, vucol::BufferType::Auto);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> b_dis(1, max_batch);
    std::uniform_int_distribution<int> m_dis(1, max_m);
    std::uniform_int_distribution<int> n_dis(1, max_n);
    std::uniform_int_distribution<int> p_dis(1, max_p);
    std::uniform_real_distribution<float> f_dis(0.0, 1.0);
    for(int i=0; i < gemm_test_iter; i++){
        // Generate Random vectors and alpha value
        uint32_t batch = b_dis(gen);
        uint32_t m = m_dis(gen);
        uint32_t n = n_dis(gen);
        uint32_t p = p_dis(gen);
        // batch = 1;
        // m = 4;
        // n = 4;
        // p = 4;
        float alpha = f_dis(gen);
        float beta = f_dis(gen);
        // alpha = 1.0;
        // beta = 0.0;

        std::vector<float> a(batch * m * n, 0.0);
        std::vector<float> b(batch * n * p, 0.0);
        std::vector<float> c(batch * m * p, 0.0);
        std::vector<float> cpu_c(batch * m * p, 0.0);

        const bool is_a_trans = f_dis(gen) > 0.5f;
        const bool is_b_trans = f_dis(gen) > 0.5f;
        const bool is_c_trans = false;

        for(uint64_t j=0; j < batch * m * n; j++){
            a[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * n * p; j++){
            b[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * m * p; j++){
            float rc = f_dis(gen);
            rc = 0.0f;
            c[j] = rc;
            cpu_c[j] = rc;
        }
        // printf("m=%d, n=%d, p=%d\n", m, n, p);
        // printf("[A]\n");
        // print_vector(a);
        // printf("[B]\n");
        // print_vector(b);
        // printf("[C]\n");
        // print_vector(c);
        // Run on CPU
        SimpleBLAS::gemm(is_a_trans, is_b_trans, alpha, beta, batch, m, n, p, a, b, cpu_c);
        // printf("[CPU_C](after solve)\n");
        // print_vector(cpu_c);

        bufferA.write(a.data(), sizeof(float) * batch * m * n);
        bufferB.write(b.data(), sizeof(float) * batch * n * p);
        bufferC.write(c.data(), sizeof(float) * batch * m * p);
        vublas::GemmArguments gemm_args = {
            .b = batch,
            .m = m,
            .n = n,
            .p = p,
            .alpha = alpha,
            .beta = beta,
            .a_stride = m * n,
            .b_stride = n * p,
            .c_stride = m * p,
            .a_m_stride = is_a_trans ? 1 : n,
            .a_n_stride = is_a_trans ? m : 1,
            .b_n_stride = is_b_trans ? 1 : p,
            .b_p_stride = is_b_trans ? n : 1,
            .c_m_stride = is_c_trans ? 1 : p,
            .c_p_stride = is_c_trans ? m : 1
        };
        auto token = execute_plan(
            ctx,
            gemm(bufferA, bufferB, bufferC, gemm_args)
        );
        token.wait();
        bufferC.read(c.data(), sizeof(float) * batch * m * p);
        // Test!
        const bool gemm_res = is_equal_tensor(c, cpu_c);
        EXPECT_TRUE(gemm_res) << "GEMM result is not equal to CPU result!";
        if(!gemm_res){
            printf("Transposed(A:%d, B:%d)\n", is_a_trans ? 1 : 0, is_b_trans ? 1 : 0);
            break;
        }
    }
}

TEST(GemvNaiveTest, BasicAssertion){
    vucol::Context ctx;
    // For extremely small P case of GEMV, arguments should be manually modified.
    // So, I modified the arguments by hand.
    vublas::GemvNaiveFP32 gemv(ctx, 16, 1, 2, 1, 2, 1, 16, 1, 4);
    auto bufferA = ctx.createBuffer(sizeof(float) * max_m * max_n, vucol::BufferType::Auto);
    auto bufferX = ctx.createBuffer(sizeof(float) * max_batch * max_n, vucol::BufferType::Auto);
    auto bufferY = ctx.createBuffer(sizeof(float) * max_batch * max_m, vucol::BufferType::Auto);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> b_dis(1, max_batch);
    std::uniform_int_distribution<int> m_dis(1, max_m);
    std::uniform_int_distribution<int> n_dis(1, max_n);
    std::uniform_real_distribution<float> f_dis(0.0, 1.0);
    for(int i=0; i < gemm_test_iter; i++){
        uint32_t batch = b_dis(gen);
        uint32_t m = m_dis(gen);
        uint32_t n = n_dis(gen);
        float alpha = f_dis(gen);
        float beta = f_dis(gen);

        std::vector<float> a(m * n, 0.0);
        std::vector<float> x(batch * n, 0.0);
        std::vector<float> y(batch * m, 0.0);
        std::vector<float> cpu_y(batch * m, 0.0);

        const bool is_a_trans = f_dis(gen) > 0.5f;

        for(uint64_t j=0; j < m * n; j++){
            a[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * n; j++){
            x[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * m; j++){
            float ry = f_dis(gen);
            y[j] = ry;
            cpu_y[j] = ry;
        }

        bufferA.write(a.data(), sizeof(float) * m * n);
        bufferX.write(x.data(), sizeof(float) * batch * n);
        bufferY.write(y.data(), sizeof(float) * batch * m);
        vublas::GemvArguments gemv_args = {
            .b = batch,
            .m = m,
            .n = n,
            .alpha = alpha,
            .beta = beta,
            .a_m_stride = is_a_trans ? 1 : n,
            .a_n_stride = is_a_trans ? m : 1,
            .x_n_stride = 1,
            .x_b_stride = n,
            .y_m_stride = 1,
            .y_b_stride = m
        };
        run_cpu_gemv(gemv_args, a, x, cpu_y);

        auto token = execute_plan(
            ctx,
            gemv(bufferA, bufferX, bufferY, gemv_args)
        );
        token.wait();
        bufferY.read(y.data(), sizeof(float) * batch * m);

        const bool gemv_res = is_equal_tensor(y, cpu_y);
        EXPECT_TRUE(gemv_res) << "GemvNaiveFP32 result is not equal to CPU result!";
        if(!gemv_res){
            printf("Transposed(A:%d)\n", is_a_trans ? 1 : 0);
            break;
        }
    }
}

TEST(GemmOutPlaceNaiveTest, BasicAssertion){
    vucol::Context ctx;
    vublas::GemmOutPlaceNaiveFP32 gemm(
        ctx, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );
    auto bufferA = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_n, vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(sizeof(float) * max_batch * max_n * max_p, vucol::BufferType::Auto);
    auto bufferC = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_p, vucol::BufferType::Auto);
    auto bufferOutC = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_p, vucol::BufferType::Auto);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> b_dis(1, max_batch);
    std::uniform_int_distribution<int> m_dis(1, max_m);
    std::uniform_int_distribution<int> n_dis(1, max_n);
    std::uniform_int_distribution<int> p_dis(1, max_p);
    std::uniform_real_distribution<float> f_dis(0.0, 1.0);
    for(int i=0; i < gemm_test_iter; i++){
        uint32_t batch = b_dis(gen);
        uint32_t m = m_dis(gen);
        uint32_t n = n_dis(gen);
        uint32_t p = p_dis(gen);
        float alpha = f_dis(gen);
        float beta = f_dis(gen);

        std::vector<float> a(batch * m * n, 0.0);
        std::vector<float> b(batch * n * p, 0.0);
        std::vector<float> c(batch * m * p, 0.0);
        std::vector<float> out_c(batch * m * p, 0.0);
        std::vector<float> cpu_c(batch * m * p, 0.0);
        std::vector<float> original_c(batch * m * p, 0.0);

        const bool is_a_trans = f_dis(gen) > 0.5f;
        const bool is_b_trans = f_dis(gen) > 0.5f;
        const bool is_c_trans = false;

        for(uint64_t j=0; j < batch * m * n; j++){
            a[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * n * p; j++){
            b[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * m * p; j++){
            float rc = f_dis(gen);
            c[j] = rc;
            cpu_c[j] = rc;
            original_c[j] = rc;
        }

        SimpleBLAS::gemm(is_a_trans, is_b_trans, alpha, beta, batch, m, n, p, a, b, cpu_c);

        bufferA.write(a.data(), sizeof(float) * batch * m * n);
        bufferB.write(b.data(), sizeof(float) * batch * n * p);
        bufferC.write(c.data(), sizeof(float) * batch * m * p);
        bufferOutC.write(out_c.data(), sizeof(float) * batch * m * p);
        vublas::GemmArguments gemm_args = {
            .b = batch,
            .m = m,
            .n = n,
            .p = p,
            .alpha = alpha,
            .beta = beta,
            .a_stride = m * n,
            .b_stride = n * p,
            .c_stride = m * p,
            .a_m_stride = is_a_trans ? 1 : n,
            .a_n_stride = is_a_trans ? m : 1,
            .b_n_stride = is_b_trans ? 1 : p,
            .b_p_stride = is_b_trans ? n : 1,
            .c_m_stride = is_c_trans ? 1 : p,
            .c_p_stride = is_c_trans ? m : 1
        };
        auto token = execute_plan(
            ctx,
            gemm(bufferA, bufferB, bufferC, bufferOutC, gemm_args)
        );
        token.wait();
        bufferOutC.read(out_c.data(), sizeof(float) * batch * m * p);
        bufferC.read(c.data(), sizeof(float) * batch * m * p);

        const bool gemm_res = is_equal_tensor(out_c, cpu_c);
        const bool input_c_res = is_equal_tensor(c, original_c);
        EXPECT_TRUE(gemm_res) << "GemmOutPlaceNaiveFP32 result is not equal to CPU result!";
        EXPECT_TRUE(input_c_res) << "GemmOutPlaceNaiveFP32 modified input C!";
        if(!gemm_res || !input_c_res){
            printf("Transposed(A:%d, B:%d)\n", is_a_trans ? 1 : 0, is_b_trans ? 1 : 0);
            break;
        }
    }
}

TEST(GemmOutPlaceNaiveTest, SupportsDistinctOutputStride){
    vucol::Context ctx;
    vublas::GemmOutPlaceNaiveFP32 gemm(
        ctx, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );

    const uint32_t batch = 2;
    const uint32_t m = 3;
    const uint32_t n = 4;
    const uint32_t p = 2;
    const uint32_t out_p_stride = 1;
    const uint32_t out_m_stride = p + 2;
    const uint32_t out_c_stride = m * out_m_stride;
    const float alpha = 1.25f;
    const float beta = 0.5f;
    const float sentinel = -777.0f;

    std::vector<float> a(batch * m * n, 0.0f);
    std::vector<float> b(batch * n * p, 0.0f);
    std::vector<float> c(batch * m * p, 0.0f);
    std::vector<float> original_c(batch * m * p, 0.0f);
    std::vector<float> cpu_c(batch * m * p, 0.0f);
    std::vector<float> out_c(batch * out_c_stride, sentinel);
    std::vector<float> expected_out_c(batch * out_c_stride, sentinel);

    for(uint64_t i=0; i < a.size(); i++){
        a[i] = float(i % 11) * 0.1f + 0.25f;
    }
    for(uint64_t i=0; i < b.size(); i++){
        b[i] = float(i % 7) * 0.2f + 0.1f;
    }
    for(uint64_t i=0; i < c.size(); i++){
        c[i] = float(i % 5) * 0.15f + 0.05f;
        original_c[i] = c[i];
        cpu_c[i] = c[i];
    }

    SimpleBLAS::gemm(false, false, alpha, beta, batch, m, n, p, a, b, cpu_c);
    for(uint32_t batch_id=0; batch_id < batch; batch_id++){
        for(uint32_t r=0; r < m; r++){
            for(uint32_t col=0; col < p; col++){
                const uint64_t compact_idx =
                    uint64_t(batch_id) * m * p + uint64_t(r) * p + col;
                const uint64_t out_idx =
                    uint64_t(batch_id) * out_c_stride + uint64_t(r) * out_m_stride + col * out_p_stride;
                expected_out_c[out_idx] = cpu_c[compact_idx];
            }
        }
    }

    auto bufferA = ctx.createBuffer(sizeof(float) * a.size(), vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(sizeof(float) * b.size(), vucol::BufferType::Auto);
    auto bufferC = ctx.createBuffer(sizeof(float) * c.size(), vucol::BufferType::Auto);
    auto bufferOutC = ctx.createBuffer(sizeof(float) * out_c.size(), vucol::BufferType::Auto);

    bufferA.write(a.data(), sizeof(float) * a.size());
    bufferB.write(b.data(), sizeof(float) * b.size());
    bufferC.write(c.data(), sizeof(float) * c.size());
    bufferOutC.write(out_c.data(), sizeof(float) * out_c.size());

    vublas::GemmOutPlaceArguments gemm_args = vublas::GemmOutPlaceArguments::sameOutputLayout({
        .b = batch,
        .m = m,
        .n = n,
        .p = p,
        .alpha = alpha,
        .beta = beta,
        .a_stride = m * n,
        .b_stride = n * p,
        .c_stride = m * p,
        .a_m_stride = n,
        .a_n_stride = 1,
        .b_n_stride = p,
        .b_p_stride = 1,
        .c_m_stride = p,
        .c_p_stride = 1
    });
    gemm_args.out_c_stride = out_c_stride;
    gemm_args.out_c_m_stride = out_m_stride;
    gemm_args.out_c_p_stride = out_p_stride;

    auto gemm_token = execute_plan(
        ctx,
        gemm(bufferA, bufferB, bufferC, bufferOutC, gemm_args)
    );
    gemm_token.wait();
    bufferOutC.read(out_c.data(), sizeof(float) * out_c.size());
    bufferC.read(c.data(), sizeof(float) * c.size());

    EXPECT_TRUE(is_equal_tensor(out_c, expected_out_c));
    EXPECT_TRUE(is_equal_tensor(c, original_c));
}

TEST(GemvOutPlaceNaiveTest, BasicAssertion){
    vucol::Context ctx;
    vublas::GemvOutPlaceNaiveFP32 gemv(
        ctx, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );
    auto bufferA = ctx.createBuffer(sizeof(float) * max_m * max_n, vucol::BufferType::Auto);
    auto bufferX = ctx.createBuffer(sizeof(float) * max_batch * max_n, vucol::BufferType::Auto);
    auto bufferY = ctx.createBuffer(sizeof(float) * max_batch * max_m, vucol::BufferType::Auto);
    auto bufferOutY = ctx.createBuffer(sizeof(float) * max_batch * max_m, vucol::BufferType::Auto);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> b_dis(1, max_batch);
    std::uniform_int_distribution<int> m_dis(1, max_m);
    std::uniform_int_distribution<int> n_dis(1, max_n);
    std::uniform_real_distribution<float> f_dis(0.0, 1.0);
    for(int i=0; i < gemm_test_iter; i++){
        uint32_t batch = b_dis(gen);
        uint32_t m = m_dis(gen);
        uint32_t n = n_dis(gen);
        float alpha = f_dis(gen);
        float beta = f_dis(gen);

        std::vector<float> a(m * n, 0.0);
        std::vector<float> x(batch * n, 0.0);
        std::vector<float> y(batch * m, 0.0);
        std::vector<float> out_y(batch * m, 0.0);
        std::vector<float> cpu_y(batch * m, 0.0);
        std::vector<float> original_y(batch * m, 0.0);

        const bool is_a_trans = f_dis(gen) > 0.5f;

        for(uint64_t j=0; j < m * n; j++){
            a[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * n; j++){
            x[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * m; j++){
            float ry = f_dis(gen);
            y[j] = ry;
            cpu_y[j] = ry;
            original_y[j] = ry;
        }

        bufferA.write(a.data(), sizeof(float) * m * n);
        bufferX.write(x.data(), sizeof(float) * batch * n);
        bufferY.write(y.data(), sizeof(float) * batch * m);
        bufferOutY.write(out_y.data(), sizeof(float) * batch * m);
        vublas::GemvArguments gemv_args = {
            .b = batch,
            .m = m,
            .n = n,
            .alpha = alpha,
            .beta = beta,
            .a_m_stride = is_a_trans ? 1 : n,
            .a_n_stride = is_a_trans ? m : 1,
            .x_n_stride = 1,
            .x_b_stride = n,
            .y_m_stride = 1,
            .y_b_stride = m
        };
        run_cpu_gemv(gemv_args, a, x, cpu_y);

        auto token = execute_plan(
            ctx,
            gemv(bufferA, bufferX, bufferY, bufferOutY, gemv_args)
        );
        token.wait();
        bufferOutY.read(out_y.data(), sizeof(float) * batch * m);
        bufferY.read(y.data(), sizeof(float) * batch * m);

        const bool gemv_res = is_equal_tensor(out_y, cpu_y);
        const bool input_y_res = is_equal_tensor(y, original_y);
        EXPECT_TRUE(gemv_res) << "GemvOutPlaceNaiveFP32 result is not equal to CPU result!";
        EXPECT_TRUE(input_y_res) << "GemvOutPlaceNaiveFP32 modified input Y!";
        if(!gemv_res || !input_y_res){
            printf("Transposed(A:%d)\n", is_a_trans ? 1 : 0);
            break;
        }
    }
}

TEST(GemvOutPlaceNaiveTest, SupportsDistinctOutputStride){
    vucol::Context ctx;
    vublas::GemvOutPlaceNaiveFP32 gemv(
        ctx, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );

    const uint32_t batch = 3;
    const uint32_t m = 4;
    const uint32_t n = 3;
    const uint32_t out_y_m_stride = 2;
    const uint32_t out_y_b_stride = m * out_y_m_stride + 3;
    const float alpha = 0.75f;
    const float beta = 0.25f;
    const float sentinel = -333.0f;

    const uint64_t out_y_size =
        uint64_t(batch - 1) * out_y_b_stride + uint64_t(m - 1) * out_y_m_stride + 1;

    std::vector<float> a(m * n, 0.0f);
    std::vector<float> x(batch * n, 0.0f);
    std::vector<float> y(batch * m, 0.0f);
    std::vector<float> original_y(batch * m, 0.0f);
    std::vector<float> cpu_y(batch * m, 0.0f);
    std::vector<float> out_y(out_y_size, sentinel);
    std::vector<float> expected_out_y(out_y_size, sentinel);

    for(uint64_t i=0; i < a.size(); i++){
        a[i] = float(i % 9) * 0.11f + 0.2f;
    }
    for(uint64_t i=0; i < x.size(); i++){
        x[i] = float(i % 6) * 0.17f + 0.05f;
    }
    for(uint64_t i=0; i < y.size(); i++){
        y[i] = float(i % 4) * 0.13f + 0.07f;
        original_y[i] = y[i];
        cpu_y[i] = y[i];
    }

    vublas::GemvArguments gemv_args = {
        .b = batch,
        .m = m,
        .n = n,
        .alpha = alpha,
        .beta = beta,
        .a_m_stride = n,
        .a_n_stride = 1,
        .x_n_stride = 1,
        .x_b_stride = n,
        .y_m_stride = 1,
        .y_b_stride = m
    };
    run_cpu_gemv(gemv_args, a, x, cpu_y);

    for(uint32_t batch_id=0; batch_id < batch; batch_id++){
        for(uint32_t r=0; r < m; r++){
            const uint64_t compact_idx = uint64_t(batch_id) * m + r;
            const uint64_t out_idx = uint64_t(batch_id) * out_y_b_stride + uint64_t(r) * out_y_m_stride;
            expected_out_y[out_idx] = cpu_y[compact_idx];
        }
    }

    auto bufferA = ctx.createBuffer(sizeof(float) * a.size(), vucol::BufferType::Auto);
    auto bufferX = ctx.createBuffer(sizeof(float) * x.size(), vucol::BufferType::Auto);
    auto bufferY = ctx.createBuffer(sizeof(float) * y.size(), vucol::BufferType::Auto);
    auto bufferOutY = ctx.createBuffer(sizeof(float) * out_y.size(), vucol::BufferType::Auto);

    bufferA.write(a.data(), sizeof(float) * a.size());
    bufferX.write(x.data(), sizeof(float) * x.size());
    bufferY.write(y.data(), sizeof(float) * y.size());
    bufferOutY.write(out_y.data(), sizeof(float) * out_y.size());

    vublas::GemvOutPlaceArguments gemv_out_args = vublas::GemvOutPlaceArguments::sameOutputLayout(gemv_args);
    gemv_out_args.out_y_m_stride = out_y_m_stride;
    gemv_out_args.out_y_b_stride = out_y_b_stride;

    auto gemv_token = execute_plan(
        ctx,
        gemv(bufferA, bufferX, bufferY, bufferOutY, gemv_out_args)
    );
    gemv_token.wait();
    bufferOutY.read(out_y.data(), sizeof(float) * out_y.size());
    bufferY.read(y.data(), sizeof(float) * y.size());

    EXPECT_TRUE(is_equal_tensor(out_y, expected_out_y));
    EXPECT_TRUE(is_equal_tensor(y, original_y));
}

TEST(ReductionNaiveTest, ComputesDotProductWithBatchStrides){
    vucol::Context ctx;
    vublas::DotProductNaiveFP32 dot(ctx, 32, 2);

    constexpr uint32_t batch = 2;
    constexpr uint32_t n = 67;
    const uint32_t a_n_stride = 2;
    const uint32_t a_b_stride = n * a_n_stride + 3;
    const uint32_t b_n_stride = 3;
    const uint32_t b_b_stride = n * b_n_stride + 2;
    const uint32_t out_b_stride = 2;
    const uint32_t out_n_stride = 1;
    const float sentinel = -777.0f;

    const uint64_t a_size =
        uint64_t(batch - 1) * a_b_stride + uint64_t(n - 1) * a_n_stride + 1;
    const uint64_t b_size =
        uint64_t(batch - 1) * b_b_stride + uint64_t(n - 1) * b_n_stride + 1;
    const uint64_t out_size = uint64_t(batch - 1) * out_b_stride + 1;

    std::vector<float> a(a_size, sentinel);
    std::vector<float> b(b_size, sentinel);
    std::vector<float> out(out_size, sentinel);
    std::vector<float> expected_out(out_size, sentinel);

    for(uint32_t batch_id=0; batch_id < batch; batch_id++){
        float acc = 0.0f;
        for(uint32_t i=0; i < n; i++){
            const uint64_t a_idx =
                uint64_t(batch_id) * a_b_stride + uint64_t(i) * a_n_stride;
            const uint64_t b_idx =
                uint64_t(batch_id) * b_b_stride + uint64_t(i) * b_n_stride;
            const float a_value =
                float(int32_t(i % 11) - 5) * 0.25f + float(batch_id);
            const float b_value =
                float(int32_t(i % 7) - 3) * 0.5f - float(batch_id) * 0.25f;
            a[a_idx] = a_value;
            b[b_idx] = b_value;
            acc += a_value * b_value;
        }

        const uint64_t out_idx = uint64_t(batch_id) * out_b_stride;
        expected_out[out_idx] = acc;
    }

    auto bufferA = ctx.createBuffer(sizeof(float) * a.size(), vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(sizeof(float) * b.size(), vucol::BufferType::Auto);
    auto bufferOut = ctx.createBuffer(sizeof(float) * out.size(), vucol::BufferType::Auto);

    bufferA.write(a.data(), sizeof(float) * a.size());
    bufferB.write(b.data(), sizeof(float) * b.size());
    bufferOut.write(out.data(), sizeof(float) * out.size());

    vublas::BinaryReductionArguments args = {
        .b = batch,
        .n = n,
        .a_b_stride = a_b_stride,
        .a_n_stride = a_n_stride,
        .b_b_stride = b_b_stride,
        .b_n_stride = b_n_stride,
        .out_b_stride = out_b_stride,
        .out_n_stride = out_n_stride
    };

    auto dot_token = execute_plan(
        ctx,
        dot(bufferA, bufferB, bufferOut, args)
    );
    dot_token.wait();
    bufferOut.read(out.data(), sizeof(float) * out.size());

    EXPECT_TRUE(is_equal_tensor(out, expected_out));
}

TEST(ElementWiseTemplateTest, AppliesUnaryOperationOutOfPlace){
    constexpr std::string_view squarePlusOne = R"(
float operation(float x){
    return fma(x, x, 1.0f);
}
)";
    vucol::Context ctx;
    vublas::UnaryElementwiseTemplateFP32 operation(ctx, squarePlusOne, 32);

    constexpr uint32_t size = 67;
    std::vector<float> input(size);
    std::vector<float> output(size, -1.0f);
    std::vector<float> expected(size);
    for(uint32_t i=0; i < size; i++){
        input[i] = float(int32_t(i) - 31) * 0.25f;
        expected[i] = input[i] * input[i] + 1.0f;
    }

    auto bufferInput =
        ctx.createBuffer(sizeof(float) * input.size(), vucol::BufferType::Auto);
    auto bufferOutput =
        ctx.createBuffer(sizeof(float) * output.size(), vucol::BufferType::Auto);
    bufferInput.write(input.data(), sizeof(float) * input.size());
    bufferOutput.write(output.data(), sizeof(float) * output.size());

    execute_plan(
        ctx,
        operation(
            bufferInput,
            bufferOutput,
            vublas::UnaryElementwiseArguments{.size = size}
        )
    ).wait();
    bufferOutput.read(output.data(), sizeof(float) * output.size());

    EXPECT_TRUE(is_equal_tensor(output, expected));
}

TEST(ElementWiseTemplateTest, AppliesBinaryOperationOutOfPlace){
    constexpr std::string_view multiplyAdd = R"(
float operation(float x, float y){
    return fma(x, y, x);
}
)";
    vucol::Context ctx;
    vublas::BinaryElementwiseTemplateFP32 operation(ctx, multiplyAdd, 32);

    constexpr uint32_t size = 67;
    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> output(size, -1.0f);
    std::vector<float> expected(size);
    for(uint32_t i=0; i < size; i++){
        a[i] = float(int32_t(i % 13) - 6) * 0.5f;
        b[i] = float(int32_t(i % 9) - 4) * 0.25f;
        expected[i] = a[i] * b[i] + a[i];
    }

    auto bufferA = ctx.createBuffer(sizeof(float) * a.size(), vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(sizeof(float) * b.size(), vucol::BufferType::Auto);
    auto bufferOutput =
        ctx.createBuffer(sizeof(float) * output.size(), vucol::BufferType::Auto);
    bufferA.write(a.data(), sizeof(float) * a.size());
    bufferB.write(b.data(), sizeof(float) * b.size());
    bufferOutput.write(output.data(), sizeof(float) * output.size());

    execute_plan(
        ctx,
        operation(
            bufferA,
            bufferB,
            bufferOutput,
            vublas::BinaryElementwiseArguments{.size = size}
        )
    ).wait();
    bufferOutput.read(output.data(), sizeof(float) * output.size());

    EXPECT_TRUE(is_equal_tensor(output, expected));
}

TEST(ReductionNaiveTest, ComputesSumAndAvgWithBatchStrides){
    vucol::Context ctx;
    vublas::SumNaiveFP32 sum(ctx);
    vublas::AvgNaiveFP32 avg(ctx);

    constexpr uint32_t batch = 2;
    constexpr uint32_t n = 5;
    const uint32_t a_n_stride = 2;
    const uint32_t a_b_stride = n * a_n_stride + 3;
    const uint32_t out_b_stride = 2;
    const uint32_t out_n_stride = 1;
    const float sentinel = -777.0f;

    const uint64_t a_size =
        uint64_t(batch - 1) * a_b_stride + uint64_t(n - 1) * a_n_stride + 1;
    const uint64_t out_size = uint64_t(batch - 1) * out_b_stride + 1;

    std::vector<float> a(a_size, sentinel);
    std::vector<float> sum_out(out_size, sentinel);
    std::vector<float> avg_out(out_size, sentinel);
    std::vector<float> expected_sum(out_size, sentinel);
    std::vector<float> expected_avg(out_size, sentinel);

    const float values[batch][n] = {
        {1.0f, -2.0f, 3.5f, 4.0f, -1.0f},
        {-3.0f, 7.0f, 2.0f, -5.0f, 9.0f}
    };

    for(uint32_t batch_id=0; batch_id < batch; batch_id++){
        float acc = 0.0f;
        for(uint32_t i=0; i < n; i++){
            const uint64_t a_idx =
                uint64_t(batch_id) * a_b_stride + uint64_t(i) * a_n_stride;
            a[a_idx] = values[batch_id][i];
            acc += values[batch_id][i];
        }

        const uint64_t out_idx = uint64_t(batch_id) * out_b_stride;
        expected_sum[out_idx] = acc;
        expected_avg[out_idx] = acc / float(n);
    }

    auto bufferA = ctx.createBuffer(sizeof(float) * a.size(), vucol::BufferType::Auto);
    auto bufferSumOut = ctx.createBuffer(sizeof(float) * sum_out.size(), vucol::BufferType::Auto);
    auto bufferAvgOut = ctx.createBuffer(sizeof(float) * avg_out.size(), vucol::BufferType::Auto);

    bufferA.write(a.data(), sizeof(float) * a.size());
    bufferSumOut.write(sum_out.data(), sizeof(float) * sum_out.size());
    bufferAvgOut.write(avg_out.data(), sizeof(float) * avg_out.size());

    vublas::UnaryReductionArguments args = {
        .b = batch,
        .n = n,
        .a_b_stride = a_b_stride,
        .a_n_stride = a_n_stride,
        .out_b_stride = out_b_stride,
        .out_n_stride = out_n_stride
    };

    auto sum_token = execute_plan(ctx, sum(bufferA, bufferSumOut, args));
    sum_token.wait();
    auto avg_token = execute_plan(ctx, avg(bufferA, bufferAvgOut, args));
    avg_token.wait();

    bufferSumOut.read(sum_out.data(), sizeof(float) * sum_out.size());
    bufferAvgOut.read(avg_out.data(), sizeof(float) * avg_out.size());

    EXPECT_TRUE(is_equal_tensor(sum_out, expected_sum));
    EXPECT_TRUE(is_equal_tensor(avg_out, expected_avg));
}

TEST(ReductionNaiveTest, ComputesMaxAndMinValuesAndIndices){
    vucol::Context ctx;
    vublas::MaxNaiveFP32 max(ctx);
    vublas::MinNaiveFP32 min(ctx);

    constexpr uint32_t batch = 2;
    constexpr uint32_t n = 6;
    const uint32_t a_n_stride = 2;
    const uint32_t a_b_stride = n * a_n_stride + 4;
    const uint32_t out_value_b_stride = 2;
    const uint32_t out_value_n_stride = 1;
    const uint32_t out_index_b_stride = 2;
    const uint32_t out_index_n_stride = 1;
    const float value_sentinel = -555.0f;
    const uint32_t index_sentinel = 999u;

    const uint64_t a_size =
        uint64_t(batch - 1) * a_b_stride + uint64_t(n - 1) * a_n_stride + 1;
    const uint64_t out_value_size =
        uint64_t(batch - 1) * out_value_b_stride + 1;
    const uint64_t out_index_size =
        uint64_t(batch - 1) * out_index_b_stride + 1;

    std::vector<float> a(a_size, value_sentinel);
    std::vector<float> max_values(out_value_size, value_sentinel);
    std::vector<float> min_values(out_value_size, value_sentinel);
    std::vector<float> expected_max_values(out_value_size, value_sentinel);
    std::vector<float> expected_min_values(out_value_size, value_sentinel);
    std::vector<uint32_t> max_indices(out_index_size, index_sentinel);
    std::vector<uint32_t> min_indices(out_index_size, index_sentinel);
    std::vector<uint32_t> expected_max_indices(out_index_size, index_sentinel);
    std::vector<uint32_t> expected_min_indices(out_index_size, index_sentinel);

    const float values[batch][n] = {
        {1.0f, 7.0f, -3.0f, 7.0f, -3.0f, 2.0f},
        {4.0f, -8.0f, 6.0f, 6.0f, -8.0f, 5.0f}
    };

    for(uint32_t batch_id=0; batch_id < batch; batch_id++){
        uint32_t max_idx = 0;
        uint32_t min_idx = 0;
        for(uint32_t i=0; i < n; i++){
            const uint64_t a_idx =
                uint64_t(batch_id) * a_b_stride + uint64_t(i) * a_n_stride;
            a[a_idx] = values[batch_id][i];

            if(values[batch_id][i] > values[batch_id][max_idx]){
                max_idx = i;
            }
            if(values[batch_id][i] < values[batch_id][min_idx]){
                min_idx = i;
            }
        }

        const uint64_t value_idx = uint64_t(batch_id) * out_value_b_stride;
        const uint64_t index_idx = uint64_t(batch_id) * out_index_b_stride;
        expected_max_values[value_idx] = values[batch_id][max_idx];
        expected_min_values[value_idx] = values[batch_id][min_idx];
        expected_max_indices[index_idx] = max_idx;
        expected_min_indices[index_idx] = min_idx;
    }

    auto bufferA = ctx.createBuffer(sizeof(float) * a.size(), vucol::BufferType::Auto);
    auto bufferMaxValues = ctx.createBuffer(sizeof(float) * max_values.size(), vucol::BufferType::Auto);
    auto bufferMinValues = ctx.createBuffer(sizeof(float) * min_values.size(), vucol::BufferType::Auto);
    auto bufferMaxIndices = ctx.createBuffer(sizeof(uint32_t) * max_indices.size(), vucol::BufferType::Auto);
    auto bufferMinIndices = ctx.createBuffer(sizeof(uint32_t) * min_indices.size(), vucol::BufferType::Auto);

    bufferA.write(a.data(), sizeof(float) * a.size());
    bufferMaxValues.write(max_values.data(), sizeof(float) * max_values.size());
    bufferMinValues.write(min_values.data(), sizeof(float) * min_values.size());
    bufferMaxIndices.write(max_indices.data(), sizeof(uint32_t) * max_indices.size());
    bufferMinIndices.write(min_indices.data(), sizeof(uint32_t) * min_indices.size());

    vublas::IndexedUnaryReductionArguments args = {
        .b = batch,
        .n = n,
        .a_b_stride = a_b_stride,
        .a_n_stride = a_n_stride,
        .out_value_b_stride = out_value_b_stride,
        .out_value_n_stride = out_value_n_stride,
        .out_index_b_stride = out_index_b_stride,
        .out_index_n_stride = out_index_n_stride
    };

    auto max_token = execute_plan(
        ctx,
        max(bufferA, bufferMaxValues, bufferMaxIndices, args)
    );
    max_token.wait();
    auto min_token = execute_plan(
        ctx,
        min(bufferA, bufferMinValues, bufferMinIndices, args)
    );
    min_token.wait();

    bufferMaxValues.read(max_values.data(), sizeof(float) * max_values.size());
    bufferMinValues.read(min_values.data(), sizeof(float) * min_values.size());
    bufferMaxIndices.read(max_indices.data(), sizeof(uint32_t) * max_indices.size());
    bufferMinIndices.read(min_indices.data(), sizeof(uint32_t) * min_indices.size());

    EXPECT_TRUE(is_equal_tensor(max_values, expected_max_values));
    EXPECT_TRUE(is_equal_tensor(min_values, expected_min_values));
    EXPECT_EQ(max_indices, expected_max_indices);
    EXPECT_EQ(min_indices, expected_min_indices);
}

TEST(MatMulNaiveTest, BasicAssertion){
    run_matmul_test<vublas::MatMulNaiveFP32>("MatMulNaiveFP32");
}

TEST(NaiveTemplateTest, AppliesRuntimeReluEpilogue){
    constexpr std::string_view relu = R"(
float epilogue(float value){
    return max(value, 0.0f);
}
)";
    vucol::Context ctx;
    auto bufferA = ctx.createBuffer(sizeof(float), vucol::BufferType::Auto);
    auto bufferB = ctx.createBuffer(sizeof(float), vucol::BufferType::Auto);
    auto bufferC = ctx.createBuffer(sizeof(float), vucol::BufferType::Auto);
    auto bufferOut = ctx.createBuffer(sizeof(float), vucol::BufferType::Auto);

    const float a = -2.0f;
    const float b = 3.0f;
    const float initial = 1.0f;
    float result = initial;
    bufferA.write(&a, sizeof(a));
    bufferB.write(&b, sizeof(b));

    const vublas::GemmArguments gemmArgs = {
        .b = 1, .m = 1, .n = 1, .p = 1,
        .alpha = 1.0f, .beta = 0.0f,
        .a_stride = 1, .b_stride = 1, .c_stride = 1,
        .a_m_stride = 1, .a_n_stride = 1,
        .b_n_stride = 1, .b_p_stride = 1,
        .c_m_stride = 1, .c_p_stride = 1
    };
    const vublas::MatMulArguments matmulArgs = {
        .b = 1, .m = 1, .n = 1, .p = 1,
        .a_stride = 1, .b_stride = 1, .c_stride = 1,
        .a_m_stride = 1, .a_n_stride = 1,
        .b_n_stride = 1, .b_p_stride = 1,
        .c_m_stride = 1, .c_p_stride = 1
    };
    const vublas::GemvArguments gemvArgs = {
        .b = 1, .m = 1, .n = 1,
        .alpha = 1.0f, .beta = 0.0f,
        .a_m_stride = 1, .a_n_stride = 1,
        .x_n_stride = 1, .x_b_stride = 1,
        .y_m_stride = 1, .y_b_stride = 1
    };

    vublas::GemmNaiveTemplateFP32 gemm(
        ctx, relu, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );
    bufferC.write(&initial, sizeof(initial));
    execute_plan(ctx, gemm(bufferA, bufferB, bufferC, gemmArgs)).wait();
    bufferC.read(&result, sizeof(result));
    EXPECT_FLOAT_EQ(result, 0.0f);

    vublas::GemmOutPlaceNaiveTemplateFP32 gemmOutPlace(
        ctx, relu, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );
    bufferC.write(&initial, sizeof(initial));
    bufferOut.write(&initial, sizeof(initial));
    execute_plan(
        ctx,
        gemmOutPlace(
            bufferA,
            bufferB,
            bufferC,
            bufferOut,
            vublas::GemmOutPlaceArguments::sameOutputLayout(gemmArgs)
        )
    ).wait();
    bufferOut.read(&result, sizeof(result));
    EXPECT_FLOAT_EQ(result, 0.0f);

    vublas::MatMulNaiveTemplateFP32 matmul(
        ctx, relu, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );
    bufferOut.write(&initial, sizeof(initial));
    execute_plan(ctx, matmul(bufferA, bufferB, bufferOut, matmulArgs)).wait();
    bufferOut.read(&result, sizeof(result));
    EXPECT_FLOAT_EQ(result, 0.0f);

    vublas::GemvNaiveTemplateFP32 gemv(
        ctx, relu, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );
    bufferC.write(&initial, sizeof(initial));
    execute_plan(ctx, gemv(bufferA, bufferB, bufferC, gemvArgs)).wait();
    bufferC.read(&result, sizeof(result));
    EXPECT_FLOAT_EQ(result, 0.0f);

    vublas::GemvOutPlaceNaiveTemplateFP32 gemvOutPlace(
        ctx, relu, 1, 1, 32, 4, 2, 4, 16, 1, 4
    );
    bufferC.write(&initial, sizeof(initial));
    bufferOut.write(&initial, sizeof(initial));
    execute_plan(
        ctx,
        gemvOutPlace(bufferA, bufferB, bufferC, bufferOut, gemvArgs)
    ).wait();
    bufferOut.read(&result, sizeof(result));
    EXPECT_FLOAT_EQ(result, 0.0f);
}
