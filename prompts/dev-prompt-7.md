# Improve GemmContiguousNaiveFP32 performance
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## User Prompt Summary

The user wants to improve the FP32 GEMM path for contiguous matrices. The current contiguous shader is slower than the shared-memory GEMM despite assuming contiguous layout. The goal is for `GemmContiguousNaiveFP32` to become the best-performing GEMM implementation.

## What to Implement

- Optimize `GemmContiguousNaiveFP32.comp` for contiguous FP32 GEMM.
- Add register accumulation so each invocation accumulates multiple output values before writing to memory.
- Use subgroup operations for in-tile computation.
- Assume the maximum subgroup size is 32.
- Avoid mutual vectorization if it gets in the way of performance.

## How to Implement

- Inspect the existing GEMM shaders and host dispatch configuration to preserve project conventions.
- Target the contiguous-memory assumptions directly in the shader.
- Prefer a tiled algorithm where a subgroup cooperatively loads/uses fragments and each lane accumulates multiple results in registers.
- Keep changes scoped to existing shader/host interfaces unless a dispatch-size adjustment is necessary.
- Do not add external libraries.
- Do not compile or execute tests in the agent environment.

## Conversation Update

- The user only allows reading `shaders/GemmSharedFP32.comp` and `shaders/GemmContiguousNaiveFP32.comp`.
- Do not read prior/other shader code.
- Any non-`prompts` file edit still requires explicit user approval.
- The user clarified that avoiding mutual/vectorized computation was intended; remove the `vec4` accumulator/writeback style from the contiguous GEMM compute path and use scalar register accumulation.
- The user explicitly rejected `vec4` C writeback because the register reshuffling is hurting the optimized path. Final C update should use scalar stores, while preserving the contiguous shader's row-major A/shared-memory assumptions.
- The user clarified that SSBO layout may also be changed when needed for optimization. Keep A/B vectorized loads for contiguous tile staging, but change C to scalar `float[]` so final writeback has no `vec4` accumulator packing or component-store path. Add fast paths for common `alpha=1, beta=0` / `beta=0` cases.
- The user asked to lightly modify `tests/PerformanceTest.cpp` to add repeated GEMM benchmarking. It is acceptable for each iteration to overwrite C with `beta=0`; the goal is to reduce measurement noise and expose whether the current socl/operator structure is dominated by per-call submit/sync overhead.
- The user has updated GEMM to expose a `DispatchToken` and now wants every `Operator` subclass to return `DispatchToken` from `execute()` and `operator()` instead of returning `void`, so dispatches can be composed/synchronized outside individual operator calls.
- When editing `.cpp` files, replace `ctx.submitAndWait()` with `ctx.submitAsync()` and return the resulting `DispatchToken`.
- Existing correctness tests should store the returned token and call `token.wait()` before reading output buffers.
- The performance benchmark should collect async tokens for all repeated iterations, then wait after all submissions to avoid per-iteration wait overhead.
