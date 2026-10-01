# Development Prompt 8
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## User Request

Improve `GemmContiguousNaiveFP32.comp` by replacing its naive FP32 GEMM
implementation with a high-performance, outer-product-based implementation.
Analyze the existing shader and its host-side dispatch contract thoroughly before
optimizing it.

Follow-up benchmark request:

- Compare the optimized contiguous GEMM against cuBLAS available in the
  `climate` Conda environment.
- Match the existing strided-batched FP32 workload and report both strict FP32
  and TF32-enabled cuBLAS results where supported.
- Change the default benchmark arguments to
  `64 4096 1024 1024 128 32 128 8 8 50`.
- Place the standalone cuBLAS benchmark and its build script under `tests`;
  the script must assume the Conda environment is already active.
- Compile the host-only cuBLAS benchmark as ordinary C++ and link CUDA Runtime
  and cuBLAS directly; do not use `nvcc` because the benchmark has no CUDA
  device code.

## What To Implement

- Convert the contiguous FP32 GEMM shader to an outer-product accumulation
  strategy.
- Improve data reuse and arithmetic intensity while preserving numerical
  correctness, buffer layout, and edge handling.
- Preserve the existing host-facing interface unless the current dispatch
  contract makes a shader-only optimization impossible.

## How To Implement

- Inspect the shader, descriptor layout, push constants, workgroup dimensions,
  dispatch calculation, and relevant tests or benchmarks.
- Keep vec4 global A/B tile staging, but store A in K-major shared-memory order
  so lane-adjacent row reads do not alias the same shared-memory bank.
- For 32-lane subgroups, map lanes along C's physically contiguous dimension:
  lanes span P for normal C and M for transposed C.
- Load the opposite operand with conflict-free source lanes and distribute it
  with subgroup shuffle operations, then accumulate a scalar register outer
  product. Fall back to the thread-tile outer product when the subgroup/tile
  geometry cannot cover the output block exactly.
- Size the accumulator from the specialized thread tile instead of always
  reserving the maximum 8x8 register tile.
- Preserve the contiguous path's existing aligned, block-divisible dimension
  contract; do not add scalar boundary fallbacks to the shader.
- Avoid new dependencies and follow existing project conventions.
- Keep the cuBLAS comparison isolated from the soclBLAS library target so CUDA
  remains an optional benchmark-only dependency.
- Use `cublasSgemmStridedBatched`, CUDA event timing, warm-up iterations, and
  the same batch/dimensions/iteration count as the soclBLAS performance test.
- Request permission before editing any file outside `prompts`.
- Do not compile or execute the project in the agent environment; verification
  is limited to static inspection and a clearly documented developer test plan.
