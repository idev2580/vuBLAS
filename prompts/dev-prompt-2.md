# Development Prompt 2
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## User Request
Extend the naive GEMM approach with workgroup shared memory and add it as a new FP32 operator named `GemmSharedFP32`.

Follow-up naming correction:
- Rename the concrete shared-memory FP32 operation to `GemmSharedFP32`.

## What To Implement
- Add a new shared-memory GEMM FP32 operator separate from the existing `GemmNaiveFP32`.
- Name the concrete operator `GemmSharedFP32`.
- Add a new compute shader that uses shared memory to cache A/B tiles before accumulating C.
- Keep the existing naive GEMM implementation intact.
- Register the new source and shader in the existing CMake structure.
- Update `PerformanceTest.cpp` to use `GemmSharedFP32` and a more suitable default shared-memory tile size.

## How To Implement
- Follow existing soclBLAS operator patterns for headers, source files, shader embedding, and CMake source registration.
- Reuse the existing `Gemm`/`GemmArguments` interface where possible.
- Start with a conservative tiled FP32 shader using workgroup shared memory, register accumulation, and no new external dependencies.
- Use a default shared GEMM tile of `tile_m=64`, `tile_n=16`, and `tile_p=64` after making per-thread output tile size configurable.
- Allow the performance executable to override GEMM and tile dimensions from command-line arguments.
- Avoid compiling or executing the project in the agent environment.
