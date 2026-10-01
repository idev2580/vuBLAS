# Development Prompt 0
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## User Request
Implement DotProduct and Max/Min/Avg/Sum in:
- `include/soclblas/ops/DotProductNaive.hpp`
- `include/soclblas/ops/ReductionNaive.hpp`
- `src/ops/DotProductNaive.cpp`
- `src/ops/ReductionNaive.cpp`

Also add shaders for these operations under `shaders/`.

Follow-up correction:
- Even though the classes are named `Naive`, the shaders must use subgroup operations for efficient reductions.
- Do not use 256-thread workgroups.
- Prefer a 32-lane shape and keep the shaders workable on wave64 hardware.
- Avoid GLSL reserved keywords and reserved-like identifiers in shader variable/member names.

Follow-up test request:
- Add simple tests for reduction operations only.

Follow-up test clarification:
- Add DotProduct coverage as well because dot product is also a reduction operation.

## What To Implement
- Add/complete the naive dot product operator.
- Add/complete naive reduction operators for maximum, minimum, average, and sum.
- Add compute shaders for dot product and reduction operations.
- Implement shader reductions with subgroup features instead of shared-memory 256-thread reductions.
- Review shader identifiers for GLSL keyword conflicts.
- Add simple coverage for `MaxNaive`, `MinNaive`, `AvgNaive`, and `SumNaive`.
- Add simple coverage for `DotProductNaive`.

## How To Implement
- Follow the existing soclBLAS operator patterns and naming conventions.
- Reuse libraries and abstractions already present in the repository.
- Keep the implementation scoped to the requested naive operator headers and source files.
- Add shader files in `shaders/` and dispatch them from the naive operators.
- Register `ReductionNaive.cpp` and the new shader files in the existing CMake source lists.
- Use a small subgroup-oriented default workgroup size, not 256 threads.
- Add reduction tests following the existing GoogleTest and buffer usage pattern.
- Do not introduce new external dependencies.
- Do not compile or execute the project in the agent environment.
