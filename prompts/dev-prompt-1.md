# Development Prompt 1
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## User Request
Change only the test-side CMake structure.

Existing correctness tests should continue to use GoogleTest, while performance tests should be built as a separate executable.

Follow-up request:
- Also implement a GEMM-based performance test that runs random matrices with a very large batch and measures GPU FLOPs.

Follow-up correction:
- Do not use repeated performance iterations to inflate the measured workload.
- The performance workload should come from a large GEMM batch.
- Randomized repeated runs are for correctness tests, not performance tests.
- Choose a default single-run workload that is very large but should fit on a 4GB VRAM GPU.

## What To Implement
- Split test sources so correctness tests and performance tests are not compiled into the same executable.
- Keep correctness tests using GoogleTest.
- Add a separate executable target for performance tests.
- Scope changes to the test CMake structure.
- Add an independent `PerformanceTest.cpp` entry point for GEMM GPU FLOPs measurement.
- Measure one large batched GEMM workload instead of looping over iterations.

## How To Implement
- Inspect the current `tests/` sources to identify the performance test source.
- Update the CMake test source lists to export separate correctness and performance variables.
- Update the top-level test target wiring so GoogleTest is only used for correctness tests.
- Link the performance executable against `soclblas` without GoogleTest.
- Generate random FP32 input matrices, time a single large batched GEMM call, and report GFLOP/s.
- Provide command-line overrides for batch and GEMM dimensions so the default large batch can be adjusted by the developer.
- Use default dimensions that allocate about 3GiB for A/B/C combined, leaving room on a 4GB VRAM GPU.
- Do not add external dependencies.
- Do not compile or execute the project in the agent environment.
