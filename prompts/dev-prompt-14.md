# FetchContent-compatible CMake configuration
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Make soclBLAS usable as a CMake `FetchContent` dependency.
- Ensure generated SPIR-V headers are written to the directory exposed through the `soclblas` target's include paths.
- Avoid building soclBLAS tests and benchmarks by default when the project is consumed as a dependency.
- Preserve the existing standalone development behavior as much as possible.

## How to implement

- Replace the top-level `${CMAKE_BINARY_DIR}` shader output path with the shader subdirectory's `${CMAKE_CURRENT_BINARY_DIR}`.
- Add a `SOCLBLAS_BUILD_TESTS` option whose default follows `PROJECT_IS_TOP_LEVEL`.
- Only add the `tests` subdirectory and define its test/benchmark targets when that option is enabled.
- Keep the change minimal and use only the project's existing dependencies.
- Do not compile or execute the project in the agent environment.
