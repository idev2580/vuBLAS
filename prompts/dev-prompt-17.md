# Runtime GLSL epilogue shader templates
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Keep the existing precompiled `GemmNaiveFP32` and `MatMulNaiveFP32`
  implementations unchanged.
- Add separate GEMM, GEMM out-of-place, Gemv, Gemv out-of-place, and MatMul
  implementations that compile shader templates at runtime.
- Name the new operators `GemmNaiveTemplateFP32`,
  `GemvNaiveTemplateFP32`, and `MatMulNaiveTemplateFP32` so their relationship
  to the Naive implementation is explicit.
- Add corresponding `GemmOutPlaceNaiveTemplateFP32` and
  `GemvOutPlaceNaiveTemplateFP32` operators.
- Embed the complete GLSL template sources in the soclBLAS library binary.
- Accept only direct GLSL epilogue source from the caller; do not add built-in
  ReLU helpers, enums, template files on disk at runtime, or installation rules.
- Insert the caller-provided `epilogue()` definition into the embedded template
  and compile the rendered GLSL through socl's runtime shaderc compiler.

## How to implement

- Put near-copies of the current Naive GEMM and MatMul shaders under
  `shaders/templates/`, replacing only the epilogue-definition location with a
  unique marker and calling `epilogue()` during final writeback.
- Define the epilogue contract as `float epilogue(float value)` with no output
  coordinates or additional resources.
- Generate a C++ byte-array source from each template during the CMake build and
  link the generated source privately into the `soclblas` library.
- Expose new explicit-configuration C++ operators for the runtime-template
  paths, while preserving all existing Naive operator behavior.
- Implement the Gemv template operator as a thin adapter over the GEMM template
  operator, reusing the same embedded GEMM shader template and the existing
  Gemv-to-GEMM argument conversion.
- Render and compile the template when a runtime-template operator is created,
  then reuse the resulting pipeline for subsequent dispatches from that object.
- Add deterministic correctness tests using a directly supplied GLSL ReLU
  function and inputs containing negative results.
- Do not add external dependencies; use the existing socl shaderc support.
- Do not compile or execute the project in the agent environment.
