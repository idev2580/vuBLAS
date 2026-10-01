# soclDNN integration audit
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to investigate

- Evaluate whether soclBLAS can be used as the foundation of a separate
  soclDNN library where activations use the elementwise shader templates and
  Softmax has a dedicated shader.
- Find concrete integration, packaging, CMake, API, runtime compilation,
  resource-lifetime, and execution-model problems that would affect a
  downstream library.

## How to investigate

- Review the top-level and subdirectory CMake configuration for downstream
  `add_subdirectory`, `FetchContent`, install/export, generated shader, and
  dependency-tool requirements.
- Review public headers and Elementwise implementation for ABI/API usability,
  buffer sizing, asynchronous dispatch, descriptor ownership, and pipeline
  creation costs.
- Review the underlying socl APIs available in the configured dependency when
  needed to distinguish confirmed defects from assumptions.
- Report findings in severity order with exact file references and practical
  remediation guidance.
- Do not modify files outside the prompts directory and do not run tests or
  executables.
