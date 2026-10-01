# Require wave32 for subgroup-dependent shaders
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Find every soclBLAS shader whose indexing, shuffle operations, workgroup
  layout, or specialization invariants assume a subgroup size of 32.
- Require subgroup size 32 when creating every corresponding shader pipeline.

## How to implement

- Inspect shader source and templates for explicit or implicit wave32
  assumptions instead of applying the requirement indiscriminately.
- Add `requiredSubgroupSize = 32` to each matching
  `socl::ShaderPipelineCreateInfo` in the associated operator implementation.
- Keep shaders that do not depend on a fixed subgroup size unchanged.
- Rely on socl's existing subgroup-size capability validation so unsupported
  devices fail during pipeline creation rather than executing with an invalid
  subgroup layout.
- Do not add libraries or change unrelated operator behavior.
- Do not compile or execute the project in the agent environment.
