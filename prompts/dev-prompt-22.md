# Separate dispatch planning from execution
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Revise `DispatchPlan` so every deferred dispatch safely owns the pipeline,
  buffer bindings, access declarations, push-constant bytes, and dispatch
  dimensions needed at recording time.
- Revise `ExecutionPlan` into an owned sequence of dispatch plans.
- Separate recording commands into an existing socl batch from the convenience
  path that begins, records, and submits a batch.

## How to implement

- Represent each buffer binding explicitly with its descriptor binding number,
  `socl::Buffer` handle, and `socl::BufferAccess` value.
- Store `socl::ShaderPipeline`, buffer bindings, and push constants by value so
  plans do not retain references, spans, or raw pointers to caller-owned data.
- Give `DispatchPlan` a recording helper that creates and populates a logical
  descriptor set, selects the pipeline, binds the set, pushes the owned bytes,
  and dispatches through a supplied `socl::Context`.
- Let `ExecutionPlan::record()` append all dispatches to a context that is
  already recording, and provide `execute()` as a one-plan convenience wrapper
  around `begin()`, `record()`, and `submitAsync()`.
- Use standard-library facilities and the existing socl API only.
- Do not compile or execute the project in the agent environment.
