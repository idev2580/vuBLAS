# Separate dispatch resources from command recording
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Introduce `DispatchPlanResource` as the prepared, context-specific resource
  state for a `DispatchPlan`.
- Move logical `DescriptorSet` creation and buffer binding out of
  `DispatchPlan::record()` and into `DispatchPlan::allocate()`.
- Cache the allocated resource inside its owning `DispatchPlan` so repeated
  recording reuses the same logical descriptor set without adding resource
  ownership or cache policy to `ExecutionPlan`.

## How to implement

- Let `DispatchPlan::allocate(Context&)` create a compatible logical
  `DescriptorSet`, bind the plan's fixed buffers once, and return the prepared
  `DispatchPlanResource`.
- Store the prepared resource on `DispatchPlan`; `allocate(Context&)` creates it
  at most once and rejects reuse with a different Context.
- Let `DispatchPlan::record(Context&)` ensure its own resource is allocated,
  then select the pipeline, bind the cached resource, push owned constants, and
  dispatch.
- Keep `ExecutionPlan` as a simple sequence that appends plans and calls each
  plan's `record()` without owning a parallel resource cache.
- Validate push-constant byte size against the pipeline declaration so a
  pipeline that requires constants cannot dispatch with missing or stale data.
- Keep immutable per-dispatch native descriptor snapshots and their in-flight
  lifetime management inside socl.
- Do not compile or execute the project in the agent environment.
