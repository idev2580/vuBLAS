# Make operators produce dispatch plans
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Change soclBLAS operators so operation calls construct and return owned
  `DispatchPlan` values instead of beginning command recording and returning
  `socl::DispatchToken` values.
- Remove per-call command submission and mutable descriptor-set ownership from
  operators.
- Preserve each operator's argument validation, push constants, buffer binding
  semantics, access declarations, and dispatch-size calculations in the
  generated plan.

## How to implement

- Update the common `Operator` interface and every concrete operator header and
  implementation to return `DispatchPlan`.
- Keep reusable shader pipelines and specialization configuration on operator
  instances, but move per-call pipeline handles, buffers, copied argument bytes,
  and dispatch dimensions into each returned plan.
- Express every descriptor binding explicitly with its binding number and the
  appropriate `socl::BufferAccess` mode.
- Remove operator-owned descriptor sets and all operator-side calls to
  `Context::begin()`, `Context::use()`, `Context::bind()`, `Context::push()`,
  `Context::dispatch()`, and `Context::submitAsync()`.
- Use the existing `ExecutionPlan` types, the standard library, and socl only.
- Do not compile or execute the project in the agent environment.
