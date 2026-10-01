# Deferred Execution Architecture

## Goal

Support dynamic eager-mode workloads while recording multiple operations into
one command buffer and submitting them as a batch. Pipelines must be reusable
across calls, while every recorded dispatch receives immutable descriptor
bindings that remain alive until GPU completion.

## Current Problems

- Each operator owns one `DescriptorSet`, updates it for every call, starts a
  command buffer, and submits immediately.
- Reusing an operator before completion either updates a descriptor set that is
  in flight or makes multiple recorded dispatches observe the final descriptor
  contents.
- One submission per operation prevents efficient chaining of GEMM,
  Elementwise, reductions, and future DNN operations.
- vucol has no public compute-to-compute barrier recording API.
- `Context::createDescriptorSet` currently creates a separate descriptor pool
  with `maxSets = 1` for every set, which is unsuitable for eager dispatches.
- Command buffers do not retain all pipelines, descriptor sets, and buffers
  referenced by earlier recorded dispatches.

## Target Model

### Operator

A reusable, immutable operation definition.

- Owns the shader pipeline and specialization configuration.
- Validates typed arguments and calculates dispatch dimensions.
- Produces a `DispatchPlan`; it does not own per-call descriptor state.
- The same Operator instance and pipeline can serve any number of calls.

### DispatchPlan

A declarative description of one GPU dispatch.

- Holds a copy of the `ShaderPipeline` handle.
- Holds ordered buffer bindings with `Read`, `Write`, or `ReadWrite` access.
- Owns copied push-constant bytes; it must not retain caller-owned pointers.
- Holds the X/Y/Z dispatch group counts.
- Contains no descriptor pool, command buffer, or submission policy.

Proposed shape:

```cpp
enum class BufferAccess {
    Read,
    Write,
    ReadWrite,
};

struct BufferBinding {
    uint32_t binding;
    vucol::Buffer buffer;
    BufferAccess access;
};

struct DispatchPlan {
    vucol::ShaderPipeline pipeline;
    std::vector<BufferBinding> bindings;
    std::vector<std::byte> pushConstants;
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
};
```

### ExecutionPlan

A sequence of `DispatchPlan` objects for one logical operation.

- Single-pass operations return one dispatch.
- Multi-pass operations such as Softmax may return reduction, transform, and
  normalization dispatches with dependencies between them.
- The plan can be constructed dynamically; no static computation graph is
  required.

### Executor

Consumes plans and records them into the active command batch.

- Allocates a fresh descriptor set for every dispatch.
- Populates the set once and never updates it until its submission completes.
- Selects pipelines, binds descriptor sets, pushes constants, and dispatches.
- Tracks prior buffer accesses and inserts required barriers.
- Flushes according to eager execution policy and returns a completion token.
- Reclaims descriptor pools and other batch resources only after the fence is
  complete.

### CommandBatch

Owns all mutable and in-flight state for one submission.

- Owns the command buffer and one or more descriptor pool blocks.
- Retains every descriptor set used by recorded dispatches.
- Retains pipeline and buffer handles until GPU completion.
- Is moved into the submission/completion token on flush.
- Resets and recycles descriptor pools only after its fence signals.

## Recording Flow

```cpp
executor.begin();

executor.enqueue(gemm.plan(x, weight, temp1, gemmArgs));
executor.enqueue(relu.plan(temp1, temp2, elementCount));
executor.enqueue(gemm.plan(temp2, weight2, temp3, gemmArgs2));
executor.enqueue(softmax.plan(temp3, output, softmaxArgs));

auto token = executor.flush();
```

For every enqueue, the Executor should:

1. Inspect the plan's buffer access declarations.
2. Record any required compute memory barrier.
3. Allocate a descriptor set compatible with the plan's pipeline layout.
4. Bind all buffers and update the descriptor set exactly once.
5. Record pipeline selection, descriptor binding, push constants, and dispatch.
6. Transfer the plan and descriptor set into `CommandBatch` ownership.

## Barrier Rules

Track the last access to each underlying buffer within the active batch.

| Previous access | Current access | Barrier |
| --- | --- | --- |
| Read | Read | No |
| Read | Write | Yes |
| Write | Read | Yes |
| Write | Write | Yes |
| ReadWrite | Any | Yes |
| Any | ReadWrite | Yes |

Begin with a conservative global compute barrier between dependent dispatches.
Later, narrow barriers to affected buffers if vucol exposes buffer barriers.
Pipeline changes and descriptor bindings do not provide memory visibility by
themselves.

## Eager Execution Policy

Enqueue operations as they are called. Flush the active batch when any of the
following occurs:

- The CPU requests tensor data or explicit synchronization.
- The active descriptor pool or command budget reaches its configured limit.
- Execution switches to another device or Context.
- Memory pressure requires completion before storage can be reclaimed.
- The caller explicitly requests a flush.

Do not require the graph to be known ahead of time. Output buffers are created
before enqueue, so later eager operations can immediately reference them while
the producing dispatch is still only recorded.

## Implementation Phases

### 1. vucol command recording

- Add a compute-to-compute memory barrier API.
- Introduce a command batch/encoder abstraction, or formally support repeated
  `use`, `bind`, `push`, and `dispatch` calls between one `begin` and submit.
- Make submission tokens retain all GPU resources referenced by a batch.

### 2. Descriptor allocation

- Replace the pool-per-set implementation with reusable pool blocks.
- Allocate one immutable descriptor set per recorded dispatch.
- Add fence-based pool reset and recycling.
- Grow pools in bounded blocks and flush or allocate another block on capacity
  exhaustion.

### 3. vuBLAS plan API

- Add `BufferAccess`, `BufferBinding`, and `DispatchPlan` types.
- Change operators to create plans without calling `begin` or `submitAsync`.
- Keep the current immediate-call API as a convenience wrapper that creates a
  one-operation batch and submits it.
- Ensure all deferred argument data is copied into the plan.

### 4. Executor

- Implement enqueue, hazard tracking, barrier insertion, and flush.
- Preserve descriptors, pipelines, buffers, and command buffers until fence
  completion.
- Define thread-affinity or locking rules for one Executor/Context.

### 5. DNN integration

- Cache reusable activation Operators such as ReLU and Sigmoid.
- Implement Softmax as a multi-dispatch `ExecutionPlan` with explicit scratch
  buffers and access declarations.
- Add eager chaining tests with distinct tensor buffers and repeated use of the
  same Operator in one batch.

## Acceptance Checks

- Record the same Elementwise Operator multiple times with different buffers;
  every dispatch must observe its own bindings.
- Chain producer and consumer dispatches without intermediate CPU waits and
  verify correct results.
- Destroy caller-side temporary plans after enqueue without invalidating the
  active or in-flight batch.
- Recycle descriptor pools only after the associated fence completes.
- Support a dynamically constructed eager sequence without a prebuilt graph.
- Keep the existing immediate operator calls source-compatible where practical.
- Run validation layers without descriptor lifetime, synchronization, or
  resource destruction errors.
