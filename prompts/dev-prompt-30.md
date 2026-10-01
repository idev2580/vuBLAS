# Propagate SOCL buffer binding ranges through soclBLAS
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Audit every soclBLAS API and internal execution path that accepts or binds an
  SOCL buffer after SOCL added buffer binding `offset` and `size` support.
- Add `include/soclblas/BufferView.hpp` and change every public operation input,
  inout, and output from a bare buffer to this `(buffer, offset, size)` view.
- Preserve correct element indexing and bounds behavior when operations target
  subranges of larger buffers.
- Keep existing bare `socl::Buffer` operation calls source-compatible by treating
  them as whole-buffer views.
- Update the affected headers, implementations, execution-plan binding, and tests
  consistently; the user explicitly approved these non-prompt changes.
- Diagnose a runtime undefined-symbol failure where `soclblas_tests` requests the
  new `MatMul::execute(std::span<const BufferView>, ...)` ABI.

## How to implement

- Define a small `soclblas::BufferView` value type holding `socl::Buffer`, byte
  `offset`, and byte `size`, including a whole-buffer converting constructor.
- Trace all buffer parameters from public soclBLAS constructors and argument
  objects through execution plans, descriptor writes, and dispatch recording.
- Change `Operator::execute()` and each typed operation overload to accept
  `BufferView`, and propagate views through wrapper operators such as GEMV.
- Store each view's byte offset and size in the dispatch plan and call SOCL's
  range-aware `DescriptorSet::bindBuffer()` overload without changing shader
  indexing or push-constant strides.
- Add tests using non-zero, alignment-compliant offsets for input and output
  ranges, while retaining existing whole-buffer calls as compatibility coverage.
- Compare the test executable's dynamic dependencies and required symbols with
  the build-tree and installed soclBLAS shared libraries to identify stale or
  mixed build artifacts; do not rebuild or execute tests in the agent environment.
- Do not add external libraries, configure, compile, or execute code in the
  agent environment.
