# Rename socl → vucol, soclBLAS → vuBLAS everywhere

## What to implement

- Rename every `socl` reference to `vucol` and every `soclBLAS` reference to
  `vuBLAS` across source code, directories, documentation, and install paths.
- Case variants in scope: `soclBLAS` → `vuBLAS`, `soclblas`/`soclblas` →
  `vublas`, `SOCLBLAS` → `VUBLAS`, `SOCL` → `VUCOL`, `socl` → `vucol`
  (includes `socl::`, `<socl/...>`, `SOCL::socl`, `find_package(SOCL)`).
- Rename filesystem entries: `include/soclblas/` → `include/vublas/`,
  `cmake/soclBLASConfig.cmake.in` → `cmake/vuBLASConfig.cmake.in`
  (plus generated `soclBLASTargets.cmake` / `soclBLASConfig*.cmake` names).
- Update build/install names: `project(soclBLAS)` → `project(vuBLAS)`,
  lib target `soclblas` → `vublas`, alias `soclBLAS::soclblas` →
  `vuBLAS::vublas`, export `soclBLASTargets` → `vuBLASTargets`,
  option `SOCLBLAS_BUILD_TESTS` → `VUBLAS_BUILD_TESTS`,
  `SOCLBLAS_GLSLANG_VALIDATOR` → `VUBLAS_GLSLANG_VALIDATOR`,
  custom target `soclblas_compile_shaders` → `vublas_compile_shaders`,
  install prefix `/opt/socl` → `/opt/vucol`,
  doc dir `doc/soclBLAS` → `doc/vuBLAS`,
  cmake dir `lib/cmake/soclBLAS` → `lib/cmake/vuBLAS`.
- Update C++ names: namespace `soclblas` → `vublas`
  (`soclblas::detail` → `vublas::detail`), all `#include <soclblas/...>`
  → `<vublas/...>`, shader marker `__SOCLBLAS_*__` → `__VUBLAS_*__`.
- Update external dependency references: `#include <socl/...>` →
  `<vucol/...>`, `"socl/..."` → `"vucol/..."`, `socl::` → `vucol::`,
  `SOCL::socl` → `VUCOL::vucol`, `find_dependency(SOCL ...)` →
  `find_dependency(VUCOL ...)`.
- Update tests/binaries/comments: `soclblas_tests`,
  `soclblas_performance_tests`, `soclblas_performance_sweep_tests` →
  `vublas_*` equivalents; `./build/soclblas_performance_tests` usage comments
  in `tests/PerformanceTest.cpp`; `socl_tile_*` log labels in
  `tests/CublasGemmBenchmark.cpp` (check whether these mean this project's
  tiles or external CUBLAS comparison before renaming).
- Update active docs: `AGENTS.md` title/body, `TODO.md` (`socl` barrier API,
  `socl::` snippets, `soclBLAS plan API`, `socl command recording`),
  `CMakeLists.txt` messages (`"Embedding soclBLAS shader templates"`,
  `"glslangValidator is required to build soclBLAS"`), `README`-type text if
  present.
- Out of scope by default: past `prompts/dev-prompt-*.md` history records
  (rewriting them would falsify history); only this new prompt uses the new
  names going forward. Confirm with user if they want history rewritten too.

## How to implement

- Inventory first with case-insensitive regex `(?i)socl` over `CMakeLists.txt`,
  `src/`, `include/`, `tests/`, `shaders/`, `cmake/`, `TODO.md`, `AGENTS.md`,
  `analyze_sweep.py` (no compile/execution in agent environment).
- Apply replacements longest-name-first to avoid partial overlap:
  `soclBLAS` → `vuBLAS`, `soclblas` → `vublas`, `SOCLBLAS` → `VUBLAS`,
  then remaining `SOCL` → `VUCOL`, `socl` → `vucol`, `Socl` → `Vucol`
  (if any); verify no `socl`/`SOCL` (case-insensitive) remains outside
  historical `prompts/` afterwards.
- Use `git mv` semantics for directory/file renames (`include/soclblas` →
  `include/vublas`, `cmake/soclBLASConfig.cmake.in` →
  `cmake/vuBLASConfig.cmake.in`) so git tracks them as renames.
- Keep behavior identical: rename only, no logic changes, no new external
  libraries; do not configure, compile, or execute code in the agent
  environment (developer's work per `AGENTS.md`).
- Per `AGENTS.md`, this prompt file is written first; every other file change
  requires explicit user approval before editing.
