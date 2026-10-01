# Package soclBLAS as an installable library
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Make soclBLAS installable as a standalone CMake config package alongside the
  existing SOCL installation, forcing the configured install prefix to
  `/opt/socl` even when an older build cache contains another prefix.
- Link against the installed SOCL package through its exported `SOCL::socl`
  target.
- Keep generated SPIR-V headers private to the soclBLAS implementation.
- Detect `glslangValidator` as a required soclBLAS build tool.
- Keep GoogleTest and its `FetchContent` setup entirely inside the test-enabled
  block.
- Make soclBLAS tests disabled by default.

## How to implement

- Add project versioning, GNU install directories, target export rules, public
  header installation, and generated config/version files for a relocatable
  `soclBLAS` CMake package.
- Force `/opt/socl` as the install prefix when soclBLAS is configured as the
  top-level project, including when an existing cache contains another prefix,
  without changing a parent project's prefix when used through
  `add_subdirectory()`.
- Add a namespaced build-tree alias and export the installed library as
  `soclBLAS::soclblas`.
- Use build/install generator expressions for public include directories and
  propagate the C++20 requirement from the library target.
- Change the SOCL dependency to `find_package(SOCL CONFIG REQUIRED)` and link
  `SOCL::socl` publicly.
- Move the `MatMulNaiveFP32_SPIRV` include from the public header into its source
  file, then make the generated shader include directory private.
- Resolve `glslangValidator` during configuration and use its resolved path in
  shader custom commands.
- Default `SOCLBLAS_BUILD_TESTS` to `OFF`; only include FetchContent, fetch
  GoogleTest, create tests, and register GoogleTest discovery when the option is
  enabled, and enable CTest in that block.
- Do not add external libraries and do not configure, compile, install, or run
  tests in the agent environment.
