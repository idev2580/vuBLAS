# Diagnose the socl include compilation failure
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

- Determine why soclBLAS cannot compile due to missing socl includes even though
  socl is installed under `/opt/socl` and VS Code has a `cmake.environment`
  entry.
- Report the concrete root cause and the configuration change that would fix it.
- Do not modify project files while diagnosing.

## How to implement

- Inspect the VS Code CMake environment configuration and the project's CMake
  package lookup/link setup.
- Inspect the installed `/opt/socl` directory, especially its CMake package
  configuration and exported target include directories.
- Compare the configured environment variable with what `find_package(SOCL)`
  actually consumes.
- Use only read-only inspection; do not configure, compile, or execute the
  project, and do not add external libraries.
