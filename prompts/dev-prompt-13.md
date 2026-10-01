# Development Prompt 13
> **IMPORTANT:** Historical record from before the rename — `socl` is the former name of `vucol`, and `soclBLAS` is the former name of `vuBLAS` (same dependency, project, and library throughout).

## What to implement

Complete `tests/PerformanceSweepTest.cpp` as a standalone performance sweep for
`soclblas::GemmNaiveFP32`.

The sweep should reuse the existing buffer initialization and benchmark every
valid combination of:

- `subgroup_tile_m`
- `subgroup_tile_n`
- `subgroup_tile_p`
- `subgroup_tile_cnt_m`
- `subgroup_tile_cnt_p`
- `shared_tile_n_multiplier`
- `reg_tile_m`
- `reg_tile_p`
- `k_unroll`

For each tested combination, report the complete configuration and its measured
performance so that configurations can be compared after one run.

Use a fixed batch size of 32. Measure every valid configuration eight separate
times and print every individual measurement.

Fix `k_unroll` to 1 for every configuration because larger values can make the
long-running Vulkan dispatch exceed the AMDGPU gfx-ring timeout.

For the next focused RX 9070 XT sweep, vary only:

- subgroup shape: `(1, 32)`, `(2, 16)`, `(4, 8)`, `(8, 4)`
- `shared_tile_n_multiplier`: `2`, `3`, `4`, `5`, `6`, `8`

Fix the remaining tuning parameters to the current best configuration:

- `subgroup_tile_n = 1`
- `subgroup_tile_cnt_m = 4`
- `subgroup_tile_cnt_p = 2`
- `reg_tile_m = 8`
- `reg_tile_p = 4`
- `k_unroll = 1`

## How to implement

- Add the minimum CMake wiring needed to build the sweep as a standalone
  executable, separately from the existing performance benchmark.
- Define the candidate values clearly near the top of the file and iterate over
  their Cartesian product.
- Apply the following sweep filters:
  - `subgroup_tile_m * subgroup_tile_p == 32`
  - `reg_tile_m * reg_tile_p >= 32`, so every shader invocation owns at least
    32 output accumulator values.
- Filter out configurations whose A/B shared arrays exceed 64 KiB, using the
  same specialized tile-size formula as `GemmNaiveFP32.comp`.
- Do not add any other filtering condition for workgroup size or expected
  runtime.
- Construct a separately specialized `GemmNaiveFP32` instance for each valid
  configuration.
- For each configuration, execute and wait for one GEMM at a time, repeat that
  measurement eight times, and retain no 20-iteration benchmark loop.
- Use simple `printf` output. Print each measurement in the form
  `(configuration)[repeat]:(result)`, where the configuration contains all nine
  tuning parameters and the result is the measured TFLOP/s.
- Continue the sweep when an individual pipeline configuration or measurement
  throws an exception.
- Print counts for total candidate combinations, statically skipped
  combinations, successful measurements, and runtime failures.
- Do not add external libraries.
- Do not compile or execute the benchmark in the agent environment.

## Sweep-result filtering

Add a small shell script dedicated to `sweep_rx9070xt_naive.txt`. Read that
file directly and print each successful measurement line whose result is greater
than 6 TFLOP/s. Preserve the original line, ignore headers and error rows, write
only to standard output, and do not add dependencies.

Replace the broad Cartesian-product sweep with an explicit list containing the
42 unique configurations selected by that greater-than-6-TFLOP/s filter. Test
only those configurations and increase the repeat count from 2 to 8 for deeper
measurement. Keep the existing output format and validity checks.

Change the sweep output configuration order to match `PerformanceTest` and the
`GemmNaiveFP32` constructor:

`subgroup_tile_m subgroup_tile_n subgroup_tile_p subgroup_tile_cnt_m
subgroup_tile_cnt_p shared_tile_n_multiplier reg_tile_m k_unroll reg_tile_p`

Separate configuration values with spaces instead of commas for direct copying
into the performance-test command line. Apply the same order and separators to
successful and error output.

For each of the 42 focused base configurations, test all four subgroup-grid
orientations containing eight subgroups:

- `(subgroup_tile_cnt_m, subgroup_tile_cnt_p) = (1, 8)`
- `(subgroup_tile_cnt_m, subgroup_tile_cnt_p) = (2, 4)`
- `(subgroup_tile_cnt_m, subgroup_tile_cnt_p) = (4, 2)`
- `(subgroup_tile_cnt_m, subgroup_tile_cnt_p) = (8, 1)`

Since each subgroup contains 32 invocations, every generated workgroup contains
256 invocations. Keep the shared-memory validity filter, so generated variants
that exceed 64 KiB are skipped.

Use one measurement per configuration for the next broad scan. Preserve the
existing 42 base configurations and add nearby candidates around the strongest
observed `(1, 32)`, `(4, 8)`, and `(8, 4)` subgroup shapes, focusing on missing
`shared_tile_n_multiplier` and `k_unroll` combinations. Apply all four
256-invocation subgroup-grid orientations to the added base configurations too.

Turn the benchmark into a two-stage sweep:

1. Measure every valid generated configuration once and retain configurations
   whose result is strictly greater than 7.0 TFLOP/s in a runtime list.
2. After the broad scan finishes, recreate and measure each retained
   configuration ten times, printing every individual result.

Mark broad-scan results with `[scan]`, keep `[0]` through `[9]` for the deep
measurements, and print the number of selected configurations between the two
stages. Preserve the performance-test parameter order and space separators.

Immediately before deep measurement, print a large, fixed hash-delimited marker
block containing `DEEP MEASUREMENT`. Print no broad-scan output below that
marker, so a later Python analysis script can discard everything through the
marker and parse only the ten-repeat result rows that follow it.
