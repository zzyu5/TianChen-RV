# RVV microkernel self-check runtime-n separation

## Goal

Make the explicit direct RVV i32-vadd microkernel self-check harness exercise
the generated runtime-callable ABI with explicit runtime `n` values, instead of
using descriptor-local `element_count` as the only runtime trip count. This
advances the target/export runtime ABI boundary: the generated callable
function is still the same bounded RVV dataflow-derived source, but its
evidence harness now proves that runtime element count is caller-controlled ABI
state while `element_count` remains descriptor-local bounded capacity metadata.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- The worktree was clean before this task was created.
- Latest HEAD at task start is `c8a10df feat: prove rvv dataflow microkernel on ssh`.
- Latest Hermes audit/review input is under
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0095-20260509T005515Z/`.
- The previous completed round made the generated RVV microkernel C source
  dataflow-driven and proved the explicit self-check source on real `ssh rvv`.
- The current direct self-check harness still allocates arrays sized by
  descriptor-local `element_count` and calls the ABI function once with
  `n = element_count`.
- The RVV+scalar dispatch self-check harness already uses explicit runtime
  counts `7` and `16`, so direct RVV microkernel evidence should use the same
  parameter-layering discipline.

## Requirements

- Preserve the primary implementation stack: C++ / MLIR / LLVM / TableGen /
  CMake / lit / FileCheck.
- Keep `tcrv.exec` compute-free and keep RVV compute/dataflow semantics
  plugin-local to `tcrv_rvv` and target/export code.
- Do not add another smoke/probe/report-only artifact.
- Change the C++ RVV target exporter so
  `tcrv-translate --tcrv-export-rvv-microkernel-self-check-c` emits a bounded
  harness that calls the same generated ABI function for explicit runtime
  counts smaller than and equal to the bounded capacity.
- Preserve descriptor-local `element_count` as harness capacity / selected
  descriptor metadata only; it must not be described as shape, AVL, VL, or the
  only runtime trip count.
- Keep the default `--tcrv-export-rvv-microkernel-c` artifact library-style
  with no hidden `main` or self-check helper.
- Update focused lit/script checks that inspect the generated harness marker or
  source shape.
- Update durable spec/README wording only where it describes the direct
  microkernel self-check harness contract.
- Run local compiler checks and, if available, a real `ssh rvv` self-check run
  for the updated generated harness source.

## Acceptance Criteria

- [x] Direct RVV microkernel self-check C contains explicit runtime count cases
      such as `7` and `element_count`, and the success marker reports those
      runtime counts rather than only `elements=<element_count>`.
- [x] The generated harness verifies that entries past runtime `n` are not
      written for the smaller runtime count.
- [x] Existing dataflow provenance validation remains required before local
      dry-run or remote evidence accepts generated source.
- [x] Focused lit/FileCheck expectations are updated for the new harness
      source semantics.
- [x] Relevant spec/README text preserves the parameter-layering invariant.
- [x] `git diff --check` passes.
- [x] Relevant build/lit checks pass, or the exact missing dependency is
      documented.
- [x] Real `ssh rvv` evidence is run for the updated generated self-check
      source if the host is reachable.
- [x] One coherent commit is created and the worktree is clean when complete.

## Out Of Scope

- Generic high-level lowering or arbitrary RVV kernel emission.
- New RVV dialect generalization beyond the finite i32-vadd dataflow slice.
- Runtime probing, performance, throughput, latency, or benchmark claims.
- Python replacement of compiler internals.
- Broad negative fixture matrices or dashboard/report-only work.

## Technical Notes

Relevant materials inspected before choosing this owner:

- `AGENTS.md`
- `README.md`
- `CMakeLists.txt`
- `.trellis/spec/index.md`
- `.trellis/spec/capability-model/capability-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0095-20260509T005515Z/repo_audit.md`
- `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0095-20260509T005515Z/review_input.md`
- `lib/Target/RVV/RVVMicrokernel.cpp`
- `scripts/rvv_microkernel_e2e.py`
- `test/Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir`
- `test/Scripts/rvv-microkernel-e2e.test`

Verification run:

- `cmake --build build --target tcrv-translate -j2`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -a Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir Scripts/rvv-microkernel-e2e.test` from `build/test`
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`
- `python3 scripts/rvv_microkernel_e2e.py --run-id manual-rvv-runtime-n-self-check --overwrite --self-check-harness --input test/Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir`

Real `ssh rvv` result:

- Artifact dir: `artifacts/tmp/rvv_microkernel_e2e/manual-rvv-runtime-n-self-check`
- Remote compile: `clang -O2 -march=rv64gcv -mabi=lp64d rvv_microkernel.c -o rvv_microkernel`, exit code 0.
- Remote run: `./rvv_microkernel`, exit code 0.
- Observed stdout marker: `tcrv_rvv_microkernel_ok runtime_counts=7,16`.
