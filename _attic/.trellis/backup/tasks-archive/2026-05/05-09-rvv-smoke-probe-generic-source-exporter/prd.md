# register rvv smoke probe generic source exporter

## Goal

Register the RVV smoke-probe standalone C source export path as a built-in
target artifact exporter so a planned smoke-probe RVV path can be exported
through `tcrv-translate --tcrv-export-target-source-artifact`, while preserving
the existing direct `--tcrv-export-rvv-smoke-probe-c` behavior.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Live HEAD before this task is `dbf2b03 feat: use profile compile facts in RVV smoke export`.
* Worktree was clean before task creation.
* The direct RVV smoke-probe exporter already validates selected RVV paths,
  `tcrv_rvv.lowering_boundary`, relation-provider target profile capability
  lookup, selected march, and optional selected mabi.
* The generic target artifact front doors select only registry-backed
  emission-plan candidates after execution-plan coherence preflight.
* The current profile-backed smoke fixture reaches the direct smoke exporter,
  but its planned emission route is not a smoke-probe target artifact route.

## Requirements

* Add the smallest C++ target/export API needed to register the existing RVV
  smoke-probe C emitter in `TargetArtifactExporterRegistry`.
* Do not duplicate smoke-probe C source generation logic.
* Preserve direct `tcrv-translate --tcrv-export-rvv-smoke-probe-c` behavior.
* Keep target artifact front-door selection registry-driven.
* Keep RVV-specific selection and metadata in plugin-local or target-local
  code; do not add RVV route branches to generic core passes, generic
  capability classes, or `tcrv.exec`.
* Use an existing source artifact kind suitable for standalone C source.
* Preserve exact-first `TargetCapabilitySet` lookup and explicit module-level
  target profile relation-provider behavior.
* Make no RVV runtime, correctness, or performance claim without `ssh rvv`
  evidence.

## Acceptance Criteria

* A focused lit/FileCheck test runs the module-level target-profile
  smoke-probe fixture through:

```bash
tcrv-opt %s --tcrv-execution-planning-pipeline | \
  tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s
```

* The generic output proves the same RVV smoke-probe C artifact surface as the
  direct route: `standalone-c-source`, RVV-owned route/origin metadata,
  profile-backed `selected_march`, optional `selected_mabi`, required capability
  `@smoke_rvv_profile`, `riscv_vector.h`, and RVV intrinsic use.
* Existing direct smoke-probe route stays passing.
* C++ registry assertions cover the smoke-probe route metadata if lit does not
  otherwise prove registry metadata.
* Required checks pass:
  `git diff --check`, CMake configure under `artifacts/tmp/tianchenrv-build`,
  and `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.

## Out Of Scope

* Python compiler-internal implementation.
* New `tcrv.exec` compute operations or generic tensor/tile IR.
* Generic front-door target-family special cases.
* Runtime ABI changes, runtime SSA/control changes, descriptor-local sample
  size changes, hardware fact rewrites, new IME/Sophgo/offload work, or RVV
  runtime/correctness/performance evidence.

## Technical Notes

* Required inspection found the user-listed
  `lib/Target/BuiltinTargetArtifactExporters.cpp` path does not exist at HEAD;
  the actual file is `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.
* `TargetArtifactExport.cpp` already recognizes `standalone-c-source` as a
  source artifact kind and gives it `.c` bundle extension behavior.
* Existing microkernel generic source export must remain unchanged for ordinary
  RVV i32-vadd microkernel fixtures.
