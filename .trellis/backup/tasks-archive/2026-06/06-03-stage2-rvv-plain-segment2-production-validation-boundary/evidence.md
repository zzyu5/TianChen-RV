# Evidence

## Checks

```bash
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk cmake --build build --target tcrv-opt tcrv-translate -j2
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-interleave-unit-load'   # workdir: build/test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-deinterleave-unit-store' # workdir: build/test
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/plain-segment2-interleave-explicit-dry-run --run-id explicit-segment2-interleave-unit-load --overwrite --op-kind segment2_interleave_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/plain-segment2-interleave-pre-realized-dry-run --run-id pre-realized-segment2-interleave-unit-load --overwrite --op-kind segment2_interleave_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/plain-segment2-deinterleave-explicit-dry-run --run-id explicit-segment2-deinterleave-unit-store --overwrite --op-kind segment2_deinterleave_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/plain-segment2-deinterleave-pre-realized-dry-run --run-id pre-realized-segment2-deinterleave-unit-store --overwrite --op-kind segment2_deinterleave_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-03-stage2-rvv-plain-segment2-production-validation-boundary
rtk git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir | rtk rg -n "^[+].*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|__riscv_.*_i32m1|source-front-door|source-artifact|emission_plan|descriptor|selected route|route id|artifact name)"
rtk git diff --check
```

## Results

* `tianchenrv-target-artifact-export-test` built and passed.
* `tcrv-opt` and `tcrv-translate` rebuilt successfully for generated-bundle
  checks.
* `segment2-interleave-unit-load` lit filter initially exposed one stale
  negative FileCheck diagnostic in
  `test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`;
  after updating that fixture to the new provider-owned plain segment2
  destination-memory diagnostic, the filter passed 4 tests.
* `segment2-deinterleave-unit-store` lit filter passed 4 tests.
* All four generated-bundle dry-runs reported `dry_run_success`.
* `task.py validate` passed with 21 `implement.jsonl` entries and 17
  `check.jsonl` entries.
* `git diff --check` passed.
* Added-line authority scan over changed production/test files returned no hits
  for legacy `i32m1`, descriptor/source-front-door/source-artifact, route-id,
  artifact name, or mirror-only route authority drift.

## Runtime Evidence Reuse

No `ssh rvv` run was required for this round. The implementation only tightened
provider-owned plain segment2 facts and target artifact validation; it did not
change route emission, generated bundle behavior, runtime ABI order, or runtime
semantics. Runtime correctness evidence is reused from the archived
`segment2_interleave_unit_load` and `segment2_deinterleave_unit_store` tasks,
which covered counts `0,1,16,17,257`.
