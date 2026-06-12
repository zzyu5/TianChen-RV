# RVV source-front-door i32m1 arithmetic family production closure

## Goal

Close the existing bounded RVV `i32m1` source-front-door arithmetic family as
one coherent production compiler path:

```text
source MLIR func/scf/vector/arith add|sub|mul
  -> RVV plugin source front door
  -> selected tcrv.exec / tcrv_rvv boundary
  -> plugin-owned materialized EmitC route
  -> generated object + declaration-only ABI header + bundle
  -> external C ABI harness
  -> ssh rvv correctness evidence
```

The module owner is the production RVV source-artifact bundle front door and
the plugin-owned construction path for the already supported add/sub/mul family.
The Python evidence script is only final evidence tooling.

## Background

- Commit `25113c6` closed the source-derived add generated-bundle ABI proof
  with real `ssh rvv` evidence for runtime counts `7`, `16`, and `23`.
- The compiler already names add/sub/mul in the RVV source front door, EmitC
  route provider, construction protocol, target artifact mapping, fixtures, and
  evidence script expectations.
- The next bottleneck is family coherence: add must not remain the only proven
  production source-artifact bundle path when sub and mul are already part of
  the bounded `i32m1` arithmetic slice.

## Scope

- Inventory whether add/sub/mul already reach the production path from source
  MLIR through `tcrv-translate --tcrv-source-artifact-bundle-front-door`.
- If production C++ is missing, stale, or add-special-cased, repair only the
  plugin-local C++/MLIR route, construction mapping, or target export
  validation needed for the existing family.
- If production code already works, avoid wrapper or compatibility additions
  and make the module truthful with focused production lit/script coverage plus
  final real RVV evidence for the full family.
- The evidence script may be adjusted only to consume generated artifacts,
  validate/report sanitized evidence, and drive the external C ABI harness. It
  must not implement compiler behavior.

## Non-Goals

- No new RVV dtypes, SEW/LMUL modes, tail or mask policies.
- No new source language forms or operation families beyond existing
  add/sub/mul.
- No descriptor tables, descriptor route IDs, direct C/source-export compute
  paths, compatibility wrappers, legacy modes, scalar fallback compute bodies,
  Python compiler-core logic, or core passes that special-case RVV semantics.
- No script-only, report-only, broad smoke-matrix, or guardrail-only result as
  the main achievement.

## Requirements

1. The one-command source-artifact bundle front door must work for the existing
   add, sub, and mul source fixtures.
2. Each selected variant must expose:
   - origin plugin `rvv-plugin`;
   - the operation-specific selected variant identity;
   - the operation-specific runtime ABI name;
   - ordered runtime ABI parameters `lhs, rhs, out, n`;
   - materialized EmitC route provenance;
   - RISC-V relocatable object handoff;
   - declaration-only runtime-callable C header;
   - object/header bundle component group.
3. Generated headers must contain declarations only and no RVV intrinsic or
   compute bodies.
4. The external C ABI harness must link and call the generated object/header
   artifacts for all three operations.
5. Real `ssh rvv` evidence must cover add, sub, and mul with several runtime
   counts, including at least one non-one-vector count. This round will use
   `7`, `16`, and `23`.
6. Unsupported or stale source/front-door inputs must still fail closed; this
   round must not weaken existing negative coverage.
7. Public evidence, generated headers, bundle indexes, and touched routes must
   reject descriptor, direct-C, source-export, self-check, RVV direct
   microkernel, raw-log, and credential-like residue.

## Acceptance Criteria

- [ ] `tcrv-translate --tcrv-source-artifact-bundle-front-door` succeeds for
      the existing add, sub, and mul source fixtures.
- [ ] Focused production lit coverage proves source-front-door add/sub/mul
      selected boundary, emission plan, generated object/header, declaration
      header shape, and bundle component-group metadata.
- [ ] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes.
- [ ] The evidence script dry-run coverage consumes generated artifacts and
      reports the full add/sub/mul family without becoming compiler logic.
- [ ] One sanitized real `ssh rvv` generated-bundle ABI run covers add/sub/mul
      and runtime counts `7`, `16`, and `23`, printing per-op PASS markers.
- [ ] Targeted residue scans over touched RVV plugin/target/script/tests show
      no restored descriptor, direct-C, source-export, or old route authority.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv` runs if practical; if not, the reason is recorded.
- [ ] Trellis task status, completion notes, archive state, and commit are
      truthful.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-i32m1-source-frontdoor-family-production-closure`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `cmake --build build --target tcrv-translate tcrv-opt -j2`
- Focused lit for the RVV source-front-door fixtures, target artifact exporter
  coverage, source-artifact bundle front-door coverage, and evidence script
  dry-run coverage touched by this task.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-source-frontdoor-family-add-sub-mul --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- Targeted residue scans over touched RVV plugin/target/script/tests.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/tasks/archive/2026-05/05-18-rvv-source-frontdoor-generated-bundle-runtime-abi-bridge/prd.md`
- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `lib/Transforms/ExecutionPlanningPipeline.cpp`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir`
- `test/Target/RVV/vector-source-target-artifact-exporters.mlir`
- `scripts/rvv_generated_bundle_abi_e2e.py`

## Technical Notes

- Source materialization must remain RVV plugin-owned.
- Common translate orchestration must stay family-neutral and consume enabled
  source-front-door pass registrations.
- Target artifact export must continue through selected emission-plan
  candidates, plugin-owned `TCRVEmitCLowerableRoute` builders, common
  materialized EmitC helpers, MLIR EmitC C/C++ emission, and registered target
  artifact exporters.
- Runtime ABI parameter order is part of the acceptance contract:
  `lhs, rhs, out, n`.

## Definition of Done

- The existing add/sub/mul source-front-door family is proven through generated
  bundle artifacts, external C ABI harness, and real `ssh rvv` correctness
  evidence.
- Any production C++ changes are narrowly tied to family coherence; if no C++
  changes are needed, the evidence and tests make that explicit.
- Focused checks pass or failures are recorded as real blockers without
  restoring stale descriptor/direct-C/source-export behavior.
- The Trellis task is finished, archived, and committed as one coherent commit.

## Completion Notes

- Inventory found no production C++ gap. The RVV source front door, EmitC route
  provider, construction protocol, target artifact bundle, and evidence script
  already carry add/sub/mul as the existing bounded `i32m1` arithmetic family.
- No compiler C++ was changed. This round made the module truthful by replacing
  the add-only generated-bundle dry-run lit coverage with a full-family
  add/sub/mul dry-run test that validates the generated bundle evidence and
  external C ABI harnesses for all three operations.
- The production one-command source-artifact bundle front door was proven
  locally for add/sub/mul by
  `test/Scripts/rvv-generated-bundle-abi-e2e-source-family-dry-run.test` and
  by the existing `source-artifact-bundle-front-door-rvv.mlir` target artifact
  coverage.
- Real RVV evidence was refreshed under
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260518T-rvv-source-frontdoor-family-add-sub-mul`.
  The remote run printed:

  ```text
  tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
  PASS op=add counts=7,16,23
  tcrv_rvv_generated_bundle_abi_sub_ok counts=7,16,23
  PASS op=sub counts=7,16,23
  tcrv_rvv_generated_bundle_abi_mul_ok counts=7,16,23
  PASS op=mul counts=7,16,23
  ```

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-i32m1-source-frontdoor-family-production-closure`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `cmake --build build --target tcrv-translate tcrv-opt -j2`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-generated-bundle-abi-e2e-self-test.test Scripts/rvv-generated-bundle-abi-e2e-source-family-dry-run.test ../test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir ../test/Target/RVV/vector-source-target-artifact-exporters.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir` passed 8/8.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-source-frontdoor-family-add-sub-mul --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- `git diff --check`
- Old manual pipe scan over `scripts/rvv_generated_bundle_abi_e2e.py` found no
  `source-artifact-front-door-pipeline`, `tcrv-export-target-artifact-bundle`,
  `tcrv_opt`, or `--tcrv-opt`.
- Targeted descriptor/direct-C/source-export residue scan over the RVV
  plugin/target, evidence script, and touched tests found only negative
  FileCheck assertions, explicit rejection lists/self-tests, and RVV target
  fail-closed rejection code.
- `cmake --build build --target check-tianchenrv -j2` passed 125/125 tests.

## Spec Update Review

No `.trellis/spec/` edit was required. Existing RVV plugin and
lowering-runtime specs already require this add/sub/mul source-front-door
family shape, common source-artifact bundle front door, declaration-only
header, selected EmitC route, and `ssh rvv` evidence boundary.
