# RVV source-front-door generated-bundle runtime ABI bridge

## Goal

Close the runtime ABI bridge for the existing bounded RVV source-front-door
`i32m1` add path:

```text
source MLIR func/scf/vector/arith
  -> RVV plugin source front door
  -> selected RVV extension-family boundary
  -> plugin-owned materialized EmitC/object route
  -> generated object + declaration-only ABI header + bundle
  -> external C ABI harness
  -> ssh rvv correctness evidence
```

This round does not add RVV family coverage. The required proof is one coherent
source-derived add path where the generated bundle and the external C ABI
consumer agree on the same selected function, runtime ABI name, ordered
parameters `lhs, rhs, out, n`, object route, and declaration-only header.

## Current Repo Evidence

- Current HEAD is `829c902`, which added focused lit coverage proving that the
  source-level add fixture reaches object/header/bundle artifact structure.
- The source front door is implemented in
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`; it materializes selected
  `tcrv.exec`/`tcrv_rvv` IR from the exact supported `func/scf/vector/arith`
  pattern.
- RVV target support already registers the construction-template object,
  header, and bundle artifact route through
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- `scripts/rvv_generated_bundle_abi_e2e.py` is evidence tooling, not compiler
  core. It invokes the one-command source artifact bundle front door, verifies
  generated bundle records, writes an external C harness, and can compile/run
  that harness on `ssh rvv`.
- The remaining module-level gap is to make the runtime-count and harness
  evidence contract explicit and focused for the source-derived add proof, then
  refresh real `ssh rvv` evidence.

## Scope

- Tighten the RVV generated-bundle ABI evidence harness so runtime evidence
  requires several runtime `n` values and at least one bounded non-one-vector
  stress count.
- Keep the harness as Python evidence tooling only. It may parse artifacts,
  generate an external C consumer, stage files to `ssh rvv`, and record
  sanitized evidence. It must not implement compiler IR, lowering, selection,
  emission, descriptor logic, scalar fallback compute, or runtime glue.
- Add focused script/lit coverage that runs the evidence tool in dry-run mode
  on the real source-derived add fixture and checks generated evidence/harness
  records.
- Refresh real `ssh rvv` evidence for the source-derived add proof with
  runtime counts that include non-one-vector cases.
- Repair only narrowly related target-export/runtime-ABI plumbing if the
  generated source-derived bundle cannot be consumed as-is.

## Non-Goals

- No new RVV operation coverage such as sub/mul as the main result.
- No broader SEW/LMUL/dtype/source-language support.
- No descriptor tables, descriptor route IDs, direct C/source-export compute
  paths, compatibility wrappers, legacy modes, scalar fallback compute, or
  Python compiler-core behavior.
- No core pass changes that special-case RVV semantics.
- No broad smoke matrix and no standalone evidence artifact as the only
  repository change.

## Requirements

1. The evidence tool must consume the production
   `tcrv-translate --tcrv-source-artifact-bundle-front-door` path for the add
   fixture, not the old manual `tcrv-opt | tcrv-translate` chain.
2. The generated bundle must contain exactly one object and one header record
   for selected variant `@vector_source_rvv_i32_add`, runtime ABI
   `rvv-i32m1-add-callable-c-abi.v1`, route
   `rvv-i32m1-arithmetic-emitc-route-family`, and ordered parameters
   `lhs, rhs, out, n`.
3. The generated declaration-only header must expose the unmangled external C
   ABI prototype:

   ```c
   void tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add(
       const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
   ```

4. The external harness must allocate `lhs`, `rhs`, and `out`, call only the
   generated header/object ABI, validate all output elements, and print a
   deterministic PASS marker.
5. Runtime evidence must include several `n` values and at least one bounded
   non-one-vector stress count. The focused acceptance run will use
   `n = 7, 16, 23`.
6. Public evidence and generated headers/bundle indexes must reject descriptor,
   direct-C, source-export, self-check, RVV direct microkernel, raw-log, and
   credential-like residue.
7. Focused script/lit coverage must prove the source-derived add dry-run path
   writes evidence for the source artifact bundle front door, selected ABI
   identity, runtime count contract, generated harness boundary, and PASS
   marker.
8. Real `ssh rvv` evidence must be refreshed for the source-derived add proof
   and recorded under a sanitized artifact path.

## Acceptance Criteria

- [ ] `scripts/rvv_generated_bundle_abi_e2e.py --self-test` covers runtime
      count contract failures and still passes.
- [ ] A focused lit test runs the evidence tool in dry-run mode for
      `--op-kind add` and verifies the source-derived bundle/harness evidence.
- [ ] The existing RVV source artifact exporter lit coverage still passes.
- [ ] Real `ssh rvv` run for `--op-kind add --runtime-count 7 --runtime-count
      16 --runtime-count 23` succeeds and prints
      `tcrv_rvv_generated_bundle_abi_add_ok` plus `PASS op=add`.
- [ ] Targeted scans over touched RVV target/plugin/harness surfaces show no
      restored descriptor/direct-C/source-export route authority.
- [ ] Trellis task status, completion notes, and journal are truthful.
- [ ] The task is finished, archived, and committed as one coherent commit.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-source-frontdoor-generated-bundle-runtime-abi-bridge`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `cmake --build build --target tcrv-translate tcrv-opt -j2`
- Focused lit for:
  - `test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test`
  - the new source-add dry-run script test
  - `test/Target/RVV/vector-source-target-artifact-exporters.mlir`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-source-frontdoor-generated-bundle-add --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- `git diff --check`
- Targeted residue scans over `scripts/rvv_generated_bundle_abi_e2e.py`,
  touched tests, `lib/Target/RVV`, `lib/Plugin/RVV`, and
  `lib/Target/TargetArtifactExport.cpp`.
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/tasks/archive/2026-05/05-18-05-18-rvv-vector-source-artifact-bridge/prd.md`
- `test/Target/RVV/vector-source-target-artifact-exporters.mlir`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `scripts/rvv_remote_probe.py`
- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `lib/Target/TargetArtifactExport.cpp`

## Definition of Done

- The source-derived add runtime ABI bridge is proven through generated
  bundle, external C ABI harness, and `ssh rvv` correctness evidence.
- The evidence tooling remains outside compiler core and does not restore old
  descriptor/direct-C/source-export authority.
- Focused checks pass or failures are recorded as real blockers without
  papering over stale expectations.
- The Trellis task is finished, archived, and committed.

## Completion Notes

- Tightened `scripts/rvv_generated_bundle_abi_e2e.py` so runtime ABI evidence
  now requires several distinct runtime `n` counts and at least one bounded
  non-one-vector stress count `n >= 17`.
- The evidence root and per-op evidence JSON now record the runtime-count
  contract that was validated for the run.
- Repaired a stale script cwd assumption: built-in source fixtures now resolve
  relative to the repository root when the script is invoked from another
  working directory, such as lit's build/test execution root.
- Added
  `test/Scripts/rvv-generated-bundle-abi-e2e-source-add-dry-run.test`, which
  runs the evidence tool in dry-run mode on the real source-derived add fixture
  and checks the generated evidence JSON plus external C ABI harness.
- Refreshed real `ssh rvv` evidence for the add proof under
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260518T-rvv-source-frontdoor-generated-bundle-add`.
  The remote run printed:

  ```text
  add case n=7 ok
  add case n=16 ok
  add case n=23 ok
  tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
  PASS op=add counts=7,16,23
  ```

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-source-frontdoor-generated-bundle-runtime-abi-bridge`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `cmake --build build --target tcrv-translate tcrv-opt -j2`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-generated-bundle-abi-e2e-self-test.test Scripts/rvv-generated-bundle-abi-e2e-source-add-dry-run.test ../test/Target/RVV/vector-source-target-artifact-exporters.mlir`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-source-frontdoor-generated-bundle-add --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- `git diff --check`
- `rg -n "source-artifact-front-door-pipeline|tcrv-export-target-artifact-bundle|tcrv_opt|--tcrv-opt" scripts/rvv_generated_bundle_abi_e2e.py; test $? -eq 1`
- Targeted descriptor/direct-C/source-export residue scan over
  `scripts/rvv_generated_bundle_abi_e2e.py`, the touched script tests,
  `test/Target/RVV/vector-source-target-artifact-exporters.mlir`,
  `lib/Target/RVV`, `lib/Plugin/RVV`, and
  `lib/Target/TargetArtifactExport.cpp`. Matches were forbidden-token
  rejection code, self-test negative cases, and FileCheck
  `implicit-check-not` assertions only.
- `cmake --build build --target check-tianchenrv -j2` passed 125/125 tests.

## Spec Update Review

No `.trellis/spec/` edit was required. Existing RVV plugin,
lowering-runtime EmitC route, and emission-runtime specs already require this
bridge shape, runtime parameter order, non-descriptor target artifact route,
and `ssh rvv` evidence boundary. This round tightened evidence tooling and
focused coverage for that existing contract.
