# Stage2 RVV masked memory movement executable slice

## Goal

Complete one bounded Stage2 RVV mask-aware unit-stride memory movement
executable slice on the corrected typed `tcrv_rvv` surface. The concrete route
is signed i32 / SEW32 / LMUL m1:

```text
dst[i] = mask[i] ? src[i] : old_dst[i]
```

The source data buffer, mask buffer or mask vector role, destination buffer,
old-destination preservation requirement, runtime `n` / AVL, vector
dtype/config, mask element/config facts, source and destination unit-stride
memory forms, tail/mask policy, inactive-lane policy, and ABI roles must be
explicit typed facts. The route must start from a selected `tcrv.exec` RVV
boundary and typed or pre-realized `tcrv_rvv` body, flow through RVV
selected-body realization if needed, RVV plugin-owned route planning/provider
output, neutral common EmitC/export, generated artifact evidence, and real
`ssh rvv` correctness evidence if executable correctness is claimed.

This is one bounded masked unit-stride source-to-destination memory movement
slice. It is not a broad masked-memory framework, masked gather/scatter matrix,
segmented memory task, high-level frontend task, or one-intrinsic wrapper
route.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV masked memory movement executable slice`.
- Module owner: RVV plugin-owned mask-aware unit-stride memory movement path
  for one bounded i32 SEW32 LMUL m1 masked-load to masked/unit-stride-store
  executable slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `28f1ecc4 rvv: add indexed scatter memory executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- The immediately prior task completed the indexed scatter slice with typed
  source/index/destination roles, selected-body realization, route
  planning/provider output, generated-bundle dry-run, real `ssh rvv`
  correctness for counts `7,16,23`, full `check-tianchenrv`, and a clean
  worktree.
- Existing Stage2 memory movement covers strided source load, indexed gather,
  and indexed scatter. The current memory gap is mask-aware unit-stride memory
  semantics where active lanes update and inactive lanes preserve old
  destination values.
- Existing masked arithmetic policy work is useful only as mask/tail policy
  precedent. Masked memory movement must be represented as typed
  source/mask/destination memory structure consumed by RVV plugin-owned
  realization and route planning, not as arithmetic-only policy, route-id,
  artifact-name, descriptor, direct-C, ABI-string, or harness-side sentinel
  behavior.
- `tcrv.exec` declares ABI/runtime roles only. It must not own masked memory
  semantics, inactive-lane policy, mask config, dtype/config, route support, or
  intrinsic spelling.
- Common EmitC/export must stay neutral and only materialize provider-built
  route payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, route-id
  authority, artifact-name authority, or common/export RVV semantic authority.

## Requirements

1. Keep scope to one signed i32 / SEW32 / LMUL m1 masked unit-stride memory
   movement route with `src`, `mask`, `dst`, and `n/AVL` ABI/runtime roles.
2. The selected boundary or typed body must structurally carry source
   `mem_window`, mask `mem_window` or mask vector role, destination
   `mem_window`, source unit-stride memory form, destination unit-stride memory
   form, vector dtype/config, mask element/config facts, tail/mask policy,
   inactive-lane policy, runtime `n/AVL`, old-destination preservation
   requirement, and ABI roles.
3. If a pre-realized body is used, RVV selected-body realization may
   materialize only legal generic typed masked memory structure. It must not
   change computation, dtype semantics, parameter roles, selected variant
   origin, required capabilities, dispatch/fallback behavior, inactive-lane
   semantics, or runtime `n` / AVL values.
4. RVV route planning must derive ABI order, vector C types, mask C type,
   mask-load or mask-materialization leaves, memory intrinsic leaves,
   inactive-lane policy mirrors, header/artifact metadata mirrors, and
   fail-closed diagnostics from typed source/mask/destination/config/runtime
   facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer RVV masked memory form, inactive-lane
   semantics, mask type, dtype, SEW, LMUL, or intrinsic choices.
6. Missing mask role, invalid mask config, unsupported inactive-lane policy,
   missing old-destination preservation when required, missing `n/AVL`,
   mismatched dtype/config, stale route-id authority, unsupported memory form,
   and incomplete typed masked memory body structure must fail closed with
   targeted diagnostics.
7. Generated bundle evidence must use non-trivial mask patterns and
   sentinel-filled destination buffers so correctness proves active masked
   lanes update and inactive plus tail lanes preserve sentinel values.
8. Real `ssh rvv` evidence is required for any runtime/correctness claim.

## Acceptance Criteria

- [ ] `tcrv_rvv` accepts a valid typed masked unit-stride memory movement body
      and rejects missing/wrong mask operands, missing mask role, invalid mask
      config, unsupported inactive-lane policy, missing old-destination
      preservation when required, wrong memory form, missing AVL/runtime roles,
      type/config mismatches, stale route-id authority, and incomplete masked
      memory body structure.
- [ ] RVV selected-body realization materializes the pre-realized masked memory
      body into legal generic typed `setvl`, mask load or equivalent
      mask-vector materialization, source unit-stride load, old-destination
      preserving masked store or equivalent active-lane update structure if
      this slice uses a pre-realized fixture path.
- [ ] RVVEmitCRoutePlanning derives route metadata, runtime ABI order, data
      vector C type, mask C type, mask intrinsic leaf, unit-load/store
      intrinsic leaves, inactive-lane policy mirrors, and target
      header/artifact facts from typed body/config/runtime facts.
- [ ] Positive route/materialization tests prove typed masked memory facts
      reach provider-derived metadata and generated output through
      `TCRVEmitCLowerableRoute`.
- [ ] Negative fail-closed tests cover missing mask role, invalid mask config,
      unsupported inactive-lane policy, missing old-destination preservation
      when required, mismatched dtype/config, missing AVL/runtime roles, stale
      route-id authority, unsupported memory form, and incomplete typed body
      structure.
- [ ] Generated-bundle dry-run passes for the masked memory slice at counts
      `7,16,23`.
- [ ] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving active
      masked updates and inactive/tail sentinel preservation.
- [ ] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, selected-body realization, route planning/provider,
      materializer/export, and generated-bundle paths pass.
- [ ] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [ ] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No masked gather/scatter matrix, segmented memory, atomics, broad
  mask-policy matrix, dtype/LMUL clone batch,
  high-level Linalg/Vector/StableHLO frontend lowering, one-intrinsic wrapper
  dialect, dashboard, global tuning database, report-only inventory,
  source-front-door positive route, or performance claim.
- No macc/reduction/compare/select/broadcast/conversion side quest.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper preserving old i32 route authority.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative masked
   memory movement structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the masked memory movement op at counts
   `7,16,23`.
8. Run real `ssh rvv` correctness for the masked memory movement op at counts
   `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/implementation-stack/compiler-stack-contract.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-indexed-scatter-memory-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-indexed-gather-memory-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-strided-memory-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-tail-mask-policy-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-selected-body-masked-add-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-masked-add-route-semantics/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-conversion/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-reduce-add-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-macc-executable-slice/prd.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing masked-add and memory-movement generated-bundle evidence tests.

## Definition Of Done

- One bounded masked unit-stride source-to-destination memory movement route is
  represented, verified, route-supported, materialized through the production
  RVV provider path, dry-run validated, and runtime-validated on `ssh rvv` if
  executable correctness is claimed.
- Existing Stage2 slices remain intact.
- The task report distinguishes route-supported, generated-bundle dry-run, and
  executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
