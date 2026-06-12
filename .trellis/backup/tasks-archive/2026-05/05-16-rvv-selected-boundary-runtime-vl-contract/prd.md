# RVV selected-boundary config and runtime-VL contract

## Goal

Define one coherent RVV i32m1 selected-boundary config/runtime-VL contract and
make the existing bounded source-seed add route preserve and validate it from
RVV seed materialization through RVV plugin validation, emission planning,
EmitC route construction, target artifact/header or bundle metadata, and real
RVV object/header export.

This round is about the existing i32m1 path only:

```text
bounded source seed
  -> RVV selected boundary
  -> RVV extension-family ops
  -> common EmitC materialization
  -> target artifact/header/bundle metadata
  -> ssh rvv correctness evidence
```

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean at HEAD `7717877`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the supplied Direction Brief.
- The previous archived task added
  `--tcrv-source-seed-artifact-front-door-pipeline`, composing plugin-owned
  source-seed materializers with generic legality/capability/emission-plan and
  coherence checks.
- Existing RVV source-seed output materializes one selected `rvv-plugin`
  variant containing explicit `tcrv_rvv.runtime_abi_value` bindings for
  `lhs`, `rhs`, `out`, and `n`, followed by `tcrv_rvv.setvl`, selected
  `tcrv_rvv.with_vl`, two `tcrv_rvv.i32_load` ops, one `tcrv_rvv.i32_add`,
  and one `tcrv_rvv.i32_store`.
- Existing RVV dialect verification already validates bounded metadata on
  `setvl`, `with_vl`, dataflow ops, and runtime ABI value bindings.
- Existing `RVVConfigContract` validates that the bounded i32m1 arithmetic
  route has SEW32, LMUL m1, tail/mask agnostic policy, and a visible
  `with_vl` consumer of the `setvl` VL result.
- Existing RVV EmitC route construction validates runtime ABI role/type/order
  and fails before route payload construction on malformed boundary/dataflow.
- The current gap is contract exposure and coherence across planning/export
  metadata: emission-plan diagnostics and target/header/bundle metadata should
  carry the same explicit RVV config/runtime-VL contract that the seed and
  EmitC route validate.

## Requirements

- Keep the change bounded to the existing RVV i32m1 arithmetic path,
  preferably the source-seed add proof path.
- Reuse or extend plugin-local RVV helpers for the config/runtime-VL contract
  only when they are consumed by real RVV validation, EmitC route construction,
  or target export.
- The selected-boundary contract must validate exactly one `tcrv_rvv.setvl`
  and one selected `tcrv_rvv.with_vl` for the bounded artifact route.
- The selected-boundary contract must require SEW=32, LMUL=m1, tail agnostic,
  and mask agnostic policy for the route that can produce object/header
  artifacts.
- Runtime AVL/VL must stay as SSA/control flow: `n` is the runtime ABI
  element-count value feeding `setvl`, and `with_vl`, load, compute, and store
  ops must consume the same visible VL token.
- Runtime ABI validation must require the stable callable C ABI parameter
  order and C names: `lhs`, `rhs`, `out`, `n`.
- Emission-plan diagnostics must expose the selected RVV config/runtime-VL
  contract in structured metadata rather than only relying on an implicit
  hand-shaped body.
- Target artifact/header or bundle metadata must expose the same RVV
  config/runtime-VL contract carried by the emission plan and validated by the
  EmitC route.
- EmitC route materialization must fail closed on mismatched policy, missing
  runtime AVL/VL, stale selected-boundary residue, wrong ABI role order, or
  any dataflow op using a different VL.
- The source-seed artifact front door must continue to produce the RVV object
  and callable header through the existing target export path.
- Common/core code may remain a generic metadata carrier, but it must not
  learn RVV intrinsic names, RVV arithmetic semantics, RVV source marker names,
  or RVV-specific runtime ABI interpretation.
- Do not introduce descriptor-driven computation, direct C semantic exporters,
  Python compiler-core logic, compatibility wrappers, new artifact routes, or
  new source shapes/op families.

## Acceptance Criteria

- [x] RVV source-seed materialization emits the selected i32m1 boundary with
      explicit ABI values for `lhs`, `rhs`, `out`, `n`, SEW32 LMUL m1
      `setvl`, selected `with_vl`, and same-VL load/add/store dataflow.
- [x] RVV plugin validation and EmitC route construction use one plugin-owned
      config/runtime-VL contract and reject mismatched SEW, LMUL, tail/mask
      policy, missing `setvl`/`with_vl`, non-visible VL, missing runtime AVL,
      wrong ABI roles/names/order, stale selected-boundary residue, and
      different-VL dataflow.
- [x] Emission-plan diagnostics expose the RVV config/runtime-VL contract,
      including SEW32, LMUL m1, tail/mask agnostic policy, `setvl`/`with_vl`
      boundary identity, AVL source, VL use, and callable ABI order.
- [x] RVV header or bundle metadata exposes the same config/runtime-VL contract
      and remains generated through the existing target artifact route.
- [x] Positive lit/FileCheck coverage proves the source-seed front door reaches
      supported RVV emission-plan metadata with the explicit contract and that
      target header/bundle metadata carries the same contract.
- [x] Negative lit/FileCheck coverage proves fail-closed behavior for policy
      mismatch, missing runtime AVL/VL, stale selected-boundary residue, wrong
      ABI role order/name, and dataflow using a different VL.
- [x] Focused C++ tests for RVV dialect/plugin/construction/target support are
      updated where the contract API or metadata changes.
- [x] Focused build, focused RVV lit, `git diff --check`, and
      `check-tianchenrv` if practical pass.
- [x] The RVV source-seed artifact path is rerun and linked/executed on
      `ssh rvv` for the i32m1 add proof.
- [x] Changed-surface scans show no descriptor/direct-C/Python compiler-core
      route and no RVV semantic branch in common/core orchestration.

## Validation Results

- Focused build passed:
  `cmake --build build --target TianChenRVRVVDialect
  TianChenRVRVVPlugin TianChenRVTarget tcrv-opt tcrv-translate
  tianchenrv-target-artifact-export-test -j2`.
- Focused RVV/plugin build passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-construction-protocol-common-test
  tianchenrv-rvv-dialect-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  `./build/bin/tianchenrv-construction-protocol-common-test`, and
  `./build/bin/tianchenrv-rvv-dialect-test`.
- Focused lit passed from `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'rvv-i32m1-selected-boundary-seed|i32m1-add-object-artifact|i32m1-selected-dispatch-artifact|i32m1-sub-selected-dispatch-artifact|i32m1-mul-selected-dispatch-artifact|i32m1-object-stale-route-op|i32m1-object-missing-contract-metadata|i32m1-artifact-ambiguous-selected|source-seed-artifact-front-door'`,
  12/12 selected tests passed.
- Full practical check passed:
  `cmake --build build --target check-tianchenrv -j2`, 110/110 lit tests
  passed.
- `git diff --check` passed.
- RVV source-seed artifact evidence directory:
  `artifacts/tmp/rvv_selected_boundary_runtime_vl_contract/20260516T150644Z`.
- Generated artifact files include `seed.planned.mlir`, `seed.o`, `seed.h`,
  `bundle/tianchenrv-target-artifact-bundle.index`, and bundle header/object
  outputs. `file` reported `seed.o` as a RISC-V ELF relocatable object.
- Artifact metadata scan confirmed the same RVV config/runtime-VL entries in
  the emission-plan diagnostic, generated header, bundle header, and bundle
  index.
- `ssh rvv` link/run evidence:
  `artifacts/tmp/rvv_selected_boundary_runtime_vl_contract/20260516T150644Z/ssh_rvv_link_run.log`.
  Remote run printed
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`
  and exited with status 0.
- Common/core RVV semantic branch scan returned only existing neutral comments
  under Exec/Transforms and no new common/core RVV semantic branch.
- Descriptor/direct-C/Python compiler-core scan returned no descriptor-driven
  compute, direct source/C exporter, or Python compiler-core route in the
  changed surfaces. The only `descriptor` matches were pre-existing bounded
  RVV target route descriptor variable names in RVV-owned target support code.

## Self-Repair

- Added a generic artifact metadata carrier instead of teaching common
  emission-readiness, coherence, or target export code RVV-specific semantics.
  RVV-specific validation remains in `RVVConfigContract` and the RVV target
  support bundle.
- Updated hand-authored RVV emission-plan fixtures to carry the same explicit
  metadata expected from plugin-produced emission plans.
- Added a negative RVV target artifact test for missing/mismatched contract
  metadata so target export fails closed before accepting stale selected
  boundary residue as artifact-ready.

## Spec Update Judgment

No spec update was needed. The existing RVV plugin, EmitC route, variant
pipeline, and testing specs already require plugin-owned RVV boundary
validation, selected-plan-driven artifact export, explicit metadata, focused
lit/FileCheck, and `ssh rvv` evidence. This round implements that contract for
the existing i32m1 route rather than adding a new durable architecture rule.

## Definition Of Done

- The task context files are curated and truthful.
- Implementation changes production/default RVV validation/export behavior,
  not just comments, reports, or helper-only code.
- Focused tests cover both positive metadata propagation and negative
  fail-closed cases.
- Task status and workspace journal are updated.
- Task is finished/archived when complete.
- One coherent commit is created when complete; if unfinished, the task remains
  open with the exact next continuation point.

## Out Of Scope

- No generic MLIR vector lowering, scalable-vector lowering, new SEW/LMUL
  variants, dtype families, op families, RVV sub/mul source seeds, high-level
  tensor lowering, TensorExt/IME/offload work, performance tuning, descriptor
  or binary-family registries, direct source/C semantic exporters, Python
  compiler-core logic, GCC-default route, compatibility wrapper, state-machine
  ledger, broad smoke matrix, or docs/tests-only completion.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior PRD read:
  `.trellis/tasks/archive/2026-05/05-16-05-16-registry-composed-source-seed-front-door/prd.md`.
- Relevant journal read:
  `.trellis/workspace/codex/journal-8.md` sessions 93, 94, 95, 96, and 97.
- Primary implementation surfaces inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
