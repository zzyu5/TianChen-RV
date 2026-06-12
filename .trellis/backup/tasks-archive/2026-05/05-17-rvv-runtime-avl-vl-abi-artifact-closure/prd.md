# RVV runtime AVL/VL ABI artifact closure

## Goal

Close the runtime AVL/VL ABI boundary for the existing bounded RVV i32m1
materialized EmitC target artifact route. The selected RVV typed body,
emission-plan metadata, materialized EmitC route, declaration-only header,
bundle index, and `ssh rvv` evidence must agree that callable ABI parameter
`n` is the runtime AVL consumed by `tcrv_rvv.setvl`, that `setvl` produces the
VL consumed by `tcrv_rvv.with_vl`, and that the current supported artifact is
only a one-VL i32m1 arithmetic slice.

## What I Already Know

- Current HEAD before the task is `cad75b2 tensorextlite: activate common emitc
  route`; the worktree was clean and no `.trellis/.current-task` existed.
- The previous RVV task rebuilt the selected materialized EmitC object, header,
  and bundle path from one selected emission-plan candidate.
- Current RVV selected seed already materializes explicit
  `tcrv_rvv.runtime_abi_value`, `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
  `tcrv_rvv.i32_load`, `tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store` ops.
- Current RVV route construction already binds the `setvl` AVL operand to the
  `runtime-element-count` ABI role and emits `__riscv_vsetvl_e32m1(n)` followed
  by load/compute/store calls using local produced `vl`.
- `RVVConfigContract` already defines an artifact metadata contract for
  SEW32/LMUL m1/tail agnostic/mask agnostic plus runtime AVL/VL identity, but
  the emission plan currently only publishes `rvv_emitc_lowerable_route` and
  `rvv_arithmetic_op`.
- Current generated header declares `lhs, rhs, out, n` but does not state that
  `n` is the runtime AVL or that the artifact is a one-VL slice.
- Current bundle index carries runtime ABI parameters and route metadata, but
  does not yet carry the RVV runtime AVL/VL contract metadata.

## Requirements

- Publish the RVV i32m1 config/runtime AVL/VL artifact metadata from the
  supported RVV emission plan, together with existing route and arithmetic-op
  provenance.
- Validate selected RVV target artifact candidates against the same metadata
  contract before object, header, or bundle export.
- Keep the runtime ABI parameter order as `lhs, rhs, out, n`; make `n` the
  explicit runtime AVL identity, not descriptor-local element count metadata or
  a hardcoded artifact count.
- Make generated declaration-only headers expose bounded metadata comments for
  runtime AVL, produced VL, VL scope, and the one-VL i32m1 slice limit, while
  preserving the body-free header rule.
- Ensure bundle metadata/index preserves the same runtime AVL/VL contract for
  the object/header component group.
- Keep all compiler implementation in C++/MLIR/LLVM/TableGen/CMake/lit/
  FileCheck. Python may only be used for existing tooling/evidence capture.

## Acceptance Criteria

- [x] Positive lit evidence shows selected RVV i32m1 add path with runtime ABI
      value `n`, `tcrv_rvv.setvl %n`, produced VL consumed by
      `tcrv_rvv.with_vl`, and emission-plan artifact metadata for
      `runtime_avl_source`, `vl_def`, `vl_scope`, `runtime_abi_order`, and
      one-VL bounded slice support.
- [x] Positive header evidence shows the declaration-only header prototype
      `lhs, rhs, out, n` plus comments tying `n` to runtime AVL and the
      produced VL to `setvl`/`with_vl`.
- [x] Positive bundle evidence shows object and header records preserve the
      same selected variant, route, ABI name/kind, runtime ABI parameter order,
      and RVV runtime AVL/VL artifact metadata.
- [x] C++ target exporter tests reject missing runtime AVL/VL metadata, stale
      descriptor-local element-count metadata, hardcoded artifact element
      counts, stale runtime ABI order, and mismatched setvl/with_vl metadata.
- [x] Existing negative dialect/route coverage continues to reject missing AVL
      ABI, mismatched setvl/with_vl SSA, wrong SEW/LMUL/policy config, and
      unsupported extra RVV ops/multi-VL bodies.
- [x] Fresh artifact evidence and `ssh rvv` compile/run uses the bundled
      header/object path with at least two supported AVL values inside the
      bounded one-VL contract, or the missing rebuild gap is reported without
      restoring descriptor/direct-C behavior.
- [x] Targeted scans over touched RVV plugin/target/test surfaces show no
      restored descriptor route authority, no direct C semantic exporter, no
      source-export route, and no tests protecting old paths.

## Non-Goals

- No new RVV op family, dtype, LMUL, generic vector lowering, high-level
  frontend lowering, TensorExtLite target artifact, scalar fallback compute,
  descriptor table/adapter, direct C semantic exporter, source-export route,
  compatibility wrapper, Python compiler-core behavior, or common/core RVV
  semantic branch.
- No generic multi-VL loop implementation in this round. If the current route
  remains one-VL, it must say so and fail closed for unmodeled multi-iteration
  route shapes.
- No new artifact ledger, state machine, checkpoint protocol, or prompt/report
  treated as implementation achievement.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Prior task PRDs read:
  `.trellis/tasks/archive/2026-05/05-17-rvv-materialized-emitc-header-bundle-abi-packaging-bridge/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-17-05-17-tensorextlite-common-emitc-route-activation/prd.md`.
- Primary code surfaces:
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`,
  `test/Target/RVV/source-seed-target-artifact-header.mlir`,
  `test/Target/RVV/source-seed-target-artifact-object.mlir`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Definition Of Done

- Runtime AVL/VL metadata is production-consumed by RVV emission plan and target
  artifact validation, not just documented.
- Focused C++ tests, lit tests, `git diff --check`, and targeted scans are
  recorded.
- Artifact output and `ssh rvv` evidence are refreshed or the exact blocker is
  reported as missing new-architecture work.
- Trellis task status and workspace journal are updated truthfully.
- One coherent commit records the completed closure, or the task remains open
  with the exact next continuation point.

## Implementation Summary

- Extended the RVV i32m1 artifact metadata contract with explicit
  `runtime_avl_abi_parameter = n`, `bounded_slice =
  one-vl-i32m1-arithmetic`, and `multi_vl = unsupported`.
- Made `RVVExtensionPlugin::buildVariantEmissionPlan` publish the full
  RVV config/runtime AVL/VL artifact metadata contract in the supported
  selected emission-plan diagnostic.
- Made the RVV target artifact candidate preflight verify route arithmetic
  provenance plus the full `tcrv_rvv.*` metadata contract before object,
  header, or bundle export.
- Made the declaration-only header print runtime AVL, produced VL definition,
  VL scope, bounded slice, and multi-VL support comments from the same selected
  candidate metadata used by object/bundle export.
- Added C++ negative coverage for missing runtime AVL/VL metadata, stale
  runtime ABI order, stale VL scope metadata, and descriptor/hardcoded
  element-count residue.
- Updated lit coverage for emission-plan metadata, declaration-only header
  metadata, and bundle index metadata on both object and header records.

## Validation

- Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`.
- Focused C++:
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-i32m1-selected-boundary-seed|source-seed-target-artifact-header|source-seed-target-artifact-object|rvv-first-slice-vl-contract-negative|rvv-first-slice-config-vl-contract-negative|rvv-first-slice-materialization-missing-abi|rvv-first-slice-materialization-negative'`
  from `build/test`, 8/8 selected tests passed.
- Full check:
  `cmake --build build --target check-tianchenrv -j2`, 106/106 lit tests
  passed.
- `git diff --check` passed.
- Artifact evidence:
  `artifacts/tmp/rvv_runtime_avl_vl_abi_artifact_closure/20260516T192409Z`
  contains selected plan MLIR, declaration-only header, RISC-V relocatable
  object, bundle index, bundled object/header files, readobj output, and remote
  compile/run log.
- Local object evidence:
  `/usr/lib/llvm-20/bin/llvm-readobj -h rvv_target_artifact.o` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`, and `Type: Relocatable`.
- Real `ssh rvv` header+object evidence:
  `tcrv_rvv_runtime_avl_vl_header_bundle status=PASS runtime_avl_values=2,4 bounded_slice=one-vl-i32m1-arithmetic`.
- Targeted scans over touched RVV plugin/target/tests found no restored
  historical RVV descriptor target route identifiers or old
  `tcrv-rvv-i32m1-{add,sub,mul}` artifact route strings. Matches were limited
  to negative guard text such as header `implicit-check-not` lines, an Offload
  source-route absence assertion, and RVV dialect prose rejecting direct-C
  export.
- Targeted element-count residue scan found no descriptor-local or hardcoded
  artifact element-count metadata in touched RVV plugin/target/test surfaces.

## Status

Complete. The task can be archived after Trellis finish bookkeeping and one
coherent commit.
