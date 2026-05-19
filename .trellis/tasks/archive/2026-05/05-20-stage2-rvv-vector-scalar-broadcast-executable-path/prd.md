# Stage2 RVV vector-scalar broadcast executable path

## Goal

Add one bounded Stage2 RVV vector-scalar broadcast path that imports an explicit
runtime scalar ABI value into a selected `tcrv.exec` RVV variant, realizes it
into typed low-level `tcrv_rvv` body structure, routes it through the RVV
provider-owned EmitC route, generates a target artifact, and proves correctness
on real `ssh rvv` for representative runtime counts.

The concrete module goal for this round is one i32 SEW32 LMUL m1 add path:

```text
lhs vector load + runtime scalar rhs splat -> generic tcrv_rvv.binary add -> out store
```

## Direction Source

- Hermes Direction Brief: `Stage2 RVV vector-scalar broadcast executable path`.
- Current HEAD before implementation: `f345bbb0 rvv: close cmp select generated bundle abi`.
- Initial repo checks in this round:
  - `pwd`: `/home/kingdom/phdworks/TianchenRV`
  - `git status --short`: clean
  - `git log --oneline -8`: `f345bbb0`, `09ca76d0`, `647c0ee6`,
    `90497af1`, `e47ff870`, `86ef664a`, `81d3a85f`, `11efca89`
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## Repository Facts Already Checked

- `.trellis/spec/index.md` and `.trellis/spec/extension-plugins/rvv-plugin.md`
  require the active RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export ->
  target artifact -> `ssh rvv` evidence for runtime/correctness claims.
- `.trellis/spec/lowering-runtime/emitc-route.md` and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md` forbid common
  EmitC/export, route ids, artifact names, C ABI strings, descriptors,
  source-front-door metadata, or status fields from becoming route or compute
  authority.
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md` keeps
  selected-body realization and route provider decisions inside the origin
  plugin.
- `.trellis/spec/testing/mlir-testing-contract.md` requires focused lit/unit
  coverage for typed body legality, selected-body realization, provider
  materialization, fail-closed diagnostics, and real `ssh rvv` output for any
  executable correctness claim.
- Current explicit RHS broadcast support is a pointer memory form:
  `tcrv_rvv.broadcast_load` consumes an RHS buffer runtime ABI value and the
  provider emits a scalar load such as `rhs[0]` before RVV broadcast. That is
  not the scalar runtime ABI import requested here.
- Current pre-realized binary selected body supports add/sub/mul
  `vector-rhs-load` and strided add `strided-load-store`. It does not support a
  scalar runtime RHS role, scalar splat op, scalar-broadcast memory form, or
  scalar-broadcast runtime ABI signature.
- The latest compare/select executable ABI closure changed only evidence
  tooling and proved `ssh rvv` PASS for counts `7,16,23`; it did not add this
  scalar ABI boundary.
- Stage1 guardrails remain active: retained i32 add-like behavior is allowed
  only as an ordinary instance of the corrected generic typed surface. Do not
  add finite `tcrv_rvv.i32_*` helpers, `!tcrv_rvv.i32m*` positive authority,
  `RVVI32M1*` route tables, source-front-door positives, descriptors, or common
  EmitC RVV semantic branches.

## Requirements

1. Introduce a bounded explicit runtime scalar ABI role for the RHS scalar
   value, with stable C ABI spelling and verifier/type checks for the selected
   scalar value.
2. Add a non-dtype-prefixed typed `tcrv_rvv` scalar splat/broadcast dataflow op
   or equivalent vector-level op that consumes the explicit RHS runtime scalar
   and active VL token, producing a generic `!tcrv_rvv.vector<i32, "m1">`.
3. Extend the pre-realized binary selected-body path with one scalar-broadcast
   add memory form. The pre-realized body must bind lhs buffer, RHS scalar,
   out buffer, runtime n/AVL, op kind, SEW, LMUL, policy, memory form, and ABI
   ownership explicitly.
4. RVV selected-body realization must consume the pre-realized scalar facts into
   concrete `setvl/with_vl/load/splat/binary/store` typed `tcrv_rvv`
   structure before route construction.
5. RVV route planning/provider must derive the scalar-broadcast route from the
   typed body/config/runtime facts and fail closed for unsupported scalar role,
   missing scalar splat, missing n/AVL, wrong dtype/config/policy, unsupported
   memory form, and incomplete body structure.
6. Add a distinct provider-derived scalar-broadcast runtime ABI route/signature
   for the generated artifact, rather than reusing a pointer RHS ABI name for a
   scalar argument.
7. Common EmitC/materializer and target export must remain neutral consumers of
   provider-built route payloads. They may materialize generic operands supplied
   by the provider but must not infer scalar broadcast semantics.
8. Extend generated-bundle evidence tooling only as an external validation
   harness for the new path; scripts must not become compiler truth.
9. Keep the task bounded to one i32 SEW32 LMUL m1 vector-scalar add path.

## Acceptance Criteria

- [ ] PRD, implement/check context, and task metadata describe this bounded
      vector-scalar broadcast path and do not drift into broad broadcast or
      dtype coverage.
- [ ] `tcrv_rvv` has a generic typed scalar splat/broadcast body surface whose
      scalar operand comes from explicit runtime ABI facts, not parameter names,
      route ids, artifact names, helper names, C strings, descriptors, or
      scripts.
- [ ] Selected-body realization turns the scalar pre-realized body into
      explicit `setvl/with_vl/load/splat/binary/store` structure.
- [ ] RVV route planning/provider recognizes the scalar-broadcast add route and
      emits provider-derived metadata, runtime ABI parameters, intrinsic leaves,
      and call steps from typed body/config/runtime facts.
- [ ] Generated target header/object/bundle exists for the new scalar-broadcast
      path and records scalar-broadcast memory/runtime facts as mirrors only.
- [ ] Real `ssh rvv` generated-bundle correctness evidence passes representative
      counts `7,16,23` if executable status is claimed.
- [ ] Positive tests cover dialect parsing/verification, selected-body
      realization, route planning/materialization, target artifact generation,
      and generated-bundle dry-run for the new path.
- [ ] Negative tests fail closed for at least missing/wrong scalar role,
      missing n/AVL, unsupported memory form/policy/config, mismatched scalar
      dtype, and incomplete scalar-broadcast body structure.
- [ ] Active-authority scan shows no reintroduced positive `RVVI32M1`,
      `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door/source-seed, descriptor/direct-C/source-export, or
      common/export RVV semantic authority.
- [ ] Focused build/lit/unit/script checks, `git diff --check`, task
      validation, and worktree cleanliness checks pass.

## Non-Goals

- No broad dtype, LMUL, scalar kind, predicate, layout, conversion, reduction,
  contraction, or broadcast framework expansion.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive RVV route.
- No one-intrinsic wrapper dialect and no dtype-prefixed helper op family.
- No descriptor-driven computation, descriptor-driven C/source export,
  direct-C exporter revival, or compatibility wrapper preserving legacy i32
  route authority.
- No performance claim.
- No Template/Toy/TensorExtLite/IME/Offload/future-plugin side work.
- No dashboard, report-only inventory, broad smoke matrix, or helper-only
  completion as the main result.

## Validation Plan

1. Validate Trellis context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-vector-scalar-broadcast-executable-path`
2. Build focused compiler/test targets:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
3. Run focused C++ tests:
   `build/bin/tianchenrv-rvv-dialect-test`,
   `build/bin/tianchenrv-rvv-extension-plugin-test`,
   `build/bin/tianchenrv-construction-protocol-common-test`,
   `build/bin/tianchenrv-target-artifact-export-test`.
4. Run focused lit for the new scalar splat verifier, selected-body
   realization, provider/materialization, target artifact fixture, and negative
   fail-closed fixtures.
5. Run script checks:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
6. Run generated-bundle dry-run for the new path with counts `7,16,23`.
7. Run real `ssh rvv` generated-bundle correctness evidence for the new path
   with counts `7,16,23` if the executable path is reached.
8. Run active-authority scans over touched active RVV include/lib/script/test
   paths.
9. Run `git diff --check`.
10. Run `check-tianchenrv` if shared compiler behavior changes enough that the
    broader lit gate is justified.

## Initial Code Surface

- `include/TianChenRV/Support/RuntimeABI.h`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Focused RVV dialect, conversion, target, script, and plugin tests.

## Definition Of Done

- [ ] The production/default RVV selected-body path consumes the new scalar
      runtime ABI facts through plugin-local realization and provider route
      planning.
- [ ] Focused checks pass and any failures are repaired.
- [ ] Task status, PRD, and journal notes stay truthful.
- [ ] Task is finished/archived if complete.
- [ ] One coherent commit records the implementation, validation, task
      closeout, and evidence, or the task remains open with an exact
      continuation point.
