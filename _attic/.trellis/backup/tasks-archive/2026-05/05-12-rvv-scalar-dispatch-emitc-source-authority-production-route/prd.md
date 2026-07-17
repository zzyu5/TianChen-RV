# RVV+scalar dispatch EmitC source authority production route

## Goal

Rewire the production/default RVV+scalar dispatch C source wrapper so the
dispatcher function body is emitted from a verified MLIR EmitC module translated
by `mlir::emitc::translateToCpp`, using selected `tcrv.exec.dispatch` /
`tcrv.exec.case` / `tcrv.exec.fallback` IR, selected component emission plans,
and the IR-backed runtime ABI parameter model as the source of dispatch control.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean at task creation.
- Task start HEAD is `b6a2d6f feat(scalar): use emitc source authority for fallback source`.
- `.trellis/.current-task` was absent at session start, so this task was
  created as `.trellis/tasks/05-12-rvv-scalar-dispatch-emitc-source-authority-production-route/`.
- The archived scalar task
  `.trellis/tasks/archive/2026-05/05-12-scalar-emitc-source-authority-production-route/`
  and the archived RVV source-authority task remain closed and must not be
  reopened.
- RVV and scalar component callable bodies already use the common
  `TCRVEmitCLowerableRoute` to MLIR EmitC / MLIR Cpp emitter source-authority
  route.
- Current dispatch export already validates selected RVV case, scalar fallback,
  component route ids, selected family metadata, IR-backed callable ABI mirrors,
  `tcrv.exec.dispatch`, `tcrv.exec.fallback`, and selected-case
  `runtime_guard` linkage.
- Current dispatch source output still prints the top-level dispatcher wrapper
  via a handwritten C control renderer in `lib/Target/Builtin/RVVScalarDispatch.cpp`.

## Requirements

- Rewire the default RVV+scalar dispatch source route so the dispatcher wrapper
  body comes from `emitTCRVEmitCLowerableRouteAsCppSource` or the same common
  MLIR EmitC / MLIR Cpp emitter authority surface used by RVV and scalar
  components.
- Build the dispatch route from IR-selected semantics: selected kernel,
  selected RVV dispatch-case variant, selected scalar fallback variant,
  selected component callable symbols, selected family identity, the
  `runtime_guard`-linked dispatch availability parameter, and ABI-ordered
  runtime parameters.
- Add only generic common materializer support for a dispatch-control route
  shape: branch on a runtime guard, call the selected RVV component with the
  ABI-ordered callable parameters, return, then call the selected scalar
  fallback component.
- Do not add RVV/scalar family semantic branches to common materializer code.
  The route payload must carry callable symbol names, ABI mappings, and control
  provenance.
- Generated source must record dispatch-specific source-authority/provenance
  comments distinct from RVV/scalar component body authority comments.
- The exporter must fail closed for missing or stale dispatch facts, including
  missing/malformed runtime guard, wrong guard ABI role, missing callable ABI
  roles, wrong parameter ordering, missing component emission plan, stale RVV
  component route, stale scalar component route, and route-family mismatch.
- Preserve component ownership: RVV and scalar component bodies remain emitted
  by their target/plugin-owned component source-authority routes; dispatch only
  emits runtime control glue and calls those component functions.
- Keep `tcrv.exec` compute-free and keep descriptors out of the production
  dispatch computation/control source authority. Descriptor-local metadata may
  only validate bounded legacy mirrors where existing component routes require
  it.
- Python remains tooling-only for runners and evidence parsing.

## Acceptance Criteria

- [ ] `lib/Target/Builtin/RVVScalarDispatch.cpp` production/default dispatch
      source no longer renders the dispatcher wrapper through
      `printDispatcherFunction`-style handwritten C control text.
- [ ] A common MLIR EmitC source-authority materializer supports a bounded
      generic dispatch-control route shape without family-name branches.
- [ ] Generated dispatch source records
      `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter` and a
      dispatch-specific route/source-authority comment for the wrapper.
- [ ] Dispatch wrapper source calls the selected RVV and scalar component
      symbols in the ABI-ordered callable parameter order and branches only on
      the IR-linked dispatch availability guard parameter.
- [ ] Existing component source authority remains separate: RVV/scalar callable
      component bodies continue to be emitted by their own target/plugin routes.
- [ ] Focused C++ and lit/FileCheck tests prove dispatch source authority,
      ABI ordering, guard linkage, component route fail-closed behavior, and
      no stale handwritten production renderer dependency.
- [ ] Dry-run dispatch bundle validation passes for one bounded family such as
      `i32-vmul`.
- [ ] One bounded `ssh rvv` validation is attempted if the environment is
      available; any claim is limited to that exact dispatch slice.
- [ ] `git diff --check`, Trellis validation, and focused build/lit checks pass.
- [ ] `check-tianchenrv -j2` is run if focused checks pass and the build tree is
      usable.
- [ ] One coherent commit records the completed module, or the task remains
      open with the exact continuation point.

## Non-Goals

- No new RVV/scalar family, dtype, LMUL, operation matrix, benchmark,
  throughput, latency, or performance claim.
- No compute semantics in `tcrv.exec`; `tcrv.exec.dispatch` remains execution
  control.
- No descriptor-driven computation and no descriptor-to-C dispatch renderer as
  the production path.
- No moving RVV/scalar semantic decisions into common passes or common EmitC
  materializer branches.
- No Python compiler implementation.
- No helper-only, metadata-only, wrapper-only, smoke-only, or report-only
  closeout.
- No migration of self-check harness source to EmitC unless it is the single
  blocker for production dispatch source/object validation.

## Technical Approach

Add a small dispatch-control source-authority surface to the existing EmitC
materializer route payload. Target-owned dispatch code will construct a
`TCRVEmitCLowerableRoute` whose ABI mappings are the full dispatcher ABI
parameters, whose steps encode the selected dispatch case call and fallback
call, and whose guard operand names the IR-linked dispatch-availability
runtime parameter. The common materializer will recognize the route shape by
generic source roles, emit an `emitc.if` over the guard value, emit an
`emitc.call_opaque` to the primary callable followed by `emitc.return`, and
emit an `emitc.call_opaque` to the fallback callable after the if.

The RVV+scalar dispatch exporter will keep target-owned selection, family,
component authority, and runtime ABI validation. After those checks, it will
emit metadata and embedded component sources as today, but the top-level
dispatcher wrapper function body will come from MLIR EmitC / MLIR Cpp emitter
translation.

## Decision (ADR-lite)

**Context**: RVV and scalar component source-authority migrations are complete,
but the top-level dispatcher wrapper remains a handwritten production C
renderer despite already consuming typed selected dispatch IR and ABI facts.

**Decision**: Promote a generic dispatch-control route shape in the existing
EmitC source-authority materializer and use it for the production
RVV+scalar dispatch wrapper.

**Consequences**: Generated dispatcher formatting may change according to the
MLIR Cpp emitter. Tests should assert source authority, guard linkage,
callable symbols, ABI order, and fail-closed behavior instead of exact legacy
handwritten wrapper formatting.

## Minimal Validation

- Build focused owners:
  `TianChenRVConversionEmitC`, `TianChenRVBuiltinTargetArtifactExporters`,
  `TianChenRVRVVTarget`, `TianChenRVScalarTarget`, `tcrv-translate`,
  `tianchenrv-emitc-lowerable-interface-test`, and
  `tianchenrv-target-artifact-export-test` as applicable.
- Run focused C++ tests for EmitC lowerable/source authority and target
  artifact export.
- Run focused lit/FileCheck under `test/Target/RVVScalarDispatch/`, relevant
  `test/Target/TargetArtifactBundleExport/` fixtures, and
  `test/Scripts/rvv-scalar-dispatch-e2e.test` plus
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`.
- Run one dry-run dispatch bundle validation through
  `scripts/rvv_scalar_dispatch_e2e.py` using the target-artifact bundle /
  plan-and-export front door for `i32-vmul`.
- Attempt one fresh bounded `ssh rvv` RVV+scalar dispatch validation if the
  environment is available.
- Run `git diff --check`, Trellis task validation, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and the build tree is usable.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Prior PRDs read:
  - `.trellis/tasks/archive/2026-05/05-12-scalar-emitc-source-authority-production-route/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-emitc-source-authority-production-route/prd.md`
- Initial source focus:
  - `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h`
  - `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
  - `include/TianChenRV/Target/RVVScalarDispatch.h`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
  - `lib/Support/RuntimeABIContract.cpp`
  - `lib/Support/RuntimeABICallablePlan.cpp`
  - `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`

## Definition Of Done

This task is complete only when the production/default RVV+scalar dispatch
source wrapper is MLIR EmitC / MLIR Cpp emitter source-authority backed, the
RVV/scalar component source authorities remain target/plugin owned, focused
checks pass, Trellis state is truthful, and one coherent commit records the
module. If unfinished, leave the task open with the exact missing production
call site, materializer gap, failing fixture, dry-run/ssh evidence blocker, or
validation target.
