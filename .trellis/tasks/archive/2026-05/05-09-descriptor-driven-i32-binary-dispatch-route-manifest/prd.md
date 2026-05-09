# Descriptor-driven i32 binary dispatch route manifest

## Goal

Make the RVV+scalar i32 binary dispatch route, ABI, artifact-kind, component,
self-check, and diagnostic surface descriptor-driven for the existing finite
families `i32-vadd`, `i32-vsub`, and `i32-vmul`.

The module should remove active compiler/tool duplication that made the vmul
dispatch module require coordinated hand edits in target/export code,
`tcrv-translate`, route names, ABI metadata, self-check markers, and tests.

## Why Now

HEAD `fa8a1ab` completed the prior `i32-vmul` dispatch bundle and `ssh rvv`
evidence module. Add/sub/mul dispatch now works end to end, but route and ABI
metadata are still split across the registry, `RVVScalarDispatch.cpp`,
`tcrv-translate`, and Python runner expectations. The next maintainability
blocker is making the active C++ route surface consume one bounded manifest as
the route source of truth before adding any new arithmetic family.

## Requirements

- Create a bounded C++ dispatch route manifest/API for the existing finite
  `i32-vadd`, `i32-vsub`, and `i32-vmul` families.
- The manifest must be target-owned and derive family facts from
  `I32BinaryFamilyRegistry` plus dispatch-specific stable route fields.
- The manifest must enumerate source/header/object/self-check routes,
  artifact kinds, runtime ABI kind/name, component group, external ABI name,
  function/header stems, self-check markers, and diagnostic family labels.
- `lib/Target/Builtin/RVVScalarDispatch.cpp` must resolve families and validate
  route/ABI/artifact metadata through the manifest or manifest-backed helper
  APIs.
- `tools/tcrv-translate/tcrv-translate.cpp` must register RVV+scalar dispatch
  translate routes from the manifest-backed API rather than maintaining an
  independent add/sub/mul route list.
- Direct source/header/object/self-check translate routes must fail closed when
  their route family does not match the selected RVV+scalar dispatch pair.
- Keep the Python evidence runner as runner/orchestration tooling only. If no
  emitted compiler manifest command is added this round, do not expand Python
  compiler semantics; keep any script changes minimal and truthful.
- Preserve existing route ids, ABI names, generated artifact names,
  self-check markers, add/sub/mul arithmetic semantics, and prior evidence
  meanings unless a compatibility fix is documented here and tested.

## Acceptance Criteria

- [x] A C++ manifest/API covers exactly `i32-vadd`, `i32-vsub`, and
      `i32-vmul`.
- [x] Manifest entries expose distinct source/header/object/self-check route
      ids and descriptor-equivalent runtime ABI/component metadata for every
      family.
- [x] Built-in target/export route registration consumes the manifest-backed
      family surface and still registers add/sub/mul dispatch source, header,
      and object routes.
- [x] `tcrv-translate` registers dispatch source/header/object/self-check
      routes by iterating the manifest-backed API.
- [x] Direct translate routes reject stale-family mismatches, for example a
      vmul direct route must not export vadd or vsub selected dispatch
      artifacts.
- [x] Existing generic bundle/source/header/object behavior remains stable for
      add/sub/mul.
- [x] Focused C++ or lit coverage proves manifest coverage and distinct route,
      ABI, component, marker, operator, and intrinsic metadata.
- [x] Focused translate-route tests prove add/sub/mul dispatch routes are
      exposed and route/family mismatches fail closed.
- [x] Focused RVVScalarDispatch tests prove stale-family mismatch rejection for
      all three families.
- [x] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` passes.
- [x] Focused script dry-run tests for add/sub/mul dispatch bundles pass.
- [x] Focused bundle/export lit tests for add/sub/mul routes touched by this
      migration pass.
- [x] `git diff --check` passes.
- [x] CMake configure with repository LLVM/MLIR paths passes.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes before finish/archive.
- [x] Trellis task validation, finish/archive, journal update, and one coherent
      commit are completed if the module is finished.

## Scope Boundary

This task is bounded to the already implemented finite i32 binary dispatch
families. It may refactor route metadata and registration APIs, but it must not
add generic arithmetic, generic RVV lowering, new `tcrv.exec` compute ops, or
extension-specific semantic branches in generic core passes.

Parameter layering remains explicit:

- hardware facts and selected march/mabi stay capability/toolchain facts;
- compile-time family and route metadata stay descriptor/manifest facts;
- runtime `n` and `rvv_available` stay runtime ABI/control values;
- descriptor-local element-count and microkernel shape facts stay
  exporter/plugin-local metadata.

## Non-goals

- No new arithmetic family beyond `i32-vadd`, `i32-vsub`, and `i32-vmul`.
- No new `ssh rvv` runtime correctness claim unless generated C/ABI behavior
  changes and a fresh focused remote run becomes necessary.
- No performance benchmarking or ratio claims.
- No i64/e64, masks, widening/narrowing, new RVV policy families,
  dynamic-shape frontend expansion, StableHLO/TOSA lowering, or generic RVV
  lowering.
- No new `tcrv.exec` compute ops.
- No extension-specific semantic branches in generic core passes.
- No Python implementation of compiler registry, plugin proposal, lowering,
  emission, route selection, runtime ABI decisions, or source generation.
- No docs-only, smoke-only, wrapper-only, report-only, or metadata-only closeout.

## Technical Notes

- Repo root: `/home/kingdom/phdworks/TianchenRV`
- Starting HEAD: `fa8a1ab feat(rvv): add vmul dispatch bundle evidence`
- Starting worktree: clean
- No active Trellis task existed before this task was created.
- Prior PRDs read:
  - `.trellis/tasks/archive/2026-05/05-09-i32-vmul-rvv-scalar-dispatch-ssh-rvv-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-descriptor-backed-i32-vmul-standalone-artifact-path/prd.md`
- Current code finding:
  - `I32BinaryFamilyRegistry.h` already contains finite add/sub/mul registry
    facts and dispatch descriptors.
  - `RVVScalarDispatch.cpp` already loops registry descriptors for composite
    exporter registration, but still exposes add-named direct source/header/
    object functions and family-specific matcher wrappers.
  - `tcrv-translate.cpp` still hand-registers dispatch translate routes; direct
    source/header/object translate routes are vadd-only while self-check routes
    are manually listed for add/sub/mul.
  - `scripts/rvv_scalar_dispatch_e2e.py` remains Python runner/evidence
    tooling with duplicated family expectations; this round should not expand
    that into compiler semantics.

## Continuation Rule If Unfinished

Keep this task open. Record exactly which consumers are manifest-backed and
which still duplicate route/ABI metadata: RVVScalarDispatch target code,
`tcrv-translate` registration, e2e runner expectations, C++ tests, lit tests,
or task archive. Do not archive or claim the dispatch route surface is
descriptor-driven until at least target/export code and translate route
registration consume the manifest-backed API.

## Completion Notes

This task was completed as a continuation from a dirty failed round. The final
implementation keeps source/header/object/self-check dispatch route metadata in
the target-owned C++ manifest, registers `tcrv-translate` direct routes by
iterating that manifest, registers add/sub/mul target artifact composite
source/header/object routes through manifest entries, and rejects stale-family
direct route use before artifact output. No generated add/sub/mul route id,
runtime ABI name, artifact kind, self-check marker, arithmetic operator,
intrinsic, or prior `ssh rvv` evidence meaning was intentionally changed.
