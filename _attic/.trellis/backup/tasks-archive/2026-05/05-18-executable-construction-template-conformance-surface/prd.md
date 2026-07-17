# Executable construction-template conformance surface

## Goal

Define one bounded reusable C++ construction-template conformance surface for
executable extension-family routes. TensorExtLite must consume this surface on
its default selected path so its role sequence, selected lowering boundary,
EmitC route mapping, and artifact/export readiness checks are validated through
generic construction-template contracts rather than TensorExtLite-only
orchestration.

## Current HEAD Evidence

- Repo root is `/home/kingdom/phdworks/TianchenRV`; the worktree was clean
  before this task.
- Current HEAD is `289f426 plugin: materialize tensorextlite selected artifact
  path`.
- `.trellis/.current-task` did not exist at session start, so this task was
  created from the Direction Brief before source edits.
- TensorExtLite already materializes the selected
  `configure -> load_frag -> tile_mma -> store_frag` role sequence, one
  selected `tcrv_tensorext_lite.lowering_boundary`, a plugin-owned
  `TCRVEmitCLowerableRoute`, emission-plan metadata, and object/header/bundle
  target artifact evidence.
- The current bottleneck is that the reusable construction model verifies
  manifests and typed role objects, while selected role-sequence discovery,
  selected-boundary coherence, route-readiness metadata, and target artifact
  readiness are still checked mostly through TensorExtLite-local code.

## Requirements

- Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck. Python may only be used for task tooling or validation
  wrappers.
- Add or repair a common C++/MLIR construction-template conformance surface
  around existing construction protocol data. The surface may express generic
  concepts such as selected role sequence, selected boundary attributes,
  route mapping, artifact metadata, and fail-closed validation.
- TensorExtLite must consume the reusable surface on its production/default
  selected path, not only in a test fixture.
- Common/core code may learn generic construction-template concepts only. It
  must not learn TensorExtLite fragment semantics, operation-specific compute
  meaning, intrinsic names, fragment layout, dtype, or tile semantics.
- Preserve TensorExtLite plugin ownership for family-local role definitions,
  role-step callee mapping, EmitC route construction, runtime ABI identity, and
  target artifact route callbacks.
- Preserve fail-closed behavior for duplicate, partial, reordered, stale, or
  missing selected role sequences; stale construction metadata; mismatched
  typed-role/interface realization; missing or duplicate selected lowering
  boundaries; stale route mapping; fallback-only/no-route selections; stale
  source-front-door residue; and descriptor/direct-C/source-export residue.
- Do not add a new plugin, new TensorExtLite feature, new role family, RVV/IME/
  Offload/scalar expansion, descriptor compatibility, direct C semantic
  exporter, source-export route, Python compiler-core behavior, or
  extension-specific semantic branch in common/core passes.

## Acceptance Criteria

- TensorExtLite still materializes the same selected
  `configure -> load_frag -> tile_mma -> store_frag` role sequence and one
  selected `tcrv_tensorext_lite.lowering_boundary`.
- TensorExtLite role-sequence validation and route-readiness validation flow
  through the reusable construction-template conformance surface on the
  production plugin path.
- TensorExtLite selected-boundary validation and target artifact readiness use
  the common conformance surface for generic selected-boundary / artifact
  checks, while keeping TensorExtLite-specific ABI and handoff data plugin-local.
- Focused C++ tests prove the common surface rejects duplicate/partial/reordered
  role sequences, stale route mapping or metadata, missing selected boundary,
  and mismatched construction evidence.
- Existing TensorExtLite object/header/bundle and runtime ABI harness evidence
  remains valid.
- Focused scans over touched common/TensorExtLite plugin and target files show
  no descriptor-driven computation, direct-C/source-export route, Python
  compiler-core path, or common/core family-specific semantic branch was
  introduced.

## Out Of Scope

- New TensorExtLite runtime correctness or performance claims.
- New high-level frontend lowering.
- New RVV, IME, Offload, scalar, vendor, or hardware target behavior.
- Descriptor-driven computation, compatibility layers, direct source printers,
  or construction-metadata-generated C/C++ output.
- Broad smoke/report matrix or helper-only progress that is not consumed by
  the production TensorExtLite path.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-18-tensorextlite-construction-template-artifact-closure/prd.md`.
- Main code surfaces:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h`,
  `lib/Plugin/TensorExtLite/Construction/TensorExtLiteConstructionProtocol.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`, and
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`.
- Main checks:
  focused plugin/common C++ tests, focused TensorExtLite target artifact lit
  tests, `check-tianchenrv` if feasible, and targeted residue scans.
