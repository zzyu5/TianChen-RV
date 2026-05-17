# Journal - codex (Part 9)

> Continuation from `journal-8.md` (archived at ~2000 lines)
> Started: 2026-05-17

---



## Session 101: TensorExtLite materialized EmitC artifact bundle bridge

**Date**: 2026-05-17
**Task**: TensorExtLite materialized EmitC artifact bundle bridge
**Branch**: `main`

### Summary

Added a TensorExtLite object/header bundle component contract over the existing materialized EmitC object route and declaration-only header composite, with zero-argument ABI bundle metadata and focused C++/lit coverage.

### Main Changes

- Added a TensorExtLite materialized EmitC bundle component group shared by the object exporter and object-backed declaration-only header composite.
- Added TensorExtLite header composite bundle metadata preserving component group, external ABI name, runtime ABI kind/name, and object handoff identity.
- Relaxed the generic grouped bundle component contract to accept a shared zero-argument ABI signature while still rejecting one-sided empty or otherwise mismatched signatures.
- Added `runtime_abi_parameter_count` to bundle index records so zero-argument bundles are explicit rather than inferred from missing parameter lines.
- Added focused TensorExtLite bundle lit coverage for source-front-door -> materialized EmitC -> object/header bundle, including object readobj/symbol checks and declaration-only header checks.
- Updated lowering-runtime and testing specs to document shared empty ABI signatures for zero-argument object/header bundles.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] focused lit filter `TensorExtLite|tensorext-lite|TargetArtifactBundleExport|vector-source-target-artifact-object`: 15/15 passed
- [OK] focused lit filter `tensorext-lite-source-front-door-target-artifact-bundle|tensorext-lite-target-artifact-header`: 2/2 passed
- [OK] `cmake --build build --target check-tianchenrv -j2`: 112/112 lit tests passed
- [OK] `git diff --check`
- [OK] manual bundle evidence: shared component group `tensorext-lite-fragment-mma-materialized-emitc-bundle.v1`, `artifact_count: 2`, object/header records with `runtime_abi_parameter_count: 0`, and RISC-V relocatable object symbol `tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice`.

### Status

[OK] Completed and archived.


## Session 102: Common executable plugin construction template

**Date**: 2026-05-17
**Task**: Common executable plugin construction template
**Branch**: `main`

### Summary

Added a code-consumed common materialized EmitC object/header bundle construction surface, migrated RVV and TensorExtLite target-support registration to it, documented the wired interface, and validated focused C++/lit plus full check-tianchenrv.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 121: Executable construction gate closure for remaining builtin plugins

**Date**: 2026-05-18
**Task**: Executable construction gate closure for remaining builtin plugins
**Branch**: `main`

### Summary

Closed the remaining builtin executable construction registration bypass by
wiring Toy and Template into the existing registry-time conformance gate.

### Main Changes

- Created Trellis task
  `05-18-executable-construction-gate-remaining-builtins` from the Direction
  Brief and repaired the Brief's stale construction-protocol spec path to the
  live `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  source of truth.
- Added Toy and Template
  `verifyExecutableConstructionConformance()` overrides so
  `ExtensionPluginRegistry::registerPlugin()` validates both plugins before
  proposal, lowering, EmitC route construction, or artifact export can consume
  them.
- Reworked Toy and Template construction readiness checks to call the common
  `ConstructionConformanceGateSpec` path for manifest, typed-role realization,
  validation spec, and construction artifact metadata before their existing
  route/bundle checks.
- Added Toy/Template construction artifact metadata helpers and verifiers.
- Extended `ConstructionProtocolCommonTest` with valid registration coverage
  for RVV, TensorExtLite, Toy, and Template, plus fail-closed registry coverage
  for stale Toy manifest metadata, stale Toy artifact metadata, and stale
  Template artifact metadata.
- Added no new route, artifact kind, extension semantics, descriptor/direct-C/
  source-export authority, compatibility layer, Python compiler-core logic, or
  family-specific common branch.

### Testing

- [OK] Trellis context validation for the task.
- [OK] `cmake --build build --target tianchenrv-construction-protocol-common-test -j2`
- [OK] `./build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `cmake --build build --target tianchenrv-toy-extension-plugin-test tianchenrv-template-extension-plugin-test tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] Toy, Template, RVV, and TensorExtLite plugin smoke tests.
- [OK] Construction-capable builtin hook scan showed RVV, TensorExtLite, Toy,
  and Template gated at registration; Scalar/Offload no-construction scan
  returned no matches.
- [OK] Common construction family-name scan returned no matches.
- [OK] Touched-surface descriptor/direct-C/source-export residue scan returned
  no matches.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 121: Executable construction protocol registry gate

**Date**: 2026-05-18
**Task**: Executable construction protocol registry gate
**Branch**: `main`

### Summary

Made executable construction conformance a production plugin registry gate and
proved the same common gate on RVV and non-RVV TensorExtLite construction
surfaces.

### Main Changes

- Created Trellis task `05-18-executable-construction-protocol-registry-gate`
  from the Direction Brief and wrote the PRD before source edits.
- Added common `ConstructionConformanceGateSpec` and
  `verifyConstructionConformanceGate`, covering manifest, typed role
  realization, optional role steps, and construction artifact metadata through
  existing construction conformance helpers.
- Added `ExtensionPlugin::verifyExecutableConstructionConformance` and called
  it from `ExtensionPluginRegistry::registerPlugin`, so plugin registration
  and built-in extension-bundle setup fail closed before publishing invalid
  executable construction artifacts.
- Wired RVV and TensorExtLite plugin overrides into their existing
  construction protocol ready checks. RVV remains the first real executable
  artifact user; TensorExtLite is the reusable non-RVV proof.
- Added no descriptor route, direct-C/source-export path, compatibility
  wrapper, new RVV SEW/LMUL/dtype/op family, target artifact kind, or Python
  compiler-core path.

### Git Commits

Included in the final task commit for this round.

### Testing

- [OK] Trellis context validation for the task.
- [OK] Focused construction/RVV/TensorExtLite build:
  `cmake --build build --target tianchenrv-construction-protocol-common-test tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] Common construction family-name scan returned no matches.
- [OK] Descriptor/direct-C/source-export/Python compiler-core scan over
  changed include/lib/test plugin files returned no matches.
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 121: RVV executable construction conformance adoption

**Date**: 2026-05-18
**Task**: RVV executable construction conformance adoption
**Branch**: `main`

### Summary

Moved the bounded RVV i32m1 add/sub/mul executable construction path onto the
common construction conformance surface for selected role sequencing, selected
lowering-boundary checks, and construction artifact metadata validation without
changing RVV typed semantics or generated target payloads.

### Main Changes

- Created and archived Trellis task
  `05-18-rvv-executable-construction-conformance-adoption` from the Direction
  Brief.
- Extended common selected executable role-sequence inspection to accept an
  extension-owned ordered operation list plus construction-order evidence.
- Added RVV i32m1 executable role-step metadata and routed RVV selected EmitC
  route readiness through common selected role-sequence conformance.
- Routed RVV selected `tcrv_rvv.with_vl` boundary checks through common
  lowering-boundary conformance while preserving RVV-local config/VL and typed
  op legality.
- Routed generic RVV construction artifact metadata checks through common
  `verifyConstructionArtifactMetadata` before RVV-local target checks.
- Added fail-closed RVV coverage for out-of-order selected role ops and common
  construction protocol coverage for the RVV role-step surface.
- Added no new RVV family coverage, descriptor route, direct-C/source-export
  path, compatibility wrapper, Python compiler-core logic, or common RVV
  semantic branch.

### Git Commits

Included in the final task commit for this round.

### Testing

- [OK] Trellis context validation for the task.
- [OK] Focused construction/RVV/target artifact build.
- [OK] `tianchenrv-construction-protocol-common-test`
- [OK] `tianchenrv-rvv-extension-plugin-test`
- [OK] `tianchenrv-target-artifact-export-test`
- [OK] Focused RVV materialized EmitC/source-seed/target artifact lit passed
  14/14 from `build/test`.
- [OK] Focused common/RVV residue scans found no descriptor-driven route
  authority, direct-C semantic exporter, source-export route, Python
  compiler-core behavior, or extension-specific semantic branch in common
  construction code.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 121: Executable construction-template conformance surface

**Date**: 2026-05-18
**Task**: Executable construction-template conformance surface
**Branch**: `main`

### Summary

Added a reusable C++ construction-template conformance surface and rewired the
TensorExtLite selected path to consume it for executable role-sequence,
selected-boundary, and artifact metadata validation.

### Main Changes

- Created Trellis task `05-18-executable-construction-template-conformance-surface`
  from the Direction Brief and wrote the PRD before source edits.
- Added common `plugin::construction` conformance APIs for:
  executable role steps,
  selected executable role-sequence inspection/collection,
  selected lowering-boundary coherence,
  and ordered construction artifact metadata validation.
- Replaced TensorExtLite-local selected role-sequence scanning and role-step
  validation with the common conformance surface while keeping role semantics,
  callee mapping, ABI identity, and target export callbacks plugin-local.
- Replaced TensorExtLite plugin/target selected-boundary string and capability
  checks with the common selected-boundary conformance verifier.
- Promoted the reusable conformance API contract into
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`.
- Added duplicate selected-role negative coverage and a static proof that
  TensorExtLite executable role steps use the common conformance model.
- Added no descriptor adapter, direct-C/source-export route, Python compiler
  core, new TensorExtLite feature, new plugin family, or common/core
  family-semantic branch.

### Git Commits

Included in the final task commit for this round.

### Testing

- [OK] Trellis context validation for the task.
- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-construction-protocol-common-test`
- [OK] `./build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `cmake --build build --target check-tianchenrv` -> 122/122 passed.
- [OK] Focused changed-source scan found no descriptor/direct-C/source-export/
  Python compiler-core residue in touched common/TensorExtLite plugin/target
  files.
- [OK] Common construction source scan found no TensorExtLite/RVV/IME/Offload/
  Toy/Template family semantic branch in touched common conformance files.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 121: TensorExtLite executable construction-template artifact closure

**Date**: 2026-05-18
**Task**: TensorExtLite executable construction-template artifact closure
**Branch**: `main`

### Summary

Closed the remaining TensorExtLite default selected-path gap by making the
plugin-owned lowering-boundary materializer create or reuse the same ordered
construction-template role sequence and direct lowering-boundary surface that
the TensorExtLite target artifact exporter already requires.

### Main Changes

- Created and archived Trellis task
  `05-18-tensorextlite-construction-template-artifact-closure` from the
  Direction Brief.
- Replaced the TensorExtLite active-route no-boundary result with production
  selected-surface materialization in
  `TensorExtLiteExtensionPlugin::materializeSelectedLoweringBoundary`.
- The default TensorExtLite selected path now materializes
  `configure -> load_frag -> tile_mma -> store_frag` role ops inside the
  selected variant body when no role sequence exists, validates an existing
  complete role sequence when present, and fail-closes on partial or duplicate
  role materialization.
- The same hook now creates or reuses one direct
  `tcrv_tensorext_lite.lowering_boundary` and validates it before returning a
  materialized boundary result.
- Updated the TensorExtLite plugin C++ test so the generic proposal/selection
  path proves an active route, supported emission plan, route provenance, and
  materialized EmitC module without relying on hand-written role ops.
- Added partial role-sequence negative coverage proving stale partial typed
  role realization is not auto-completed into artifact authority.
- Added no descriptor adapter, compatibility wrapper, direct-C semantic
  exporter, source-export route, Python compiler-core path, RVV expansion, or
  common/core TensorExtLite branch.

### Git Commits

Included in the final task commit for this round.

### Testing

- [OK] Trellis context validation for the task.
- [OK] `cmake --build build --target tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] Focused TensorExtLite lit from `build/test` passed 14/14.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.
- [OK] TensorExtLite residue scan found only fail-closed/prohibitive test
  strings and source-only stale-residue rejection text.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 123: RVV construction-template metadata closure on real hardware path

**Date**: 2026-05-17
**Task**: RVV construction-template metadata closure on real hardware path
**Branch**: `main`

### Summary

Closed the existing RVV i32m1 add/sub/mul materialized EmitC
object/header/bundle route against the executable construction-template record
proven by TensorExtLite, while preserving RVV as the real `ssh rvv` hardware
evidence path.

### Main Changes

- Created Trellis task
  `05-17-rvv-construction-template-metadata-closure` from the Direction Brief
  and wrote the PRD before source edits.
- Added RVV construction-template artifact metadata APIs for selected EmitC
  route, arithmetic op, source-op/source-role/interface provenance,
  construction protocol, extension archetype, semantic role graph, bounded
  interface summaries, EmitC route mapping, evidence profile, runtime ABI
  contract, bundle component group, and object handoff.
- Added RVV target artifact mapping API for header route, header artifact kind,
  bundle component group, object handoff, and EmitC-to-C++ translate route.
- Rewired RVV emission plans, declaration-only header evidence, and
  object/header bundle metadata to consume and validate the same construction
  metadata record plus existing RVV config/runtime-VL metadata.
- Added fail-closed C++ coverage for missing construction metadata, stale
  route-only metadata, stale source-op provenance, fallback-only selection,
  mismatched runtime ABI parameters, descriptor/direct-C/source-export residue,
  and ambiguous selected candidates.
- Updated RVV source-front-door, materialized target artifact, and source
  bundle lit coverage for construction metadata in plans, headers, and bundle
  indexes with declaration-only header checks.

### Evidence

- Dry-run artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-construction-template-metadata-closure-dry`.
- `ssh rvv` artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-construction-template-metadata-closure-ssh`.
- Real `ssh rvv` add/sub/mul external C ABI runs passed `n=1,7,16,17,257` and
  printed `tcrv_rvv_generated_bundle_abi_add_ok`,
  `tcrv_rvv_generated_bundle_abi_sub_ok`,
  `tcrv_rvv_generated_bundle_abi_mul_ok`, and matching `PASS op=...` markers.

### Testing

- [OK] Trellis context validation for the task.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] Focused lit from `build/test` passed 6/6 for RVV source-front-door,
  materialized target artifact, and source bundle coverage.
- [OK] RVV generated bundle ABI dry-run for add/sub/mul counts
  `1,7,16,17,257`.
- [OK] RVV generated bundle ABI `ssh rvv` run for add/sub/mul counts
  `1,7,16,17,257`.
- [OK] Focused descriptor/direct-C/source-export residue scan left only
  rejection checks, negative fixtures, and `implicit-check-not` assertions.
- [OK] Core/common RVV branch scan left only prohibitive `Passes.td` wording.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] Completed and archived. Ready for final task commit.

### Next Steps

- None - task complete


## Session 121: RVV generated-bundle external C ABI correctness proof

**Date**: 2026-05-17
**Task**: RVV generated-bundle external C ABI correctness proof
**Branch**: `main`

### Summary

Closed a fresh current-HEAD proof for the existing public
`tcrv-translate --tcrv-source-artifact-bundle-front-door` RVV add/sub/mul
bundle ABI path after the binary self-check deletion round. The compiler route
was already present; this round strengthened the evidence tool's fail-closed
checks and re-ran real `ssh rvv` external C ABI evidence.

### Main Changes

- Created Trellis task
  `05-17-rvv-generated-bundle-external-c-abi-proof` from the Direction Brief.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` to reject forbidden
  descriptor/direct-C/source-export/self-check/raw-log/credential-like residue
  in bundle index, artifact metadata, and generated headers.
- Added self-test negative coverage for descriptor metadata residue,
  direct-C/source-export metadata residue, header self-check residue, header
  credential residue, and sanitizer redaction of private-key, bearer-token,
  environment-token, and password-like text.
- Strengthened remote runtime validation so each operation must report both
  `tcrv_rvv_generated_bundle_abi_<op>_ok` and `PASS op=<op>`.
- Added no compiler route, no compatibility wrapper, no descriptor adapter, no
  direct-C/source-export path, no scalar fallback compute, and no Python
  compiler-core behavior.

### Evidence

- Dry-run artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-generated-bundle-external-c-abi-dry`.
- `ssh rvv` artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-generated-bundle-external-c-abi-ssh`.
- Remote toolchain facts recorded in per-op evidence:
  `remote_arch=riscv64`, `clang_path=/usr/bin/clang`, and
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote add/sub/mul runs passed `n=1,7,16,17,257` and printed:
  `tcrv_rvv_generated_bundle_abi_add_ok`,
  `tcrv_rvv_generated_bundle_abi_sub_ok`,
  `tcrv_rvv_generated_bundle_abi_mul_ok`,
  `PASS op=add counts=1,7,16,17,257`,
  `PASS op=sub counts=1,7,16,17,257`, and
  `PASS op=mul counts=1,7,16,17,257`.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `cmake --build build --target tcrv-translate -j2`
- [OK] Dry-run generated bundle ABI command for add/sub/mul counts
  `1,7,16,17,257`.
- [OK] `ssh rvv` generated bundle ABI command for add/sub/mul counts
  `1,7,16,17,257`.
- [OK] Focused lit from `build/test` passed 3/3:
  `Scripts/rvv-generated-bundle-abi-e2e-self-test.test`,
  `Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`,
  `Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-fail-closed.mlir`.
- [OK] Focused residue scan left only intentional rejection logic, negative
  self-test fixtures, and existing fail-closed test coverage.
- [OK] `git diff --check`
- [OK] Trellis context validation for the task.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 122: TensorExtLite executable plugin construction template closure

**Date**: 2026-05-17
**Task**: TensorExtLite executable plugin construction template closure
**Branch**: `main`

### Summary

Closed the bounded TensorExtLite first-slice plugin construction workflow as a
copyable non-RVV executable template: source marker, explicit selected
TensorExtLite role ops, complete construction protocol metadata, plugin-owned
EmitC route, materialized EmitC, and object/header/bundle artifact evidence.

### Main Changes

- Created Trellis task
  `05-17-05-17-tensorextlite-executable-plugin-construction-template-closure`
  from the Direction Brief.
- Extended TensorExtLite fragment-MMA artifact metadata from 8 to 12 entries,
  adding extension archetype, common interface realization, EmitC route
  mapping, and evidence profile to the existing route/source/protocol/role
  record.
- Rewired TensorExtLite emission plans, declaration header evidence, and
  object/header bundle metadata to consume and validate the same complete
  construction protocol record.
- Updated TensorExtLite plugin C++ tests, target artifact export C++ fixtures,
  source-front-door lit, materialized EmitC lit, target bundle lit, and local
  runtime ABI evidence checks for the complete record.
- Added no RVV coverage, descriptor adapter, direct-C/source-export route,
  scalar fallback compute, compatibility wrapper, Python compiler-core path, or
  core/common TensorExtLite semantic branch.

### Evidence

- Local ABI/bundle evidence:
  `artifacts/tmp/tensorextlite_runtime_abi_e2e/codex-tensorextlite-template-closure`.
- Evidence consumed `--tcrv-source-artifact-bundle-front-door`, verified the
  generated object/header bundle metadata, compiled a native ABI harness
  against generated C++ and declaration header surfaces, and recorded native
  call trace `configure,load_frag,tile_mma,store_frag`.
- Self-repair: first full `check-tianchenrv` failed because
  `TargetArtifactExportTest.cpp` still had a handwritten 8-entry TensorExtLite
  emission-plan fixture. Updated that fixture to the 12-entry protocol record,
  then reran focused target-artifact and full checks successfully.

### Testing

- [OK] Trellis context validation for the task.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] Focused TensorExtLite lit from `build/test` passed 12/12.
- [OK] `python3 -m py_compile scripts/tensorextlite_runtime_abi_e2e.py`
- [OK] `python3 scripts/tensorextlite_runtime_abi_e2e.py --artifact-root artifacts/tmp/tensorextlite_runtime_abi_e2e --run-id codex-tensorextlite-template-closure --input test/Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir --materialized-input test/Target/TensorExtLite/tensorext-lite-target-artifact-header.mlir --tcrv-translate build/bin/tcrv-translate --clangxx clang++ --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test -j2 && ./build/bin/tianchenrv-target-artifact-export-test`
- [OK] Focused residue scan left only rejection checks, negative fixtures, and
  `implicit-check-not` assertions.
- [OK] Core/common branch scan left only prohibitive `Passes.td` text.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] Completed and archived. Ready for final task commit.

### Next Steps

- None - task complete


## Session 118: Built-in target registration wrapper erasure

**Date**: 2026-05-17
**Task**: Built-in target registration wrapper erasure
**Branch**: `main`

### Summary

Deleted the target-side built-in aggregate registration wrappers that hid the
extension bundle front door, then rewired active tool and target-test callers
to pass explicit `ExtensionBundleRegistry` plus `ExtensionPluginRegistry`
pairs into the surviving bundle-aware target registration surfaces.

### Main Changes

- Created Trellis task
  `05-17-05-17-builtin-target-registration-wrapper-erasure` from the Direction
  Brief.
- Removed
  `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &)`,
  `registerBuiltinTargetArtifactExporters(TargetArtifactExporterRegistry &, const ExtensionPluginRegistry &)`,
  and
  `registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &)`.
- Kept the bundle-aware target artifact exporter and target translate route
  registration surfaces.
- Rewired `tcrv-translate` target route registration and
  `TargetArtifactExportTest.cpp` callers to construct/populate explicit
  bundle/plugin registries.
- Updated plugin-protocol and lowering-runtime specs so durable target
  registration examples no longer present hidden wrappers as the correct API.
- No compatibility alias, replacement wrapper, new target route, new artifact
  behavior, descriptor path, direct C exporter, source-export path, Python
  compiler-core logic, or extension-specific common/core/tool branch was added.

### Testing

- [OK] Trellis context validation for the task.
- [OK] Focused build:
  `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test -j2`.
- [OK] Rewritten C++ test:
  `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] Translate command surface probe for target artifact and materialized
  EmitC translate routes.
- [OK] Deleted target wrapper signature scan over `include`, `lib`, `tools`,
  `test`, and `.trellis/spec`.
- [OK] Wrapper-protecting call/test scan over active code and specs.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target tcrv-opt check-tianchenrv -j2` -> 122/122
  passed.

### Status

[OK] Completed. Ready for archive and final task commit.


## Session 118: Legacy built-in plugin registration wrapper erasure

**Date**: 2026-05-17
**Task**: Legacy built-in plugin registration wrapper erasure
**Branch**: `main`

### Summary

Deleted the legacy plugin-only built-in registration wrapper and rewired active
tests to the canonical extension bundle frontdoor.

### Main Changes

- Created Trellis task
  `05-17-legacy-builtin-plugin-registration-wrapper-erasure` from the Direction
  Brief and wrote a deletion/refactor-only PRD.
- Removed `registerBuiltinExtensionPlugins` from
  `include/TianChenRV/Plugin/BuiltinExtensionPlugins.h` and
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp`.
- Rewired scalar, Toy, offload, target artifact export, and variant selection
  test callers to create an `ExtensionBundleRegistry` and call
  `registerBuiltinExtensionBundlePlugins` directly.
- Removed the explicit `compatibilityPlugins` coverage and
  `legacy built-in plugin registration delegates through bundle frontdoor`
  assertion from `TargetArtifactExportTest.cpp`.
- Preserved `tcrv-opt` and `tcrv-translate` on the canonical bundle frontdoor.
- Updated plugin-protocol specs so future built-in aggregate registration keeps
  bundle identity visible and does not reintroduce a plugin-only wrapper.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-legacy-builtin-plugin-registration-wrapper-erasure`
- [OK] Active residue scan:
  `rg -n "registerBuiltinExtensionPlugins|compatibilityPlugins|legacy built-in plugin registration" include lib tools test`
  -> no matches.
- [OK] Active code/spec residue scan:
  `rg -n "registerBuiltinExtensionPlugins|compatibilityPlugins|legacy built-in plugin registration" include lib tools test .trellis/spec`
  -> no matches.
- [OK] Focused build for touched tests and tools:
  `cmake --build build --target tianchenrv-scalar-extension-plugin-test tianchenrv-toy-extension-plugin-test tianchenrv-offload-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-variant-selection-test tcrv-opt tcrv-translate -j2`.
- [OK] Focused C++ tests:
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-toy-extension-plugin-test`,
  `tianchenrv-offload-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, and
  `tianchenrv-variant-selection-test`.
- [OK] Tool registration probes with `build/bin/tcrv-opt --help-hidden` and
  `build/bin/tcrv-translate --help`.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 107: Template materialized EmitC construction-template route

**Date**: 2026-05-17
**Task**: Template materialized EmitC construction-template route
**Branch**: `main`

### Summary

Made the Template extension-family construction template executable enough to
prove selected Template role IR can flow through a plugin-owned materialized
EmitC route, MLIR EmitC C/C++ emission, generated C++ syntax compile, and a
declaration-only header artifact route without descriptor, direct-C, or
source-export authority.

### Main Changes

- Replaced the Template manifest's metadata-only no-active route with
  `template-extension-compute-skeleton-emitc-route`.
- Added `TemplateEmitCRouteProvider` to consume selected
  `tcrv_template.compute_skeleton` through `TCRVEmitCLowerableOpInterface`.
- Rewired Template emission readiness, emission planning, selected boundary
  validation, and target-support registration around the plugin-owned route.
- Added Template target support for `tcrv-template-emitc-to-cpp` and a
  runtime-callable header artifact route.
- Added C++ and lit coverage for positive materialization/C++/header evidence
  and fail-closed stale/missing/prohibited metadata cases.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-template-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-template-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused lit from `build/test` with `--filter='Template|template'`: 15/15 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 122/122 passed.
- [OK] `git diff --cached --check`
- [OK] focused production-surface residue scan found no Template no-active route, metadata-diagnostic, descriptor, source-export, direct-C, or `tcrv_template.lowering_boundary` residue.

### Status

[OK] Completed. Ready for archive and commit.


## Session 107: TensorExtLite materialized bundle runtime ABI proof

**Date**: 2026-05-17
**Task**: TensorExtLite materialized bundle runtime ABI proof
**Branch**: `main`

### Summary

Extended the TensorExtLite first-slice artifact proof from generation-only to
local runtime ABI consumption evidence. The production target bundle still
contains the existing RISC-V relocatable object/header pair; the local native
execution proof compiles the same generated materialized EmitC C++ source into
a host proof object, links a harness against the generated declaration header,
and runs the generated ABI function.

### Main Changes

- Created `.trellis/tasks/05-17-tensorextlite-runtime-abi-proof/` with PRD,
  implement context, and check context for this bounded round.
- Added `scripts/tensorextlite_runtime_abi_e2e.py` as evidence tooling only.
  It invokes `tcrv-opt`, `tcrv-translate`, `clang++`, and `llvm-readobj`;
  it does not implement compiler IR, lowering, emission, descriptors, or
  runtime glue.
- Added `test/Target/TensorExtLite/tensorext-lite-runtime-abi-harness.test`
  to run the ABI proof under lit when local RISC-V object clang and native
  clang++ are available.
- Added `tianchenrv-local-native-clangxx` detection and `clang++` substitution
  to `test/lit.cfg.py`.
- Evidence runner outputs include post-planning MLIR, generated C++ source,
  generated header, target object, object/header bundle, bundle index, harness
  source, native proof object, command records, run output, and negative
  fail-closed diagnostics.

### Testing

- [OK] `python3 -m py_compile scripts/tensorextlite_runtime_abi_e2e.py`
- [OK] `python3 scripts/tensorextlite_runtime_abi_e2e.py --artifact-root artifacts/tmp/tensorextlite_runtime_abi_e2e --run-id manual-smoke --tcrv-opt ./build/bin/tcrv-opt --tcrv-translate ./build/bin/tcrv-translate --clangxx /usr/lib/llvm-20/bin/clang++ --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] Evidence path:
  `artifacts/tmp/tensorextlite_runtime_abi_e2e/manual-smoke`
- [OK] Harness PASS tied to selected variant, origin plugin, construction
  protocol, EmitC route, runtime ABI name, zero ABI params, bundle component
  group, and native call trace `configure,load_frag,tile_mma,store_frag`.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused lit from `build/test` with
  `--filter='tensorext-lite-runtime-abi-harness'`: 1/1 passed.
- [OK] focused lit from `build/test` with
  `--filter='TensorExtLite|tensorext-lite'`: 14/14 passed.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 116/116 lit
  tests passed.
- [OK] targeted production scans found no TensorExtLite descriptor route
  authority, direct C semantic exporter, source-export route, Python
  compiler-core path, or common/core TensorExtLite semantic branch.

### Status

[OK] Completed. Ready for archive and commit.


## Session 106: TensorExtLite source-to-artifact production path closure

**Date**: 2026-05-17
**Task**: TensorExtLite source-to-artifact production path closure
**Branch**: `main`

### Summary

Created the Trellis task from the direction brief, wrote the PRD, inspected the
current TensorExtLite plugin/target/source-front-door surfaces, and verified
that current HEAD already closes the TensorExtLite source-only MLIR to
object/header/bundle production path through the common source-artifact and
materialized EmitC interfaces.

### Main Findings

- The current route is already production-wired:
  TensorExtLite source marker -> TensorExtLite-owned selected variant and
  ordered role ops -> TensorExtLite EmitC-lowerable route provenance -> common
  source-artifact front-door pipeline -> supported emission plan -> target
  object/header/bundle exporters.
- No source-code blocker was found, so no compiler, target, script, or test
  source files were changed.
- The round deliberately avoided adding another evidence harness because the
  requested non-RVV proof is already represented by the existing production
  path and focused lit/C++ coverage.
- No `.trellis/spec/` update was needed; the existing plugin-protocol and
  lowering-runtime specs already define this TensorExtLite route and its
  fail-closed boundaries.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-construction-protocol-common-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-construction-protocol-common-test`
- [OK] focused lit from `build/test`: TensorExtLite source-front-door,
  EmitC materialization, target object/header/bundle, unsupported and
  non-materialized negatives, and target artifact registry coverage; 13/13
  passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 115/115 lit tests
  passed.
- [OK] `git diff --check`

### Status

[OK] Completed. Ready for archive and commit.


## Session 103: RVV generated bundle ABI execution proof on ssh rvv

**Date**: 2026-05-17
**Task**: RVV generated bundle ABI execution proof on ssh rvv
**Branch**: `main`

### Summary

Added a bounded generated-bundle ABI evidence runner for the current RVV
i32m1 vector-source add path and proved that the generated declaration-only
header plus generated relocatable object can be compiled, linked, and run by an
external C harness on `ssh rvv`.

### Main Changes

- Created Trellis task `05-17-rvv-generated-bundle-abi-execution-proof` from the Hermes brief and wrote a PRD around the current RVV generated object/header bundle ABI proof.
- Added `scripts/rvv_generated_bundle_abi_e2e.py` to invoke the existing `tcrv-opt` source artifact front door and `tcrv-translate` target artifact bundle exporter, verify generated bundle metadata, generate an external C ABI consumer, and run it on `ssh rvv`.
- Added local self-test coverage for missing header/object-style failures, stale ABI order, missing multi-VL metadata, intrinsic body residue in the generated header, and mismatched selected variant.
- Added `test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test`.
- Updated `.trellis/spec/testing/mlir-testing-contract.md` to state the durable boundary for live RVV generated-bundle ABI correctness evidence.

### Runtime Evidence

- Evidence directory: `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-generated-bundle-abi-ssh`.
- Generated bundle command:
  `build/bin/tcrv-opt test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir --tcrv-source-artifact-front-door-pipeline | build/bin/tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-generated-bundle-abi-ssh/generated_bundle`.
- Generated artifacts:
  `tianchenrv-target-artifact-bundle.index`,
  `artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o`,
  `artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h`,
  and `rvv_generated_bundle_abi_harness.c`.
- `ssh rvv` compile/link/run succeeded for runtime counts `1,7,16,17,257` and printed `tcrv_rvv_generated_bundle_abi_ok counts=1,7,16,17,257`.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] local dry-run bundle verifier for counts `1,7,16,17,257`
- [OK] focused lit: `Scripts/rvv-generated-bundle-abi-e2e-self-test.test Target/RVV/vector-source-target-artifact-object.mlir Target/RVV/vector-source-target-artifact-header.mlir`, 3/3 passed
- [OK] `cmake --build build --target check-tianchenrv -j2`: 113/113 lit tests passed
- [OK] `git diff --check`
- [OK] targeted scans found descriptor/direct-C/source-export strings only in fail-closed validation, forbidden-token lists, explanatory non-authority text, or `CHECK-NOT` assertions.

### Status

[OK] Completed. Ready for archive and commit.


## Session 104: RVV vector-source arithmetic-family selected-boundary materializer

**Date**: 2026-05-17
**Task**: RVV vector-source arithmetic-family selected-boundary materializer
**Branch**: `main`

### Summary

Generalized the production RVV vector-source front-door materializer from the
add-only source matcher into a bounded i32m1 add/sub/mul arithmetic-family
materializer. The selected source arithmetic op now carries into explicit
`tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, or `tcrv_rvv.i32_mul` ops and then
through the existing materialized EmitC object/header/bundle route.

### Main Changes

- Created Trellis task `05-17-rvv-vector-source-arithmetic-family-materializer`
  from the Hermes brief and wrote the PRD around the production RVV
  front-door boundary.
- Replaced `BoundedI32AddSourcePattern` with an op-kind-aware
  `BoundedI32ArithmeticSourcePattern`.
- Recognized exactly `arith.addi`, `arith.subi`, and `arith.muli` in the
  bounded vector/scf source body and mapped them to
  `tcrv_rvv.i32_add/sub/mul`.
- Preserved fail-closed behavior for stale `tcrv_rvv.lowering_seed`,
  pre-existing `tcrv.exec`/`tcrv_rvv` residue, wrong ABI shape, wrong vector
  type, unsupported arithmetic, and generated-header descriptor/direct-C/
  source-export residue.
- Added sub and mul source-front-door lit fixtures and expanded RVV target
  object/header/bundle export coverage for add/sub/mul selected variants.
- Did not touch `scripts/rvv_generated_bundle_abi_e2e.py`; no new sub/mul
  runtime correctness claim was made.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused lit from `build/test`: `Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`, `Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir`, `Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir`, `Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir`, `Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-negative.mlir`, `Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-disabled.mlir`, `Target/RVV/vector-source-target-artifact-object.mlir`, `Target/RVV/vector-source-target-artifact-header.mlir`: 8/8 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 115/115 lit tests passed.
- [OK] `git diff --check`
- [OK] targeted scans found descriptor/direct-C/source-export strings only in fail-closed diagnostics, forbidden-token validation, or `CHECK-NOT` assertions.

### Runtime Evidence

No fresh `ssh rvv` run was performed because the ABI runner and bundle mechanics
were not changed. The previous `b99f31a` add-path proof remains the runtime ABI
consumption evidence for the unchanged generated object/header bundle
mechanics; this round's new sub/mul behavior is covered at compiler and target
artifact export level.

### Status

[OK] Completed. Ready for archive and commit.


## Session 105: RVV vector-source arithmetic family ssh-rvv runtime closure

**Date**: 2026-05-17
**Task**: RVV vector-source arithmetic family ssh-rvv runtime closure
**Branch**: `main`

### Summary

Generalized the generated-bundle ABI evidence runner to cover source-derived
RVV add/sub/mul, captured per-op generated artifacts, and proved all three
generated header/object bundles on real `ssh rvv`.

### Main Changes

- Created Trellis task
  `05-17-05-17-rvv-vector-source-arithmetic-family-runtime-closure` from the
  direction brief and wrote the PRD around hardware runtime closure.
- Reworked `scripts/rvv_generated_bundle_abi_e2e.py` from add-only constants
  into per-op expectations for add/sub/mul source fixtures, selected variants,
  runtime ABI names, generated function symbols, EmitC route metadata, harness
  expected arithmetic, and PASS markers.
- Default runner behavior now covers add, sub, and mul in one run; `--op-kind`
  supports focused single-op or repeated-op subsets, and `--input` is limited
  to exactly one selected op kind.
- Each op records its own `source.mlir`, generated bundle index, generated
  object/header, generated external ABI harness, per-op `evidence.json`, remote
  compile log, and remote run log.
- The harness remains an external ABI consumer of generated header/object only;
  it does not contain RVV intrinsic bodies or become compiler lowering/runtime
  fallback.
- No `.trellis/spec/` update was needed; the existing testing/lowering specs
  already define this generated-bundle ABI evidence contract and Python
  evidence-tooling boundary.

### Git Commits

This commit.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] local dry-run for add/sub/mul:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --run-id codex-rvv-vector-source-family-dry --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused lit from `build/test`: script self-test plus RVV
  add/sub/mul source-front-door and target object/header coverage, 6/6 passed.
- [OK] real `ssh rvv` run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --run-id codex-rvv-vector-source-family-ssh --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257 --timeout 180`
- [OK] remote PASS output:
  `PASS op=add counts=1,7,16,17,257`,
  `PASS op=sub counts=1,7,16,17,257`,
  `PASS op=mul counts=1,7,16,17,257`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 115/115 lit tests passed.
- [OK] `git diff --check`
- [OK] targeted scans found descriptor/direct-C/source-export strings only in
  fail-closed validation, forbidden-token lists, non-authority docstrings,
  existing target preflight rejection, or `CHECK-NOT` assertions.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 106: TensorExtLite construction protocol consumption

**Date**: 2026-05-17
**Task**: TensorExtLite construction protocol consumption
**Branch**: `main`

### Summary

Rewired TensorExtLite source-front-door, EmitC route, plugin emission plan, and target artifact bundle to consume the plugin-owned construction protocol for role, route, runtime ABI, and artifact metadata coherence.

### Main Changes

### Summary

Made TensorExtLite's existing fragment-MMA source-to-artifact path consume one plugin-owned construction protocol instead of preserving duplicated role, route, runtime ABI, and artifact metadata constants across production surfaces.

### Main Changes

- Extended `TensorExtLiteConstructionProtocol` with protocol-owned role-step, source-op/source-role, EmitC-lowerable interface, route, runtime ABI, artifact metadata, header, bundle, and evidence-profile accessors.
- Rewired the TensorExtLite source front door to materialize configure/load_frag/tile_mma/store_frag role ops from protocol role steps.
- Rewired the TensorExtLite EmitC route provider to use protocol role order, role lookup, route identity, and call-opaque callee mapping.
- Rewired the TensorExtLite plugin emission plan and target support bundle to use protocol runtime ABI, artifact metadata, header route, bundle group, object handoff, and EmitC-to-C++ route data.
- Rewrote focused TensorExtLite plugin and target artifact C++ tests so stale protocol or metadata mismatch fails before artifact export.
- Preserved the existing production/default source marker -> role ops -> materialized EmitC -> object/header/bundle behavior.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-construction-protocol-common-test`
- [OK] focused lit from `build/test` with `--filter='TensorExtLite|tensorext-lite'`: 13/13 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 115/115 lit tests passed.
- [OK] `git diff --check`
- [OK] targeted scans found no TensorExtLite descriptor route authority, direct C semantic exporter, source-export route, Python compiler-core path, or common/core TensorExtLite semantic branch.

### Status

[OK] Completed. Ready for archive and commit.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 107: Template materialized EmitC object bundle packaging bridge

**Date**: 2026-05-17
**Task**: Template materialized EmitC object bundle packaging bridge
**Branch**: `main`

### Summary

Extended the already selected Template materialized EmitC construction route
from generated C++ plus declaration header into a coherent construction-template
source/header/object/bundle packaging bridge.

### Main Changes

- Created Trellis task `05-17-template-emitc-object-bundle` from the direction
  brief and wrote the PRD around Template object/header/bundle packaging without
  runtime, performance, or hardware claims.
- Extended `TemplateEmitCConstructionRoute` with object route, header route,
  header artifact kind, bundle component group, and object handoff metadata.
- Rewired Template target support to register a selected relocatable object
  route and an object-backed declaration header composite through the common
  materialized EmitC object/header bundle helper.
- Added local `clang++` object packaging from MLIR EmitC-generated C++ source;
  the object is a relocatable packaging proof only, not a RISC-V runtime claim.
- Preserved `tcrv-template-emitc-to-cpp` as the selected generated C++ route,
  now tied to the same selected object candidate validation.
- Added focused C++ and lit coverage for Template object exporter shape,
  header composite behavior, bundle metadata coherence, route mismatch and
  fail-closed behavior, and generated object/header/bundle artifacts.
- Updated durable specs for the Template construction-template object/header/
  bundle opt-in contract.

### Testing

- [OK] `cmake --build build --target tianchenrv-template-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-template-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 124/124 lit tests passed.
- [OK] `git diff --check`
- [OK] bounded residue scans over Template plugin/target/tests and common/core
  target surfaces found no descriptor-driven compute authority, direct C
  semantic exporter, source-export route, or extension-specific core branch.
- [WARN] `clang-format` was unavailable in the local toolchain; C++ formatting
  was manually inspected and whitespace was checked with `git diff --check`.

### Status

[OK] Completed. Ready for archive and commit.


## Session 108: Toy materialized EmitC object-bundle bridge

**Date**: 2026-05-17
**Task**: Toy materialized EmitC object-bundle bridge
**Branch**: `main`

### Summary

Rebuilt Toy target artifact export from header-only to materialized EmitC object plus object-backed declaration header and bundle; focused Toy/target tests and check-tianchenrv passed.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 109: RVV runtime-callable C ABI link proof

**Date**: 2026-05-17
**Task**: RVV runtime-callable C ABI link proof
**Branch**: `main`

### Summary

Tightened RVV generated bundle ABI evidence checks for extern-C header guards and unmangled selected symbols; proved add-path external C harness on ssh rvv.

### Main Changes

- Created and archived Trellis task
  `05-17-rvv-runtime-callable-c-abi-link-proof` from the Hermes brief, with PRD
  scope locked to the selected RVV vector-source add runtime-callable C ABI
  proof.
- Kept the compiler route unchanged; the only code change is evidence tooling
  in `scripts/rvv_generated_bundle_abi_e2e.py`.
- Tightened generated header verification so the public declaration must sit
  inside the C++ `extern "C"` guard required by the C ABI contract.
- Tightened generated object verification so `llvm-readobj --symbols` must
  show the unmangled selected add symbol and must not show a C++-mangled
  selected symbol.
- Extended the script self-test fake bundle to include the required extern-C
  guard and added a negative self-test for a missing guard.
- Proved the add route on `ssh rvv` with an external C harness that includes
  the generated header, links the generated object, and prints
  `PASS op=add counts=1,7,16,17,257`.

### Git Commits

(Committed in this round; see final report for hash.)

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] add-only dry-run with `/usr/lib/llvm-20/bin/llvm-readobj`, recording
  `extern_c_guard: true` and `unmangled_selected_symbol: true`.
- [OK] focused lit from `build/test` with filter
  `Scripts/rvv-generated-bundle-abi-e2e-self-test|Target/RVV/vector-source-target-artifact-(object|header)`.
- [OK] `ssh rvv` add harness run:
  `tcrv_rvv_generated_bundle_abi_add_ok counts=1,7,16,17,257` and
  `PASS op=add counts=1,7,16,17,257`.
- [OK] `git diff --check`
- [INFO] `check-tianchenrv` was not run because this round did not change
  common EmitC or target artifact C++ code.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 110: Emission-runtime direct-route residue erasure

**Date**: 2026-05-17
**Task**: Emission-runtime direct-route residue erasure
**Branch**: `main`

### Summary

Removed stale emission-runtime contract wording that could read as direct RVV,
scalar fallback, RVV+scalar dispatch, descriptor-body, source/header/object, or
self-check route authority outside the materialized EmitC route.

### Main Changes

- Created and archived Trellis task
  `05-17-emission-runtime-direct-route-residue-erasure` from the Direction
  Brief, with deletion/refactor-only PRD scope.
- Rewrote `.trellis/spec/lowering-runtime/emission-runtime-contract.md` so
  target artifact generation requires selected extension-family IR plus a
  materialized EmitC/runtime route before target-owned object/header/bundle
  facts can be usable.
- Replaced positive direct-route wording for direct RVV microkernel object/
  header paths, scalar fallback helpers, RVV+scalar dispatch helpers, dispatch
  bundle records, descriptor body policy, and self-check object helpers with
  deleted/fail-closed or future-rebuild-only language.
- Preserved the current bounded RVV path only as explicit typed RVV IR ->
  selected materialized EmitC route -> MLIR EmitC C/C++ emission -> target
  object/header/bundle packaging.
- Added no compatibility layer, legacy route, descriptor adapter, helper
  wrapper, compiler code, tests, scripts, or evidence artifacts.

### Git Commits

(Committed in this round; see final report for hash.)

### Testing

- [OK] Trellis context validation for implement/check JSONL.
- [OK] Focused `rg` scans over the touched spec plus directly related
  `lib/Target`, `test/Target`, and `scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `git diff --check`
- [INFO] No build, lit, `ssh rvv`, or `check-tianchenrv` run because this round
  changed only Trellis task docs and the lowering-runtime contract spec.

### Status

[OK] **Completed**

### Next Steps

- Continue deletion only if future scans find more positive descriptor/direct-C
  route-authority wording; otherwise rebuild work should remain gated by the
  explicit extension-family IR -> materialized EmitC route.


## Session 111: Common source-artifact bundle front door

**Date**: 2026-05-17
**Task**: Common source-artifact bundle front door
**Branch**: `main`

### Summary

Added a one-command `tcrv-translate` production front door that runs
plugin-registered source front doors, shared source-artifact planning/coherence,
and target artifact bundle export without a manual `tcrv-opt | tcrv-translate`
chain.

### Main Changes

- Created and archived Trellis task
  `05-17-common-source-artifact-bundle-front-door` from the Direction Brief,
  with PRD scope locked to a common workflow bridge.
- Added `--tcrv-source-artifact-bundle-front-door` to `tcrv-translate`.
- The new route collects enabled `SourceFrontDoorPassRegistration` entries,
  runs `buildSourceArtifactFrontDoorPipeline`, then exports the selected bundle
  through the existing target artifact bundle exporter.
- Added `--tcrv-disable-builtin-plugins` to `tcrv-translate` as a bounded
  fail-closed test switch for empty/unregistered source front-door coverage.
- Added RVV and Toy lit coverage proving source input can reach object/header
  bundle output through the one-command front door.
- Added fail-closed lit coverage for disabled plugins, no matching source
  front door, and no supported selected artifact route.
- Updated the lowering-runtime EmitC route spec with the durable command,
  contract, error matrix, good/base/bad cases, tests, and wrong-vs-correct
  boundary for the common source-artifact bundle front door.

### Git Commits

(Committed in this round; see final report for hash.)

### Testing

- [OK] `ninja -C build tcrv-translate`
- [OK] `ninja -C build check-tianchenrv` -> 128/128 passed.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-17-common-source-artifact-bundle-front-door`
- [OK] `git diff --check`
- [OK] Focused production scan:
  `rg -n "descriptor|direct-C|direct C|direct-c|source-export|source_export|source authority|source_authority" tools/tcrv-translate/tcrv-translate.cpp`
  returned no matches.
- [OK] Focused production scan:
  `rg -n "rvv|toy|tensorext|TensorExt|RVV|Toy" tools/tcrv-translate/tcrv-translate.cpp`
  returned no matches.
- [INFO] `clang-format` was not available on this machine; the C++ edit was
  manually formatted to match surrounding style.
- [INFO] No fresh `ssh rvv` run was needed because this change did not alter
  the generated RVV object/header ABI route; it only adds a one-command
  front door over the already proved artifact path.

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete.


## Session 116: Core RVV source-to-exec named-absence fixture erasure

**Date**: 2026-05-17
**Task**: Core RVV source-to-exec named-absence fixture erasure
**Branch**: `main`

### Summary

Deleted active lit named-absence fixtures for historical core
RVV/source-seed public option spellings while preserving the current
plugin-owned RVV vector source front door and source artifact front-door
pipeline coverage. This round stayed deletion-only and did not add replacement
routes, compatibility aliases, descriptor adapters, source front doors, EmitC
routes, artifact routes, or production compiler behavior.

### Main Changes

- Created and archived Trellis task
  `05-17-core-rvv-source-to-exec-named-absence-fixture-erasure`.
- Removed the deleted selected-boundary seed negative RUN line from
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`.
- Removed the matching `OLD-SEED-REMOVED` FileCheck prefix and checks.
- Removed the deleted source-seed pipeline negative RUN line from
  `test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-disabled.mlir`.
- Removed the matching `OLD-PIPELINE-REMOVED` FileCheck prefix and checks.
- Reviewed spec update need; no `.trellis/spec/` edit was needed because the
  testing, variant-pipeline, and lowering-runtime specs already encode the
  no durable named absence fixture rule.

### Testing

- [OK] Focused lit through build site config for the two touched transform
  tests: 2/2 passed.
- [OK] Exact old-option and `OLD-*-REMOVED` prefix scan over touched transform
  test directories and relevant specs: no matches.
- [OK] Active option scan over `test/Transforms/RVV` and
  `test/Transforms/SourceFrontDoor` reports only current
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door` and
  `--tcrv-source-artifact-front-door-pipeline` invocations.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 123/123 passed.

### Status

[OK] Completed and archived. Ready for commit.

### Next Steps

- None - task complete.


## Session 115: Plan-and-export target artifact bundle route erasure

**Date**: 2026-05-17
**Task**: Plan-and-export target artifact bundle route erasure
**Branch**: `main`

### Summary

Deleted the legacy `tcrv-translate
--tcrv-plan-and-export-target-artifact-bundle` wrapper route. The public target
bundle exporter now remains only the coherence-gated exporter for already
planned/materialized MLIR, while source-level positive bundle workflows remain
on the plugin source artifact bundle front door.

### Main Changes

- Created Trellis task
  `05-17-plan-and-export-target-artifact-bundle-route-erasure` from the
  Direction Brief and wrote its PRD/context before source edits.
- Removed the `planAndExportTargetArtifactBundle` helper from
  `tools/tcrv-translate/tcrv-translate.cpp`.
- Removed the `TranslateFromMLIRRegistration` for
  `tcrv-plan-and-export-target-artifact-bundle`.
- Deleted the old no-viable wrapper lit test:
  `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`.
- Rewired source artifact bundle front-door negative coverage to use a neutral
  in-file no-artifact input instead of the deleted wrapper fixture.
- Updated lowering-runtime, variant-pipeline, and testing specs so the route is
  documented as deleted/absent rather than a current in-process
  planning/export workflow.

### Testing

- [OK] `cmake --build build --target tcrv-translate -j2`
- [OK] `tcrv-translate --help` scan found no deleted route option.
- [OK] Invoking the deleted option failed as an unknown command-line argument
  and emitted no bundle completion marker.
- [OK] Focused lit from `build/test` with filter
  `TargetArtifactBundleExport|Template|Toy`: 26/26 passed.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 123/123 passed.
- [OK] Targeted scans over code, tests, and specs found no active deleted route
  registration, helper, diagnostic, or manual execution-planning artifact pipe
  residue. Remaining `plan-and-export` mentions are negative spec constraints
  that the wrapper must stay absent.

### Status

[OK] Completed. Ready for archive and commit.

### Next Steps

- None - task complete.


## Session 113: Legacy manual source-artifact pipe cleanup

**Date**: 2026-05-17
**Task**: Legacy manual source-artifact pipe cleanup
**Branch**: `main`

### Summary

Removed or rewrote positive target-artifact tests that still advertised the
manual `tcrv-opt --tcrv-source-artifact-front-door-pipeline | tcrv-translate`
pipe as the source-to-artifact workflow. Source-input bundle coverage now uses
the one-command source artifact bundle front door, while retained low-level
exporter coverage starts from explicit materialized IR.

### Main Changes

- Created and completed Trellis task
  `05-17-legacy-manual-source-artifact-pipe-cleanup`.
- Deleted RVV/Toy/TensorExtLite positive source-to-object/header tests that
  depended on the old manual source-front-door pipe.
- Added scoped materialized-module exporter tests for RVV, Toy, and
  TensorExtLite object/header/bundle surfaces.
- Extended RVV source bundle front-door lit coverage from add-only to
  add/sub/mul through
  `tcrv-translate --tcrv-source-artifact-bundle-front-door`.
- Rewrote TensorExtLite source-to-bundle lit coverage to use the one-command
  source bundle front door.
- Updated `scripts/tensorextlite_runtime_abi_e2e.py` so source-input bundle
  generation uses the one-command front door and lower-level EmitC/header/object
  checks use an explicit materialized target fixture.
- Added missing-output-directory fail-closed coverage to the source bundle
  front-door negative lit test.

### Testing

- [OK] `python3 -m py_compile scripts/tensorextlite_runtime_abi_e2e.py`
- [OK] Focused lit filter from `build/test` covering touched source-bundle and
  materialized-exporter tests -> 10/10 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 126/126 passed.
- [OK] `git diff --check`
- [OK] Manual-pipe positive workflow scan over target tests and evidence/tool
  surfaces returned no matches.

### Status

[OK] **Completed**

### Next Steps

- None - task complete.


## Session 114: Legacy execution-planning artifact pipe test cleanup

**Date**: 2026-05-17
**Task**: Legacy execution-planning artifact pipe test cleanup
**Branch**: `main`

### Summary

Removed the remaining positive Template/Toy target tests that accepted a
manual `tcrv-opt --tcrv-execution-planning-pipeline | tcrv-translate` chain as
the successful input path for low-level target artifact/header/bundle or
Template EmitC-to-C++ routes. Retained low-level coverage now consumes explicit
materialized/planned IR fixtures, while source-level bundle coverage remains on
the one-command front door.

### Main Changes

- Created Trellis task
  `05-17-legacy-execution-planning-artifact-pipe-test-cleanup` from the
  Direction Brief and repaired its PRD/context before source edits.
- Rewrote `test/Target/Template/template-target-artifact-object.mlir` into
  materialized/planned Template object/header/bundle exporter coverage with
  direct `tcrv-translate` invocations.
- Rewrote `test/Target/Template/template-emitc-to-cpp.mlir` and
  `template-emitc-to-cpp-compile.test` to consume the Template materialized
  fixture directly.
- Rewrote Template stale-route fail-closed coverage to mutate the materialized
  fixture rather than a manual planning-pipeline stream.
- Deleted stale positive header tests:
  `test/Target/Template/template-target-artifact-header.mlir` and
  `test/Target/Toy/toy-target-artifact-header.mlir`.
- Reviewed spec update need; no `.trellis/spec/` change was needed because the
  durable deletion/source-front-door/materialized-fixture boundary is already
  documented.

### Testing

- [OK] Manual Template direct probes for object, header, EmitC-to-C++ source,
  and bundle output from `template-target-artifact-object.mlir`.
- [OK] Focused lit from `build/test` with the touched Template/Toy/source
  bundle filter: 9/9 passed.
- [OK] `git diff --check`
- [OK] Forbidden manual-pipe scan over Template/Toy/Target bundle tests,
  scripts, `tools/tcrv-translate/tcrv-translate.cpp`, and
  `lib/Transforms/ExecutionPlanningPipeline.cpp`: no matches.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 124/124 passed.
- [OK] No production code changed; no fresh `ssh rvv` proof required.

### Status

[OK] Completed. Ready for archive and commit.


## Session 112: RVV source-bundle runtime ABI execution proof

**Date**: 2026-05-17
**Task**: RVV source-bundle runtime ABI execution proof
**Branch**: `main`

### Summary

Moved the RVV generated bundle runtime ABI evidence from the old manual
`tcrv-opt | tcrv-translate` chain to the current one-command
`tcrv-translate --tcrv-source-artifact-bundle-front-door` production path, then
proved the generated header/object bundle can be consumed by an external C ABI
caller on `ssh rvv`.

### Main Changes

- Created Trellis task
  `05-17-05-17-rvv-source-bundle-runtime-abi-proof` from the Direction Brief.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so local bundle generation
  invokes the source-artifact bundle front door directly:
  `build/bin/tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=<bundle-dir> <source.mlir>`.
- Removed the script's `--tcrv-opt` argument and old manual source-front-door
  pipe from this evidence path.
- Preserved the external caller boundary: generated harness includes the
  generated header, links the generated object, and calls only the generated
  runtime-callable C ABI symbol.
- Added self-test negative coverage for stale object route, stale header route,
  and stale runtime ABI identity.
- Reviewed spec update need; no `.trellis/spec/` change was needed because the
  durable route and evidence rules already exist.

### Runtime Evidence

- Dry-run artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-source-bundle-frontdoor-dry`.
- `ssh rvv` artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-source-bundle-frontdoor-ssh`.
- Remote run passed add/sub/mul for `n=1,7,16,17,257` and printed:
  `tcrv_rvv_generated_bundle_abi_add_ok`,
  `tcrv_rvv_generated_bundle_abi_sub_ok`, and
  `tcrv_rvv_generated_bundle_abi_mul_ok`.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Dry-run source-bundle evidence command with add/sub/mul and counts
  `1,7,16,17,257`.
- [OK] `ssh rvv` source-bundle evidence command with add/sub/mul and counts
  `1,7,16,17,257`.
- [OK] Focused lit:
  `Scripts/rvv-generated-bundle-abi-e2e-self-test.test`,
  `Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`,
  `Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-fail-closed.mlir`.
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 128/128 passed.
- [OK] `git diff --check`
- [OK] Script scan confirms old pipe entry is gone:
  `rg -n "source-artifact-front-door-pipeline|tcrv-export-target-artifact-bundle|tcrv_opt|--tcrv-opt" scripts/rvv_generated_bundle_abi_e2e.py; test $? -eq 1`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete.


## Session 117: Legacy source-front-door named-absence fixture erasure

**Date**: 2026-05-17
**Task**: Legacy source-front-door named-absence fixture erasure
**Branch**: `main`

### Summary

Deleted the remaining Toy selected-boundary seed unknown-option fixture while preserving current source-front-door registry coverage.

### Main Changes

- Created and archived Trellis task `05-17-legacy-source-front-door-named-absence-fixture-erasure` from the Direction Brief.
- Deleted `test/Transforms/Toy/toy-template-selected-boundary-seed-deleted.mlir`, removing the stale `--tcrv-toy-materialize-template-selected-boundary-seed` named-absence fixture.
- Preserved current active source-front-door tests for RVV, Toy, TensorExtLite, and the generic source-artifact front-door pipeline.
- Reviewed spec-update need; no `.trellis/spec/` change was needed because the existing source-front-door and deleted-route constraints already cover this behavior.
- Checks: focused old-option scans passed, focused lit from `build/test` passed 4/4, `git diff --check` passed, and `cmake --build build --target check-tianchenrv -j2` passed 122/122.


### Git Commits

Included in the final task commit for this round.

### Testing

- [OK] Focused selected-boundary/source-seed option scan returned no matches.
- [OK] Active unknown-option scan left only current disabled-builtins
  source-front-door option coverage.
- [OK] Focused lit from `build/test` passed 4/4.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 119: Built-in non-plugin target artifact placeholder erasure

**Date**: 2026-05-17
**Task**: Built-in non-plugin target artifact placeholder erasure
**Branch**: `main`

### Summary

Deleted the remaining empty non-plugin built-in target artifact exporter placeholder, made the built-in target artifact registration path directly bundle-aware, tightened the emission-runtime spec, and validated focused target registration coverage plus full check-tianchenrv.

### Main Changes

- Created and archived Trellis task
  `05-17-05-17-builtin-non-plugin-target-artifact-placeholder-erasure` from
  the Direction Brief.
- Deleted the empty
  `registerBuiltinNonPluginTargetArtifactExporters` helper from
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.
- Rewired `registerBuiltinTargetArtifactExporters` so it directly delegates to
  `registerTargetArtifactExportersForEnabledExtensionBundles` with the caller's
  visible `ExtensionBundleRegistry` and `ExtensionPluginRegistry`.
- Tightened `.trellis/spec/lowering-runtime/emission-runtime-contract.md` so
  built-in target artifact registration has no pre-bundle or non-plugin lane.
- Added no compatibility wrapper, renamed helper, descriptor path,
  direct-C/source-export path, target route, artifact kind, or extension
  feature.

### Git Commits

Included in the final task commit for this round.

### Testing

- [OK] Trellis context validation for the task.
- [OK] Placeholder and deleted wrapper signature scans returned no matches.
- [OK] Protective residue scan left only the new prohibitive spec sentence:
  "There is no pre-bundle or non-plugin built-in target artifact exporter
  lane."
- [OK] `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tcrv-translate --help` route probe showed generic target
  artifact routes plus RVV, Template, and TensorExtLite materialized EmitC
  translate routes.
- [OK] Focused lit from `build/test` passed 8/8 for RVV, Toy, Template, and
  TensorExtLite target artifact / EmitC translate coverage.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 120: Binary self-check artifact residue erasure

**Date**: 2026-05-17
**Task**: Binary self-check artifact residue erasure
**Branch**: `main`

### Summary

Deleted the obsolete binary self-check expectation public target header and removed stale active testing guidance for the historical self-check harness/object/evidence bridge.

### Main Changes

- Created and archived Trellis task `05-17-binary-self-check-artifact-residue-erasure` from the Direction Brief.
- Deleted `include/TianChenRV/Target/BinarySelfCheckExpectation.h`, removing `BinarySelfCheckArithmeticKind`, `BinarySelfCheckExpectation`, `makeBinarySelfCheckExpectationError`, `getRuntimeABIPointeeScalarCType`, and `buildBinarySelfCheckExpectationFromRuntimeABI` from the public target API.
- Rewrote `.trellis/spec/testing/mlir-testing-contract.md` so historical RVV+scalar dispatch self-check harness/object/evidence bridge surfaces are deleted test surfaces rather than active conditional testing guidance.
- Added no replacement route, compatibility wrapper, descriptor adapter, direct-C source exporter, self-check generator, target artifact kind, or Python compiler-core path.
- Checks run: focused `BinarySelfCheck*` scan, self-check residue scan, descriptor/direct-C/source-export scan, focused target artifact build, `tianchenrv-target-artifact-export-test`, `git diff --check`, and `check-tianchenrv`.


### Git Commits

Included in the final task commit for this round.

### Testing

- [OK] Trellis context validation for the task.
- [OK] Deleted public API scan returned no matches outside the archived task
  PRD.
- [OK] Self-check residue scan left only prohibitive or historical-deletion
  spec wording.
- [OK] Descriptor/direct-C/source-export focused scan left existing
  fail-closed rejection code/tests and negative `implicit-check-not`
  assertions, not active self-check authority.
- [OK] `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete
