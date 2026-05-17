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
