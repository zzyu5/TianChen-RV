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
