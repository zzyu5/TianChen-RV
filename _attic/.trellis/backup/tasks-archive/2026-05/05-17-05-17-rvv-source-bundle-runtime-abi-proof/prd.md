# RVV source-bundle runtime ABI execution proof

## Goal

Prove that the current one-command RVV source-artifact bundle front door emits a generated object/header bundle that an external runtime ABI consumer can compile, link, and run on the real RVV machine via `ssh rvv`.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`; HEAD before this task is `7441edf translate: add source artifact bundle front door`; the worktree was clean before task creation.
- No `.trellis/.current-task` existed, so this task was created from the Direction Brief before source edits.
- The previous archived task `05-17-common-source-artifact-bundle-front-door` made `tcrv-translate --tcrv-source-artifact-bundle-front-door` the production one-command bridge from supported source MLIR to target artifact bundle export.
- `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir` proves the local bundle records for the RVV add source fixture, but it does not compile/link/run a generated-header external consumer on `ssh rvv`.
- `scripts/rvv_generated_bundle_abi_e2e.py` already checks RVV generated object/header bundle records, writes an external C harness, and can run it on `ssh rvv`, but its bundle generation still uses the older manual `tcrv-opt --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact-bundle` chain.

## Boundaries

- This is a runtime ABI consumption/evidence task for the existing RVV i32m1 source add/sub/mul path.
- The harness is only an external C ABI consumer of compiler-produced header/object artifacts; it must not duplicate or synthesize the RVV compute body.
- The compiler path under proof is:

```text
source MLIR
  -> plugin source front door
  -> selected RVV boundary
  -> plugin-owned EmitC route
  -> materialized EmitC
  -> bundle object/header
  -> external C ABI consumer on ssh rvv
```

- Do not add a new front-door command, descriptor route, direct C semantic exporter, source-export compatibility path, scalar fallback runtime path, new RVV family, or family-specific semantic branch in common translate code.
- Do not treat artifacts, reports, prompt edits, or broad smoke tests as the main achievement.

## Requirements

- Change RVV generated bundle ABI evidence so bundle generation uses one `tcrv-translate --tcrv-source-artifact-bundle-front-door` command with `--tcrv-target-artifact-bundle-output-dir`, not the manual `tcrv-opt | tcrv-translate --tcrv-export-target-artifact-bundle` chain.
- Preserve bundle index/header/object validation for selected variant, selected role, route, owner plugin, runtime ABI kind/name, ordered ABI parameters, unmangled symbol, object record, header record, and multi-VL runtime metadata.
- Preserve the external C consumer boundary: the harness includes the generated header, links the generated object, calls only the generated ABI symbol, and checks nontrivial runtime `n` values that cross the AVL/VL loop boundary.
- Keep negative coverage fail-closed for missing bundle index, missing header, missing object, stale/mismatched ABI parameter order, stale/mismatched selected variant, stale/mismatched arithmetic metadata, missing multi-VL metadata, forbidden header compute/body residue, and missing C ABI guard.
- Add negative self-test coverage for stale or mismatched object/header route and runtime ABI identity if absent.
- Record evidence under `artifacts/tmp/rvv_generated_bundle_abi_e2e/` with enough command/output metadata to tie the run back to the one-command source front door.

## Acceptance Criteria

- [x] The evidence tool invokes `tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=... <source.mlir>` for RVV source bundle generation.
- [x] Dry-run local evidence validates the generated bundle index, generated header, generated object, and unmangled runtime-callable symbol.
- [x] Real `ssh rvv` evidence compiles and links the external consumer against the generated header/object and runs runtime counts that include values smaller than, equal to, greater than, and much greater than common vector chunk sizes.
- [x] Evidence ties the run to selected variant, route, runtime ABI name, ordered ABI parameters, object record, header record, and source front-door command.
- [x] Negative self-tests fail closed for missing, stale, mismatched, or route/ABI-incoherent bundle/header/object metadata.
- [x] Focused lit/self-test coverage passes for the changed evidence tool.
- [x] Targeted scans over touched runtime/evidence/translate surfaces show no descriptor authority, direct-C/source-export semantic path, handwritten RVV compute body in compiler code, or family-specific common-translate branch was added.

## Out of Scope

- New RVV lowering, new source route, new exporter, new descriptor compatibility route, scalar fallback runtime execution, generic RVV dtype/LMUL expansion, Python compiler-core logic, broad evidence matrices, or new artifact ledger/state-machine protocol.
- Rewriting compiler lowering/exporter behavior unless the evidence tool reveals a real compiler contract bug in the current selected RVV bundle route.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/extension-plugins/index.md`, and
  `.trellis/spec/testing/index.md`.
- Relevant implementation/evidence files:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.

## Completion Evidence

- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so bundle generation now records and runs:

```text
build/bin/tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=<bundle-dir> <source.mlir>
```

- Removed the script's old `--tcrv-opt` dependency and no longer uses `tcrv-opt --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact-bundle` for this evidence path.
- Added negative self-test coverage for stale object route, stale header route, and stale runtime ABI identity.
- Dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-source-bundle-frontdoor-dry`.
- Real RVV evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-source-bundle-frontdoor-ssh`.
- `ssh rvv` output proved add/sub/mul external C ABI consumers compiled, linked, and ran counts `1,7,16,17,257`.
- Spec update review: no `.trellis/spec/` edit was needed; the durable architecture rule already exists in the lowering-runtime source-artifact bundle front-door and RVV plugin specs.

## Checks

- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --run-id codex-rvv-source-bundle-frontdoor-dry --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --run-id codex-rvv-source-bundle-frontdoor-ssh --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- [x] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-generated-bundle-abi-e2e-self-test.test Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-fail-closed.mlir`
- [x] `cmake --build build --target check-tianchenrv -j2`
- [x] `git diff --check`
- [x] `rg -n "source-artifact-front-door-pipeline|tcrv-export-target-artifact-bundle|tcrv_opt|--tcrv-opt" scripts/rvv_generated_bundle_abi_e2e.py; test $? -eq 1`
