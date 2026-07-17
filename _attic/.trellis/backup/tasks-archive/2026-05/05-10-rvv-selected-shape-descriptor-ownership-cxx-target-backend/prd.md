# RVV selected-shape descriptor ownership in the C++ target backend

## Goal

Move the finite RVV selected i32 vector-shape contract used by the existing
RVV-primary plus scalar-fallback i32 dispatch exporter into target-owned C++
descriptor/API ownership. The current `i32m1` and `i32m2` semantics already
exist and have runtime evidence from the previous task; this round is about
compiler/backend ownership and fail-closed validation across source, header,
object, bundle, and evidence consumer paths.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting inspected state is clean at HEAD `5db3128`.
- There was no active Trellis task, so this task was created explicitly as
  `.trellis/tasks/05-10-rvv-selected-shape-descriptor-ownership-cxx-target-backend/`.
- The previous `rvv-i32m2-vsub-dispatch-artifact-path` task is archived and
  must not be reopened. Its PRD records real direct and bundle `ssh rvv`
  evidence for `i32-vsub i32m2` at commit `5db3128`.
- A target-owned finite shape config already exists in
  `include/TianChenRV/Target/RVV/RVVVectorShape.h`, but the dispatch exporter
  does not yet consume it as the C++ source of truth for selected-plan metadata,
  capability IDs, suffixes, and intrinsic consistency.
- `scripts/rvv_scalar_dispatch_e2e.py` still carries a Python
  `RVV_VECTOR_SHAPE_SPECS` table with semantic facts that should be compiler
  output or descriptor-derived truth, not evidence-script truth.

## Requirements

- Keep compiler truth in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may orchestrate tools, parse generated artifacts, validate
  evidence, and write sanitized summaries, but it must not own compiler shape
  semantics.
- Consolidate the existing finite RVV i32 selected vector-shape config into a
  bounded target-owned descriptor/API that covers only the currently proven
  shapes: `i32m1` and `i32m2`.
- The descriptor/API must expose the shape facts needed by target/export code:
  shape id/name, SEW, LMUL, tail and mask policy, C vector type, vector
  intrinsic suffix, `vsetvl` suffix, capability IDs, selected-plan metadata
  names/roles/notes, and diagnostic spelling.
- The existing RVV+scalar dispatch exporter must consume that C++ descriptor
  before accepting selected artifacts. It must fail closed on:
  - missing, unknown, partial, duplicate, or mismatched selected-plan shape
    metadata;
  - missing or inconsistent finite shape capability IDs on the selected RVV
    variant;
  - shape suffix or intrinsic spelling inconsistency between selected metadata
    and the generated embedded RVV source;
  - stale `i32m1`/`i32m2` selected-shape metadata mixed into the wrong export.
- Preserve existing generated output names, ABI names, route IDs, component
  groups, self-check markers, and evidence meanings unless a compatibility
  correction is unavoidable and tested.
- Keep `tcrv.exec` compute-free. RVV-specific selected-shape semantics must
  remain in RVV target/plugin-owned code and not move into core passes.
- Adjust Python evidence validation to consume generated source comments,
  bundle index metadata, and emitted artifact strings. It may keep a bounded
  CLI selector and fixture-default mapping for `i32m1`/`i32m2`, but any
  remaining table must be documented as evidence-layer routing, not compiler
  semantic truth.

## Acceptance Criteria

- C++ descriptor/API tests prove the finite shape descriptor covers `i32m1` and
  `i32m2`, includes selected-plan metadata fields, and exposes capability IDs,
  vector type, vector suffix, and setvl suffix from one target-owned surface.
- Focused dispatch export coverage proves both `i32m1` and `i32m2` dispatch
  exports consume the C++ shape descriptor/API and emit the expected selected
  metadata, capability IDs, vector type, setvl suffix, and intrinsic suffix.
- Negative coverage proves at least one selected-plan metadata mismatch fails
  before a misleading dispatch artifact or bundle record is accepted.
- Python evidence dry-runs still pass for existing direct and bundle paths, but
  shape validation reads generated artifact facts instead of inventing the full
  compiler shape contract in Python.
- `git diff --check` passes.
- The local build target `tcrv-translate tcrv-opt` passes when the configured
  build directory exists.
- Focused tests covering changed behavior pass.
- `check-tianchenrv` is run after C++ changes. If blocked, record the exact
  non-secret blocker and keep the task open.

## Out of Scope

- New arithmetic families, dtypes, LMULs, RVV ops, scalar ops, generic vector
  lowering, or new runtime ABI surface.
- Any compiler IR, lowering, selection, emission, ABI, or plugin decision
  implemented in Python.
- Generic `tcrv.exec` compute semantics or RVV-specific branches in core
  passes.
- Broad smoke matrices or new runtime/correctness/performance claims.
- New `ssh rvv` evidence unless ABI-visible generated source/object behavior
  changes enough to create a new correctness claim.
- Committing `artifacts/tmp`, raw remote logs, credentials, or environment-
  specific build outputs.

## Technical Notes

- Specs read for this task:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Prior task reference read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i32m2-vsub-dispatch-artifact-path/prd.md`
- Primary implementation surfaces:
  - `include/TianChenRV/Target/RVV/RVVVectorShape.h`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - `test/Target/I32BinaryFamilyRegistryTest.cpp`
  - `test/Target/RVVScalarDispatch/`
  - `test/Target/TargetArtifactBundleExport/`
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`
- Known remaining duplication allowed by this PRD if fully removing it becomes
  too large: Python may keep the finite CLI shape names and default fixture
  routing needed to select the checked-in `i32m1` or `i32m2` inputs. It must not
  retain authoritative SEW/LMUL/type/suffix/capability tables as compiler truth.

## Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tcrv-opt -j2`
- Focused C++ target test for the descriptor/registry/exporter surface.
- Focused lit for RVV scalar dispatch and target artifact bundle selected-shape
  exports and mismatch failures.
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Focused direct dry-run for `i32-vsub --vector-shape=i32m2`.
- Focused bundle dry-run for `i32-vsub --vector-shape=i32m2`.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Current Status

Completed. The finite RVV i32 selected-shape descriptor/API is now owned by
the C++ RVV target surface and consumed by the RVV plugin selected-plan
metadata path plus the RVV+scalar dispatch exporter. The exporter validates
the selected variant capability IDs, selected-plan metadata, embedded source
shape comments, vector type, setvl suffix, intrinsic suffix, and arithmetic
intrinsic spelling against the descriptor before producing dispatch source or
header artifacts. Python evidence validation now consumes generated artifact
comments and emitted intrinsic strings; the remaining Python shape duplication
is limited to the bounded CLI selector/default fixture routing and self-test
fixture payloads used to exercise the parser without invoking the compiler.

Validation completed:
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tcrv-opt -j2`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i32-vsub --vector-shape=i32m2 --lower-linalg-frontend --run-id codex-rvv-shape-cxx-direct-dry --overwrite`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-rvv-shape-cxx-bundle-dry --overwrite`
- manual selected-plan metadata mismatch probe expecting
  `selected_plan_metadata 'tcrv_rvv.selected_vector_suffix' vector suffix must be 'i32m2'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

No new real `ssh rvv` evidence was produced because this round changes
compiler ownership and fail-closed validation only, without making a new
runtime correctness or performance claim. The previous `5db3128` evidence
remains the latest runtime evidence for the affected `i32-vsub i32m2` path.
