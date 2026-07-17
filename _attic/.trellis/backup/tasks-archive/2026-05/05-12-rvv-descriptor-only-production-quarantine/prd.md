# Quarantine RVV Descriptor-Only Production Selected Export

## Goal

Complete the next structural RVV descriptor-exit step: direct
`tcrv_rvv.lowering_descriptor` metadata by itself must no longer resolve,
materialize, or export a supported production selected RVV binary route for the
typed-source finite RVV binary families. A supported selected export must be
backed by an already typed `tcrv_rvv.*_microkernel` body, or by
frontend/default typed-family authority that materializes the typed body before
selected emission/export.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Branch is `main`; the worktree was clean before this task was created.
* Current HEAD before this task is
  `c632e07 feat(rvv): require typed body for default vadd`.
* No `.trellis/.current-task` existed at session start; this task was created
  as `.trellis/tasks/05-12-rvv-descriptor-only-production-quarantine/`.
* The archived predecessor
  `.trellis/tasks/archive/2026-05/05-12-descriptor-exit-rvv-default-vadd-body-materialization/`
  completed the bounded default descriptorless `i32-vadd` typed-body
  materialization path and should not be reopened.
* Specs keep `tcrv.exec` compute-free, keep RVV semantics in the RVV extension
  family, and define the current main route as extension family ops -> common
  EmitC -> RVV intrinsic C/C++ artifact.
* Current code still has descriptor-only production surfaces:
  `resolveRVVBinaryFamilyForProposal` rejects direct descriptor-only
  `i32-vadd` only, while direct descriptor-only `i32-vsub`, `i32-vmul`, and
  i64 finite families can still resolve through descriptor metadata.
* Current selected lowering-boundary materialization still auto-materializes
  typed RVV microkernel ops from `tcrv_rvv.lowering_descriptor` for descriptor
  families other than `i32-vadd`.
* Current selected emission/export already requires a matching typed
  microkernel attachment before a supported plan is emitted; the remaining
  structural leak is that descriptor-only selected-boundary materialization can
  manufacture that attachment.
* Existing focused tests still preserve descriptor-only production success for
  i32 sub and i64 finite routes. Those tests need to become typed-body success,
  frontend/default typed-body success, or explicit legacy quarantine coverage.

## Requirements

* Direct descriptor-only RVV binary variants for the typed-source finite family
  set (`i32`/`i64` add/sub/mul) must fail before supported selected
  lowering-boundary materialization or selected emission/export.
* When a typed `tcrv_rvv.*_microkernel` body exists, optional
  `tcrv_rvv.lowering_descriptor` metadata may remain only as a fail-closed
  compatibility check. Family, shape, element count, selected variant, role,
  required capabilities, required march, ABI roles, and dataflow body must
  agree.
* Frontend-derived finite RVV binary paths must remain supported through the
  descriptorless typed-family proposal path, materialized RVV family body, and
  common EmitC-backed target artifact route.
* The descriptorless default `i32-vadd` typed-body materialization path from
  the predecessor remains in scope as a bounded non-descriptor compatibility
  path, but it must not reattach descriptor compute authority.
* Route metadata may still include descriptor-local selected-config fields as
  legacy compatibility metadata when an actual typed body exists. It must not
  be the compute authority.
* The finite family registry remains target metadata for route ids, ABI names,
  intrinsic names, and bounded family constants. It must not be treated as IR
  compute authority.
* Tests that preserve descriptor-only production success must be updated or
  deleted. Descriptor-only tests that remain must be named/checked as
  legacy/quarantine or compatibility fail-closed coverage.

## Acceptance Criteria

* [x] `resolveRVVBinaryFamilyForProposal` rejects descriptor-only finite RVV
      binary families, not just `i32-vadd`, when no typed body exists.
* [x] Selected lowering-boundary materialization rejects descriptor-only finite
      RVV binary variants before auto-materializing a microkernel attachment.
* [x] A typed body plus matching optional descriptor still succeeds and records
      descriptor metadata only as compatibility metadata.
* [x] A typed body plus mismatched optional descriptor fails closed before
      selected export.
* [x] Frontend-derived finite binary routes still materialize typed RVV body ops
      and export through the common EmitC-backed target artifact route.
* [x] Descriptorless default `i32-vadd` still materializes typed body authority
      and does not reattach `tcrv_rvv.lowering_descriptor`.
* [x] Focused C++ and lit/FileCheck coverage prove descriptor-only quarantine,
      typed-body success, frontend/default typed success, and compatibility
      mismatch rejection.

## Out Of Scope

* New RVV arithmetic families, dtypes, LMULs, benchmarks, performance claims,
  runtime correctness claims, or broad smoke matrices.
* Descriptor-to-C exporters or direct descriptor-to-C production behavior.
* Generic RVV lowering, MLIR vector/scalable-vector lowering, IME, AME,
  Sophgo/offload, TensorExt, Template, Toy, or unrelated plugin work.
* Python compiler behavior. Python may only drive Trellis validation or
  existing tooling checks.
* `ssh rvv` evidence; this is a structural compiler migration and makes no
  fresh RVV runtime, correctness, or performance claim.

## Minimal Validation

* Run `git diff --check`.
* Build focused targets:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
* Run focused C++ tests for RVV planning, selected lowering-boundary,
  extension plugin, and target artifact export behavior.
* Run focused lit/FileCheck coverage for descriptor-only quarantine, typed body
  success, and at least one frontend/default typed-source success path through
  target artifact export.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and local time/tooling allows.
* Run Trellis task validation before finish/archive and again on the archived
  task path if completed.

## Technical Notes

* `.trellis/spec/architecture/unified-riscv-mlir.md` defines descriptor-driven
  computation as implementation debt and keeps `tcrv.exec` compute-free.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the correct route as
  typed extension op -> generated lowerable interface -> target/plugin mapping
  -> common EmitC route.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  selected emission/export to fail closed for stale, unsupported, or spoofed
  artifact routes.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires supported RVV
  microkernel export to consume verified `setvl` / `with_vl` /
  load/arithmetic/store typed RVV body structure.
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md` makes
  bounded linalg body/types the frontend compute authority and rejects stale
  legacy descriptor metadata before creating exec IR.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused C++ and
  lit/FileCheck coverage for MLIR behavior and explicitly keeps local
  compiler checks separate from RVV runtime/correctness evidence.

## Round Result

Completed in this round:

* Added `isTypedSourceRVVBinaryFamily` as the shared RVV planning predicate for
  finite typed-source binary families.
* Changed direct proposal resolution so descriptor-only `i32`/`i64` RVV binary
  families are legacy-quarantined unless a typed body or frontend/default
  typed-family source is present.
* Changed selected lowering-boundary materialization so a direct
  `tcrv_rvv.lowering_descriptor` for a typed-source finite RVV binary family
  cannot auto-materialize a typed microkernel attachment by itself.
* Kept descriptor metadata as fail-closed compatibility only when a typed RVV
  microkernel body already exists; matching descriptor metadata can still
  cross-check body family/shape/element-count, and mismatches still reject
  selected export.
* Updated direct RVV microkernel lit fixtures for i32 sub/mul and i64
  add/sub/mul to use frontend-derived typed-family authority instead of
  descriptor-only variant authority.
* Updated focused C++ tests so descriptor-only i64 direct proposal records a
  recoverable quarantine decline and produces no production candidate, while
  typed-body/direct and frontend/default routes remain supported.

Self-repair performed:

* Adjusted the RVV extension plugin test after observing that registry proposal
  declines are recoverable decline records, not fatal `collectVariantProposals`
  errors.
* Repaired the i64 missing-cap lit fixture so i64m1 shape authority comes from
  explicit typed-source selected config requirements, with `rvv.i64_m1.sew64`
  unavailable, rather than from a legacy descriptor.

Validation completed:

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* Focused lit/FileCheck with `-j1` for descriptor-only quarantine, direct typed
  i32/i64 microkernel exports, emission readiness, and i64 missing-cap
  fail-closed coverage.
* Focused lit/FileCheck with `-j1` for frontend-derived linalg-to-exec
  i32/i64 typed routes through target artifact export.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 206/206 lit tests.

No `ssh rvv` command was run or claimed. This round is a structural compiler
migration and makes no fresh RVV hardware runtime, correctness, or performance
claim.
