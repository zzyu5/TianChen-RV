# RVV Binary Descriptor Authority Deletion

## Goal

Delete RVV descriptor-driven binary compute authority as a deletion-only Wrong
Logic Deletion Campaign round. This supersedes the earlier capability-gated
selected-config legality task; no selected-config rebuild feature work is in
scope for this round.

## Why Now

The completed `43dd5b1` task proved one i32m2 selected-config artifact/runtime
path, but current HEAD still contains RVV descriptor-era authority surfaces:
`RVVBinaryFamilyRegistry`, `RVVBinaryDescriptor`, descriptor-shaped
RVV+scalar dispatch records, legacy lowering descriptor mirrors, and descriptor
element-count metadata that can still participate in route/runtime/artifact
claims. The campaign rule is deletion before rebuild, so this task removes
that wrong logic instead of adding another legality feature.

## Scope

- Delete or rewrite RVV binary descriptor family/route headers and type names
  that make descriptors the source of family, intrinsic, artifact, dispatch,
  RuntimeABI, or C compute semantics.
- Delete or rewrite descriptor-shaped RVV+scalar dispatch family records and
  route manifests.
- Delete or rewrite descriptor element-count metadata in RVV selected config,
  runtime length, RVV microkernel export, scalar dispatch, bundle/index
  evidence, and tests.
- Remove direct legacy `tcrv_rvv.lowering_descriptor` authority and tests that
  protect descriptor-as-legal-input behavior.
- Keep retained family tables only if they are non-semantic route/package
  records consumed after typed family/body/source authority is established.

## Acceptance Criteria

- [x] No capability-gated selected-config feature work is carried forward unless
      it is strictly deletion of descriptor wrong logic.
- [x] `RVVBinaryFamilyRegistry` and `RVVBinaryDescriptor` are deleted or
      production code is rewired away from those descriptor authority objects.
- [x] `DispatchBinaryFamilyDescriptor` and related descriptor-shaped scalar
      dispatch records are deleted or rewritten as non-authoritative route
      records.
- [x] Descriptor element-count naming no longer appears as RVV selected
      config/runtime/route authority; any retained capacity count is named as
      non-semantic artifact/component capacity.
- [x] Tests that exist only to protect descriptor legal input are deleted or
      rewritten as fail-closed removal coverage.
- [x] No compatibility layer, helper-only wrapper, replacement feature, new
      family/dtype/shape, Python compiler semantics, direct descriptor-to-C
      exporter, or generic core RVV branch is added.
- [x] If deletion exposes build/lit failures, record exact failures as missing
      new-architecture gaps and do not restore descriptor logic to make checks
      pass.
- [x] Final changeset is deletion/ref-scan oriented and the committed worktree
      is clean.

## Non-Goals

- No capability-gated selected-config implementation.
- No new RVV family, dtype, SEW, LMUL, tail, mask, frontend, runtime, EmitC
  route, plugin template, common interface, evidence matrix, or performance
  claim.
- No helper-only, wrapper-only, compatibility-mode, quarantine-mode,
  legacy-mode, metadata-only, report-only, PRD-only, journal-only, smoke-only,
  or artifact-packaging milestone.
- No descriptor element count or vector shape as selected config, runtime, or
  route authority.
- No computation semantics in `tcrv.exec`.
- No restoring wrong descriptor paths to make checks pass.

## Minimal Evidence

- Focused deletion inventory of descriptor-driven RVV binary compute objects
  deleted, rewritten, or retained.
- Bounded ref-scan over touched RVV target/plugin/transform/support/test files
  for `RVVBinaryFamilyRegistry`, `RVVBinaryDescriptor`,
  `DispatchBinaryFamilyDescriptor`, `lowering_descriptor`,
  `descriptor_element_count`, descriptor-to-C, explicit-only route authority,
  and direct C semantic exporter references.
- `git diff --check` and `git diff --cached --check`.
- Focused build/lit checks if the tree remains buildable. If deletion leaves
  expected missing-architecture failures, report the exact compiler/test errors
  without restoring descriptor logic.
- Trellis context validation before finish/archive when task state is coherent.

## Read Files

- `.trellis/spec/index.md`
- `.trellis/spec/architecture/design-boundaries.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`
- `.trellis/tasks/archive/2026-05/05-14-rvv-lmul-m2-selected-config-artifact-closure/prd.md`
