# RVV smoke-probe descriptor residue deletion

## Goal

Delete the RVV smoke-probe descriptor as an active compiler input. RVV selected
legality and selected lowering-boundary behavior must not depend on
`tcrv_rvv.smoke_probe_descriptor`, `standalone-c-toolchain-smoke-probe`, or a
standalone direct-C smoke-probe descriptor frontdoor. Future RVV executable
behavior must come from explicit `tcrv_rvv` extension-family IR and a
materialized MLIR EmitC route, not from a named descriptor attribute.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `4af7962`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-15-scalar-descriptorless-default-materialization-deletion/prd.md`
  deleted scalar descriptorless default microkernel materialization and left
  this RVV descriptor residue as a bounded follow-up.
- `.trellis/spec/index.md` states descriptor-driven computation is invalid as
  long-term architecture and the compiler stack must remain
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- `.trellis/spec/architecture/design-boundaries.md` rejects descriptor-driven
  microkernel/exporter frameworks and independent backend dialect framing.
- `.trellis/spec/extension-plugins/rvv-plugin.md` still documents
  `tcrv_rvv.smoke_probe_descriptor` as deleted negative input and, in one
  older section, as a selected standalone smoke-probe route.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` still permits
  stale smoke-probe descriptor fixtures as negative coverage.
- Active code/test scan found descriptor residue in:
  `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `test/Plugin/RVVBinaryVariantLegalityTest.cpp`, and `README.md`.

## Requirements

- Remove the `tcrv_rvv.smoke_probe_descriptor` constant and all special-case
  legality/boundary trigger logic from active RVV plugin code.
- Remove the standalone smoke-probe descriptor fixture from
  `RVVBinaryVariantLegalityTest.cpp`; do not preserve it as negative-only
  quarantine.
- Rewrite active specs and README text that keep
  `tcrv_rvv.smoke_probe_descriptor`, `standalone-c-toolchain-smoke-probe`, or
  smoke-probe descriptor handling alive.
- Keep selected RVV legality governed only by explicit typed RVV
  extension-family body authority, selected vector-shape/capability metadata,
  required march/capacity metadata, and existing non-descriptor boundaries.
- Do not introduce any replacement descriptor, compatibility mode, wrapper,
  negative-only legacy branch, or new extension route.
- Record any newly exposed missing RVV executable path as a rebuild gap rather
  than restoring the descriptor path.

## Acceptance Criteria

- [x] Active non-archive code/spec/tests no longer contain
      `tcrv_rvv.smoke_probe_descriptor`.
- [x] Active non-archive code/spec/tests no longer contain
      `standalone-c-toolchain-smoke-probe`.
- [x] Active non-archive code/spec/tests no longer contain smoke-probe
      descriptor handling or descriptor-specific RVV selected-legality trigger
      logic.
- [x] RVV selected-boundary validation no longer calls plugin legality because
      a descriptor attr is present.
- [x] RVV variant legality no longer has descriptor-specific rejection logic;
      descriptor metadata is not recognized as a compiler input.
- [x] No Common EmitC lowering, new RVV lowering, new RVV op, new artifact
      export route, runtime ABI rebuild, ssh-rvv evidence path, wrapper,
      compatibility shim, or broad smoke matrix is added.
- [x] Focused ref-scans for `smoke_probe_descriptor`,
      `standalone-c-toolchain-smoke-probe`, and `smoke-probe descriptor` are
      reported truthfully for active non-archive surfaces.
- [x] Focused RVV plugin legality/selected-boundary tests and directly
      affected lit tests pass, or failures are recorded as expected deletion
      gaps without restoring the descriptor path.
- [x] `git diff --check`, Trellis validation, finish/archive, final clean
      `git status --short`, and one coherent commit are produced if complete.

## Non-goals

- No common EmitC lowering implementation.
- No new RVV lowering.
- No new RVV extension ops.
- No new target artifact/export routes.
- No new runtime ABI modeling.
- No `ssh rvv` evidence, smoke matrices, correctness claims, or performance
  claims.
- No scalar cleanup unless a scalar file is directly coupled to this descriptor
  residue.
- No compatibility shims, wrapper branches, replacement descriptors, or
  negative-only descriptor quarantine.

## Minimal Evidence

- Focused ref-scan for:
  `smoke_probe_descriptor`, `standalone-c-toolchain-smoke-probe`, and
  `smoke-probe descriptor` in active RVV plugin code, active specs/README, and
  directly affected tests.
- Focused build/run of RVV binary variant-legality and selected-boundary C++
  tests.
- Affected lit/FileCheck tests if direct search or build failures identify
  them.
- Full `check-tianchenrv` if practical after focused validation.
- `git diff --check` and `git diff --cached --check`.
- `python3 ./.trellis/scripts/task.py validate` before finish/archive.

## Technical Notes

- The current deletion target is not the RVV executable microkernel body path;
  explicit typed `tcrv_rvv.*_microkernel` bodies remain the valid bounded
  authority for existing first-slice tests.
- Required march, capacity metadata, selected vector-shape metadata, and typed
  body validation may remain because they are non-descriptor selected-path
  metadata surfaces.
- If removing the descriptor residue exposes that no standalone RVV smoke
  probe compiler frontdoor exists, that is the intended rebuild gap for this
  deletion round.

## Completion Evidence

- Removed `kRVVSmokeProbeDescriptorAttrName`,
  `rejectDeletedSmokeProbeDescriptorAttr`, and descriptor-specific property
  view/legality branching from
  `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`.
- Removed the descriptor-triggered selected-legality branch from
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`.
- Removed the standalone smoke-probe descriptor fixture and expectation from
  `test/Plugin/RVVBinaryVariantLegalityTest.cpp`.
- Rewrote `README.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md` so the former
  standalone smoke-probe compiler front doors are deleted without preserving a
  descriptor attribute, descriptor fixture, or descriptor-specific negative
  quarantine.
- Focused ref-scan over `lib`, `include`, `test`, `README.md`, and
  `.trellis/spec` found no active occurrences of
  `tcrv_rvv.smoke_probe_descriptor`, `smoke_probe_descriptor`,
  `standalone-c-toolchain-smoke-probe`, `smoke-probe descriptor`,
  `smoke descriptor`, `smoke.*descriptor`, or `descriptor.*smoke`.
- A wider standalone smoke-probe scan still finds deleted route/option
  documentation such as `tcrv-export-rvv-smoke-probe-c` and
  `rvv-smoke-probe-standalone-c-source`; those are route-absence contracts, not
  descriptor inputs or descriptor handling.
- Checks passed:
  `git diff --check`;
  `cmake --build build --target tianchenrv-rvv-binary-variant-legality-test tianchenrv-rvv-selected-lowering-boundary-test -j2`;
  `./build/bin/tianchenrv-rvv-binary-variant-legality-test`;
  `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`;
  `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test -j2`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVSmokeProbe/rvv-smoke-probe-route-deleted.mlir`
  from `build/test`;
  `cmake --build build --target check-tianchenrv -j2` with 114/114 passing.
- No self-repair was needed after focused or full checks.
- Missing new-architecture gap: there is no descriptor-based standalone RVV
  smoke-probe compiler frontdoor. That is intentional for this deletion round;
  future executable or probe behavior must be rebuilt from explicit
  extension-family IR, common EmitC/artifact routes, or external probe tooling
  with recorded `ssh rvv` evidence.
