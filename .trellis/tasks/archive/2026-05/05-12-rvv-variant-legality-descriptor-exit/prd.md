# RVV variant legality descriptor exit

## Goal

Rewire RVV binary variant legality so direct finite RVV binary variants are
legal only when typed RVV authority is present. Legacy
`tcrv_rvv.lowering_descriptor` may remain only as optional mirror metadata
after typed authority has established the bounded family/body/source route; it
must not make an empty-body or descriptor-only direct RVV variant legal.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean before this task was created.
- Task start HEAD is
  `b251b38 feat(rvv): exit descriptor authority in plugin planning`.
- `.trellis/.current-task` was absent at session start, so this task was
  created as `.trellis/tasks/05-12-rvv-variant-legality-descriptor-exit/`.
- The archived predecessor
  `.trellis/tasks/archive/2026-05/05-12-rvv-plugin-proposal-planning-descriptor-exit/`
  completed direct RVV proposal planning descriptor exit and must not be
  reopened.
- Recent archived descriptor-exit tasks have already moved RVV selected
  emission, selected lowering production, i64 target export, and finite
  registry surfaces toward typed selected-plan/body authority with descriptor
  metadata quarantined as legacy mirror data.
- Current live evidence still shows the next boundary can preserve old
  authority: `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp` still looks up
  `tcrv_rvv.lowering_descriptor`, and
  `test/Plugin/RVVBinaryVariantLegalityTest.cpp` still accepts empty
  `tcrv.exec.variant` bodies with descriptor metadata as legal.
- Specs keep `tcrv.exec` compute-free, keep RVV semantics in RVV
  plugin/target code, define the current route as extension family ops ->
  EmitC -> intrinsic/runtime C/C++, and treat descriptor-driven computation as
  bounded debt rather than architecture.

## Requirements

- `verifyRVVBinaryVariantLegality` must no longer accept a direct finite RVV
  binary variant solely because legacy descriptor metadata, selected vector
  shape metadata, required march, policy, and capability requirements are
  present.
- A legal direct finite RVV binary variant must carry typed authority from one
  of the current supported typed paths:
  - an explicit typed `tcrv_rvv.*_microkernel` body under the variant; or
  - explicit typed selected-source metadata that unambiguously names the RVV
    family/body source before later selected lowering materializes the body.
- Descriptor mirror validation may run only after typed authority has
  established family, dtype, selected vector shape/config, route, and source
  authority. A stale descriptor mirror must fail closed and diagnostics must
  state that descriptor metadata is non-authoritative mirror metadata.
- Empty-body descriptor-only direct variants must be rejected for at least
  bounded i32 and i64 finite binary examples.
- The default descriptorless typed i32 route and the existing typed i64 route
  must remain legal.
- The smoke-probe route must stay separate from finite RVV binary compute
  legality and must not become a descriptor-driven compute backdoor.
- Legality diagnostics should name the missing typed RVV authority boundary,
  not suggest adding a descriptor.
- Tests that encode descriptor-driven legality as production behavior must be
  rewritten or deleted rather than preserved as compatibility success.
- Keep all RVV-specific legality logic in RVV plugin/target code. Do not add
  compute semantics to `tcrv.exec` or family-specific branches to generic core
  orchestration.

## Acceptance Criteria

- [x] Descriptor-only finite RVV binary variants with empty bodies fail plugin
      legality for both i32 and i64 examples.
- [x] Descriptor mirror validation is post-typed-authority only; matching
      descriptor mirrors are accepted only beside typed authority, and stale
      mirrors fail closed with non-authoritative mirror diagnostics.
- [x] Descriptorless typed i32 legality remains legal.
- [x] Existing typed i64 legality remains legal.
- [x] Smoke-probe legality remains separate from finite RVV binary compute
      legality.
- [x] Focused C++ tests cover descriptor-only failure, typed-body success,
      descriptor mirror success, stale mirror failure, and smoke-probe
      separation.
- [x] Focused neighboring RVV planning, selected-lowering-boundary, extension
      plugin, and RVV artifact/linalg lit checks remain coherent.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Trellis task validation passes on the active task path and on the
      archived path if completed.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
      -j2` runs if the existing build tree is usable; otherwise the blocker is
      recorded and focused checks pass.

## Non-Goals

- No new arithmetic family, dtype, LMUL, broad matrix, scheduler, tuning, cost
  model, runtime ABI expansion, or performance work.
- No compiler core, dialects, passes, plugin registry, capability model,
  lowering, or emission logic in Python.
- No descriptor-to-C export, descriptor-selected compute semantics, or new
  descriptor-owned tests.
- No changes to the default native compiler route and no new GCC/vendor
  compiler default claim.
- No fresh `ssh rvv` run or runtime/correctness/performance claim; this is a
  structural compiler legality migration.
- No helper-only, metadata-only, coverage-only, broad smoke, or report-only
  closeout.

## Minimal Validation

- Build and run `tianchenrv-rvv-binary-variant-legality-test`.
- Build and run focused neighboring tests:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`, and
  `tianchenrv-rvv-extension-plugin-test`.
- Run focused lit filters from the build test directory for RVV artifact /
  linalg paths that depend on finite RVV legality, especially
  `linalg-i32-vadd-to-exec` and `rvv.*artifact`.
- Run `git diff --check` and `git diff --cached --check`.
- Run `python3 ./.trellis/scripts/task.py validate` on the active task path and
  on the archived task path if the task is completed.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if the existing build tree is usable.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Prior PRDs read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-plugin-proposal-planning-descriptor-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-selected-emission-descriptor-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-lowering-boundary-materialization/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-i64-target-export-body-authority/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-descriptor-registry-quarantine/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-descriptor-only-production-quarantine/prd.md`
- Initial source focus:
  - `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVBinaryVariantLegality.h`
  - `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `test/Plugin/RVVBinaryVariantLegalityTest.cpp`
  - neighboring RVV planning / selected-boundary / extension-plugin tests and
    RVV artifact lit fixtures.

## Definition Of Done

The task is complete only when RVV binary variant legality is typed-authority
first in production code, descriptor-only finite binary variants fail closed,
descriptors are mirror-only after typed authority, focused and available full
checks pass, Trellis state is truthful, and one coherent commit records the
completed module. If unfinished, leave the task open and record the exact
remaining descriptor-only legality path and producer boundary to fix next.
