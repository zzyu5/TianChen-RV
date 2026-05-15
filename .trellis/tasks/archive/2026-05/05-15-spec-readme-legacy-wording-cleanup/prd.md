# Spec/README legacy wording cleanup

## Goal

Delete or rewrite stale repository wording that still legitimizes
descriptor-driven computation, descriptor-backed family modules, selected
descriptors, descriptor-to-source generation, scalar/offload descriptor
examples, direct C semantic exporters, or independent-backend framing as valid
TianChen-RV architecture.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule. The round is documentation, prompt, and test wording cleanup only.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `0938701`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- `.trellis/spec/index.md` defines TianChen-RV as a unified TCRV RISC-V MLIR
  and rejects descriptor-driven computation as the long-term architecture.
- `.trellis/spec/architecture/design-boundaries.md` rejects independent backend
  dialect framing and descriptor-driven microkernel/exporter frameworks.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says RVV is an extension
  family inside one TCRV system and direct C / descriptor-derived RVV routes are
  deleted or fail-closed until a materialized MLIR EmitC route exists.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` says direct
  descriptor-to-C string export and direct runtime-callable C compute-body
  exporters are not the architecture.
- The archived previous task
  `.trellis/tasks/archive/2026-05/05-15-rvv-selected-source-metadata-quarantine-deletion/prd.md`
  deleted selected-source/source-kind metadata as RVV legality, planning,
  dialect, test, and spec authority.
- The workspace journal records recent deletion rounds for selected-source
  metadata, RVV descriptor authority, RVV body-verifier descriptors, RVV
  smoke-probe descriptor residue, and offload descriptor-only target artifacts.
- `README.md` still contains positive or stale wording for a deleted exported
  RVV smoke-probe command, scalar `lowering_descriptor`, and offload runtime
  descriptor export examples.

## Requirements

- Rewrite README/spec/prompt/test wording so descriptors are not described as
  transition architecture, compatibility aid, migration aid, bounded production
  slice, semantic source, selected production input, or route authority.
- Rewrite direct C exporter wording so old RVV/scalar/dispatch direct C paths
  appear only as deleted legacy routes or fail-closed residue, never as current
  compute authority.
- Rewrite scalar/offload examples that present descriptor artifacts or
  descriptor metadata as valid active production output.
- Keep extension families framed as TCRV extension families inside one unified
  system, not independent backends or backend dialects.
- Keep `tcrv.exec.kernel` framed as execution envelope / callable execution
  plan envelope, not mathematical compute IR or hardware IR body.
- If test names, comments, or fixture text protect old descriptor/direct-export
  logic, delete or rewrite that wording within this owner.
- Do not modify compiler implementation, lowering, plugin code, registries,
  runtime behavior, helper scripts, or new architecture.
- Do not add compatibility layers, descriptor tests, negative-test frameworks,
  evidence matrices, helper wrappers, or replacement routes.

## Acceptance Criteria

- [x] README wording no longer tells users to export a deleted RVV smoke-probe
      C route or present scalar/offload descriptors as active compute or
      artifact authority.
- [x] `.trellis/spec/` wording no longer presents descriptors, selected
      descriptors, descriptor-backed modules, descriptor-to-source routes, or
      direct C semantic exporters as valid architecture.
- [x] `scripts/codex_serial_supervisor_prompt.md` remains aligned with the
      deletion-campaign red lines or is tightened if scans expose stale
      legitimacy language.
- [x] Test comments, names, and fixture text found by bounded scans no longer
      legitimize descriptor/direct-export/independent-backend paths; remaining
      occurrences are historical, negative, deletion-target, or non-semantic
      packaging references.
- [x] Any remaining old code names are clearly described as deleted historical
      residue, fail-closed metadata, or non-semantic packaging rather than
      production compute authority.
- [x] Focused bounded ref-scans report the remaining `descriptor`,
      `lowering_descriptor`, `RVVBinaryDescriptor`,
      `RVVBinaryFamilyRegistry`, `descriptor-backed`, `selected descriptor`,
      `migration aid`, `compatibility`, `transition`, `direct C exporter`,
      `runtime-callable-c-source`, `independent backend`, `backend dialect`,
      `scalar descriptor`, and `offload descriptor` references truthfully.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass.

## Non-goals

- No compiler implementation changes.
- No lowering, EmitC, plugin, registry, core pass, runtime, or target behavior
  changes.
- No new architecture or rebuild path.
- No helpers, wrappers, compatibility paths, negative-test frameworks,
  descriptor tests, or evidence matrices.
- No broad build/test matrix unless a test file is renamed or materially
  changed.
- No `ssh rvv` runtime, correctness, or performance claim.
- No Trellis task-state cleanup beyond normal create/finish/archive lifecycle.

## Minimal Evidence

- Bounded ref-scans over README, `.trellis/spec/`,
  `scripts/codex_serial_supervisor_prompt.md`, and directly relevant tests for:
  `descriptor`, `lowering_descriptor`, `RVVBinaryDescriptor`,
  `RVVBinaryFamilyRegistry`, `descriptor-backed`, `selected descriptor`,
  `migration aid`, `compatibility`, `transition`, `direct C exporter`,
  `runtime-callable-c-source`, `independent backend`, `backend dialect`,
  `scalar descriptor`, and `offload descriptor`.
- Markdown/text sanity checks if available.
- `git diff --check`.
- `git diff --cached --check`.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-15-spec-readme-legacy-wording-cleanup`.
- Focused lit/unit checks only if test files are renamed or materially changed.

## Technical Notes

- Descriptor-like terms can remain only when they are explicitly historical,
  fail-closed, deleted-route residue, or non-semantic packaging. They must not
  act as computation, lowering, route, ABI, source, or artifact authority.
- Offload handoff can remain as metadata handoff semantics, but descriptor-only
  target artifact output should not be described as active production behavior.
- Current intended compiler path remains extension family ops -> EmitC ->
  intrinsic/vendor builtin/runtime C/C++ -> native compiler.
- This PRD intentionally excludes implementation rewiring because the owner is
  documentation, prompt, README, spec, and test wording cleanup.

## Completion Evidence

- Rewrote `README.md` so it no longer instructs users to export the deleted RVV
  smoke-probe C route, no longer presents scalar `lowering_descriptor` strings
  as active compute authority, and no longer presents offload descriptor-only
  target artifact export as supported production behavior.
- Rewrote active `.trellis/spec/` wording so descriptor/direct-export paths are
  historical residue, deletion targets, or fail-closed implementation debt, not
  transition architecture, migration evidence, compatibility aid, semantic
  source, production input, or evidence authority.
- Replaced descriptor-as-current-input wording in plugin protocol, variant
  pipeline, core dialect, RVV, scalar, capability-model, testing, and
  lowering/runtime specs with typed extension-family body, selected-path
  metadata, or fail-closed residue language.
- Updated C++ test failure messages/comments that named descriptor-backed or
  selected-descriptor paths so they describe typed-family fixtures, selected
  metadata, legacy wrappers, or deleted-route residue instead.
- Added no compatibility layer, helper wrapper, descriptor test, negative-test
  framework, implementation code, lowering route, plugin route, runtime
  behavior, or new architecture.
- Focused residual scan for `migration aid`, `transition architecture`,
  `transition slices`, `bounded descriptors`, `plugin-specific descriptor`,
  `descriptor-backed`, `selected descriptor`, `supported descriptor emission`,
  `scalar descriptor`, `offload descriptor`, `runtime handoff descriptor`, and
  `generated self-check harness` reports only deleted/fail-closed/historical
  wording, negative tests, or explicitly forbidden old-route references.
- Focused C++ build passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-lowering-boundary-test
  tianchenrv-target-artifact-export-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  `./build/bin/tianchenrv-rvv-binary-planning-test`,
  `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`,
  `./build/bin/tianchenrv-rvv-lowering-boundary-test`, and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- `git diff --check`, `git diff --cached --check`, and
  `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-15-spec-readme-legacy-wording-cleanup` passed.
- No local `markdownlint` or `mdl` command was available.
