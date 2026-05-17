# Binary self-check artifact residue erasure

## Goal

Delete the obsolete binary self-check expectation API and directly related
target artifact residue. After this round, public target headers, target
exporters, tools, and active tests must not preserve self-check harness compute
semantics as a target artifact concept.

This is a Wrong Logic Deletion Campaign round. It is deletion/refactor-only:
remove stale self-check authority before any rebuild or new executable route.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD
  `22c6207 plugin: erase non-plugin target artifact placeholder`; worktree
  clean; no `.trellis/.current-task` existed.
- The previous completed target cleanup removed the non-plugin built-in target
  artifact placeholder lane and left target artifact registration bundle-only.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` states the
  active route is extension family ops to materialized EmitC to C/C++ emitter
  to native compiler, and that descriptor/direct-C/self-check object routes are
  deletion targets or future rebuild gaps.
- `include/TianChenRV/Target/BinarySelfCheckExpectation.h` currently defines
  `BinarySelfCheckArithmeticKind`, `BinarySelfCheckExpectation`,
  `makeBinarySelfCheckExpectationError`,
  `getRuntimeABIPointeeScalarCType`, and
  `buildBinarySelfCheckExpectationFromRuntimeABI`.
- That header derives add/sub/mul expectation semantics and scalar expression
  text from runtime ABI role metadata, including wording about self-check
  harness emission.
- The task brief reports no direct C++ or target-test consumers were found; the
  implementation must verify this against the current checkout before deleting.

## Boundaries

- Start from `include/TianChenRV/Target/BinarySelfCheckExpectation.h` and
  directly related target artifact surfaces.
- Delete the header if current scans confirm it has no active non-wrong
  consumer.
- If hidden consumers appear, delete or rewrite those consumers away from
  self-check compute/harness authority instead of restoring the helper.
- Remove active references, comments, docs, tests, build/install surfaces, or
  artifact expectations that preserve self-check compute-body generation as a
  target artifact concept.
- Remaining self-check mentions, if any, must be prohibitive historical
  deletion wording or external evidence notes, not active compiler API.

## Non-Goals

- No replacement self-check generation.
- No new target artifact routes, header/bundle rebuilds, RVV/Template/
  TensorExtLite expansion, source-export paths, direct C semantic exporters,
  descriptor adapters, compatibility wrappers, or legacy mode.
- No Python compiler-core logic.
- No `ssh rvv` runtime/correctness/performance evidence as the main result.
- No broad unrelated refactors or test matrix expansion beyond focused
  deletion validation unless a focused failure requires it.

## Requirements

- Delete public `BinarySelfCheckExpectation` API surfaces if they have no active
  non-wrong consumer.
- Remove any build/install exposure for the deleted header.
- Remove or rewrite active target/exporter/test/tool references to:
  `BinarySelfCheckExpectation`, `BinarySelfCheckArithmeticKind`,
  `buildBinarySelfCheckExpectationFromRuntimeABI`, self-check harness emission,
  self-check object routes, self-check route authority, runtime-callable C
  source export, and descriptor/direct-C self-check residue.
- Preserve the current bundle/plugin-owned target artifact route boundary.
- Do not introduce compatibility aliases or a renamed helper that carries the
  same self-check expectation semantics.

## Acceptance Criteria

- [x] No public `BinarySelfCheckExpectation` remains.
- [x] No public `BinarySelfCheckArithmeticKind` remains.
- [x] No public `buildBinarySelfCheckExpectationFromRuntimeABI` remains.
- [x] No active helper or comment still advertises self-check harness emission,
  self-check object output, or self-check target artifact route authority.
- [x] `include/TianChenRV/Target`, `lib/Target`, `tools/tcrv-translate`, and
  `test/Target` scans show no active self-check compute-body generation
  surface.
- [x] Specs and tests do not protect self-check compute-body generation.
  Remaining self-check mentions are prohibitive deletion wording or historical
  evidence notes only.
- [x] No replacement route, compatibility wrapper, descriptor adapter, direct-C
  source exporter, or Python compiler-core path is added.
- [x] Focused target artifact/export build or tests affected by the deletion
  pass.
- [x] `check-tianchenrv` is run if practical; otherwise the exact blocker is
  recorded.
- [x] `git diff --check` passes.

## Completion Evidence

- Deleted `include/TianChenRV/Target/BinarySelfCheckExpectation.h`.
- Removed the public target-layer API definitions:
  `BinarySelfCheckArithmeticKind`, `BinarySelfCheckExpectation`,
  `makeBinarySelfCheckExpectationError`,
  `getRuntimeABIPointeeScalarCType`, and
  `buildBinarySelfCheckExpectationFromRuntimeABI`.
- Removed self-check arithmetic expression selection and runtime ABI pointee
  type agreement logic from the public target header surface.
- Rewrote `.trellis/spec/testing/mlir-testing-contract.md` so the historical
  RVV+scalar dispatch self-check harness, self-check object, direct executable
  evidence bridge, explicit self-check source export, route-name dry-runs, and
  old bundle-mode caller generation are deleted test surfaces rather than
  active conditional testing guidance.
- No source, header, object, bundle, target route, compatibility wrapper,
  descriptor adapter, direct-C source exporter, Python compiler-core path, or
  replacement self-check generation was added.

## Checks

- [x] Trellis context validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-binary-self-check-artifact-residue-erasure`
  -> implement/check JSONL valid, 5 entries each.
- [x] Deleted public API scan:
  `rg -n "BinarySelfCheckExpectation|BinarySelfCheckArithmeticKind|buildBinarySelfCheckExpectationFromRuntimeABI" include/TianChenRV/Target lib/Target tools/tcrv-translate test/Target .trellis/spec/lowering-runtime .trellis/spec/testing .trellis/spec/plugin-protocol`
  -> no matches.
- [x] Self-check residue scan:
  `rg -n "SelfCheck|self-check|self check|self_check" include/TianChenRV/Target lib/Target tools/tcrv-translate test/Target .trellis/spec/lowering-runtime .trellis/spec/testing .trellis/spec/plugin-protocol`
  -> only prohibitive or historical-deletion spec wording remains.
- [x] Descriptor/direct-C/source-export focused scan:
  `rg -n "runtime-callable-c-source|source-export|direct-C|descriptor" include/TianChenRV/Target lib/Target tools/tcrv-translate test/Target`
  -> remaining matches are existing fail-closed residue rejection code/tests and
  negative `implicit-check-not` assertions, not active self-check route
  authority.
- [x] Focused build:
  `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test -j2`
  -> passed.
- [x] Focused C++ target artifact test:
  `./build/bin/tianchenrv-target-artifact-export-test`
  -> passed.
- [x] `git diff --check`
  -> passed.
- [x] `cmake --build build --target check-tianchenrv -j2`
  -> 122/122 lit tests passed.

## Minimal Evidence

- Focused deletion scans over:
  `include/TianChenRV/Target`, `lib/Target`, `tools/tcrv-translate`,
  `test/Target`, and relevant `.trellis/spec` files.
- Search strings must include:
  `BinarySelfCheckExpectation`, `BinarySelfCheckArithmeticKind`,
  `buildBinarySelfCheckExpectationFromRuntimeABI`,
  `self-check harness emission`, `self-check object`, `self-check route`,
  `runtime-callable-c-source`, `source-export`, `descriptor`, and `direct-C`.
- Focused build/test targets should cover target artifact/export surfaces, for
  example `tcrv-translate` and `tianchenrv-target-artifact-export-test`, plus
  the C++ target artifact test if present.
- Run `check-tianchenrv` if practical for final deletion-campaign confidence.

## Definition of Done

- Trellis task status and notes truthfully describe the deletion/refactor.
- Journal records exact obsolete surface removed, scans, checks, self-repair,
  archive status, and commit hash.
- One coherent commit is created when the task completes.

## Technical Notes

- Specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/implementation-stack/index.md`,
  `.trellis/spec/implementation-stack/compiler-stack-contract.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-05-17-builtin-non-plugin-target-artifact-placeholder-erasure/prd.md`.
- Direct starting file inspected:
  `include/TianChenRV/Target/BinarySelfCheckExpectation.h`.
