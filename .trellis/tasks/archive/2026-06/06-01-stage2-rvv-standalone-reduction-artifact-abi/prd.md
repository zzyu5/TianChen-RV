# Stage2 RVV standalone-reduction selected-body artifact ABI

## Goal

Carry exactly one existing supported standalone reduction selected body from a
selected `tcrv.exec` RVV variant through RVV plugin-local selected-body
realization, provider-owned route facts, common EmitC materialization, RVV
target artifact bundle generation, and real `ssh rvv` scalar-result ABI
correctness evidence.

This round is the scalar-result ABI complement to the previous
`widening_macc_add` selected-body-to-artifact closeout. It must prove that
runtime `n`/AVL, source memory, scalar accumulator seed, scalar result output,
typed standalone reduction semantics, and provider-derived identity/seed facts
survive into the generated bundle and external runtime consumer without route
names, descriptors, common export inference, or artifact metadata becoming
authority.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes direction brief.
* The repository started clean on `main` at
  `2ffe9cc1 rvv: close selected-body artifact abi evidence`.
* The previous selected-body artifact ABI closeout proved
  `widening_macc_add` through generated bundle export and real `ssh rvv`
  counts `0,1,16,17,257`.
* The previous standalone reduction realization task proved that
  `typed_standalone_reduce_pre_realized_body` is consumed by the RVV standalone
  reduction realization owner into explicit `setvl`, `with_vl`, source `load`,
  `standalone_reduce`, and scalar-result `store` structure before provider
  route construction.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require RVV semantics,
  dtype/config, route facts, intrinsic mapping, scalar result ABI, and artifact
  metadata mirrors to remain provider/plugin-owned.
* `scripts/rvv_generated_bundle_abi_e2e.py` already contains standalone
  reduction op-kind support and generated-bundle verification helpers, but
  this round must check whether the production/export/runtime boundary is
  sufficient for scalar-result standalone reduction evidence.

## Requirements

* Use one existing supported standalone reduction selected body, initially
  `standalone_reduce_add`, unless repository evidence shows another existing
  standalone reduction body is the only safe candidate.
* Preserve the authority chain:

  ```text
  selected tcrv.exec RVV variant
    -> typed pre-realized standalone reduction body
    -> RVV selected-body realization owner
    -> realized tcrv_rvv setvl/with_vl/load/standalone_reduce/store body
    -> RVV provider-owned route facts
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV target artifact bundle
    -> external scalar-result ABI consumer
    -> ssh rvv evidence
  ```

* The positive path must start from the selected-boundary pre-realized body and
  must realize before route construction.
* The provider route facts must carry runtime `n`/AVL, source input role,
  accumulator seed role/layout, scalar result role/layout, operation kind,
  dtype/config, scalar result vector facts, route operand binding, statement
  plan, and required RVV headers/intrinsics.
* Common EmitC/materialization and target export may consume provider-built
  route payloads and mirrors, but must not infer standalone reduction semantics
  from route ids, artifact names, manifests, C strings, or test names.
* The generated bundle and harness must preserve scalar-result ABI behavior:
  source memory input, scalar seed/identity input, scalar output storage,
  runtime `n`, and no writes outside the scalar output contract.
* Direct pre-realized route-entry or artifact export for standalone reduction
  must fail closed before route construction/export.
* If the e2e harness cannot express scalar-result ABI yet, repair that single
  production/export/runtime boundary rather than adding a broad matrix.
* Keep all changes bounded to the module owner files, directly required tests,
  task context, and workspace notes.

## Acceptance Criteria

* [x] The PRD and context files are created and validated by Trellis.
* [x] Focused evidence shows the chosen standalone reduction pre-realized body
  is realized by the RVV plugin selected-body owner before route construction.
* [x] Focused route/provider evidence shows the realized standalone reduction
  route carries runtime `n`/AVL, source input, scalar accumulator seed,
  scalar result output, operation kind, dtype/config, scalar result vector
  facts, operand binding, statement plan, and provider-derived leaves.
* [x] Generated target artifact bundle evidence shows runtime `n`, source
  memory role, scalar result role, accumulator seed/identity facts, route
  metadata mirrors, and header/object ABI survive export for the chosen
  selected body.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind standalone_reduce_add` passes for
  counts `0,1,16,17,257` or equivalent representative counts that include
  zero, one, a VL-boundary count, a tail count, and a larger count.
* [x] Direct pre-realized route-entry mode for the chosen standalone reduction
  fails closed with the expected selected-body realization diagnostic.
* [x] Real `ssh rvv` compile/run correctness evidence passes for the same
  representative counts if runtime correctness is claimed.
* [x] Focused C++ build/test targets for target artifact export and RVV plugin
  pass after the change.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes.
* [x] A bounded old-authority scan over touched plugin/provider/materializer/
  target/script/test/spec/task files classifies any remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] `git diff --check` passes.
* [x] The task is finished/archived and one coherent commit is created, or the
  exact blocker and continuation point are recorded.

## Definition of Done

* The executable compiler path for the chosen standalone reduction selected
  body is proven through generated RVV target artifact bundle export and
  external scalar-result ABI execution on `ssh rvv`.
* Runtime correctness claims are backed by real `ssh rvv` output.
* No new reduction family, dtype/LMUL clone batch, source-front-door path,
  descriptor-driven computation path, common EmitC semantic inference, or
  broad smoke matrix is added.
* Task status, workspace journal, archived task record, final commit, and final
  report truthfully describe what was completed and what was not.

## Out of Scope

* New reduction families or broad coverage expansion.
* Contraction, MAcc, computed-mask standalone reduction, runtime-scalar masked
  standalone reduction, or high-level Linalg frontend authority.
* Dtype/LMUL clone batches, dashboards, performance claims, tuning databases,
  readiness state machines, source-front-door positive routes, descriptor
  residue, direct-C/source-export routes, or common EmitC semantic inference.
* Treating route ids, artifact names, manifests, exact intrinsic spellings,
  status fields, emission-plan metadata, or test names as reduction kind,
  dtype, identity, scalar-result, policy, ABI, or route authority.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-selected-body-artifact-abi-closeout/prd.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary/prd.md`.
* Initial module owner files from the brief:
  `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `scripts/rvv_remote_probe.py`, and focused tests.

## Completion Notes

* The chosen selected body was `standalone_reduce_add`.
* The existing production compiler/export path already carried the selected
  standalone reduction body through RVV selected-body realization, provider
  route facts, common EmitC, target artifact bundle export, and external
  scalar-result ABI execution. No production C++ owner/provider/materializer/
  target changes were required.
* Added a durable `scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  regression for the retired direct pre-realized `standalone_reduce_add`
  route-entry mode. The diagnostic must name `standalone_reduce_add` and the
  required public selected lowering-boundary producer.
* Final dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-standalone-reduction-artifact-abi/final-dry-run/20260601T140023Z`.
  The evidence records `input_mode = pre-realized-selected-body`, materialized
  selected body, header prototype
  `void tcrv_emitc_pre_realized_body_standalone_reduce_add_kernel_pre_realized_body_rvv_standalone_reduce_add(const int32_t *lhs, const int32_t *acc, int32_t *out, size_t n);`,
  runtime ABI order `lhs,acc,out,n`, scalar-result runtime boundary
  `scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1`,
  accumulator seed source `acc[0]`, store target `out`, and runtime counts
  `0,1,16,17,257`.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-standalone-reduction-artifact-abi/final-direct-fail/20260601T140023Z`.
  The command exited 1 with the expected retired direct route-entry diagnostic
  before target bundle export.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-standalone-reduction-artifact-abi/final-ssh-rvv/20260601T140023Z`.
  Counts `0,1,16,17,257` passed for seeds `-11` and `17`, with scalar outputs
  checked and tail preservation reported by the external generated-bundle ABI
  harness.
* Bounded old-authority scan:
  new script diff contains no matches for the requested legacy-authority
  strings. Task-context hits are guardrail/acceptance text. Relevant existing
  module/test hits are pre-existing fail-closed legacy tests, descriptor
  rejection checks, provider-derived intrinsic leaf evidence, or selected-route
  consistency diagnostics; no new positive legacy route authority was added.
* Spec update decision: no `.trellis/spec/` update was needed. The existing RVV
  plugin, EmitC route, and MLIR testing contract already cover this behavior;
  this round only made the evidence/regression for an existing path durable.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-01-stage2-rvv-standalone-reduction-artifact-abi`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-standalone-reduction-artifact-abi/final-dry-run`
* [x] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `standalone_reduce_add`.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-standalone-reduction-artifact-abi/final-ssh-rvv`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] Bounded old-authority scan over touched script/task files and relevant
  owner/provider/materializer/target/test files.
* [x] `rtk git diff --check`
