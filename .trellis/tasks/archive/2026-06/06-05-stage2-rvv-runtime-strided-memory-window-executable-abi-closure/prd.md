# Stage2 RVV runtime-strided memory-window executable ABI closure

## Goal

Close executable ABI evidence for the runtime-strided base-memory route
mirrors added by `75cb53e7`. Starting from selected pre-realized `tcrv.exec`
RVV variants with typed `tcrv_rvv` strided-load/unit-store and
unit-load/strided-store body facts, this task proves that provider-derived
memory form and stride-source facts survive selected-body realization,
emission planning, common EmitC materialization, generated header/object bundle
export, external ABI harness compilation, and real `ssh rvv` execution.

This is a runtime/evidence closure for an already production-supported route
boundary. It must not expand RVV route coverage, move semantic authority into
the ABI harness, or let artifact names, ABI strings, route ids, metadata
mirrors, C strings, intrinsic spellings, or fixture names define RVV memory
semantics.

## What I Already Know

- The repository started on `main` with latest commit `75cb53e7 rvv: add
  runtime-strided memory route mirrors`; the worktree was clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
- `.trellis/spec/index.md` requires the RVV authority chain to flow through a
  selected typed `tcrv_rvv` body, RVV plugin legality/realization/provider,
  common `TCRVEmitCLowerableRoute`, neutral EmitC materialization, target
  artifact, and `ssh rvv` evidence for runtime correctness claims.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says base memory movement
  statement plans must come from verified typed body/config/runtime facts,
  materialization facts, memory operand-binding facts, and RVV-owned provider
  plans. Generated-bundle evidence must expose
  `base_memory_movement_boundary`, ordered statement-plan callees, and
  mirror-only route metadata.
- `.trellis/spec/lowering-runtime/emitc-route.md` says provider operand
  binding summaries must cover every exported runtime ABI parameter and that
  Common EmitC/target export must carry provider payloads unchanged, not infer
  RVV memory semantics.
- The archived route-foundation task
  `.trellis/tasks/archive/2026-06/06-05-06-05-stage2-rvv-runtime-strided-memory-window-route-foundation/`
  completed route-supported and target-header mirror support for both
  `strided_load_unit_store` and `unit_load_strided_store`, while explicitly
  leaving executable correctness out of scope.
- Existing fixtures
  `test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-unit-load-strided-store.mlir`
  already prove selected-boundary realization, provider-derived route metadata,
  and generated header artifacts for the two proof shapes.
- `scripts/rvv_generated_bundle_abi_e2e.py` already has op expectations,
  dry-run generation, metadata checks, C harness generation, and runtime
  stride/count controls for both proof shapes.
- Initial dry-run with both proof shapes, counts `1,17,257`, and byte strides
  `4,8,12` succeeded and produced per-op `base_memory_movement_boundary`
  evidence. The generated harness already checks noncontiguous source or
  destination behavior and destination tail sentinel preservation.
- Repository inspection found one bounded evidence gap: the generated harness
  did not explicitly snapshot and compare source buffers after the RVV call.
  Source preservation is applicable to both proof shapes, so this task should
  add only minimal neutral harness checks for that condition.

## Requirements

- Exercise both existing proof shapes unless a real blocker is found:
  `strided_load_unit_store` and `unit_load_strided_store`.
- Use the existing typed body/config/runtime facts as authority. The harness
  may verify behavior but must not choose route support, memory form, stride
  side, dtype, schedule, policy, or intrinsic spelling.
- Dry-run evidence must prove provider-derived memory-form and stride-source
  facts before execution:
  `base_memory_movement_boundary`, route metadata, runtime ABI order,
  provider-supported mirror, source/destination memory forms, stride-source
  mirror, and ordered statement-plan callees.
- Runtime evidence must compile and run the generated bundle through an
  external C ABI harness on `ssh rvv`.
- Runtime cases must cover multiple element counts including a VL/tail-crossing
  count, multiple positive byte strides, nontrivial source spacing or
  destination spacing, destination tail/skipped-slot sentinel preservation, and
  source preservation where applicable.
- If harness code changes are needed, keep them neutral evidence tooling:
  no compiler core implementation in Python, no descriptor-driven computation,
  no route-id/artifact-name/ABI-string/C-string semantics.
- If production compiler code is not changed, focused script self-test,
  dry-run, fixture checks, and runtime evidence are sufficient; production C++
  test targets are required only if production code changes.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` truthfully describe this
      executable-closure task and its spec basis.
- [x] Generated-bundle dry-run succeeds for both pre-realized selected-body
      proof shapes with counts `1,17,257` and byte strides `4,8,12`.
- [x] Per-op evidence records `base_memory_movement_boundary` authority as
      provider-derived typed `tcrv_rvv` body/config/runtime facts.
- [x] Per-op evidence records ordered statement-plan callees:
      `vlse32`/unit-store for `strided_load_unit_store` and
      unit-load/`vsse32` for `unit_load_strided_store`.
- [x] Generated harness checks exact host/reference values, multiple positive
      stride values, noncontiguous memory behavior, destination tail or skipped
      sentinel preservation, and source preservation.
- [x] `ssh rvv` compile/run succeeds for both proof shapes with the same
      counts and stride variants, and the output records PASS for each op.
- [x] Focused fixture checks pass if the fixtures are touched; otherwise the
      existing fixture paths are exercised by the generated-bundle script.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
      script `--self-test` pass if the script changes.
- [x] Bounded old-authority/q-name scan over touched files shows no new
      positive legacy i32 route authority, q8/q4 route authority, descriptor,
      source-front-door, route-id, artifact-name, ABI-string, intrinsic, or
      C-string semantics.
- [x] `git diff --check`, `git diff --cached --check`, and final clean
      `git status --short` are recorded.

## Out Of Scope

- New route coverage beyond runtime-strided base-memory executable proof.
- q8/q4/llama benchmarks or low-precision route expansion.
- ProviderSpec/model-name route authority.
- Computed-mask, contraction, dequant, clamp, MAcc, reduction, segment, gather,
  scatter, or dtype/LMUL clone evidence as the primary deliverable.
- High-level Linalg/Vector/StableHLO frontend or source-front-door positive
  route work.
- One-intrinsic wrapper dialects or dtype-prefixed helper families.
- Compatibility wrappers preserving legacy i32m1 authority.
- Common EmitC/export or ABI harness logic that invents RVV memory semantics,
  stride units, dtype, schedule, policy, or body shape.
- Broad smoke matrices, dashboards, prompt-only, helper-only, or report-only
  completion.

## Technical Approach

1. Read the specified specs, archived route-foundation task, fixtures, route
   planning files, target support files, and ABI e2e script.
2. Run a focused generated-bundle dry-run for both proof shapes to establish
   current behavior and locate any executable evidence gaps.
3. If needed, add the smallest neutral harness support for source preservation
   while keeping route authority in provider-derived facts.
4. Re-run py_compile/self-test, focused dry-run, and generated fixture checks.
5. Run the same proof shapes on `ssh rvv` and record compile/run correctness
   evidence.
6. Update Trellis notes, archive the task if complete, and create one coherent
   commit.

## Completion Evidence

- Task context:
  `.trellis/tasks/06-05-stage2-rvv-runtime-strided-memory-window-executable-abi-closure/`
  contains the PRD plus implementation/check spec context.
- Harness repair:
  `scripts/rvv_generated_bundle_abi_e2e.py` now snapshots and verifies the
  read-only source buffer in both generated base-memory harnesses:
  byte-for-byte source preservation for `strided_load_unit_store` and
  element-wise source preservation for `unit_load_strided_store`. The generated
  PASS markers now include `source_preserved`.
- Spec update:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now records the executable
  base-memory route harness contract: host/reference comparison over multiple
  counts and positive strides, destination tail or skipped-slot sentinel
  preservation, and read-only source preservation where applicable, with those
  checks explicitly scoped as external ABI evidence rather than route
  authority.
- Dry-run evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind strided_load_unit_store --op-kind unit_load_strided_store --runtime-count 1 --runtime-count 17 --runtime-count 257 --stride-bytes 4 --stride-bytes 8 --stride-bytes 12 --artifact-root artifacts/tmp/rvv_runtime_strided_memory_executable_closure --run-id dry-run-source-preserved --overwrite`
  succeeded with artifact root
  `artifacts/tmp/rvv_runtime_strided_memory_executable_closure/dry-run-source-preserved`.
- Provider-derived boundary evidence:
  `strided_load_unit_store` records authority as provider-derived typed
  `tcrv_rvv` body/config/runtime facts, source form `strided-load`,
  destination form `unit-stride-store`, source stride
  `runtime_abi:stride_bytes`, and ordered loop callees
  `__riscv_vsetvl_e32m1`, `__riscv_vlse32_v_i32m1`,
  `__riscv_vse32_v_i32m1`.
- Provider-derived boundary evidence:
  `unit_load_strided_store` records authority as provider-derived typed
  `tcrv_rvv` body/config/runtime facts, source form `unit-stride-load`,
  destination form `strided-store`, destination stride
  `runtime_abi:dst_stride_bytes`, and ordered loop callees
  `__riscv_vsetvl_e32m1`, `__riscv_vle32_v_i32m1`,
  `__riscv_vsse32_v_i32m1`.
- Real RVV evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind strided_load_unit_store --op-kind unit_load_strided_store --runtime-count 1 --runtime-count 17 --runtime-count 257 --stride-bytes 4 --stride-bytes 8 --stride-bytes 12 --artifact-root artifacts/tmp/rvv_runtime_strided_memory_executable_closure --run-id ssh-rvv-source-preserved --overwrite`
  succeeded with artifact root
  `artifacts/tmp/rvv_runtime_strided_memory_executable_closure/ssh-rvv-source-preserved`.
  The remote output reports PASS for both op kinds across all nine
  count/stride cases and includes `source_preserved`.
- Checks run:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  the dry-run command above; the `ssh rvv` command above; focused jq evidence
  inspections for both per-op `base_memory_movement_boundary` summaries;
  bounded old-authority/q-name diff scan; `git diff --check`;
  `git diff --cached --check`.
- Final clean status is recorded after the final commit in the worker report.
