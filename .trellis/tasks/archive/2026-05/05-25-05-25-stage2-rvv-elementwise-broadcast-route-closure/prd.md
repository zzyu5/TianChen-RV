# Stage2 RVV elementwise/broadcast route closure

## Goal

Close one bounded Stage 2 RVV elementwise/broadcast route path on the corrected typed `tcrv_rvv` surface by moving the scalar-broadcast elementwise statement sequence into the RVV-owned migrated statement-plan boundary. The representative proof path for this round is `scalar_broadcast_sub`: selected `tcrv.exec` RVV variant -> typed/pre-realized `tcrv_rvv` body with explicit scalar splat source -> RVV plugin realization/facts/statement plan -> provider-built `TCRVEmitCLowerableRoute` -> common EmitC/target artifact mirrors -> focused generated-bundle and `ssh rvv` correctness evidence if executable behavior is claimed.

## What I already know

* Hermes' brief asks for the remaining elementwise/broadcast capability boundary to be explicit typed `tcrv_rvv` and RVV-plugin facts, not inferred from route names, helper names, ABI strings, artifact metadata, or harness constants.
* The previous completed task closed plain `macc_add` through RVV-owned statement planning and provider route construction.
* Current repo inspection shows scalar-broadcast elementwise already has typed `tcrv_rvv.splat` / `tcrv_rvv.binary` surfaces, pre-realized selected-body realization, route-family plans, operand-binding facts, explicit and pre-realized target fixtures, and generated-bundle dry-run support for add/sub/mul.
* The active code path still treats only `ScalarBroadcastAdd` as an elementwise arithmetic statement-plan consumer. `ScalarBroadcastSub` and `ScalarBroadcastMul` have route-family/operand-binding facts but fall through to provider-local generic statement assembly.
* The bounded module behavior for this round is therefore to make scalar-broadcast elementwise statement planning family-owned for the existing typed scalar-broadcast route family, with representative lit/evidence centered on `scalar_broadcast_sub`.
* The compiler implementation must stay in C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck. Python may only support generated-bundle tooling/evidence.
* `tcrv.exec` owns envelope and ABI/runtime role declaration only. It must not own RVV scalar-broadcast compute semantics.
* The RVV plugin owns scalar splat/broadcast legality, selected-body realization, route-family plan verification, operand binding, statement planning, intrinsic mapping, and fail-closed diagnostics. Common EmitC/export must remain neutral.

## Assumptions

* No new dialect op is needed because `tcrv_rvv.splat`, generic vector type, and `tcrv_rvv.binary {kind = "sub"}` already exist and are enough for the representative path.
* It is acceptable to generalize the existing scalar-broadcast statement-plan predicate from add-only to the existing scalar-broadcast elementwise family, while focused acceptance and evidence remain bounded to `scalar_broadcast_sub`.
* Runtime `ssh rvv` evidence should use the existing generated-bundle ABI/e2e script for the pre-realized `scalar_broadcast_sub` fixture. If the remote environment is unavailable, the task must document the exact blocker and avoid executable correctness claims.

## Requirements

* Treat `scalar_broadcast_sub` as a scalar-broadcast elementwise statement-plan consumer when the selected body structurally carries `tcrv_rvv.splat` feeding `tcrv_rvv.binary {kind = "sub"}` under SEW32/LMUL m1 policy and runtime VL facts.
* Preserve existing typed route-family facts: operation kind, lhs vector role, RHS scalar role, output role, runtime count role, scalar splat leaf, arithmetic leaf, element dtype, SEW, LMUL, policy, and runtime AVL/VL plan.
* Provider route construction must consume the migrated RVV-owned statement plan for the scalar-broadcast path before provider-local generic statement assembly.
* Fail closed before route/artifact authority if the scalar-broadcast route lacks route-family plan facts, operand-binding facts, scalar splat leaf, vector load/store leaf, arithmetic leaf, `setvl` leaf, scalar binding, lhs/out/n bindings, or typed structure consistency.
* Keep artifact metadata as explicit mirrors only; do not make route id, status, artifact name, ABI string, test name, descriptor residue, or harness constant into scalar-broadcast authority.
* Keep common EmitC and target artifact mechanics neutral; no scalar-broadcast semantics move into common plumbing.

## Acceptance Criteria

* [x] `scalar_broadcast_sub` reaches provider construction through `RVVSelectedBodyElementwiseArithmeticRouteStatementPlan` and the migrated statement-plan boundary rather than provider-local fallback statement assembly.
* [x] The statement plan carries `setvl`, lhs load, RHS scalar splat, `vsub`, and store steps derived from typed body/config/runtime facts and route-family/operand-binding facts.
* [x] Provider/unit tests prove the scalar-broadcast sub statement-plan flags, steps, and migrated provider route are present and fail closed when required scalar-broadcast plan or facts are stale/missing.
* [x] Representative FileCheck and generated-bundle dry-run evidence for explicit/pre-realized `scalar_broadcast_sub` shows typed `tcrv_rvv.splat`, `tcrv_rvv.binary {kind = "sub"}`, scalar-broadcast route-family mirror labels, and no descriptor/direct-C/source-export/stale `tcrv_rvv.i32_*` authority.
* [x] Generated artifact mirrors keep explicit mirror labels such as `provider_supported_mirror` and remain downstream of provider route construction.
* [x] Focused `ssh rvv` correctness evidence covers runtime counts 0, 7, 16, and 23 for rhs scalars -37 and 91 on the representative scalar-broadcast sub path.
* [x] Bounded scans over touched RVV planning/provider/test/script/fixture files show no new metadata-, name-, descriptor-, ABI-, harness-, source-front-door-, or legacy i32-derived scalar-broadcast authority. Hits are limited to negative/implicit-check text, stale-route rejection tests, and existing fail-closed legacy i32 checks.
* [x] `git diff --check` passes.
* [x] Focused C++/lit checks pass, and `check-tianchenrv` passes.

## Definition of Done

* PRD and Trellis task context are truthful.
* Relevant specs have been read before implementation.
* Code/tests implement only the bounded statement-plan route closure described above.
* Focused checks and self-repair have been run.
* Task status is updated and archived if complete.
* One coherent commit is created if the task reaches completion.

## Out of Scope

* Broad arithmetic coverage beyond the existing scalar-broadcast elementwise family surface.
* New dtype or LMUL matrices, high-level Linalg/Vector frontend lowering, global performance tuning, source-front-door positive routes, dashboards, one-op-per-intrinsic wrapper growth, or compatibility wrappers preserving legacy i32 authority.
* New dtype-prefixed helper ops such as `tcrv_rvv.i32_broadcast_*`, `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, or `tcrv_rvv.i32_macc`.
* Moving scalar-broadcast elementwise semantics into common EmitC/export or target artifact plumbing.

## Technical Notes

* Task source: Hermes Direction Brief supplied on 2026-05-25.
* Required specs read: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
* Relevant prior PRD read: `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-contraction-multiply-add-route-closure/prd.md`.
* Current implementation target after repo inspection: extend the existing elementwise arithmetic statement-plan consumer from add-only scalar-broadcast to the existing scalar-broadcast elementwise route-family consumer, then add focused C++/FileCheck/generated-bundle evidence for `scalar_broadcast_sub`.
* Completed implementation: `RVVSelectedBodyElementwiseArithmeticRouteStatementPlan` now treats the existing scalar-broadcast elementwise route family as the statement-plan consumer family, not add-only.
* Spec update: `.trellis/spec/extension-plugins/rvv-plugin.md` now records scalar-broadcast `Add`/`Sub`/`Mul` as the long-term elementwise arithmetic statement-plan boundary.

## Checks Run

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tcrv-opt test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-sub-materialization.mlir --tcrv-materialize-emitc-lowerable-routes | /usr/lib/llvm-20/bin/FileCheck test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-sub-materialization.mlir`
* `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-sub.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-sub.mlir --check-prefix=PLAN`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_scalar_broadcast_sub_dry --run-id pre-realized-scalar-broadcast-sub --overwrite --op-kind scalar_broadcast_sub --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'scalar-broadcast-sub|rvv-generic-stage2-scalar-broadcast-sub'` from `build/test`, 4/4 passed
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_scalar_broadcast_sub_ssh --run-id pre-realized-scalar-broadcast-sub-ssh-rvv --overwrite --op-kind scalar_broadcast_sub --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv`
* `cmake --build build --target check-tianchenrv -j2`, 367/367 passed
* `git diff --check`
