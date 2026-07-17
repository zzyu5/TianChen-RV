# Stage2 RVV runtime AVL/VL boundary closure

## Goal

Close one bounded Stage 2 runtime AVL/VL authority path for the existing
pre-realized `i64_add` RVV selected-body artifact. The round must prove that
the selected `tcrv.exec` runtime element-count ABI value is explicitly imported
into the typed `tcrv_rvv` body, consumed by `tcrv_rvv.setvl`, carried through
RVV-owned runtime control / operand-binding / statement-plan facts, emitted as
runtime `n` driven RVV setvl/intrinsic execution, exported through the generated
artifact ABI, and validated by one real `ssh rvv` correctness run across several
runtime counts.

## Direction Source

- Direction title: `Stage2 RVV runtime AVL/VL boundary closure`.
- Module owner: RVV plugin-owned runtime AVL/VL binding from selected
  `tcrv.exec` runtime parameters into typed `tcrv_rvv` setvl/route facts,
  provider emission, generated artifact ABI, and one `ssh rvv` correctness
  check.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `a6790b8f rvv: expose real typed config hardware evidence`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the current RVV authority chain to run
  through selected `tcrv.exec` variant, typed/realized `tcrv_rvv` body, RVV
  plugin legality / realization / route provider, common EmitC, target artifact,
  and real `ssh rvv` evidence when runtime correctness is claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says
  `tcrv.exec.runtime_param` / `mem_window` declare ABI roles only; selected
  `tcrv_rvv` body must explicitly bind/import ABI values and consume `%n`
  through `setvl`, loads/stores, compute, masks, reductions, or movement ops.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  consume provider-built `TCRVEmitCLowerableRoute` payloads neutrally and not
  synthesize runtime ABI from parameter names.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires target
  export to come from materialized EmitC over provider-built routes; metadata,
  route ids, artifact names, manifests, diagnostics, and test names are mirrors
  only.
- The archived typed-config closure task proved real `ssh rvv` compile/run for
  pre-realized `i64_add` counts `7,16,23`, but this round must make the
  runtime AVL/VL boundary itself explicit and reviewable.
- Existing source already has `RVVRuntimeAVLVLControlPlan`,
  `deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody`,
  `deriveRVVRuntimeAVLVLControlPlanForRealizedBody`, and i64 add generated
  bundle support. The likely implementation is a focused closure/evidence
  hardening round, not a broad route-family rewrite.

## Requirements

1. Keep the executable closure bounded to the already route-supported
   pre-realized `i64_add` path unless live repository evidence proves that path
   is unsafe or stale.
2. The selected input must visibly import a single runtime-element-count ABI
   value named `n`, and the realized typed `tcrv_rvv` body must consume that
   value as the `tcrv_rvv.setvl` AVL.
3. RVV planning/provider artifacts must expose runtime AVL/VL facts as
   provider-derived mirrors: runtime control plan id, runtime VL contract,
   runtime AVL source `runtime_abi:n`, `setvl`/`with_vl` placement, runtime ABI
   order, operand binding for `n`, and loop/pointer-advance facts.
4. Generated RVV C/C++ evidence must prove that the emitted loop calls
   `__riscv_vsetvl_e64m1(n - offset)` or equivalent runtime-remaining AVL form,
   uses the resulting runtime VL for load/add/store intrinsics, and does not use
   a harness-only constant as VL authority.
5. Generated artifact ABI evidence must expose honest runtime count ABI
   parameter metadata and the same runtime AVL/VL mirrors in the bundle/header.
6. Missing, ambiguous, stale, or inconsistent runtime AVL/VL ownership must
   fail closed before route/artifact authority. The focused negative coverage
   must include at least one wrong runtime `n` binding diagnostic.
7. Do not add broad operation-family coverage, reductions, contractions,
   high-level frontend lowering, source-front-door positive routes, legacy i32
   authority, one-op-per-intrinsic wrappers, dtype/LMUL clone batches,
   dashboards, or broad smoke matrices.
8. Do not derive AVL/VL from route ids, helper names, artifact names, test
   names, descriptor residue, manifests, or harness-only constants.
9. Do not move RVV semantics into common EmitC, target artifact plumbing, or
   Python compiler-core structures. Python may only harden evidence tooling and
   artifact parsing.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` reference the relevant RVV
      plugin, EmitC route, emission/runtime specs, testing policy, and previous
      archived task.
- [x] Focused lit/FileCheck coverage for pre-realized `i64_add` proves the
      realized body consumes the imported runtime `%n` in `tcrv_rvv.setvl`.
- [x] Focused lit/FileCheck coverage proves the emission-plan/header mirrors
      include runtime control plan, runtime VL contract, runtime AVL source,
      runtime ABI order, route operand binding for `n`, and multi-VL loop facts.
- [x] Focused EmitC materialization coverage proves emitted RVV C/C++ uses the
      runtime `n`/remaining AVL for `__riscv_vsetvl_e64m1` and threads the
      produced VL into i64 load/add/store intrinsics.
- [x] A fail-closed test still rejects missing or inconsistent runtime AVL
      binding before route/artifact authority, with an exact diagnostic.
- [x] Root and per-op generated-bundle evidence expose a `runtime_avl_vl_boundary`
      or equivalent summary tying selected runtime ABI `n` to materialized
      `setvl`, route metadata, generated C loop/VL use, bundle ABI, runtime
      counts, and correctness oracle.
- [x] One non-dry-run generated-bundle ABI/e2e run reaches real `ssh rvv`
      compile/run for pre-realized `i64_add` across runtime counts `7,16,23`
      with deterministic `lhs[index] + rhs[index]` correctness.
- [x] Bounded scan over touched exec/RVV dialect, realization, planning,
      provider, target, script, and fixture files finds no new metadata-derived
      AVL/VL authority or positive legacy i32 route authority.
- [x] `git diff --check` passes.
- [x] Focused checks and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Trellis task status, journal, archive, and commit are truthful.

## Non-Goals

- No new RVV operation-family coverage class.
- No Stage 2 reduction, contraction, compare/select, memory, frontend, Linalg,
  Vector, StableHLO, descriptor, direct-C, source-export, or source-front-door
  positive route work.
- No legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` executable authority.
- No performance claim.
- No multi-agent or sub-agent workflow in this round.

## Technical Approach

1. Start the Trellis task and validate context.
2. Tighten focused pre-realized `i64_add` lit checks so runtime `%n` to
   `tcrv_rvv.setvl` and runtime AVL/VL provider mirrors are explicit.
3. Harden the generated-bundle ABI/e2e script to record and verify the runtime
   AVL/VL boundary in materialized MLIR, emitted RVV C/C++, bundle metadata,
   harness ABI, and root evidence.
4. Reuse the existing generated-bundle command shape:

   ```bash
   python3 scripts/rvv_generated_bundle_abi_e2e.py \
     --pre-realized-selected-body \
     --artifact-root artifacts/tmp/stage2_runtime_avl_vl_boundary_closure \
     --run-id pre-realized-i64-add-ssh-rvv \
     --overwrite \
     --op-kind i64_add \
     --runtime-count 7 \
     --runtime-count 16 \
     --runtime-count 23 \
     --tcrv-opt build/bin/tcrv-opt \
     --tcrv-translate build/bin/tcrv-translate \
     --ssh-target rvv
   ```

5. Keep validation focused on changed behavior, then run full
   `check-tianchenrv` if feasible.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-runtime-avl-vl-boundary-closure`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
4. Focused dry-run for pre-realized `i64_add` with counts `7,16,23`.
5. Focused lit/FileCheck tests covering the changed RVV target and EmitC
   materialization behavior.
6. Real `ssh rvv` generated-bundle ABI/e2e run for pre-realized `i64_add` with
   counts `7,16,23`.
7. Bounded authority scan over touched and relevant runtime AVL/VL files.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`

## Definition Of Done

The pre-realized `i64_add` path has a reviewable runtime AVL/VL authority chain
from selected runtime ABI `n` through `tcrv_rvv.setvl`, RVV route/provider
facts, emitted C loop/intrinsic VL use, generated artifact ABI, and real
`ssh rvv` correctness evidence. Unsupported or inconsistent AVL/VL bindings
fail closed before route/artifact authority. The task is finished/archived and
committed, or an exact blocker and continuation point is recorded.

## Implementation Result

- `scripts/rvv_generated_bundle_abi_e2e.py` now verifies and records a
  `runtime_avl_vl_boundary` summary for generated-bundle evidence. The summary
  ties the selected runtime ABI parameter `n` to:
  - the materialized `tcrv_rvv.runtime_abi_value` result consumed by
    `tcrv_rvv.setvl`;
  - the `tcrv_rvv.with_vl` scope consuming that setvl result;
  - provider/export mirror metadata such as `tcrv_rvv.runtime_control_plan`,
    `tcrv_rvv.runtime_vl_contract`, `tcrv_rvv.runtime_avl_source`,
    `tcrv_rvv.runtime_abi_order`, `tcrv_rvv.route_operand_binding_operands`,
    `tcrv_rvv.remaining_avl`, and `tcrv_rvv.multi_vl`;
  - emitted RVV C++ runtime loop facts, including full-chunk setvl from `n`,
    loop setvl from `n - offset`, and i64 load/add/store intrinsics consuming
    the loop VL.
- `test/Target/RVV/pre-realized-selected-body-artifact-i64-add.mlir` now checks
  that the realized body uses the imported runtime `%n` as the AVL operand to
  `tcrv_rvv.setvl`, and checks the runtime AVL/VL metadata mirrors in emission
  plans and generated headers.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-i64-add-dry-run.test`
  now checks the root/per-op `runtime_avl_vl_boundary` evidence and FileChecks
  the generated RVV C++ loop:

  ```text
  size_t full_chunk_vl = __riscv_vsetvl_e64m1(n);
  for (offset = 0; offset < n; offset += full_chunk_vl) {
    remaining = n - offset;
    loop_vl = __riscv_vsetvl_e64m1(remaining);
    vle64/vadd/vse64(..., loop_vl);
  }
  ```

## Runtime Param To Setvl Evidence

- Selected input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-i64-add.mlir`.
- Materialized evidence path:
  `artifacts/tmp/stage2_runtime_avl_vl_boundary_closure/pre-realized-i64-add-ssh-rvv/i64_add/materialized_selected_body.mlir`.
- The materialized body contains exactly one runtime element-count ABI import:

  ```text
  %3 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ...,
       role = "runtime-element-count"} : index
  %4 = tcrv_rvv.setvl %3 {... sew = 64 : i64} : index -> !tcrv_rvv.vl
  tcrv_rvv.with_vl %4 ...
  ```

- Generated C++ evidence path:
  `artifacts/tmp/stage2_runtime_avl_vl_boundary_closure/pre-realized-i64-add-ssh-rvv/i64_add/materialized_rvv_emitc.cpp`.
- The emitted C++ uses the ABI runtime count parameter and runtime remaining
  AVL:

  ```text
  size_t v5 = __riscv_vsetvl_e64m1(v4);
  for (size_t v6 = 0; v6 < v4; v6 += v5) {
    size_t v7 = v4 - v6;
    size_t v8 = __riscv_vsetvl_e64m1(v7);
    ... __riscv_vle64_v_i64m1(..., v8)
    ... __riscv_vadd_vv_i64m1(..., v8)
    ... __riscv_vse64_v_i64m1(..., v8)
  }
  ```

## Real Hardware Evidence

- Command:

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body \
    --artifact-root artifacts/tmp/stage2_runtime_avl_vl_boundary_closure \
    --run-id pre-realized-i64-add-ssh-rvv --overwrite --op-kind i64_add \
    --runtime-count 7 --runtime-count 16 --runtime-count 23 \
    --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate \
    --ssh-target rvv
  ```

- Evidence root:
  `artifacts/tmp/stage2_runtime_avl_vl_boundary_closure/pre-realized-i64-add-ssh-rvv/evidence.json`.
- Per-op evidence:
  `artifacts/tmp/stage2_runtime_avl_vl_boundary_closure/pre-realized-i64-add-ssh-rvv/i64_add/evidence.json`.
- Generated object:
  `artifacts/tmp/stage2_runtime_avl_vl_boundary_closure/pre-realized-i64-add-ssh-rvv/i64_add/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`.
- Generated header:
  `artifacts/tmp/stage2_runtime_avl_vl_boundary_closure/pre-realized-i64-add-ssh-rvv/i64_add/generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`.
- Generated harness:
  `artifacts/tmp/stage2_runtime_avl_vl_boundary_closure/pre-realized-i64-add-ssh-rvv/i64_add/rvv_generated_bundle_abi_i64_add_harness.c`.
- Correctness oracle: `int64_t` buffers with expected expression
  `lhs[index] + rhs[index]`.
- Runtime counts: `7,16,23`.
- Remote run output ended with:

  ```text
  i64_add case n=7 ok
  i64_add case n=16 ok
  i64_add case n=23 ok
  tcrv_rvv_generated_bundle_abi_i64_add_ok counts=7,16,23
  PASS op=i64_add counts=7,16,23
  ```

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-runtime-avl-vl-boundary-closure`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused dry-run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_runtime_avl_vl_boundary_closure --run-id pre-realized-i64-add-dry-focused --overwrite --op-kind i64_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [x] Focused lit:
      `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-i64-add'`
      from `build/test`.
- [x] Focused lit:
      `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-pre-realized-i64-add-dry-run'`
      from `build/test`.
- [x] Focused fail-closed lit:
      `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generic-stage2-runtime-control-negative'`
      from `build/test`.
- [x] Real `ssh rvv` generated-bundle ABI/e2e command in the Real Hardware
      Evidence section.
- [x] Bounded authority scan over touched and relevant runtime AVL/VL files.
      Hits were existing negative guards, existing spec/ODS prohibitions,
      existing supported generic typed i32 Stage 2 intrinsics, or this round's
      `runtime_counts_are_execution_cases_not_vl_authority` evidence label; no
      new metadata-derived AVL/VL authority or positive legacy route authority
      was introduced.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`, 365/365 passed.

## Spec Sync

No `.trellis/spec/**` update is required. This round implements existing RVV
plugin, unified EmitC route, emission/runtime, and testing contracts. It adds
evidence/tooling checks and focused lit coverage for a specific already-defined
runtime AVL/VL boundary, but it does not introduce a new durable architecture
rule or API.
