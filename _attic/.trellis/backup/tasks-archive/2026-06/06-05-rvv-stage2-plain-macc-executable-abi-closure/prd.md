# Stage2 RVV plain MAcc executable ABI closure

## Goal

Close one bounded executable ABI path for the already route-supported plain
`macc_add` selected-body route. The path must run from selected `tcrv.exec`
RVV variant through explicit or pre-realized typed `tcrv_rvv.macc` body,
provider-derived route facts, `TCRVEmitCLowerableRoute`, Common EmitC/export,
generated header/source/object bundle, external C ABI harness, and real
`ssh rvv` correctness evidence.

This is not another route-contract or metadata round. If live checks expose a
production gap in route emission, generated C/header/object, runtime ABI order,
operand binding, accumulator/result layout, or harness correctness, repair
that production boundary and prove the fix with focused evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV plain MAcc executable ABI closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` returned clean short status through RTK.
- Initial `git log --oneline -8` started at
  `65bb8c95 rvv: close plain macc route contract evidence`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
- The immediately previous archived task
  `06-05-rvv-stage2-plain-macc-selected-body-route` closed provider route
  facts and dry-run generated-bundle evidence for plain `macc_add`, including
  `provider_route_facts.macc_arithmetic_kind = add`, but did not run
  `ssh rvv` and did not claim new executable behavior.
- Earlier archived plain MAcc work shows the script and harness have had an
  executable path before, but this round must revalidate the current HEAD
  artifact/runtime ABI path after the latest provider-fact changes.

## Current Repository Evidence

- `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output for RVV runtime/correctness claims and has a dedicated Plain MAcc
  generated-bundle evidence contract.
- `test/Target/RVV/explicit-selected-body-artifact-macc-add.mlir` carries an
  explicit typed `tcrv_rvv.macc` body with ABI order `lhs,rhs,acc,out,n`,
  arithmetic kind `add`, accumulator layout
  `separate-i32-vector-accumulator-input`, and result layout
  `store-multiply-accumulate-result-to-output-buffer`.
- `test/Target/RVV/pre-realized-selected-body-artifact-macc-add.mlir` starts
  from `tcrv_rvv.typed_macc_pre_realized_body` and expects the RVV plugin to
  consume it into `setvl`, lhs/rhs/acc loads, `tcrv_rvv.macc`, and store.
- `scripts/rvv_generated_bundle_abi_e2e.py` already supports
  `--op-kind macc_add`, `--pre-realized-selected-body`, dry-run bundle
  verification, non-dry-run `ssh rvv` execution, default runtime counts, and
  generated scalar-oracle harness checks.
- Existing dry-run tests check provider-route facts, harness source shape,
  `macc_arithmetic_kind = add`, operand binding, selected ABI roles, and
  source/tail preservation markers.

## Requirements

- Keep scope to one plain vector-vector `macc_add` route.
- Preserve the authority chain:
  selected typed `tcrv_rvv.macc` body/config/runtime facts -> RVV
  provider-derived route facts -> Common EmitC/export -> generated bundle
  mirrors -> external harness -> `ssh rvv` runtime evidence.
- The generated evidence must expose provider-derived route facts for:
  runtime ABI order `lhs,rhs,acc,out,n`, route operand binding, arithmetic kind
  `add`, accumulator/result layout, source/result dtype, SEW, LMUL, policy,
  runtime AVL/VL boundary, required headers, C type mapping, target leaf
  profile, and `provider_supported_mirror`.
- The external harness must distinguish correct `acc + lhs * rhs` behavior
  from add-only, multiply-only, missing-accumulator, wrong operand order,
  wrong ABI order, missing source preservation, or output tail writes.
- Runtime coverage must include representative `n` values with at least:
  `0`, `1`, one full VL boundary count, one tail count, and one multi-chunk
  count. The evidence should include the current bounded set
  `0,1,16,17,257` unless live VLEN/toolchain evidence shows the script's
  existing count set is the safer current harness contract.
- Run both explicit selected-body and pre-realized selected-body generated
  bundle paths unless live evidence shows one is unsafe or stale; if blocked,
  record the exact blocker and next continuation point.
- Common EmitC/export must remain neutral. Do not move MAcc semantics,
  dtype/config inference, accumulator behavior, intrinsic choice, or ABI role
  mapping into Common EmitC or test/harness strings.

## Acceptance Criteria

- [ ] PRD and context files truthfully describe this executable closure task.
- [ ] Focused build targets needed by the generated-bundle path are current:
      `tcrv-opt`, `tcrv-translate`, and target artifact/export support.
- [ ] Generated artifact/header/source evidence exists for explicit
      `macc_add` and includes provider route facts with arithmetic kind `add`,
      ABI order `lhs,rhs,acc,out,n`, route operand binding, accumulator/result
      layout, runtime AVL/VL, required headers/types, and provider mirror.
- [ ] Generated artifact/header/source evidence exists for pre-realized
      `macc_add` and proves the pre-realized body was consumed before emission.
- [ ] External ABI harness compiles/links on the `rvv` target for the current
      generated source/header/object path.
- [ ] Real `ssh rvv` execution passes for representative counts including
      `0`, `1`, VL-boundary, tail, and multi-chunk cases, with pattern coverage
      that distinguishes vector-vector multiply, accumulator contribution,
      source preservation, and output tail preservation.
- [ ] Evidence JSON or final task notes record the generated artifact path,
      `ssh rvv` command shape, success marker, runtime counts, pattern set,
      and scalar oracle contract.
- [ ] Focused negative/preflight checks remain green for stale arithmetic kind,
      runtime ABI order, operand binding, accumulator/result layout, stale route
      or provider facts, and direct pre-realized route-entry fail-closed
      behavior where applicable.
- [ ] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor, source-front-door,
      source-artifact, route-string/artifact-name/ABI-string/test-name, exact
      intrinsic-spelling, or Common EmitC semantic authority.
- [ ] `git diff --check`, `git diff --cached --check`, Trellis validation,
      final clean worktree, archive, and one coherent commit complete the
      round if the executable closure is complete.

## Evidence Plan

- Read and use the existing explicit and pre-realized plain MAcc target
  fixtures and generated-bundle dry-run tests.
- Run focused local checks first:
  - `cmake --build build --target tcrv-opt tcrv-translate -j2`
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  - generated-bundle dry-run for explicit `macc_add`
  - generated-bundle dry-run for pre-realized `macc_add`
- Run real RVV evidence:
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind macc_add ...`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind macc_add ...`
- If local or remote evidence fails because production generated code or
  harness behavior is stale, repair the narrow production/script/test boundary
  and rerun the focused checks.
- Run target artifact/export checks and direct fail-closed checks if live
  changes touch provider/materializer/target behavior or if existing focused
  tests are needed to prove stale provider facts still reject.

## Definition Of Done

- Current HEAD has real `ssh rvv` correctness evidence for plain `macc_add`
  explicit and pre-realized generated-bundle ABI paths, or a precisely
  documented production blocker with no false executable claim.
- The Trellis task records what changed, generated artifact paths, runtime
  marker, checks, and self-repair.
- Specs are updated only if this round discovers a new durable rule not already
  captured by `.trellis/spec/`.
- The task is finished/archived and one coherent commit is created when the
  module behavior is complete.

## Out Of Scope

- Broad MAcc dtype/LMUL/op matrix expansion.
- Scalar-broadcast MAcc, computed-mask MAcc, runtime-scalar computed-mask
  MAcc, widening MAcc, reductions, Gearbox, matmul, Linalg/Vector/StableHLO
  frontend work, performance benchmarking, or autotuning.
- New one-intrinsic wrapper routes, source-front-door positive routes,
  descriptor-driven computation, or common EmitC semantic inference.
- Treating route ids, artifact names, ABI strings, test names, exact intrinsic
  spelling, manifest/status fields, or metadata mirrors as MAcc authority.

## Technical Notes

Specs and guides read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived context read:

- `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-plain-macc-selected-body-route/`
- `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-plain-macc-add-vector-vector-artifact-abi-boundary/prd.md`
- `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-unit-stride-macc-production-validation-boundary/prd.md`

Likely touched files if repairs are needed:

- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Scripts/rvv-generated-bundle-abi-e2e-macc-add-dry-run.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-macc-add-dry-run.test`
- `test/Target/RVV/explicit-selected-body-artifact-macc-add.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-macc-add.mlir`
- RVV provider/materializer/target files under
  `include/TianChenRV/Plugin/RVV`, `lib/Plugin/RVV/EmitC`, and
  `lib/Target/RVV` only if live evidence exposes a production gap.

## Completion Evidence

Completed as executable ABI closure for the current plain `macc_add`
generated-bundle path.

Production/test changes:

- Aligned the plain MAcc script self-test boundary from
  `0,1,16,23,257` to `0,1,16,17,257`, matching the executable closure count
  set used for `n = 0`, scalar one-lane, full VL chunk, tail, and multi-chunk
  coverage.
- Updated explicit and pre-realized plain MAcc generated-bundle dry-run tests
  to use `0,1,16,17,257`.
- Added dry-run FileCheck coverage for the `multiply_accumulate_boundary`
  `runtime_counts` field and generated harness `counts[]` literal.

Generated executable evidence:

- Explicit selected-body artifact:
  `/tmp/tcrv-plain-macc-executable-abi/explicit-macc-add-ssh/macc_add`
- Pre-realized selected-body artifact:
  `/tmp/tcrv-plain-macc-executable-abi/pre-realized-macc-add-ssh/macc_add`
- Both paths generated:
  - `materialized_rvv_emitc.cpp`
  - `generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`
  - `generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`
  - `rvv_generated_bundle_abi_macc_add_harness.c`
- Provider route facts in both evidence JSON files:
  - `runtime_abi_order = lhs,rhs,acc,out,n`
  - `macc_arithmetic_kind = add`
  - `accumulator_layout = separate-i32-vector-accumulator-input`
  - `result_layout = store-multiply-accumulate-result-to-output-buffer`
  - `route_operand_binding_operands = rvv-route-operand-binding:macc_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr`

Remote RVV evidence:

- Explicit command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root /tmp/tcrv-plain-macc-executable-abi --run-id explicit-macc-add-ssh --overwrite --op-kind macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
- Pre-realized command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root /tmp/tcrv-plain-macc-executable-abi --run-id pre-realized-macc-add-ssh --overwrite --op-kind macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
- Both remote runs compiled and executed successfully on `ssh rvv`.
- Final marker for both paths:
  `PASS op=macc_add counts=0,1,16,17,257 patterns=0,1`.
- The harness printed per-case success for every count/pattern pair with:
  `explicit_accumulator signed_products source_preserved tail_preserved`.

Self-repair:

- Initial dry-run used `--llvm-readobj llvm-readobj` and failed because this
  host exposes `/usr/bin/llvm-readobj-20`, not a bare `llvm-readobj` on the
  shell PATH. Re-ran with `/usr/bin/llvm-readobj-20`; lit continues to use the
  build/test substitution path.
- First lit run failed after adding `counts[]` FileCheck because the function
  call is emitted inside `run_case`, before `main` declares `counts[]`.
  Reordered the `HARNESS` check so it follows the actual harness source order.

Checks run:

- `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
- `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Explicit plain MAcc dry-run with counts `0,1,16,17,257`
- Pre-realized plain MAcc dry-run with counts `0,1,16,17,257`
- `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(macc-add|pre-realized-macc-add)-dry-run'` from `build/test`
- Explicit real `ssh rvv` generated-bundle run for `macc_add`
- Pre-realized real `ssh rvv` generated-bundle run for `macc_add`
- Direct pre-realized route-entry fail-closed reproduction for `macc_add`
- `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `rtk build/bin/tianchenrv-target-artifact-export-test`
- `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'Target/RVV/(explicit-selected-body-artifact-macc-add|pre-realized-selected-body-artifact-macc-add)'` from `build/test`
- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-05-rvv-stage2-plain-macc-executable-abi-closure`
- `rtk git diff --check`
- Bounded old-authority scan over added source/test lines. No new positive
  legacy authority hits. PRD hits are negative guardrail text only.

Spec update decision:

- No `.trellis/spec/` update was needed. The existing Plain MAcc
  generated-bundle evidence contract already requires real `ssh rvv`, multiple
  runtime counts including `n = 0` and tail, two data patterns,
  accumulator/source/tail preservation, provider route facts, and mirror-only
  artifact metadata.
