# Stage2 RVV typed config real hardware executable closure

## Goal

Close the current typed-config generated-bundle proof on real RVV hardware for
one existing route-supported selected-body artifact. The round must show a
selected `tcrv.exec` RVV variant reaching a typed/realized `tcrv_rvv` body,
provider-built route, common EmitC, generated object/header bundle, external C
ABI harness, and a real `ssh rvv` compile/run correctness result.

This is a corrective evidence and tooling round. It must not turn another
dry-run into an executable claim. If real `ssh rvv` execution is blocked, the
task must record the exact access/toolchain/runtime blocker and stop there.

## Direction Source

- Direction title: `Stage2 RVV typed config real hardware executable closure`.
- Module owner: the typed dtype/config artifact path must reach one real
  `ssh rvv` compile/run correctness result for an existing route-supported
  selected-body artifact, or record an exact blocker.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `51a0f836 rvv: prove typed config artifact executable closure`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` and `.trellis/spec/extension-plugins/rvv-plugin.md`
  require RVV runtime/correctness claims to follow the typed RVV authority
  chain and include real `ssh rvv` evidence when runtime correctness is
  claimed.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC/export
  to consume provider-built RVV route payloads neutrally. Common code must not
  infer RVV dtype, SEW, LMUL, policy, intrinsic spelling, operation kind, or
  runtime ABI from metadata.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` says target
  artifact metadata and diagnostics are mirrors only; target export must come
  from materialized EmitC over a provider-built route.
- `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output for RVV runtime/correctness claims. Generated-bundle evidence over
  runtime `n` must use a deterministic oracle and avoid substituting local
  compile-only or dry-run proof for runtime evidence.
- The archived task
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-typed-config-artifact-executable-closure/prd.md`
  selected pre-realized `i64_add` and recorded a real `ssh rvv` run. The live
  code at HEAD also already writes detailed per-op nested evidence, including
  remote compile/run command records, but the root evidence only summarizes
  `ssh_evidence` and remote output.
- `scripts/rvv_generated_bundle_abi_e2e.py` already supports non-dry-run
  execution via `--ssh-target rvv` and has typed-config checks for the
  pre-realized `i64_add` path.

## Requirements

1. Keep the selected closure narrow: use the existing pre-realized `i64_add`
   route-supported typed-config artifact path unless live evidence proves it is
   unsafe or stale.
2. The real hardware evidence must name the generated C/C++ source, generated
   object, generated header, harness, compile command, run command, runtime
   counts, deterministic oracle, remote target, remote compiler facts, and run
   output.
3. Root-level generated-bundle evidence should expose the real hardware
   summary directly instead of forcing reviewers to rediscover it only from
   nested per-op evidence. Nested evidence remains the detailed source of
   command records.
4. The route/dtype/config authority must remain structural and plugin-owned:
   element type, SEW, LMUL, policy, memory form, operation kind, required
   headers, C vector type, and intrinsic spelling are checked as provider-route
   mirrors, not as route authority from artifact names, ABI strings, route ids,
   manifests, test names, or common EmitC/export code.
5. Do not add new operation-family coverage, frontend lowering, descriptor or
   source-front-door positive paths, legacy i32 route authority, one-op-per-
   intrinsic wrappers, dtype/LMUL clone batches, dashboards, or performance
   claims.
6. If `ssh rvv` access, clang flags, runtime setup, or harness execution fails,
   repair that single blocker if feasible; otherwise record the exact command,
   stderr/stdout, and continuation point without claiming executable support.

## Acceptance Criteria

- [x] PRD/context files reference the RVV plugin, EmitC route, emission/runtime,
      testing, validation, script, focused test, and previous archived task.
- [x] Root `evidence.json` for the non-dry-run generated-bundle path exposes a
      per-op `ssh_execution_summary` or equivalent with generated artifact
      paths, compile/run commands, deterministic oracle, and captured run
      output.
- [x] The selected pre-realized `i64_add` path reaches real `ssh rvv`
      compile/run with runtime counts `7,16,23` and a deterministic
      `lhs[index] + rhs[index]` oracle.
- [x] The generated typed RVV C/C++ source and bundle still show `i64`/SEW64/
      LMUL m1/provider-derived typed config facts, including
      `vint64m1_t`, `__riscv_vsetvl_e64m1`, `__riscv_vle64_v_i64m1`,
      `__riscv_vadd_vv_i64m1`, and `__riscv_vse64_v_i64m1`.
- [x] Existing stale/unsupported typed-config fail-closed coverage remains
      intact, including the script self-test that rejects stale i64 element
      metadata.
- [x] Focused script checks, focused lit coverage, bounded authority scan,
      `git diff --check`, and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Trellis task status, journal, archive, and commit are truthful.

## Non-Goals

- No new RVV coverage class or operation-family expansion.
- No high-level frontend, Linalg, Vector, StableHLO, descriptor, direct-C,
  source-export, or source-front-door positive route.
- No legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` executable authority.
- No movement of RVV semantic choices into common EmitC/export or Python
  compiler-core logic.
- No performance claim.

## Technical Approach

1. Start this Trellis task and validate context.
2. Make the smallest script change needed so root-level evidence carries a
   reviewable hardware execution summary for each op in non-dry-run mode.
3. Reuse the existing pre-realized `i64_add` typed-config path and exact
   generated-bundle command shape:

   ```bash
   python3 scripts/rvv_generated_bundle_abi_e2e.py \
     --pre-realized-selected-body \
     --artifact-root artifacts/tmp/stage2_typed_config_real_hardware_executable_closure \
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

4. Keep the existing dry-run lit coverage as a local companion only. The
   executable acceptance comes from the non-dry-run `ssh rvv` result.
5. Run focused checks and full `check-tianchenrv` if feasible.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-typed-config-real-hardware-executable-closure`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
4. Focused dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_typed_config_real_hardware_probe --run-id pre-realized-i64-add-dry --overwrite --op-kind i64_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
5. Real `ssh rvv` run using the command in Technical Approach.
6. Focused lit from `build/test` for the pre-realized `i64_add` dry-run test.
7. Bounded authority scan over touched script/task/test files.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`

## Definition Of Done

The selected pre-realized `i64_add` typed-config generated bundle has a fresh
real `ssh rvv` compile/run correctness result; the root and nested evidence
record exact generated artifacts, compile/run commands, deterministic oracle,
and run output; no dry-run-only proof is described as executable closure; all
focused and full checks pass or a precise blocker is recorded; the task is
finished/archived and committed.

## Implementation Result

- `scripts/rvv_generated_bundle_abi_e2e.py` now writes a richer root
  `op_results[op_kind]` summary. For each op it includes:
  generated RVV C/C++ source, materialized selected-body MLIR, generated bundle
  index/object/header paths, generated header prototype, harness path,
  deterministic correctness oracle, typed-config closure facts, and, for
  non-dry-run evidence, `ssh_execution_summary` with exact scp/compile/run
  command records and captured remote output.
- The focused pre-realized `i64_add` lit test now checks those new root
  evidence fields in the dry-run companion path.
- The first real `ssh rvv` attempt reached remote compile/run success but the
  script returned non-zero because the final stdout printer still read the old
  root `remote_output` field. The fix changed printing to read
  `ssh_execution_summary.remote_output`, then the same non-dry-run command
  completed with exit code 0.

## Real Hardware Evidence

- Evidence root:
  `artifacts/tmp/stage2_typed_config_real_hardware_executable_closure/pre-realized-i64-add-ssh-rvv/evidence.json`
- Per-op evidence:
  `artifacts/tmp/stage2_typed_config_real_hardware_executable_closure/pre-realized-i64-add-ssh-rvv/i64_add/evidence.json`
- Generated C/C++:
  `artifacts/tmp/stage2_typed_config_real_hardware_executable_closure/pre-realized-i64-add-ssh-rvv/i64_add/materialized_rvv_emitc.cpp`
- Generated object:
  `artifacts/tmp/stage2_typed_config_real_hardware_executable_closure/pre-realized-i64-add-ssh-rvv/i64_add/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`
- Generated header:
  `artifacts/tmp/stage2_typed_config_real_hardware_executable_closure/pre-realized-i64-add-ssh-rvv/i64_add/generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`
- Generated harness:
  `artifacts/tmp/stage2_typed_config_real_hardware_executable_closure/pre-realized-i64-add-ssh-rvv/i64_add/rvv_generated_bundle_abi_i64_add_harness.c`
- Deterministic oracle: `int64_t` input buffers initialized by the generated
  harness, expected expression `lhs[index] + rhs[index]`, runtime counts
  `7,16,23`, pass marker `tcrv_rvv_generated_bundle_abi_i64_add_ok`.
- Typed-config closure facts in root evidence: `element_type=i64`, `sew=64`,
  `lmul=m1`, `config_contract=rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1`,
  `vector_c_type=vint64m1_t`, required header `riscv_vector.h`, and intrinsics
  `__riscv_vsetvl_e64m1`, `__riscv_vle64_v_i64m1`,
  `__riscv_vadd_vv_i64m1`, `__riscv_vse64_v_i64m1`.
- Remote compile command: recorded exactly at
  `op_results.i64_add.ssh_execution_summary.compile_command.command` in the
  root evidence. It ran on `ssh rvv`, changed into
  `/tmp/tianchenrv_rvv_generated_bundle_abi_pre-realized-i64-add-ssh-rvv_i64_add`,
  printed `remote_arch`, `clang_path`, `clang_version`, then invoked
  `clang -O2 -march=rv64gcv -mabi=lp64d -I. <harness.c> <generated.o> -o <harness>`.
- Remote compile stdout:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote run command: recorded exactly at
  `op_results.i64_add.ssh_execution_summary.run_command.command` in the root
  evidence. It ran the generated harness binary on `ssh rvv`.
- Remote run stdout:
  `i64_add case n=7 ok`,
  `i64_add case n=16 ok`,
  `i64_add case n=23 ok`,
  `tcrv_rvv_generated_bundle_abi_i64_add_ok counts=7,16,23`,
  `PASS op=i64_add counts=7,16,23`.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-typed-config-real-hardware-executable-closure`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused dry-run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_typed_config_real_hardware_probe --run-id pre-realized-i64-add-dry-after-test-update --overwrite --op-kind i64_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [x] Real `ssh rvv`:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_typed_config_real_hardware_executable_closure --run-id pre-realized-i64-add-ssh-rvv --overwrite --op-kind i64_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv`
- [x] Focused lit from `build/test`:
      `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-pre-realized-i64-add-dry-run'`
      passed 1/1.
- [x] Bounded authority scan over touched task/script/test files and relevant
      provider/target/fixture files found only negative guards, PRD non-goals,
      existing fail-closed diagnostics, or self-test residue checks.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`, 365/365 passed.

## Spec Sync

No `.trellis/spec/**` update is required. This round applies existing RVV
plugin, unified EmitC route, emission/runtime, testing, and validation
contracts. The one self-repair was an implementation-level evidence summary
printing bug, not a new durable project rule.

## Commit

`this commit`: `rvv: expose real typed config hardware evidence`
