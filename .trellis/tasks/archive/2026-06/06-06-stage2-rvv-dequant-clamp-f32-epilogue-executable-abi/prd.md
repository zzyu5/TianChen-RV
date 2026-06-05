# Stage2 RVV dequant-clamp f32 epilogue executable ABI closure

## Goal

Close the executable ABI evidence boundary for the existing
`dequant_clamp_f32_epilogue` route. The bounded path is:

```text
selected pre-realized typed tcrv_rvv dequant-clamp f32 epilogue body
  -> RVV selected-body realization
  -> provider-owned route facts
  -> TCRVEmitCLowerableRoute
  -> generated target bundle
  -> external C ABI harness
  -> real ssh rvv correctness evidence
```

The task must prove runtime ABI order `lhs, scale, lower_bound, upper_bound,
out, n`, signed i32 load to f32 dequant scaling, lower/upper clamp semantics,
f32 output mapping, runtime VL loop behavior, source preservation, and output
tail sentinel preservation on real RVV hardware.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV dequant-clamp f32 epilogue executable ABI closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `19ba45f6 rvv: close dequant clamp epilogue artifact foundation`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created before source edits.
- The archived artifact-foundation task closed provider, target/header, stale
  mirror, and generated-bundle dry-run evidence for
  `dequant_clamp_f32_epilogue`, but explicitly did not claim real `ssh rvv`
  executable correctness.
- The existing generated-bundle script already accepts
  `--op-kind dequant_clamp_f32_epilogue`.
- The existing harness branch for `dequant_clamp_f32_epilogue` calls the
  generated external ABI as
  `lhs, scale, lower_bound, upper_bound, out, n`.
- The existing harness branch includes a scalar oracle over
  `((float)lhs[index]) * scale` followed by lower/upper clamp, source
  preservation checks, output tail sentinel checks, multiple signed source
  patterns, multiple runtime scale values, and multiple ordered bound pairs.
- The current likely implementation path is to run the existing dry-run and
  non-dry-run `ssh rvv` evidence, then repair only a minimal script/bundle/ABI
  blocker if real execution fails.

## Requirements

- Start from the existing generated-bundle dry-run command for
  `dequant_clamp_f32_epilogue` in pre-realized selected-body mode.
- Produce the generated object/header/harness bundle for the existing
  `test/Target/RVV/pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue.mlir`
  fixture.
- Execute the generated bundle on real `ssh rvv`.
- Validate scalar oracle correctness for counts `0`, `1`, a VL-boundary count,
  a tail count, and a multi-chunk count.
- Cover below-bound, in-bound, and above-bound lanes, signed negative and
  positive source lanes, at least two runtime scale values, and at least two
  ordered lower/upper bound pairs.
- Validate source preservation and output tail sentinel preservation.
- If the script or generated bundle cannot execute, repair only the minimal
  production/script blocker needed to carry this existing route through the
  executable ABI.
- Preserve provider-owned route facts and target validation; do not move
  dequant or clamp semantics into common EmitC/export, route strings, artifact
  names, ABI strings, or harness-only metadata.

## Acceptance Criteria

- [x] Generated-bundle dry-run succeeds for:

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --pre-realized-selected-body \
    --op-kind dequant_clamp_f32_epilogue \
    --runtime-count 0 --runtime-count 1 --runtime-count 16 \
    --runtime-count 17 --runtime-count 257 \
    --dry-run
  ```

- [x] Non-dry-run execution succeeds on real `ssh rvv` for the same op and
      count set.
- [x] Remote output proves all required counts ran and includes scalar oracle
      success lines with below/in/above-bound coverage, signed lane coverage,
      `source_preserved`, and `tail_preserved`.
- [x] Evidence summary records the generated bundle path and remote command
      output.
- [x] If script code is touched, its self-test or focused equivalent passes.
- [x] Prior target/header artifact and negative diagnostics still pass.
- [x] A bounded old-authority scan over touched files and added diff lines
      shows no new legacy i32/source-front-door/descriptor/common-EmitC route
      authority.
- [x] `git diff --check`, `git diff --cached --check`, task validation, and
      final `git status --short` are clean after commit.

## Out Of Scope

- No `widening_product_reduce_dequant_clamp_f32` fused route.
- No broad dequant, dtype, SEW, LMUL, policy, or generated-bundle matrix.
- No high-level Linalg, Vector, StableHLO, or source frontend work.
- No per-Linalg route authority or one-intrinsic wrapper expansion.
- No performance, autotuning, tuning database, dashboard, or readiness state.
- No common EmitC invention of dequant or clamp semantics.
- No route-string, artifact-name, ABI-string, test-name, descriptor, manifest,
  source-front-door, or script metadata authority.
- No evidence-only completion without either real `ssh rvv` execution or a
  proven external hardware/toolchain blocker.

## Evidence Plan

- Read the relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Read the archived artifact-foundation task and the current
  `scripts/rvv_generated_bundle_abi_e2e.py` dequant-clamp harness branch.
- Run generated-bundle dry-run for `dequant_clamp_f32_epilogue`.
- Run non-dry-run generated-bundle evidence on `ssh rvv`.
- If non-dry-run fails, inspect generated bundle, remote compile stdout/stderr,
  remote run stdout/stderr, and repair the smallest blocker.
- Run focused regressions if files are changed:
  script self-test/focused equivalent, target artifact fixture, negative
  dialect fixture, and f32 clamp/select executable regression only if shared
  clamp/select script logic is touched.

## Technical Notes

- `test/Target/RVV/pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue.mlir`
  is the positive fixture from the prior artifact-foundation task.
- `test/Dialect/RVV/pre-realized-dequant-clamp-f32-epilogue-negative.mlir`
  covers fail-closed body verification for stale pre-realized facts.
- `scripts/rvv_generated_bundle_abi_e2e.py` contains the current generated
  bundle ABI harness. Its dequant-clamp branch already performs the ABI call,
  scalar oracle, source preservation, and tail sentinel checks.
- Production code owners to inspect only if execution fails:
  `lib/Plugin/RVV/EmitC/`, `lib/Target/RVV/`, and relevant RVV provider
  headers.

## Completion Evidence

- Generated-bundle dry-run succeeded:

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --pre-realized-selected-body \
    --op-kind dequant_clamp_f32_epilogue \
    --runtime-count 0 --runtime-count 1 --runtime-count 16 \
    --runtime-count 17 --runtime-count 257 \
    --dry-run
  ```

  Artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T184843Z`.

- Real `ssh rvv` execution succeeded for the same op/count set:

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --pre-realized-selected-body \
    --op-kind dequant_clamp_f32_epilogue \
    --runtime-count 0 --runtime-count 1 --runtime-count 16 \
    --runtime-count 17 --runtime-count 257
  ```

  Artifact directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T184905Z`.

- Generated bundle:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T184905Z/dequant_clamp_f32_epilogue/generated_bundle/`.
- Generated harness:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T184905Z/dequant_clamp_f32_epilogue/rvv_generated_bundle_abi_dequant_clamp_f32_epilogue_harness.c`.
- Remote compile evidence:
  `remote_arch=riscv64`, `clang_path=/usr/bin/clang`, and
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T184905Z/dequant_clamp_f32_epilogue/remote_run_stdout.txt`.
- Remote run summary marker:

  ```text
  tcrv_rvv_generated_bundle_abi_dequant_clamp_f32_epilogue_ok counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05 source_preserved tail_preserved
  PASS op=dequant_clamp_f32_epilogue counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05
  ```

- The remote output contains per-case scalar oracle lines for counts `0`, `1`,
  `16`, `17`, and `257`, both patterns, two scale values, and two bound
  pairs. Nonzero cases report below-bound, inside-bound, above-bound, signed
  negative, and signed positive lane coverage as applicable, and every case
  reports `source_preserved tail_preserved`.
- No production C++/TableGen/Python implementation change was required. The
  existing generated-bundle harness already carried the route through the
  external ABI and real RVV execution.
- Focused checks run:

  ```bash
  python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . \
    --filter pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue
  ```

  from `build/test`, passed 1 selected test.

  ```bash
  python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . \
    --filter pre-realized-dequant-clamp-f32-epilogue-negative
  ```

  from `build/test`, passed 1 selected test.

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
  ```

  passed.

- `git diff --check`, `git diff --cached --check`, task validation, bounded
  old-authority scan over the task diff, and final status checks were run
  before commit.
