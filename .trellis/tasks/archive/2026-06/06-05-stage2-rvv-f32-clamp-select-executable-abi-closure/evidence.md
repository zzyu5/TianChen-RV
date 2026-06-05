# Evidence

## Scope

Closed executable ABI evidence for the existing Stage2 RVV f32 clamp/select
route. Production compiler code was not changed; this round added neutral
generated-bundle harness/evidence support for `f32_clamp_select` and updated the
testing contract.

## Artifact Exercised

- Input fixture: `test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir`
- Op kind: `f32_clamp_select`
- Function: `tcrv_emitc_pre_realized_f32_clamp_select_kernel_pre_realized_rvv_f32_clamp_select`
- ABI order: `input,lower_bound,upper_bound,out,n`
- Header prototype: `void tcrv_emitc_pre_realized_f32_clamp_select_kernel_pre_realized_rvv_f32_clamp_select(const float *input, float lower_bound, float upper_bound, float *out, size_t n);`

## Evidence Commands

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind f32_clamp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --run-id stage2-f32-clamp-select-dry-run --overwrite
```

Result: `dry_run_success`.

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind f32_clamp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --run-id stage2-f32-clamp-select-ssh --overwrite
```

Result: `success`; remote `ssh rvv` compile/run passed.

## Runtime Oracle

- Counts: `0,1,16,17,257`
- Patterns: `0,1`
- Bound pairs: `[-1.5, 2.25]`, `[-8.0, -0.75]`
- Tolerance: `1e-05`
- Checks: host/reference f32 clamp, below/in/above-bound lane coverage for
  nontrivial pattern cases, source preservation, output tail sentinel
  preservation.

## Artifacts

- Dry-run evidence: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-f32-clamp-select-dry-run/evidence.json`
- SSH evidence: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-f32-clamp-select-ssh/evidence.json`
- Remote stdout copy: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-f32-clamp-select-ssh/f32_clamp_select/remote_run_stdout.txt`

## Checks

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --tcrv-materialize-selected-lowering-boundaries | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --check-prefix=REALIZED`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --check-prefix=PLAN`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --check-prefix=HEADER`
- Bounded diff authority scan: no matches.
- `git diff --check`
- `git diff --cached --check`

## Self-Repair

- Aligned f32 clamp/select harness metadata expectations with production
  provider/export mirrors for C type mapping, mask role/source/form, and route
  operand binding.
- Fixed generated C float literals so integral f32 bound values render as
  valid C literals such as `-8.0f`, not `-8f`.
