# Evidence

## Scope

Closed executable ABI evidence for the existing Stage2 RVV
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` route family. Production
compiler code was not changed in this round: the positive explicit and
pre-realized selected-body paths already produced validated generated bundles
and passed real `ssh rvv` compile/run evidence. The predecessor fallback/export
candidate gate remains the focused fail-closed boundary for unsupported
selected paths.

## Artifacts Exercised

### Explicit Selected Body

- Input fixture:
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
- Selected variant: `rvv_explicit_composite`
- Op kind: `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`
- ABI order: `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`
- Header prototype:
  `void tcrv_emitc_explicit_composite_masked_indexed_gather_macc_scatter_kernel_rvv_explicit_composite(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *gather_src, const int32_t *payload, const int32_t *acc, const uint32_t *index, int32_t *dst, size_t n);`
- Selected dispatch case mirror:
  `selected_dispatch_case_mirror:@rvv_explicit_composite;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=explicit-composite-gather-macc-scatter-case`
- Selected dispatch fallback mirror:
  `selected_dispatch_fallback_mirror:@explicit_composite_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=explicit-composite-gather-macc-scatter-fallback-envelope`

### Pre-realized Selected Body

- Input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
- Selected variant: `rvv_pre_composite`
- Materializer: `tcrv-materialize-selected-lowering-boundaries`
- Realization producer:
  `rvv-plugin-local-selected-body-realization-owner-registry`
- Op kind: `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`
- ABI order: `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`
- Header prototype:
  `void tcrv_emitc_pre_realized_composite_masked_indexed_gather_macc_scatter_kernel_rvv_pre_composite(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *gather_src, const int32_t *payload, const int32_t *acc, const uint32_t *index, int32_t *dst, size_t n);`
- Selected dispatch case mirror:
  `selected_dispatch_case_mirror:@rvv_pre_composite;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-composite-gather-macc-scatter-case`
- Selected dispatch fallback mirror:
  `selected_dispatch_fallback_mirror:@pre_composite_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-composite-gather-macc-scatter-fallback-envelope`

## Evidence Commands

Initial self-repair: the first explicit command used `--llvm-readobj
llvm-readobj` and blocked before bundle/runtime execution because the short
binary name was not on `PATH`. The corrected commands below use
`/usr/lib/llvm-20/bin/llvm-readobj`.

```bash
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root .trellis/tasks/06-07-stage2-rvv-runtime-scalar-cmp-indexed-gather-macc-scatter-artifact-abi/evidence --run-id explicit-composite-gms-ssh --overwrite --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10
```

Result: `success`; `dry_run=false`; `ssh_evidence=true`; remote compile and
run both exited `0`.

```bash
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root .trellis/tasks/06-07-stage2-rvv-runtime-scalar-cmp-indexed-gather-macc-scatter-artifact-abi/evidence --run-id pre-composite-gms-ssh --overwrite --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10
```

Result: `success`; `dry_run=false`; `ssh_evidence=true`; remote compile and
run both exited `0`.

## Remote Toolchain

Both evidence runs compiled on `ssh rvv` with:

```text
remote_arch=riscv64
clang_path=/usr/bin/clang
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
```

## Runtime Oracle

- Counts: `0,1,16,17,257`
- Runtime scalar thresholds: `-37,91`
- Patterns: `0,1`
- PASS marker:
  `tcrv_rvv_generated_bundle_abi_runtime_scalar_cmp_masked_indexed_gather_macc_scatter_ok`
- Final PASS line:
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`

Representative explicit selected-body runtime lines:

```text
runtime_scalar_cmp_masked_indexed_gather_macc_scatter case n=1 rhs_scalar=-37 pattern=0 ok runtime_scalar_cmp indexed_gather_macc_scatter active_lanes=1 inactive_lanes=0 inactive_preserved_lanes=0 noncontiguous_index_lanes=0 signed_product_lanes=1 source_preserved payload_acc_preserved tail_preserved
runtime_scalar_cmp_masked_indexed_gather_macc_scatter case n=16 rhs_scalar=-37 pattern=0 ok runtime_scalar_cmp indexed_gather_macc_scatter active_lanes=7 inactive_lanes=9 inactive_preserved_lanes=9 noncontiguous_index_lanes=14 signed_product_lanes=9 source_preserved payload_acc_preserved tail_preserved
runtime_scalar_cmp_masked_indexed_gather_macc_scatter case n=257 rhs_scalar=91 pattern=1 ok runtime_scalar_cmp indexed_gather_macc_scatter active_lanes=128 inactive_lanes=129 inactive_preserved_lanes=129 noncontiguous_index_lanes=256 signed_product_lanes=127 source_preserved payload_acc_preserved tail_preserved
```

Representative pre-realized selected-body runtime lines:

```text
runtime_scalar_cmp_masked_indexed_gather_macc_scatter case n=1 rhs_scalar=-37 pattern=1 ok runtime_scalar_cmp indexed_gather_macc_scatter active_lanes=0 inactive_lanes=1 inactive_preserved_lanes=1 noncontiguous_index_lanes=0 signed_product_lanes=1 source_preserved payload_acc_preserved tail_preserved
runtime_scalar_cmp_masked_indexed_gather_macc_scatter case n=17 rhs_scalar=91 pattern=0 ok runtime_scalar_cmp indexed_gather_macc_scatter active_lanes=7 inactive_lanes=10 inactive_preserved_lanes=10 noncontiguous_index_lanes=16 signed_product_lanes=11 source_preserved payload_acc_preserved tail_preserved
runtime_scalar_cmp_masked_indexed_gather_macc_scatter case n=257 rhs_scalar=-37 pattern=0 ok runtime_scalar_cmp indexed_gather_macc_scatter active_lanes=103 inactive_lanes=154 inactive_preserved_lanes=154 noncontiguous_index_lanes=256 signed_product_lanes=131 source_preserved payload_acc_preserved tail_preserved
```

## Boundary Facts Confirmed

- `selected_dispatch_bundle_boundary.object_header_metadata_agree = true`
  for both explicit and pre-realized runs.
- `selected_dispatch_bundle_boundary.runtime_abi_order =
  cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`.
- Provider support mirror:
  `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated`.
- Route operand binding plan:
  `rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1`.
- Route operand bindings include all exported ABI/header participants:
  `cmp_lhs`, `rhs_scalar`, `gather_src`, `payload`, `acc`, `index`, `dst`,
  and `n`.
- Runtime lines cover active gather/MAcc/scatter lanes, inactive destination
  preservation, noncontiguous indexed lanes, signed product lanes, source
  preservation, payload/acc preservation, and tail preservation.

## Raw Evidence Location

The raw generated bundles, per-op JSON, emitted C++, harness C, remote stdout,
and object/header files were generated during the commands above and moved to
the ignored scratch directory
`artifacts/tmp/stage2-rvv-runtime-scalar-cmp-indexed-gather-macc-scatter-artifact-abi/evidence/`.
This committed `evidence.md` is the durable task record.

## Fail-closed Evidence Reused

Focused fail-closed coverage for this executable boundary remains in the
existing Target/RVV fixtures and predecessor task:

- stale provider mirror rejection;
- stale runtime ABI order rejection;
- stale exec ABI binding rejection;
- missing exec binding rejection;
- stale/missing selected dispatch case/fallback mirror rejection;
- stale composite resource rejection;
- unsupported selected dispatch/fallback artifact export rejection from the
  archived fallback dispatch ABI task.

## No Source Change Justification

The bounded positive path was not dry-run-only at runtime: both selected-body
inputs generated bundles, compiled on the real RVV target, and ran the external
ABI harness successfully. No stale runtime scalar, compare, mask, index,
gather, MAcc, scatter, ABI/header, selected-path candidate, or provider mirror
fact was exposed by the non-dry-run evidence. Therefore the correct closure for
this round is the committed Trellis PRD/evidence record, not a production-code
rewrite.
