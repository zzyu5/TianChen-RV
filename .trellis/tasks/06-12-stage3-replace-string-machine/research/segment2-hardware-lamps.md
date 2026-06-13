# Segment2 family — Stage 3 换心 hardware PASS evidence (ssh rvv, real riscv64)

Date: 2026-06-13. Tools: build/bin/tcrv-opt, build/bin/tcrv-translate (post-deletion build).
Runner: scripts/rvv_generated_bundle_abi_e2e.py --ssh-target rvv (NO --dry-run, real hardware).
ssh rvv: CONNECTED, uname -m = riscv64, /usr/bin/clang present.

All 7 Segment2 op-kinds PASS on real RVV hardware (counts 1,7,16,17,257; patterns 0,1):

```
PASS op=segment2_interleave_unit_load counts=1,7,16,17,257
PASS op=segment2_deinterleave_unit_store counts=1,7,16,17,257
PASS op=computed_masked_segment2_load_unit_store counts=1,7,16,17,257 patterns=0,1
PASS op=computed_masked_segment2_store_unit_load counts=1,7,16,17,257 patterns=0,1
PASS op=computed_masked_segment2_update_unit_load counts=1,7,16,17,257 patterns=0,1
PASS op=runtime_scalar_cmp_masked_segment2_load_unit_store counts=1,7,16,17,257 patterns=0,1
PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=1,7,16,17,257 patterns=0,1
```

Each case validated on hardware: field_order_distinguishing_lanes (interleave/deinterleave
field ordering), active_lanes/inactive_lanes/inactive_preserved_lanes (masked merge), and
source_preserved + tail_preserved (no over-write). These PASS lines are the I8 hardware
evidence that the converted Segment2 family (now the ONLY emission path — the string-plan
owner is deleted) is correct on real RVV.

## Byte-identity (legacy string-plan oracle vs conversion)

Captured the legacy string-plan C (pre-change binary) for all 7 fixtures into
research/*.c, then diffed against the conversion output (--tcrv-rvv-emitc-to-cpp):
all 7 IDENTICAL, both before AND after deleting the string-plan owner. The
conversion emits the vint32m1x2_t tuple, __riscv_vcreate_v_i32m1x2 (pack),
__riscv_vget_v_i32m1x2_i32m1 (extract), __riscv_vlseg2e32_v_i32m1x2[_tumu]
(load), __riscv_vsseg2e32_v_i32m1x2[_m] (store) byte-for-byte as the legacy
oracle.

## Deletion

- Deleted lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp (667 lines,
  Segment2-only after the prior 4 owner retirements) — the shared memory
  statement-plan file is now GONE (removed from CMake).
- Removed the Segment2 dispatch entry + consumer predicate
  (isRVVSelectedBodySegment2MemoryStatementPlanConsumer) from
  RVVEmitCStatementPlanOwners.{cpp,h}, the build/get statement-plan decls, and
  the RVVSelectedBodySegment2MemoryRouteStatementPlan struct
  (RVVEmitCRoutePlanning.h).
- KEPT the route-family/description/verify provider machine
  (RVVEmitCSegment2RouteFamilyPlanOwners.cpp — pre-realized body construction +
  getRVVSelectedBodySegment2RouteFamilyProviderPlan +
  verifyRVVSelectedBodySegment2MemoryRouteProviderFacts) as the description source
  of truth, exactly like the ComputedMaskMemory precedent.

## Adversarial guard probes (conversion refuses malformed bodies -> fall back)

- wrong dst buffer width (int64 buffer feeding i32 field): emitc.func=0, real op preserved.
- segment_count=3: emitc.func=0 (refused).
- bogus destination_memory_form: emitc.func=0 (refused).
- interleave field-operand swap (segment2_store %dst,%field1,%field0): emitc.func=0
  (the interleave-operand-binding-negative still rejects, exit 1).
