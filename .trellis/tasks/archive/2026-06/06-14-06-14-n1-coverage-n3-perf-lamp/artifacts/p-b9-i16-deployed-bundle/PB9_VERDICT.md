# P-B9 verdict — i16 deferred-wide dot-reduce now produces a DEPLOYABLE bundle (.o/.h)

## Headline

**The i16 widening dot-reduce deferred-wide kernel now produces a DEPLOYABLE
target-artifact BUNDLE (.o/.h) end-to-end, ssh-rvv-validated** — closing the P-B8
deployable residual exactly the way P-B6 closed P-B5's byte residual. Both kernel
families (byte low-precision contraction + i16 dot-reduce) are now deployable
("若干 deployable").

## The wall (closed, mirror of byte P-B6)

P-B8 left the i16 deferred-wide bundle export FAIL-CLOSED: the route-family
description engine did not recognize the deferred-wide dot-reduce typed-compute
chain (errored "unsupported generic tcrv_rvv.widening_product kind
'signed_widening_product'"). This is the SAME class of multi-file work P-B6 did for
the byte chain, now for the i16 dot-reduce chain (widening_product i16m4->i32m8 +
SAME-WIDTH deferred_accumulate vadd.vv + trailing standalone_reduce i32m8->i32m1).

The wall was walked iteratively (each rebuild surfaced the next narrow-dot-reduce-
pinned site). The rejecting sites (the parallel deferred-wide DOT-REDUCE fact set):

- **RVVEmitCRouteProvider.h**: NEW op kinds WideningProductDeferredDotAccumulate
  (transient walk state) + WideningProductDeferredDotAccumulateReduceAdd (terminal,
  shares the narrow widening_dot_reduce_add route identity).
- **RVVEmitCRoutePlanning.h** (slice): NEW deferredAccumulateOp field +
  reduceInputSlotResult threads the deferred i32m8 accumulate result to the reduce.
- **RVVEmitCRouteAnalysis.cpp**: accept the wide signed i16 product relation
  (signed-i16m4xi16m4-to-i32m8); NEW recordRVVSelectedBodyDeferredAccumulate; the
  deferred-wide standalone_reduce promotion (-> terminal kind); the two body-op
  walkers + the op-count adjust (+2: 1 fused op -> 3-op chain); the epilogue ABI
  seed read from standalone_reduce; the chain typed-compute-op name; the
  canonical-role-order (product 9 / deferred_accumulate 10 / reduce 11 / store 12)
  + the role-sequence chain.
- **RVVEmitCRouteConfigBinding.cpp**: the dot-reduce leaf-profile (no scalar leaf);
  the dual-config validation (source i16m4 / accumulate i32m8 / result i32m1); the
  isWideningDotReduce ABI-verifier routing (NOT the finite-binary elementwise
  contract, which would expect const int32_t * for the i16 source).
- **RVVEmitCRouteFamilyDerivation.cpp**: the operation profiles + the two switches
  (profile + arithmetic-intrinsic llvm_unreachable group).
- **RVVEmitCRoutePlanning.cpp**: the strip->result config normalization (sew16/m4
  -> sew32/m1); the chain typed-compute-op for the role sequence; the
  stringify/master-kinds array.
- **RVVEmitCContractionRouteFamilyCommon.cpp**: isContractionDotReductionOperation
  + getContractionRuntimeABIOrder admit the new kind.
- **RVVEmitCContractionRouteFamilyPlanOwners.cpp**: the dot facts builder
  (isDeferredWideDotReduction -> wide primitive intrinsics vwmul_vv_i32m8 /
  vredsum_vs_i32m8_i32m1, chain typed-compute-op, narrow-identity leaf relation) +
  the binding-plan-id + the contraction-route-operation classifier.
- **RVVEmitCContractionRouteFamilyValidation.cpp**: the plan derivation
  (lhsSourceFacts from the product head, layouts from standalone_reduce, the wide
  primitive intrinsics, the i16m4 VectorRHSLoad memory form, the dual-config
  exception, the supported-config admit, the operand bindings, the result-config
  restore in applyRVVSelectedBodyContractionRouteFamilyPlan).
- **RVVEmitCRouteVerification.cpp**: exempt the chain typed-compute-op from the
  static-route mirror; the reduction-store-VL ('1') + the wide source/result config
  admit.
- **RVVConstructionProtocol.cpp**: accept the deferred-wide dot chain in the three
  typed-compute-op validators (route-mapping, role-sequence, metadata-facts) + the
  parallel role-step canonical order (widening_product+deferred_accumulate+
  standalone_reduce).

## Blessed boundary (I4/I7, not fabricated to force green)

The wide single-scope dot-reduce body carries ONLY structurally-derivable primitive
facts: the wide intrinsics (vwmul_vv_i32m8, vadd.vv i32m8 deferred, vredsum_vs_
i32m8_i32m1) mirror the realized ops (I5); the route IDENTITY (dot-reduce-add op,
lhs/rhs/acc/out/n ABI, result config sew32/m1, runtime ABI name) is the narrow
dot-reduce. NO low_precision_resource.* selection block + NO gearbox_cross_region_
handoff (those are byte grouped two-scope artifacts). The target-leaf-profile +
c_type_mapping honestly reflect the realized wide source (i16m4) -- the wide-ness
is the realized algorithm, not hidden.

## The bundle exports (kernel -> PLAN + HEADER, exit 0)

`tcrv-opt KERNEL --tcrv-rvv-materialize-gearbox-schedules
--tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
-> a valid PLAN (rvv_selected_body_operation=widening_dot_reduce_add,
typed_compute_op=widening_product+deferred_accumulate+standalone_reduce, config
sew32/m1, status=supported). `| tcrv-translate --tcrv-export-target-header-artifact`
-> a valid callable-C HEADER:
  void tcrv_emitc_..._rvv(const int16_t *lhs, const int16_t *rhs,
                         const int32_t *acc, int32_t *out, size_t n);

## DEPLOYED-BUNDLE correctness (I8, real ssh rvv riscv64 / clang 18.1.3)

The GENERATED .o (compiled from the emitted C++ body) is ELF64 RISC-V relocatable
carrying the WIDE winning intrinsics (objdump: vle16.v, vwmul.vv, vadd.vv,
vredsum.vs). Linked + called via the GENERATED .h prototype against a GENUINE scalar
integer oracle: **DEPLOYED_BUNDLE_ALL_PASS, abs_err=0** at n=0/1/16/17/256/257/1024/
4096/16384/65536 (incl. the n=257 prime-tail partial strip) x acc {0,7,-19}.
Evidence: deployed_bundle_ssh_rvv_stdout.txt. The DEPLOYABLE bundle path -- not the
runnable C -- runs numerically correct.

PROVENANCE: the deployed body + header regenerate BYTE-IDENTICALLY from the current
clean-built tcrv-opt/tcrv-translate.

## Tests

- pre-realized-selected-body-realize-deferred-wide-dot-reduce-autotuner-e2e.mlir:
  ADDED PLAN + HEADER export RUN lines + CHECK assertions (the deferred-wide
  bundle-export facts). The WIDE/EMITC/NARROW budget-crossover checks unchanged.
- Narrow dot-reduce + byte deferred-wide + packed-i4/clamp fixtures byte-identical.

## Build / lit

Build GREEN. lit: 601 tests, 598 pass, exactly the 3 pre-existing documented
environmental reds (self-test fake bundle expects the retired grouped two-scope
form; the two computed-masked-strided-input-widening-dot-reduce-add dry-runs). RVV
Target 180/180, Conversion 109/109, Dialect 51/51 pass. NOT git-committed (per task).

## Answers to the report questions

- Does the i16 deferred-wide dot-reduce kernel now produce a deployable .o/.h
  bundle? **YES** (ssh-rvv-validated, abs_err=0).
- Both families now deployable (若干 deployable)? **YES** — byte (P-B6) + i16
  dot-reduce (P-B9) both produce deployable bundles.
- Narrow / byte / other paths intact? **YES** — all gated on the distinct op kind /
  deferred_accumulate presence / wide relation; byte-identical fixtures + lit.
