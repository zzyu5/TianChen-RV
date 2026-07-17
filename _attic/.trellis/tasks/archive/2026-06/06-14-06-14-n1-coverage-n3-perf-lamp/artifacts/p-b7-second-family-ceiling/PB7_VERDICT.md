# P-B7 verdict — N3 性能灯 extended to a 2nd kernel family (signed i16 dot-reduce)

## Headline

**The 灯 is ON for a 2nd kernel family's emitted body.** The TianChen-RV
conversion now LOWERS a (hand-authored) deferred-wide i16 dot-reduce typed body
(i16m4 strip -> i32m8 single-widening product -> i32m8 NON-widening vadd.vv
deferred accumulate -> ONE trailing vredsum -> i32 store) to a winning C body:
measured on real `ssh rvv`, that emitted body beats genuine scalar 3.9-7.5x,
beats the compiler's OWN current dot-reduce emission 5.7-11.1x, and beats a
competent narrow naive 2.2-3.8x — at every n. What is PROVEN for i16 is
**typed-body -> C -> win** (conversion + lowering + ssh-rvv measure). The
resource-aware SELECTION/REALIZATION that would produce that deferred-wide body
from a plain dot-reduce kernel (the autotuner: kernel -> select -> realize ->
that body) is the remaining wall — the selector is confirmed i8-pinned (Stage 2b
below).

## "若干 kernel" — three distinct evidence grades (do NOT flatten)

| grade | what it proves | kernels |
|---|---|---|
| autotuner-driven e2e (select->realize->lower->win) | the N3 autotuner picks + produces + wins | **byte only** (P-B5/6) |
| emitted-body measured win (selection is the wall) | the typed body lowers to a winning C body, ssh rvv | **i16 dot-reduce** (this) |
| hand-variant ceiling only (no compiler) | a win is achievable for the family | **u8** |

So: in the STRICT N3 sense (the autotuner wins on SEVERAL kernels) "若干" is **not
yet met** — that is byte alone. In the sense of *a measured win exists on several
kernels* "若干" IS met (byte + i16 + u8). The precise answer to the task's
question is: **灯 ON for the 2nd family's emitted body; selector-driven e2e is the
wall** — not a flat "met".

## The pick: signed i16 widening dot-reduce (candidate a). Why

Both candidate families won the STEP-1 hand ceiling (i16 dot-reduce + u8). Pick
(a) for distinctness + generalization (the task criterion + advisor):
- (a) is a genuinely DISTINCT family (op `tcrv_rvv.widening_dot_reduce`, dtype
  i16, two input buffers + scalar acc-seed ABI), not a clone of the proven
  signed-i8 byte path; (c) u8 is a near-clone (same shape, unsigned).
- (a) GENERALIZES the N3 mechanism rather than replaying the byte constant:
  - a NEW non-widening accumulate op (`tcrv_rvv.deferred_accumulate`, vadd.vv) —
    the i16 product is ALREADY i32==acc width, so the deferred accumulate is
    same-width vadd.vv, NOT the byte path's widening vwadd.wv;
  - a DIFFERENT pathology + fix: the compiler's current dot-reduce emission is a
    REGRESSION (0.64-0.70x vs scalar) due to a per-iteration vredsum on the
    critical path (latency-bound) — a different failure than the byte kernel's
    under-LMUL grouped-u2. The fix is TWO levers (deferred single-reduce + max
    LMUL), vs the byte kernel's single under-LMUL lever.
  - a DIFFERENT max-legal budget answer (source i16 m4, single widening to i32 m8),
    vs the byte chain's i8 m2 -> i16 m4 -> i32 m8 (two widenings).

## What was BUILT (Stage 2a — dialect + conversion, ssh-rvv numeric-correct + 灯)

Parallel deferred-wide dot-reduce path, narrow i16mf2 dot-reduce + the entire int8
byte path BYTE-UNTOUCHED (all new logic marker-gated):
- **Dialect**: new `tcrv_rvv.deferred_accumulate` op (TableGen + verifier, i32m8 +
  i32m8 -> i32m8, requires an i16m4xi16m4->i32m8 widening product producer, I5);
  a new wide-dot-reduce product relation `signed-i16m4xi16m4-to-i32m8` (parallel
  branch in `WideningProductOp::verify`); parallel wide branches in the load,
  standalone_reduce (i32m8->i32m1 fed by deferred_accumulate), store (i32m1
  lane-0), and setvl/with_vl (SEW16/m4 strip config) verifiers. Hand-fed wide
  dot-reduce IR verifies + roundtrips; 2 negatives fail-closed
  (`test/Dialect/RVV/generic-deferred-wide-dot-reduce-dataflow.mlir`).
- **Conversion**: `isDeferredWideDotReduceBody` recognizer (keyed on the
  deferred_accumulate marker) + `emitDeferredWideDotReduceBody`/`...Epilogue`
  emitters. Structural lit
  (`test/Conversion/RVV/rvv-to-emitc-widening-dot-reduce-wide-lmul.mlir`, PASS).
- **ssh rvv NUMERIC (I8)**: the compiler-emitted body == genuine scalar oracle,
  EXACT, at n=0,1,7,31,257(prime tail),256,1024,4096,16384,65536
  (`ssh_rvv_numeric_result.txt`, DEFERRED_WIDE_DOT_REDUCE_ALL_PASS).
- **ssh rvv 灯 (I8)**: the EMITTED body (verbatim, not hand-written) wins the
  3-way — see LAMP_EMITTED_BODY.md (3.9-7.5x vs scalar / 5.7-11.1x vs the
  compiler's current emission / 2.2-3.8x vs competent naive, every n).

## The WALL (Stage 2b — selector-driven end-to-end): NOT crossed (honest stop)

The compiler EMITS the winning 2nd-family body and that body WINS, but it does not
yet PRODUCE it AUTOMATICALLY from a kernel via the selector. The wall is the
realization owner (`RVVContractionSelectedBodyRealizationOwner.cpp`): the byte
deferred-wide e2e (P-B5) gates on the byte pre-realized body op type
(`TypedWideningProductReduceDequantize...`), runs the i8-pinned selector
enumeration (`enumerateRVVLowPrecisionAccumulatorLMULRungs`, `kSourceRungs` start
at i8 mf4, TWO widening steps), and calls `realizeDeferredWideDequantBody`. A
parallel dot-reduce e2e needs: (a) a NEW selector enumeration for the i16 SINGLE-
widening chain (source rungs {mf2,m1,m2,m4}, product==acc), (b) a new
`createRealizedGenericDeferredAccumulate` op-builder + `realizeDeferredWideDotReduceBody`,
(c) threading through the dot-reduce pre-realized body branch + its stamped-fact
validators — the named multi-file wall (a full P-B5-scale commit;
gearbox-current-state.md / P-B5 "each rebuild surfaced the next narrow-pinned
site"). Time-boxed out per the task's "STOP at a genuine wall" discipline; the
byte family already proves the selector mechanism end-to-end, so the 2nd family's
selector-driven production is the cherry, not the whole N3 claim.

## 灯 status precisely (mirrors P-B4 vs P-B5)

| state | byte (1st family) | i16 dot-reduce (2nd family) | u8 |
|---|---|---|---|
| hand ceiling win (ssh rvv) | P-B1 | DONE (STEP1_CEILING.md) | DONE (insurance) |
| compiler emits correct body (ssh rvv) | P-B3 | DONE (numeric ALL_PASS) | — |
| 灯 ON on emitted body (ssh rvv) | P-B4 | **DONE (LAMP_EMITTED_BODY.md)** | — |
| selector-driven e2e | P-B5/6 | WALL (2b, not crossed) | — |

## Build / lit

Full clean rebuild GREEN. lit: 600 tests, 597 pass, exactly the 3 pre-existing
documented environmental reds (verified: all 3 FAIL at HEAD a525d630 without these
changes via stash+rebuild; the 2 new lit tests PASS). RVV dialect/conversion/emitc
160/160; narrow i16mf2 dot-reduce + the int8 byte/packed-i4/clamp paths
byte-identical. NOT git-committed (per task).
