# INC-21 — q5_0 × q8_0 coverage breadth (Family A, 5-bit weight)

A STRUCTURED, byte-exact, compiler-emitted drop-in for the REAL llama.cpp kernel
`ggml_vec_dot_q5_0_q8_0`, with the capability/resource-aware autotuner inheriting
the SAME enumerate→prune→select engine — mirroring the q4_0/q4_1/q8_0 pattern. The
ONE genuinely-new piece is the 5-bit weight reconstruction (a 4-bit nibble PLUS a
per-element 5th high bit packed in a separate 32-bit qh field + offset-binary `-16`
bias); everything else (block loop, dual fp16 scale, fp32 fold, strip/unroll/elision
shape knobs, the autotuner) reuses the existing template.

## (1) The op + the 5-bit-reconstruction lowering + the autotuner inheritance

### New op `tcrv_rvv.q5_0_q8_0_block_dot` (RVVOps.td)
Mirrors the block-dot ops (ABI: vx,vy,s,n,vl; result i32 m1; the SAME shape knobs
`integer_core_lmul`/`multi_block_factor`/`strip_elision`). Block-format mirror attrs
(I4): `qk=32`, `weight_block_stride=22` (sizeof block_q5_0), `activation_block_stride=34`,
`quant_byte_offset=6` (the q5_0 nibbles follow the fp16 scale + the 4-byte qh field),
`activation_quant_byte_offset=2` (the q8_0 int8 quants — DISTINCT from the weight's +6
because the q8_0 block carries no qh), `activation_high_byte_offset=16`,
`weight_qh_byte_offset=2`. Fail-closed verifier (I7) pins every fact + the shape-knob
legality (`elided` requires the m1 anchor).

### 5-bit reconstruction (the NEW lowering piece — RVVToEmitC.cpp)
`emitFiveBitOffsetBinaryDecodeProductValue` (added alongside the offset-binary q4_0
and unsigned q4_1 decoders, both byte-untouched). Per strip at chunk offset c:
- unsigned nibble unpack (`vand 0x0F` low / `vsrl 0x04` high, on the u8 weight lane);
- per-element 5th-bit injection from the broadcast 16-bit qh halves
  (`qhLow16 = qh & 0xFFFF`, `qhHigh16 = qh >> 16`, read once per block alongside the
  fp16 scales): a `vid + c` per-lane shift vector, `vsrl_vv`, `& 1`, `<< 4`, `vncvt`
  narrow u16→u8, `vor` into the nibble — the `vid + c` makes it correct for ANY
  chunk offset (the mf4 multi-strip path) and ANY VLEN;
- offset-binary bias: reinterpret u8→i8 (value-identity for [0,31]), `vsub 16` → i8
  in [-16,15];
- the SAME signed widening product against the plain-i8 q8 halves
  (`vwmul`/`vwmacc`) → `vwredsum` → i32 `sumi`.
The fold is ggml's q5_0 scales-first order `sumf + (d_x*d_y)*sumi` (q8_0's order, NOT
q4_0's `(sumi*d_x)*d_y`), grouped into ONE `emitc.expression` (the FMA-stable form
across the -ffp-contract matrix).

### Autotuner inheritance (DERIVED, not a per-kernel constant)
- `getRVVBlockDotDecodePrefixLength` extended with a NEW format key
  `"nibble-5bit-offset-binary"` → 8 (the LONGEST decode prefix: unsigned unpack +
  5th-bit injection + offset-binary bias). `coreLatencyDepth` is DERIVED from this
  (`kBaseProductReduceChain + prefix`), NOT a hand-set per-kernel value. Like the
  others, depth ≥ 4 saturates the {1,2,4} unroll range, so the argmin matches q4_0's.
- `computeRVVQ50ShapeCost` / `enumerateRVVQ50Q80ShapeCandidates`: the SAME
  `computeBlockDotShapeCostCore` formula + `getRVVQ40ReductionsPerHalfBlock` (q5_0's
  nibble half-block is q4_0's shape) + the SAME `deriveHasZvl128b` legality prune +
  vreg-budget prune.
- `MaterializeRVVQ50Schedule` pass (`--tcrv-rvv-materialize-q5-0-schedule`): a
  byte-for-byte sibling of the q4_1 autotuner pass, stamping the chosen shape + the
  `tcrv_rvv.q5_0_schedule.*` resource-provenance audit trail.

### Files
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlBlockDotQ50Q80Op`
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `GgmlBlockDotQ50Q80Op::verify`
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `emitFiveBitOffsetBinaryDecodeProductValue`,
  `emitQ5_0Q8_0BlockDot`, `isQ5_0Q8_0BlockDotBody`, dispatch entry
- `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h` — the new format key + q5_0
  cost/enumerate
- `lib/Plugin/RVV/RVVQ50ScheduleMaterialization.cpp` (new) + `CMakeLists.txt`
- `include/TianChenRV/Transforms/Passes.{h,td}` + `tools/tcrv-opt/tcrv-opt.cpp`
- tests: `test/Dialect/RVV/q5-0-q8-0-block-dot-dataflow.mlir`,
  `test/Conversion/RVV/rvv-to-emitc-q5-0-q8-0-block-dot.mlir`,
  `test/Conversion/RVV/rvv-q5-0-q8-0-block-dot-autotuner-divergence.mlir`

## (2) Byte-exact vs ggml q5_0 (ssh rvv, riscv64, clang 18.1.3)

Pipeline: `tcrv-opt … --tcrv-rvv-materialize-q5-0-schedule=march=… --tcrv-rvv-lower-to-emitc
| mlir-translate-20 --mlir-to-cpp`. Harness `inc21_validate.cpp`: 300 reps × 11 n in
{32..8192} random + 6 edge classes × 11 n (q5 all -16 / all +15 / qh bit pattern,
crossed with q8 +127 and q8 ±128/+127) + a negative control. PRIMARY target = ggml's
REAL RVV kernel; cross-checked against a faithful `_generic`.

| config (shape)                  | -ffp-contract | ggml-rvv FAILs | _generic FAILs |
|---------------------------------|---------------|----------------|----------------|
| fullv_autotuned (m1/4/elided)   | off/on/fast/—  | **0**          | **0**          |
| zve32x_autotuned (m1/2/robust)  | off/on/fast/—  | **0**          | **0**          |
| mf4_robust (the c≠0 strip proof)| off/on/fast/—  | **0**          | **0**          |

**0 failures** vs BOTH references, all 12 (config × fp-mode) combinations, 3367
cases each. The negative control (flip ONE qh bit → diverges) confirms the check is
non-vacuous and the qh 5th-bit field is consumed. NOTE: at every -ffp-contract mode
the `_generic` delta is also 0 — q5_0's single-term fold `(d_x*d_y)*sumi` has no
cross-statement FMA-formation ambiguity (unlike q4_1's two-term fold), so there is no
`=fast` reference artifact. Evidence: `ssh_rvv_byte_exact_stdout.txt`.

The **mf4-robust** run is the load-bearing check: at VLEN=128 only the mf4 core
exercises c≠0 strips (c=0,4,8,12), so it is the ONLY VLEN=128 config that proves the
`vid + c` 5th-bit strip-offset arithmetic. (A bug here would have ridden silently past
the autotuned m1/elided single-strip shapes — caught and fixed pre-ship; see
"Deviations".)

## (3) Capability divergence (rv64gcv vs zve32x)

The SAME attr-less `tcrv_rvv.q5_0_q8_0_block_dot` op, routed by `--march` only:
- **rv64gcv** (Zvl128b): 9 legal candidates → SELECTS (m1, factor=4, **elided**).
- **rv64gc_zve32x** (no Zvl128b): elided shapes PRUNED → 6 legal → SELECTS
  (m1, factor=2, **robust**).
One capability fact (Zvl128b) → N1 legality divergence (elided vs robust strip loop),
on a structurally NEW 5-bit kernel. Evidence: `divergence_stamp.txt`,
`rvv-q5-0-q8-0-block-dot-autotuner-divergence.mlir`.

## (4) raw()=0 + structured proof

All three emitted .cpp: `raw()` count = **0**. The lowered emitc IR has **0**
non-comment `emitc.verbatim` (the only verbatims are `tcrv_emitc.*` provenance
comments). Every value is an emitc node (variable/load/mul/add/cast/for/call_opaque).
The one sanctioned opaque piece is the scalar fp16→fp32 read + the scalar qh field
read (`emitc.call_opaque`, exactly as the siblings emit their intrinsics). Evidence:
`structured_proof.txt`.

## (5) lit + reds

Full suite: **629/632 passed**. The 3 reds are the pre-existing, documented
`Scripts/rvv-generated-bundle-abi-e2e-*strided*` / `-self-test` failures (a strided
widening dot-reduce pattern-match) — they reference NO q5_0 and predate this work.
The 3 NEW q5_0 lit tests PASS (dataflow verifier + structured lowering + autotuner
divergence). Additive proof: the 7 sibling block-dot lit tests (q4_0/q4_1/q8_0/q6_K +
their divergence) still PASS (their FileCheck pins their exact emission), and a
regenerated q4_1 kernel is BYTE-IDENTICAL to the committed artifact. Full clean
rebuild green.

## (6) Deviations

- **Activation-offset bug, caught by the harness and fixed.** The first cut reused
  the weight `quant_byte_offset` (6) for the q8_0 activation loads. q5_0 moved the
  weight nibbles to +6 (after the qh field), but the q8_0 activation int8 data is
  still at +2 — so the activation was read 4 bytes off. The ssh-rvv harness caught it
  (3366/3366 FAIL vs ggml-rvv AND _generic). Fixed by threading a distinct
  `activation_quant_byte_offset` (=2) attr (mirroring how q4_1 carries separate
  weight/activation offsets); the verifier now pins it, and there is a dedicated
  reject test. After the fix: 0 failures everywhere.
- **Aligned qh read (deployable, not just board-tolerated).** The qh 5th-bit field
  is read as TWO aligned 16-bit halves (`(uint16_t)*` at xb+2 and xb+4), NOT a single
  `uint32_t` load at xb+2 — which would be MISALIGNED for the stride-22 block base
  (UB / trap-emulate per block, the only misaligned access in the block-dot family).
  Little-endian, the two halves are value-identical to `(qh & 0xFFFF)` / `(qh >> 16)`,
  so the decode is unchanged and re-validation stayed 0-failures; the kernel is now a
  genuinely alignment-safe drop-in (matching `_generic`'s `memcpy` safety).
- **Perf not measured.** Scope is COVERAGE BREADTH; ggml's q5_0 RVV kernel is tuned
  (it uses a `vlm`/`vmnand`/masked-`vsub` mask trick + a 32-lane single-shot reduce),
  so parity-or-slower is expected. Not benchmarked here.
