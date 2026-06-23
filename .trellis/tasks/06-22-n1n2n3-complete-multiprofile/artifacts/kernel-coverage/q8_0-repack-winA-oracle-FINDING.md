# q8_0 repack — Win-A·micro (GEVM LMUL ablation) + scalar numeric oracle — FINDING

**Date:** 2026-06-24
**Scope:** stage-2 evidence for the q8_0 repack GEVM kernel `tcrv_rvv.repack_gemv_q8_0_q8_0`
(op `GgmlRepackGemvQ80Q80Op`), BUILT + merged on main (commit 7f7f5a37). Two cells:
(1) numeric oracle (correctness) and (2) Win-A·micro WIDE-vs-NARROW LMUL/strip ablation.
Mirrors the proven q4_1 workflow (`q4_1-winA-oracle-FINDING.md`). NO kernel code modified.
Measured on `ssh rvv` (Sophgo SG2044, RVV1.0, VLEN128, clang18). `taskset -c 0`, best-of-reps min.
Methodology per `N3-METHODOLOGY.md`: Win-A = tune ON(WIDE) vs OFF(NARROW), BOTH compiler-emitted &
vectorized — NO scalar-contribution multiples; the scalar oracle diff is a CORRECTNESS check (norm),
NOT a speedup.

## HEADLINE — the LMUL-width knob's winning direction is INVERTED for q8_0

**q8_0 GEVM Win-A: NARROW/WIDE ≈ 0.18× — i.e. WIDE (m1, 1×16 strip) is ~5.5× SLOWER than
NARROW (mf2, 2×8 strips).** This is the OPPOSITE direction from q4_0 GEVM (~2.1×) and q4_1 GEVM
(~1.80×), where WIDE won. The 0.18× sits one line under q4_1's 1.80× in the same column and is NOT a
typo of 1.80 — for q8_0 the wide arm LOSES. Byte-exact WIDE↔NARROW (norm 0.000e+00) proves both arms
do identical work over identical bytes, so the 5.2× is PURE LMUL-realization cost, nothing to distrust.

**This is a CLEAN POSITIVE for the resource-aware thesis (not a Win-A loss to spin away):** the
RVV1.0/VLEN128 default emit on the real SG2044 (no stamp → `integer_core_lmul` unset → `mf2`,
`half_lanes=8`) IS exactly the NARROW arm — the FASTER arm. The tune picks the winner for q8_0.
Confirmed byte-identical: emitting with `--tcrv-rvv-materialize-repack-strip-width=march=rv64gcv`
(the RVV1.0 stamp, the SG2044 target tier) is byte-for-byte the no-stamp NARROW kernel (`diff` empty).
So NARROW = the SG2044 tune's ACTUAL selection, not merely "the default." The knob's right direction
is quant-type-dependent: q4 wide wins, q8 narrow wins, and the capability-driven default lands on the
faster arm in BOTH cases (q4 via the VLEN-256 / xtheadvector wide stamp; q8 via the RVV1.0 mf2 default).

## How the kernels were produced (both compiler-emitted, host tool)
Local host (x86, LLVM/MLIR 20): `build/bin/tcrv-opt … --tcrv-rvv-lower-to-emitc |
/usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp`, from the SAME source IR
`test/Conversion/RVV/rvv-to-emitc-repack-gemv-q8-0-q8-0.mlir` (single `tcrv_rvv.repack_gemv_q8_0_q8_0`
op). NARROW = lower-to-emitc directly (RVV1.0 leaves `integer_core_lmul` unset → the fractional `mf2`
chain, `half_lanes=8`). WIDE = `--tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector`
FIRST (stamp-time only: it flips the strip-width pass's `integer_core_lmul=m1` + `half_lanes=16` branch
— RVV0.7/xtheadvector has no fractional LMUL), THEN lower-to-emitc. The q8_0 op was wired into
`RVVRepackStripWidthMaterialization.cpp` (`GgmlRepackGemvQ80Q80Op`, line 183) IDENTICALLY to q4_0/q4_1.
Emitted `.cpp` scp'd to rvv, compiled there with clang18 `-O3 -march=rv64gcv_zvfh -mabi=lp64d
-ffp-contract=fast` for BOTH arms (the xtheadvector march is STAMP-TIME only; the actual SG2044 compile
target is `rv64gcv_zvfh` for both).

LMUL spellings host-verified DISJOINT (grep on emitted `.cpp`, zero overlap):
- NARROW (mf2): {i8mf2, i16m1, i32m2, f32m2, f16m1}, **4× vle8 (2 strips)**.
- WIDE (m1):    {i8m1, i16m2, i32m4, f32m4, f16m2}, **2× vle8 (1 strip)**.

q8_0-specific invariants confirmed present in BOTH arms (vs the q4_0 nibble sibling): in-block i32
accumulation via `vwmul`(i8×i8→i16) + `vwadd_wv`(i32 += i16) — and ABSENT by construction: NO
`redsum` (lane-wise dot, no cross-lane wall), NO `vsll`/`vsra`/`vxor` (no nibble decode), NO `vwmacc`
(q4's i16 accumulate), NO `vwadd_vv` (q4's lo/hi combine). grep counts: 0 for all the absent set, 8
(NARROW = 2 strips × 4) / 4 (WIDE = 1 strip × 4) for `vwmul`+`vwadd_wv`.

## Win-A WIDE/NARROW ablation — CLEAN, 3-run consistent (rvv, measured 2026-06-24)

Best-of-reps min latency (`taskset -c 0`). WIDE is also the NOISIER arm (its min wanders 32–37M ns at
4096²; NARROW is rock-steady at ~6.7M) — best-of-reps min reported either way.

| shape (n × nc) | NARROW (mf2, 2×8) | WIDE (m1, 1×16) | **ratio NARROW/WIDE** | WIDE↔NARROW norm | oracle norm |
|---|---|---|---|---|---|
| 4096 × 4096 (run1)            | 6,712,196 ns | 36,310,409 ns | **0.185×** (WIDE 5.4× slower) | 0.000e+00 BYTE-EXACT | 1.129e-06 PASS |
| 4096 × 11008 (mlp)            | 18,023,684 ns | 100,729,605 ns | **0.179×** (WIDE 5.6× slower) | 0.000e+00 BYTE-EXACT | 1.660e-06 PASS |
| 4096 × 4096 (run3, stability) | 6,695,116 ns | 36,754,609 ns | **0.182×** (WIDE 5.5× slower) | 0.000e+00 BYTE-EXACT | 1.129e-06 PASS |

(A separate earlier pass measured 4096×11008 at 0.206× / 0.222× and 4096×4096 down to 0.208× —
the WIDE arm's noise floor; the ratio is stably in the **0.18–0.22×** band, i.e. WIDE ~4.5–5.6× slower,
direction unambiguously inverted.)

### Mechanism (inference, structurally supported — the q8 i32-per-position accumulate, NOT min/nibble)
The discriminator is the per-POSITION loop-carried accumulator, confirmed by grepping the WIDE kernels:
- **q8_0 WIDE** per-position op = `__riscv_vwadd_wv_i32m4` — a single loop-carried i32m4 (LMUL-4)
  accumulate chain, 32 serial dependent adds per block. WIDE runs ONE such 4×-latency chain;
  NARROW splits the same work into TWO independent `i32m2` chains (ILP=2 → ~2× the available
  latency-hiding). q8_0's FULL-int8 product forces the in-block accumulate up to i32 (127·127·32
  overflows i16), so the loop-carried reduction sits at LMUL-4 in WIDE and becomes a serial
  latency wall.
- **q4_1 WIDE** per-position op = `__riscv_vwmacc_vx_i16m2` (×4) — the serial chain element is at
  i16m2; its `vwadd_vv_i32m4` appears only 2×/block (the end-of-block lo/hi combine), NOT in the
  per-position chain. So q4's loop-carried reduction is LMUL-2, and widening to m1 still wins.

So m4 register pressure per se is NOT the killer (q4_1 WIDE ALSO uses i32m4 + f32m4 and won 1.80×) —
the controlling variable is the **i32-per-position accumulate (q8) vs i16-per-position (q4)**. NOT the
min-fold / nibble-unpack the task flagged: those touch the scale-fold and unpack, which are byte-for-
byte identical-work across arms and cancel from the ratio (norm 0 confirms). This is labelled an
inference about the *mechanism* (as the q4 findings labelled their min-fold reasoning); what was
MEASURED is the inverted ratio (clean, 3-run consistent at ~0.18×).

## Numeric oracle (correctness) — PASS, fp-rounding only (rvv, measured 2026-06-24)

Two correctness gates, both PASS at every shape:
1. **WIDE↔NARROW byte-exact (norm = 0.000e+00, EXACT).** Per-row integer `sumi = Σ(w_i8·a_i8)` is
   lane-wise / order-independent and the per-block fp32 fold runs in the same block order regardless of
   LMUL lane-grouping → per-element identical to the bit. This is the STRONGEST gate (norm 0 EXACTLY,
   not "small"): had the harness mis-strided/mis-offset, it would not be 0.
2. **Kernel vs SCALAR q8_0 dequant-matmul oracle: norm 1.1e-6–1.7e-6, bar < 1e-4 → PASS.** q8_0 is
   INTEGER (no fp quantization error), so the integer `sumi` matches the oracle EXACTLY; the only delta
   is fp32 grouping in the per-block `d_x·d_y·sumi` fold summed over nb blocks (kernel: vector f32m2/m4
   running accumulate; oracle: double-precision scalar running sum) — the "~fp-rounding only on the
   final scale fold" the task allowed. Both WIDE and NARROW give the IDENTICAL oracle norm (they are
   byte-identical to each other), confirming the fp delta is the scale-fold grouping, not a per-arm bug.
   Not chased to 0 — the kernel is frozen and the residual is pure fp-association of the d·d·sumi sum.

### Oracle framing (HONEST)
The oracle is a hand-derived **scalar q8_0 dequant-matmul** that consumes the SAME repacked
`block_q8_0x16` (stride 544, fp16 scales @0/32B, int8 quants @32) weights and plain `block_q8_0`
(stride 34, fp16 scale @0, int8 quants @2) activation the kernel reads, dotting the QUANTIZED int8
quants (`sumi = Σ w·a` over 32 in i32, `out = d_x·d_y·sumi`) — NO min term, NO nibble decode, NO q8_1
scaled-sum (q8_0 is symmetric Family-A). `out` is zero-init per trial, so any output element the kernel
failed to write would read 0 against a ref ~O(10) and spike the error — norm at ~1e-6 therefore
confirms the FULL nc output is correct, not spot-checked. Per N3-METHODOLOGY this is a CORRECTNESS
check (norm), not a speedup (a scalar oracle is not a methodology-valid speed baseline).
Caveat on objdump: the SG2044 binutils `objdump` does not decode RVV vtype/vector mnemonics (emits
`.word`), so a binary-level vsetvli histogram could not be produced; the disjoint-LMUL evidence is
SOURCE-level (grep on the emitted `.cpp`, sets above, zero overlap) plus the runtime byte-exact
WIDE↔NARROW agreement (norm 0) — conclusive the two arms ran distinct vectorized code.

## Headline (both cells closed, rvv provenance)
- **Win-A — q8_0 GEVM WIDE/NARROW ratio = NARROW/WIDE ≈ 0.18× (ns/call), i.e. WIDE ~5.5×
  SLOWER, direction INVERTED vs q4_0/q4_1.** Clean, 3-run consistent (0.185× / 0.179× / 0.182×).
  Both arms compiler-emitted from the SAME `tcrv_rvv.repack_gemv_q8_0_q8_0` op; only the LMUL-width
  knob + strip count differ. Mechanism (inferred): q8_0's full-int8 → i32-per-position accumulate makes
  WIDE one serial i32m4 latency chain where NARROW gets ILP=2 from its two i32m2 strips.
- **Oracle — PASS.** WIDE↔NARROW byte-exact (norm 0, EXACT); kernel vs scalar q8_0 dequant-matmul
  norm 1.1e-6–1.7e-6 (fp-rounding on the d_x·d_y·sumi block fold, bar 1e-4), across both shapes.
- **Resource-aware seal:** the RVV1.0/`rv64gcv` stamp emits byte-identical to no-stamp NARROW → the
  SG2044 tune actually SELECTS the faster (NARROW) arm for q8_0.

---

## VLEN256 ISO follow-up (X60 / `ssh k1`) — strip-count/ILP axis ISOLATED from the LMUL axis

**Date:** 2026-06-24. **Box:** SpacemiT X60, RVV1.0, **VLEN256** (`ssh k1`) — a DIFFERENT silicon
than the prior SG2044/`ssh rvv` rows above (do NOT arithmetically reconcile 5.5× vs 1.9×). **Goal:**
test whether the q8_0 GEVM WIDE-loses inversion (measured 5.5× at rvv/VLEN128) ALSO holds at VLEN256,
isolating the **strip-count/ILP axis** by holding LMUL fixed.

### Why this is a DIFFERENT experiment from the rows above (the prior 5.5× was CONFOUNDED)
The headline rows above produced WIDE via the `march=rv64gc_xtheadvector` stamp, which pins
`integer_core_lmul=m1` — so that WIDE arm changed BOTH the LMUL core (mf2→m1, i.e. i32m2→i32m4) AND
the strip count (2×8→1×16). The 5.5× therefore bundled an m4-LMUL latency penalty WITH the strip change.
This follow-up holds **LMUL fixed at the mf2 fractional core** and varies ONLY `half_lanes`:
- **NARROW** = `half_lanes=8` → TWO 8-lane `i32m2` strips (ILP-2, two independent accumulate chains).
- **WIDE**   = `half_lanes=16` → ONE 16-lane `i32m2` strip (serial, one accumulate chain).

Both compiler-emitted from the SAME `tcrv_rvv.repack_gemv_q8_0_q8_0` op via **plain
`--tcrv-rvv-lower-to-emitc` (NO march stamp)**, from two IRs byte-identical except `half_lanes: 8` vs `16`
(`vlen256-iso/iso_{NARROW_hl8,WIDE_hl16}.mlir`). **Host LMUL-identity check (load-bearing):** grep of both
`.cpp` shows IDENTICAL fractional-core spellings in both arms (`i8mf2`/`i16m1`/`i32m2`/`f32m2`/`f16m1`) and
**ZERO** whole-LMUL contaminant (`i8m1`/`i16m2`/`i32m4`/`f32m4`/`f16m2`) in EITHER. The ONLY delta is the
per-strip op count: NARROW emits each of `vle8_v_i8mf2`/`vwmul_vx_i16m1`/`vwadd_wv_i32m2`/`vse32_v_f32m2`
exactly **2×**, WIDE exactly **1×** (and the strip-width literals `…(0,8)`×2 vs `…(0,16)`×1). So the only
thing that varies is two-8-lane-strips-ILP-2 vs one-16-lane-strip-serial — the pure strip-count/ILP axis.

**Selector provenance (SEALED):** emitting from the `half_lanes=8` base via the REAL VLEN256 stamp
`--tcrv-rvv-materialize-repack-strip-width=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc` is
**BYTE-IDENTICAL** (`diff` empty, `vlen256-iso/k_gemv_STAMP256.cpp`) to the hand-set `half_lanes=16` WIDE
arm, and stays at mf2 (0 m1/m4). So `deriveRepackHalfLanes(256,16)=min(16,16)=16` → the K1/VLEN256 selector
ACTUALLY emits the WIDE (1×16) arm; "WIDE" here IS the compiler's pick, not merely a hand fixture.

### Result — the inversion does NOT hold at VLEN256; **WIDE (1×16) is ~1.9–2.0× FASTER**

clang18 `-O3 -march=rv64gcv_zvl256b_zvfh -mabi=lp64d -ffp-contract=fast`, `taskset -c 0`, best-of-reps min,
3 runs/shape (`vlen256-iso/vlen256-iso-ablation.log`). (`_zvfh` added only because the f16 scale-fold
intrinsics need it; it does NOT change VLEN.)

| shape (n × nc) | NARROW (mf2, 2×8) | WIDE (mf2, 1×16) | ratio NARROW/WIDE | **16/8 ratio (WIDE/NARROW)** | WIDE↔NARROW norm | oracle norm |
|---|---|---|---|---|---|---|
| 4096 × 4096 (3 runs)  | 13.70–13.94 M ns | 6.91–7.21 M ns | 1.93–1.98× | **≈ 0.51×** (WIDE ~1.95× faster) | 0.000e+00 BYTE-EXACT | 1.129e-06 PASS |
| 4096 × 11008 (3 runs) | 36.45–37.28 M ns | 18.31–19.21 M ns | 1.94–1.99× | **≈ 0.51×** (WIDE ~1.96× faster) | 0.000e+00 BYTE-EXACT | 1.660e-06 PASS |

The VLEN256 oracle PASS is itself the VLEN-confirmation: `vmv_v_x_i32m2(0,16)` only computes 16 lanes
correctly when hardware VLMAX≥16 (VLEN≥256); on a VLEN128 part vl clamps to 8 and half the output rows go
unwritten → the oracle would spike. WIDE's norm 1.1e-6 PASS therefore proves it ran at VLEN256.

### VERDICT — the cross-VLEN mis-selection hypothesis is **REFUTED at VLEN256** (inversion is VLEN128-specific)
At VLEN256, with the LMUL axis isolated, the single 16-lane strip is ~1.9–2.0× FASTER than the two 8-lane
strips — the OPPOSITE of the rvv/VLEN128 inversion. So the task's hypothesis (that the VLEN-derived selector
picks the SLOWER arm at VLEN256 = a confirmed cross-VLEN LIMITATION) **does NOT hold**: the prior 5.5×
WIDE-loses result was confounded with the mf2→m1 (`i32m2`→`i32m4`) LMUL change, NOT a pure strip-count/ILP
effect. The strip-count inversion is **VLEN128-specific** (and even there it is entangled with LMUL). This is
NOT framed as "the tune correctly diverges" — it is a refutation of a suspected limitation: the suspected
VLEN256 mis-selection is not real on this axis.

**Mechanism (inference, reconciles both VLENs):** `i32m2` holds VLEN/16 i32 lanes — 8 at VLEN128, 16 at
VLEN256. At VLEN256 `half_lanes=16` fills the m2 register in ONE strip; `half_lanes=8` runs TWO strips at
8/16-lane occupancy = ~2× the vector ops at half utilization → ~2× slower, matching the measured 1.9×. At
VLEN128 `half_lanes=8` was FULL utilization and a 16-lane strip would have required m4 (the latency wall the
prior rows hit). The `vlen/16` formula tracks register-fill at both VLENs — which is exactly why this is a
VLEN-dependent effect and "VLEN128-specific" is defensible.

### Artifacts (VLEN256 ISO)
- `q8_0-emit/vlen256-iso/iso_{NARROW_hl8,WIDE_hl16}.mlir` — the two source IRs (identical except `half_lanes`).
- `q8_0-emit/vlen256-iso/k_gemv_{NARROW,WIDE}.cpp` — the two mf2 compiler-emitted arms (LMUL-identical).
- `q8_0-emit/vlen256-iso/k_gemv_STAMP256.cpp` — real `march=rv64gcv_zvl256b` stamp output, BYTE-IDENTICAL to WIDE.
- `q8_0-emit/vlen256-iso/ablation_micro_q80.cpp` — harness (header updated to the ISO framing).
- `q8_0-emit/vlen256-iso/vlen256-iso-ablation.log` — canonical 3-run × 2-shape K1 log (AGREE + ORACLE + RESULT).

---

## Artifacts
- `q8_0-winA-gevm-ablation.log` — canonical 3-run rvv log (AGREE + ORACLE + RESULT lines).
- `q8_0-emit/` — the two compiler-emitted GEVM arms `k_gemv_{NARROW,WIDE}.cpp` (disjoint LMUL) and the
  harness `ablation_micro_q80.cpp` (WIDE↔NARROW agreement + scalar q8_0 dequant-matmul oracle +
  best-of-reps min timing). On rvv under `~/q80-gap-agent/`: same sources + binary `ablation_q80`.
- Emitted host-side via `build/bin/tcrv-opt … --tcrv-rvv-lower-to-emitc |
  /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp`; the WIDE arm via
  `--tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector` (stamp-time m1 trigger), both
  compiled on the SG2044 with clang18 `-O3 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=fast`.
