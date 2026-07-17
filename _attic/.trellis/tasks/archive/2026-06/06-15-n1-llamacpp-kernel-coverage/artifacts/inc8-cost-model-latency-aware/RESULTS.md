# INC-8 — latency-aware cost model: the autotuner derives Q8_0's TRUE optimum

The deliverable (the user's "if perf falls short, improve the COMPILER'S tune"
mandate): the block-quantized-dot autotuner mis-picked Q8_0's `multi_block_factor`
(it credited unroll the SAME for every kernel and picked factor=4 for both Q4_0
and Q8_0 — correct for Q4_0, ~5% too slow for Q8_0). INC-8 refines the SHARED
cost model so the optimal factor EMERGES per kernel from a DERIVED structural fact
— the integer core's latency-chain depth — NOT a per-kernel lookup. After the fix
the autotuner picks Q4_0→(m1,4,elided) [unchanged, still beats ggml] and
Q8_0→(m2,2,elided) [its measured optimum, ~5% faster than the old mb4 pick].

## (1) The refined cost model (formula + how coreLatencyDepth is derived)

File: `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`.

### The formula (capability-blind, shared by Q4_0 and Q8_0)

```
cost = kReductionUnit        * reductionsPerBlock                       (anchor)
     + kOuterLoopOverhead    / min(multiBlockFactor, coreLatencyDepth)  (saturating unroll reward)
     + kUnrollOverflowPenalty * max(0, multiBlockFactor - coreLatencyDepth)  (over-unroll penalty)
     + kStripPenalty(elision)* multiBlockFactor                         (strip-loop cost)
     + kBaseConstant
```

Constants (unchanged from the committed model, plus ONE new shared calibration
constant): kReductionUnit=600, kOuterLoopOverhead=500, kRobustStripPenalty=170,
kElidedStripPenalty=40, kBase=120, **kUnrollOverflowPenalty=250 (new)**.

The ONLY change vs the committed model is the unroll term:
- **before**: `kOuterLoopOverhead / multiBlockFactor` — monotone-decreasing in the
  factor for ANY kernel ⇒ the elided argmin was ALWAYS factor=4.
- **after**: the unroll reward divisor is CLAMPED to the latency depth
  (`min(factor, depth)`), and unrolling past the depth charges a per-extra-block
  penalty (`kUnrollOverflowPenalty * max(0, factor - depth)`). The useful overlap
  saturates at the core's dependent-op chain length; beyond it, more unroll only
  adds code/i-cache/strip-setup with no latency-hiding payoff (the measured Q8_0
  mb4 regression). Loop-control amortization is folded into the same term
  deliberately: past saturation its second-order gain is dominated by the
  per-extra-block setup cost — which is exactly the observed Q8_0 mb4 turn-up.

### How `coreLatencyDepth` is DERIVED (structural, not a constant)

It is a COMPUTED SUM read off the kernel's integer-core dependency chain (helpers
`getRVVBlockDotDecodePrefixLength` / `getRVVBlockDotCoreLatencyDepth`):

```
coreLatencyDepth = kBaseProductReduceChain (=2) + decodePrefixLength(quantFormat)
```

- `kBaseProductReduceChain = 2` — the per-block widening-product → reduce chain
  (`vwmul/vwmacc → vwredsum`) that EVERY block-dot kernel has. The floor of the
  dependent-op depth between a block's load and its scalar reduce.
- `decodePrefixLength(format)` — the dependent decode ops the core runs BEFORE the
  product, derived from the quant FORMAT:
  - **plain int8** (Q8_0): operand already int8 → **0** decode ops.
  - **nibble offset-binary** (Q4_0): the one-sided nibble unpack + offset-binary
    `−8` decode the emitter realizes (`emitOffsetBinaryDecodeProductValue`: the
    `&0x0F` / `>>4` / XOR-`0x88` / sign-extend / widen chain) → **5** ops.

So **Q8_0 depth = 2 + 0 = 2** (`vwmul → vwredsum`) and **Q4_0 depth = 2 + 5 = 7**
(decode chain → `vwmul → vwmacc → vwredsum`). The wrappers supply the derived
depth: `computeRVVQ40ShapeCost` passes `"nibble-offset-binary"`,
`computeRVVQ80ShapeCost` passes `"plain-int8"`.

The Q4_0 chain matches the task's stated 7-op chain (vxor→vsll→vsra→vsra→vwmul→
vwmacc→vwredsum); Q8_0 matches the 2-op chain (vwmul→vwredsum).

### Why the factor EMERGES (the principled mechanism)

- **Q4_0, depth 7 ≥ the unroll range {1,2,4}**: `min(factor,depth)=factor` and the
  overflow term is 0 across the whole range ⇒ the cost stays monotone-decreasing
  ⇒ the elided argmin lands at **factor=4** (its ssh-rvv optimum: a long decode
  chain has lots of latency to overlap; more blocks/iter helps monotonically).
- **Q8_0, depth 2 < 4**: factor=4 exceeds the depth ⇒ pays the overflow penalty ⇒
  the elided argmin lands at **factor=2** (its ssh-rvv optimum: a short chain
  saturates the overlap at 2 blocks, then over-unroll regresses).

The factor is NEVER written down per kernel. The per-kernel input is the DERIVED
depth; the factor is the argmin the depth produces.

### Files changed
- `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h` — the formula, the new
  `kRVVQ40UnrollOverflowPenaltyCost`, the depth-derivation helpers
  (`kRVVBlockDotBaseProductReduceChain`, `getRVVBlockDotDecodePrefixLength`,
  `getRVVBlockDotCoreLatencyDepth`), and the two wrappers (`+ <algorithm>`).
- `test/Plugin/RVVQ40Q80ShapeSelectionTest.cpp` + `.test` wrapper — updated the
  Q8_0 structural/selection asserts; added `runDepthDrivesFactorTest` (the
  derived-not-lookup proof: flipping ONLY the depth flips the argmin factor),
  `runLatencyDepthIsDerivedSumTest` (the depth is a computed sum), and
  `runQ40CostUnchangedByRefactorTest` (additivity: Q4_0 costs bit-identical).
- `test/Conversion/RVV/rvv-q8-0-q8-0-block-dot-autotuner-divergence.mlir` — the
  full-V CHECK lines now expect (m2, factor=2, elided), two elided cores.

## (2) Before/after autotuner picks (full-V + zve32x) — autotuner_picks_after.txt

| kernel | profile | BEFORE (depth-blind) | AFTER (latency-aware) |
|---|---|---|---|
| Q4_0 | full-V (rv64gcv)     | (m1, **4**, elided) c=1005 | (m1, **4**, elided) c=1005 — UNCHANGED |
| Q4_0 | zve32x               | (m1, **2**, robust) c=1310 | (m1, **2**, robust) c=1310 — UNCHANGED |
| Q8_0 | full-V (rv64gcv)     | (m2, **4**, elided) c=1005 | (m2, **2**, elided) c=1050 — **FIXED** |
| Q8_0 | zve32x               | (m2, **2**, robust) c=1310 | (m2, **2**, robust) c=1310 — UNCHANGED |

The capability divergence (elided⇔robust by Zvl128b) is PRESERVED on both kernels.
Only the Q8_0 full-V `multi_block_factor` moved (4→2), exactly the mis-pick.

## (3) ssh-rvv re-bench

**Emit-via-compiler provenance (the task constraint).** The NEW pick is exported
as the compiler's own output via the working pipeline
`tcrv-opt …--tcrv-rvv-materialize-q8-0-schedule=march=rv64gcv
--tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`
(`q8_fullv_compiler_emitted_mb2.cpp`), wrapped under the bench's 4-arg ABI by a
thin shim (`kern_q8_compiler_mb2.cpp`) that clang -O3 fully INLINES (objdump of
`kern_q8_compiler_mb2`: the main `+=2` loop has the two elided cores, the tail has
the robust core — the mb2-elided structure executes at runtime). This
compiler-emitted kernel is benched (the COMPILER-EMITTED row below) and PASSES the
byte-exact correctness gate — so "emit only via the compiler" is satisfied by a
real measured row. Its structural identity to the hand mb2 (39/39 intrinsics,
verified) lets the ABI-matched hand pair carry the FACTOR comparison without the
shim confound (see below). Evidence:
`board_microbench_compiler_emitted_stdout.txt`.

Board: `ssh rvv`, 64-core rv64imafdcv+zvfh, VLEN=128, clang 18.1.3. `taskset -c 3`
pinned, best-of-200 INTERLEAVED rounds (min rejects noise), n=4096 (128 blocks),
`-O3 -march=rv64gcv_zfh_zvfh -mabi=lp64d -ffp-contract=fast`. FAIR separate-TU
ggml baseline (direct `_Float16` FPR scale load, same as real ggml). Reproduced
across runs (best-ns stable <0.3%; quietest run shown — ggml settles at ~835 ns):

| shape | best ns/call | ratio vs ggml | role |
|---|---|---|---|
| ggml (real, fair) | 835 | 1.000x | reference |
| **m2_mb2_elided (NEW inc8 pick, ABI-matched)** | **845** | **1.012x (parity)** | the FIX — factor result |
| m2_mb1_elided (= ggml's hand-written shape) | ~845 | ~1.01x | cluster optimum |
| **m2_mb4_elided (OLD pick, ABI-matched)** | **887** | **1.062x (~6% slower)** | the mis-pick inc8 fixes |
| m2_mb2_robust (zve32x pick) | 1056 | 1.265x | correctly slower (extra strip work) |
| m2_mb1_robust | 1105 | 1.325x | robust single-block |
| COMPILER-EMITTED mb2 (provenance/byte-exact row) | 874 | 1.047x | emit-via-compiler proof, see below |

**The factor result (the thing the cost model controls), measured ABI-MATCHED.**
The clean before/after holds the emission path constant — both rows the bench's
4-arg ABI: **NEW mb2 = 1.012x vs OLD mb4 = 1.062x ⇒ the inc8 pick is ~4.7% faster
than the old mb4 pick**, meeting the task's "~5% faster than the old mb4 pick,
~ggml parity" target. (`board_microbench_stdout.txt`.) The mb2/mb1-elided cluster
is at ggml parity; mb4-elided is the ~6%-slow over-unroll the latency-aware model
now avoids. So q8_0 lands at its elided-cluster optimum.

**Why ABI-matched is the honest comparison (do NOT mix the shim in).** The
COMPILER-EMITTED row (1.047x) carries the full 8-arg ggml-ABI shim's scalar
address-arithmetic overhead; comparing it to the clean-ABI mb4 (1.062x) would
change TWO variables at once (factor AND ABI path) and yield a confounded ~1.4%.
That shim overhead is **common-mode** — it would burden a compiler-emitted mb4
exactly as it burdens compiler-emitted mb2 — so in any ABI-matched comparison
(both clean, as above, or both shimmed) it cancels and the ~4.7% factor effect is
recovered. The compiler-emitted mb2 is intrinsic-for-intrinsic IDENTICAL to the
hand mb2 (39/39 shape ops; the emitter is shape-driven, so compiler-mb4 likewise
≡ hand-mb4), so the ABI-matched hand pair faithfully represents the factor effect
on the compiler's OWN shapes. The COMPILER-EMITTED row's job is narrower and it
does it: it proves the benched shape is emit-via-compiler provenance + BYTE-EXACT
(it passes the correctness gate), NOT the factor delta.

**Q4_0 was NOT re-measured — by design, because its emission is provably
unchanged.** The latency-aware refactor is additive on q4_0:
(i) `runQ40CostUnchangedByRefactorTest` asserts every q4_0 cost is bit-identical
to the pre-refactor formula (deep depth 7 ⇒ overflow term inert);
(ii) the q4_0 autotuner pick is unchanged (m1, 4, elided), cost 1005
(`autotuner_picks_after.txt`); (iii) the q4_0 emitted IR is structurally identical
(4-core mb4, raw()=0) and the q4_0 divergence lit test passes byte-unchanged. The
~13% ggml beat is the inc6-committed measurement of that exact (m1,4,elided)
emission, INHERITED unchanged — re-running it would measure identical bytes.

## (4) Byte-exact spot-check of the NEW Q8_0 pick — board correctness gate

The microbench's correctness gate runs BEFORE timing and requires every shape's
`*s` to be `memcmp`-EQUAL (fp32 bits) to ggml's REAL hand-written RVV kernel on
the bench data. The board printed:

```
correctness gate: all shapes bit-exact vs ggml on bench data
```

which INCLUDES the **COMPILER-EMITTED mb2** kernel (and the hand mb2) — so the new
(m2, factor=2, elided) compiler emission is BYTE-EXACT vs ggml on real hardware,
not just structurally. (Byte-exactness is factor-invariant:
integer add is associative so any `multi_block_factor` yields the same `sumi`, and
the per-block fp fold runs in strict ascending order regardless of factor; the
gate confirms it empirically for the new pick.)

## (5) Proof it is principled, not overfit

1. **The depth is DERIVED, the factor EMERGES.** `runDepthDrivesFactorTest`: with
   reductions/elision/constants ALL fixed, feeding the shared formula depth=2 vs
   depth=7 flips the elided argmin factor 2↔4. The factor is a CONSEQUENCE of the
   depth, never a per-kernel constant. (The exact analog of the committed "flip
   Zvl128b flips shape" derivation test, at the cost-model level.)
2. **The depth is a COMPUTED SUM, not a kind→constant switch.**
   `runLatencyDepthIsDerivedSumTest`: depth = base product→reduce chain (2) +
   format decode prefix (q8_0: 0, q4_0: 5). A THIRD plain-int8 block-dot kernel
   inherits depth=2 for free from its format — the litmus test for "derived, not a
   lookup": no per-kernel factor is ever written down.
3. **Insensitive to the exact long-chain length (the anti-overfit signature).**
   ANY depth ≥ the max unroll factor (4) yields BIT-IDENTICAL Q4_0 costs
   (`min(factor,depth)=factor`, overflow=0 over {1,2,4}). What is structural is
   that Q4_0's chain EXCEEDS the unroll range while Q8_0's SATURATES within it;
   the precise value 7 (vs any ≥4) is immaterial. Asserted in
   `runDepthDrivesFactorTest` (depth 4 and 8 both ⇒ factor=4). The picks do NOT
   hinge on a tuned depth value.
4. **The new constant has a WIDE working plateau.** kUnrollOverflowPenalty
   ∈ **[50, 800]** (verified by sweep, step 10) ALL yield the same four required
   picks; the chosen value 250 sits dead-center. A broad plateau, not a
   knife-edge — the opposite of an overfit constant. It is a SHARED calibration
   constant (the same class as the four existing ones); what is per-kernel is the
   DERIVED depth it is compared against.
5. **Additive: Q4_0 unchanged.** `runQ40CostUnchangedByRefactorTest` asserts
   every Q4_0 cost is bit-identical to the pre-refactor depth-blind formula (the
   deep depth makes the overflow term inert). The committed Q4_0 argmin,
   divergence, and ~13% ggml beat are preserved exactly. The Q4_0 emitted IR is
   byte-identical (the q4_0 divergence lit test passes unchanged; raw()=0).

## (6) lit + reds

`check-tianchenrv`: **610/613 pass** — exactly the **3 PRE-EXISTING documented
environmental reds** (`Scripts/rvv-generated-bundle-abi-e2e-self-test` +
the two `…strided-input-widening-dot-reduce-add-dry-run` board-toolchain
self-tests in the Python e2e harness; none touched by this work). The shape
selection unit test (now 14 sub-tests incl. the 3 new derived/additivity proofs)
PASSES; the q8_0 divergence lit test (now expecting mb2-elided full-V) PASSES; the
Q4_0 / deferred-wide / q8_0 dataflow + emitc tests stay byte-identical (additive).

## (7) Honest residuals / deviations

- **Q8_0 is PARITY, not a beat** (as in inc7): ggml's q8_0 RVV kernel is already
  the m2-single-block-elided shape, so the breadth proof is "the compiler DERIVES
  ggml's optimal shape AND avoids the unroll over-reward." INC-8's contribution is
  making the autotuner's DERIVED pick land in the optimum cluster (mb2, parity)
  instead of the ~6%-slower mb4. A beat was never available on q8_0 (ggml is
  already optimal); the win is "the compiler now tunes it correctly, derived."
- **The depth taxonomy is per-FORMAT, by design.** `decodePrefixLength` keys on
  the quant format (plain-int8 / nibble-offset-binary), which is the kernel's
  structural decode signature, not a per-kernel hand value. Extending to q5_0
  (5-bit, an extra high-bit-merge op) or q4_1 (scale+min) adds a format branch
  with its own DERIVED op count — the same mechanism, not a new lookup table.
- **The unroll range is {1,2,4}** (the bounded shape knob). The depth only has to
  resolve "saturates within vs exceeds" this range; finer depth resolution is not
  needed and not claimed.
- The materialize passes stay opt-in / march-selection-live (NOT ssh-rvv-probe-
  live, I6) — same posture as the committed siblings.
