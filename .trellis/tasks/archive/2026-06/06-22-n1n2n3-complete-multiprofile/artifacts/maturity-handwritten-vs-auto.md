# Maturity: hand-written kernels vs auto-optimizing compiler (read-only investigation)

**Verdict: hand-written kernel library + capability dispatch — NOT an auto-optimizing compiler.** No
software-pipelining / auto-vectorization / auto-scheduling pass exists. Every vector instruction, the
2-strip ILP split, the lane-wise vwmacc, and all unrolling are hand-emitted as literal RVV intrinsics
in the C++ emitters. "Schedule/Gearbox/autotuner" select *which* hand-written shape to stamp; they do
not generate or schedule code.

## 1. ILP / vectorization / unroll: HAND-WRITTEN in the emitter C-emit
- `emitRepackGemvQ4_0Q8_0` (`RVVToEmitCBlockQuantLinear.cpp:2853`) emits raw intrinsic *strings*:
  `__riscv_vwmacc_vx_i16`+l16 (L3022), `__riscv_vle8_v_i8`, `vsll/vsra`, `vsetvl` — built by hand via
  `rewriter.create<emitc::CallOpaqueOp>(..., calleeString, ...)`. The emitter literally concatenates
  the LMUL suffix into the intrinsic name; nothing infers it.
- The **2-strip ILP split** is a C++ `for (int64_t h = 0; h < numHalves; ++h)` loop *in the emitter*
  (L3157/3165/3169/3191) that unrolls each strip into separate emitted loads/decodes/vwmaccs. The
  comment (L3152) even says "compute every strip's byte offset FIRST … then issue all loads — preserves
  HEAD's exact node order" — i.e. a human hand-staged the load/compute interleave for ILP. `numHalves`
  (=2 @VLEN128, 1 @VLEN256) is the only knob; the structure is fixed by hand.
- `emitQ4_KSuperBlockAux32Core` (`RVVToEmitCKQuant.cpp:902`) and `emitIQ1SQ8KBlockDot`
  (`RVVToEmitCTernaryBinary.cpp:73`) are the same: literal `__riscv_vle8_v_u8m2`, `__riscv_vadd_vv_u8m2`,
  `vluxei16` gather strings; `for (chunk…)` C++ unroll loops. emitIQ1S comment (L73-78) says it "adopts
  ggml's __riscv_vluxei16" — i.e. transcribed from a reference kernel by hand.
- ~100 distinct `__riscv_*` intrinsic name-strings are emitted by hand across the emitters; the
  `emitc::ForOp`s (102 total) are *runtime* C for-loops the emitter writes — not pipelined by any pass.

## 2. Auto scheduling / pipelining / vectorization pass: NONE
- grep `pipelin|software.pipelin|autovec|auto.schedule|tiling|loop.unroll` in `lib/Plugin/RVV/` and
  `lib/Transforms/`: **zero** real hits (only capability-planning passes: VariantSelection,
  EmissionReadiness, ExecutionPlanCoherence — these choose/legalize variants, they don't optimize code).
- **No MLIR linalg/affine/vector dialect auto-lowering**: grep for `linalg.|affine.|vector::|LinalgTiling
  |createLoopPipelining|createLoopUnroll|VectorizationPass` in Conversion + whole tree = 0 (one comment
  match only). We do NOT lower from a high-level op via upstream MLIR schedulers; we go straight to
  hand-written EmitC.

## 3. "Pipelining": the human stages ILP when writing the emitter — the compiler does not insert it
The load/compute/accumulate interleave (the "software pipelining" flavor) is baked into the emitter's
node-emission order by the author (see the explicit "issue all loads first, then decode, then vwmacc"
phasing, L3152-3198). No pass reorders for latency. The downstream C compiler (GCC/clang on-board) does
its own backend scheduling of the emitted intrinsics, but *our* layer inserts none.

## 4. The "autotuner/Gearbox/Schedule" is selection, not auto-optimization
- `selectGenericSchedule` (`RVVScheduleDescriptorRegistry.cpp:216`) enumerates a **static legal candidate
  set** (`enumerateRVVQ40Q80ShapeCandidates`, etc.) and picks: (a) a pre-recorded **offline** tuning-record
  winner ("measured-best", measured by a human-run on-board best-of-N driver, NOT at compile time), else
  (b) a **static analytic argmin** over a cost *proxy* (`computeBlockDotShapeCostCore`: latency-depth +
  vreg-footprint formula — not a measurement, not a search loop, no rdcycle/clock anywhere in the compiler).
- The knobs it stamps are coarse: `integer_core_lmul {mf2,m1}`, `multi_block_factor {1,2,4}`,
  `strip_elision`. It chooses among hand-written kernel shapes; it does not synthesize or schedule them.
- `isSchedulePinned()` lets a hand-authored shape override selection entirely — the hand-written kernel
  is the source of truth; the gearbox only picks an LMUL/strip-width variant of it.

## Honest classification
We are a **hand-written RVV kernel library wrapped in capability dispatch** (closer to a capability-aware
BLAS: pick the kernel/shape that fits the target's caps), **not** a TVM/Halide-style auto-scheduling /
auto-vectorizing compiler. There is no cost-model search, no polyhedral/affine scheduling, no
auto-pipelining, no autovectorization. Our defensible "compiler" novelty is in **N1 capability-as-IR /
N2 zero-core-branch family dispatch / N3 capability-&-resource-aware variant *selection* (Gearbox)** —
i.e. choosing & legalizing the right hand-written kernel per target — **not** in automatically optimizing
or generating the kernel code. Calling the kernels' ILP/vectorization "compiler auto-optimization" would
be dishonest: a human wrote each intrinsic and staged the ILP by hand in the emitter.
