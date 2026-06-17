# INC-30 — iq4_nl × q8_0 coverage breadth (Family A, the CODEBOOK class)

A STRUCTURED, byte-exact, compiler-emitted drop-in for the REAL llama.cpp kernel
`ggml_vec_dot_iq4_nl_q8_0`. iq4_nl opens the **CODEBOOK** quantization class — the
only remaining structural class after the 10 common block-quant + K-quant dot
kernels. The block byte SHAPE is byte-identical to block_q4_0 (stride 18, nibbles
at +2, low/high nibble ↔ q8[0..15]/q8[16..31]); the ONE genuinely-new piece is the
weight decode: the 4-bit nibble does NOT decode linearly (`nibble − 8`) — it INDEXES
a 16-entry NON-LINEAR int8 codebook `kvalues_iq4nl[16] = {−127,−104,−83,−65,−49,
−35,−22,−10,1,13,25,38,53,69,89,113}` (a LOOKUP). Everything else (block loop, dual
fp16 scale, fp32 fold, strip/unroll/elision shape knobs) reuses the Family-A template.

## (1) The op + the codebook-lookup lowering + files

### New op `tcrv_rvv.iq4_nl_q8_0_block_dot` (RVVOps.td)
Mirrors the block-dot ops (ABI: vx,vy,s,n,vl; result i32 m1; the SAME shape knobs
`integer_core_lmul`/`multi_block_factor`/`strip_elision`). Block-format mirror attrs
(I4): `qk=32`, `weight_block_stride=18` (sizeof block_iq4_nl), `activation_block_stride=34`,
`quant_byte_offset=2`, `activation_high_byte_offset=16`. The NEW structural fact is
the 16-entry codebook: `codebook : DenseI8ArrayAttr` — a STRUCTURAL fact (like the
strides), carried in the typed body and realized into the body, NOT an arithmetic
decode and NOT a string plan read. Fail-closed verifier (I7) pins every fact, the
codebook size (==16), and the m1 anchor (the codebook gather needs the table
register's VLMAX ≥ 16).

### The codebook-lookup lowering (the NEW class mechanism — RVVToEmitC.cpp)
`emitIQ4NLQ8_0BlockDot` reuses the Family-A block-loop / scale-read / store / fp32-fold
scaffolding and shares `emitOffsetBinaryProductFromDecodedValue` (the vwmul/vwmacc
asymmetric widening product) verbatim. The codebook decode REPLACES the q4_0
offset-binary `vxor/vsll/vsra` arithmetic with ggml's exact RVV gather method:
- the 16-entry codebook is emitted as a structured `static const int8_t[16]` decl
  (the task-sanctioned structured const) and broadcast-loaded ONCE above the block
  loop via `__riscv_vle8_v_i8m1(kvalues, 16)` → `values`;
- per strip: split the packed nibble byte into two UNSIGNED index lanes
  (`vand_vx_u8m1 0x0F` low / `vsrl_vx_u8m1 0x04` high), then GATHER each through the
  table — `vrgather_vv_i8m1(values, idx)` → the two signed-i8 weight lanes v0/v1;
- v0/v1 feed the SHARED `vwmul`(low ↔ q8[0..15]) / `vwmacc`(+high ↔ q8[16..31]) →
  `vwredsum` → i32 `sumi`;
- the fold is ggml's iq4_nl scales-first order `sumf + (float)sumi*(d_x*d_y)`
  (`d = d_y*d_x` in `_generic`), grouped into ONE `emitc.expression` (FMA-stable).

The codebook gather PINS the m1 anchor (an N1 capability-legality fact): vrgather's
source/index/dest share one vtype, so to index all 16 entries the `values` register's
VLMAX must be ≥ 16 — at VLEN=128 e8, m1→16 (the half-block), mf4→4 (drops 12 of 16,
silently wrong). The verifier rejects non-m1 fail-closed. The m1 half-block fits one
`vsetvl_e8m1(16)`, so the codebook class is the loop-free ("elided") form, correct
∀ VLEN≥128 — apples-to-apples with ggml's VLEN≥128 assumption; a VLEN<128 board
(zve32x) cannot host the gather (capability-legality divergence, not a coverage gap).

### Files
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlBlockDotIQ4NLQ80Op`
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `GgmlBlockDotIQ4NLQ80Op::verify`
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `emitIQ4NLQ8_0BlockDot`,
  `isIQ4NLQ8_0BlockDotBody`, dispatch entry
- tests: `test/Dialect/RVV/iq4-nl-q8-0-block-dot-dataflow.mlir`,
  `test/Conversion/RVV/rvv-to-emitc-iq4-nl-q8-0-block-dot.mlir`

## (2) Byte-exact vs ggml iq4_nl (ssh rvv, riscv64, clang 18.1.3, VLEN=128)

Pipeline: `tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`.
The validated emission is the **m1/elided** shape (one `vsetvl_e8m1(16)` half-block
cover, no inner strip loop) — the SAME shape pinned by the committed conversion lit
test (`rvv-to-emitc-iq4-nl-q8-0-block-dot.mlir` carries `integer_core_lmul="m1",
strip_elision="elided"`), so the byte-exact-validated bytes ARE the lit-pinned bytes.
Harness `inc30_validate.cpp`: 300 reps × 11 n in {32..8192} random + 6 edge classes ×
11 n + the negative controls. PRIMARY target = ggml's REAL RVV iq4_nl method
(vand/vsrl → vrgather → vwmul/vwmacc → vwredsum); cross-checked vs a faithful
`_generic`. (m1 is the only legal anchor for the codebook gather; the robust strip
form is structurally uniform with the siblings but cannot be multi-strip-exercised on
this VLEN=128 board — at m1/VLEN≥128 the half-block is always one strip, and the only
multi-strip case is VLEN<128 where the gather is itself illegal, an N1 legality fact.)

| -ffp-contract | ggml-rvv FAILs | _generic FAILs | neg-ctrl-1 (codebook) | neg-ctrl-2 (nibble) |
|---------------|----------------|----------------|-----------------------|---------------------|
| off           | **0** / 3366   | **0** / 3366   | diverges OK           | diverges OK         |
| on            | **0** / 3366   | **0** / 3366   | diverges OK           | diverges OK         |
| fast          | **0** / 3366   | **0** / 3366   | diverges OK           | diverges OK         |
| default       | **0** / 3366   | **0** / 3366   | diverges OK           | diverges OK         |

**0 failures** vs BOTH references, all 4 fp-modes, 3366 cases each. The edge set
exercises ALL 16 codebook entries (the marching-nibble pattern), the extreme entries
(nib all-0 → kvalues[0]=−127, nib all-15 → kvalues[15]=+113), and q8 ±127 saturation.
The `_generic` delta is 0 in every mode — iq4_nl's single-term fold `(float)sumi*d`
has no cross-statement FMA-formation ambiguity (unlike the two-term q4_1 fold), so
there is no `=fast` reference artifact. Evidence: `ssh_rvv_byte_exact_stdout.txt`.

### Negative controls (the codebook is the LIVE mechanism)
- **Control 1 (in-harness, reference-knob):** SAME input bytes, reference computed
  with the WRONG (linear nibble-8) codebook `{−8..7}` (= the q4_0 decode). Our kernel
  — which embeds the real `kvalues_iq4nl` — MUST diverge: `ours=416178.34` ==
  real-codebook ref, ≠ linear-codebook `25194.37`. Shows the kernel's table values are
  load-bearing.
- **Control 1b (COMPILER-KNOB, end-to-end — `ssh_rvv_codebook_knob_stdout.txt`):** the
  stronger control. TWO kernels emitted BY THE COMPILER from two MLIR inputs that
  differ in ONLY the `codebook=` attr (real `array<i8: -127..113>` vs linear
  `array<i8: -8..7>`); the emitted `.cpp` are byte-IDENTICAL except the
  `static const int8_t[16]` decl flips. On ssh rvv, same input bytes: the real-attr
  kernel = `2280072` (== real-table math), the linear-attr kernel = `138029.73`
  (== linear-table math == q4_0 on those bytes), and the two diverge. Proves the attr
  → emission → HW-result link end-to-end: flipping ONLY the compiler's codebook knob
  flips the hardware answer. Artifacts: `iq4_nl_LINEAR_codebook.cpp`,
  `inc30_codebook_knob.cpp`.
- **Control 2 (non-vacuous):** flip ONE weight nibble → our kernel diverges from a
  fresh ggml run on the unperturbed data (the nibbles are consumed; check non-vacuous).

## (3) raw()=0 + structured proof

`raw()` count in the emitted kernel = **0**. The lowered emitc IR has **0** residual
`tcrv_rvv.*` / `unrealized_conversion_cast`, and **0** unstructured C-blob verbatim:
the only verbatims are 28 `tcrv_emitc.*` provenance comments (I4 mirror) + the ONE
sanctioned `static const int8_t[16]` codebook decl. The codebook TABLE and the GATHER
are STRUCTURED (the const decl renders the verified attr entries; the gather is
`call_opaque` nodes). Evidence: `structured_proof.txt`.

## (4) lit + reds

Both NEW iq4_nl lit tests PASS (dataflow verifier + structured lowering). Full suite:
**647/650 passed**. The 3 reds are the pre-existing, documented `Scripts/`
e2e-self-test failures (a dequant-metadata self-test assertion + two computed-masked-
strided dry-runs) — Python self-tests that do NOT invoke the iq4_nl op; this change is
purely additive to three compiler files. All sibling block-dot lit tests still PASS
(their FileCheck pins their exact emission); the change is byte-additive.

## (5) The codebook lookup (the new class mechanism) — how structured

The codebook class is the FIRST structural class after the arithmetic-decode block/
K-quants: the nibble→int8 map is a 16-entry NON-LINEAR LOOKUP, not arithmetic. It is
made structural by (a) a `DenseI8ArrayAttr:$codebook` verified-size-16 op attr — the
table is a first-class structural fact, the dispatch key is the op identity; and (b)
the emission realizing it as a `static const int8_t[16]` decl + a one-shot `vle8`
broadcast + per-strip `vrgather_vv_i8m1` lookups (ggml's exact board method). The
gather reuses the existing asymmetric product/reduce half, so the whole new class is
~one decode helper. The COMPILER-KNOB control (Control 1b) is the controlled one-knob
proof: two kernels emitted by the compiler from MLIRs differing ONLY in the
`codebook=` attr produce byte-identical `.cpp` except the table decl, and on hardware
the real-attr kernel ≠ the linear-attr kernel (each matching its own table's math) —
so the codebook attr, not the surrounding structure, is what makes this kernel iq4_nl,
proven end-to-end through the compiler to the board.

## (6) What remains

- **mxfp4** (the fp4 e2m1 decode): a different 4-bit decode (a tiny float format), not
  a codebook gather — a small NEW decode helper on the same block structure.
- **The deep IQ super-block codebooks** (iq1_*/iq2_*/iq3_*): the niche tail — multi-
  level codebook + super-block scale machinery (the K-quant-style outer structure
  crossed with the codebook gather). Much larger, low-priority for a Q4_0 model.
- **Honesty:** like the Family-A siblings, this is HAND-AUTHORED-typed-body emission
  coverage (the op lowered directly), NOT autotuner-SELECTED. No schedule-materialize
  pass / capability-divergence demo was added for iq4_nl (the op pins m1 by its own
  codebook legality). Perf not measured (scope = COVERAGE; ggml's iq4_nl RVV kernel
  processes 2 blocks/iter, so parity-or-slower is expected).
