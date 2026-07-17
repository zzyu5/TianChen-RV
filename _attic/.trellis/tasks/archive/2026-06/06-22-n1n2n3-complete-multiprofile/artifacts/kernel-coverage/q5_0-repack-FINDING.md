# q5_0 repack GEVM — FINDING

**Date:** 2026-06-25 · **HW:** SG2044 (`ssh rvv`), VLEN128, clang-18
`-march=rv64gcv_zfh_zvfh -O3`, `taskset -c 2` · **Strategy:** copy-then-adapt from
the PROVEN q4_0 repack (q5_0 = q4_0 + the 5th high bit).

## Headline

A NEW RVV kernel — the **q5_0 repack GEVM** (block-as-lane, 16 weight columns
across lanes, lane-wise `vwmacc`, NO per-block `vredsum`) — is **built, byte-exact,
and lit-green**. The Win-B micro is an **honest LOSS** at VLEN128: the repack
GEVM runs **0.769×** ggml's real block-dot at the mf2 anchor (0.653× at the m1
wide-LMUL anchor). q5_0 falls in the **compute-bound regime where ggml's wide-LMUL
block-dot wins** — confirming the prior `q5_0/q5_1 block-dot LOSS` pattern now for
the repack path. The block-as-lane reduction-wall removal is real but is outweighed
by the per-nibble transposed-qh expansion cost at the single-column GEVM.

## ggml ships a q5_0 repack? NO

Grepped `llama.cpp/ggml/src/ggml-cpu/{repack.cpp,repack.h}` +
`arch/riscv/repack.cpp`: ggml's repacked block family is q4_0 / q8_0 / q4_K /
q2_K / q5_K / q6_K / iq4_nl / mxfp4 — **no `block_q5_0x16`, no q5_0 gemv**. So the
Win-B baseline is ggml's **REAL shipped RVV q5_0 block-dot**
(`ggml_vec_dot_q5_0_q8_0`, the unified `__riscv_v` body, `vlenb==16`/VLEN128 path:
`vle8` nibbles → `vlm_v_b4` qh mask → `vsub_vx_i8m2_mu` masked −16 over the whole
32-element i8m2 vector → `vwmul_vv_i16m4` → `vwredsum`), run once per output column.

## What was built (the durable foundation)

| Piece | File | Status |
|---|---|---|
| Op `GgmlRepackGemvQ50Q80Op` | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (after q4_0 op) | DONE |
| Verifier `GgmlRepackGemvQ50Q80Op::verify()` | `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (after q4_0 verifier) | DONE, fail-closed I7 |
| Emitter `emitRepackGemvQ5_0Q8_0` | `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (after q4_0 emitter) | DONE |
| Recognizer `isRepackGemvQ5_0Q8_0Body` + dispatch entry | `lib/Conversion/RVV/RVVToEmitC.cpp` | DONE |
| Declarations | `lib/Conversion/RVV/RVVToEmitCInternal.h` | DONE |
| Lit: dataflow round-trip + 7 fail-closed rejects | `test/Dialect/RVV/repack-gemv-q5-0-q8-0-dataflow.mlir` | PASS |
| Lit: emit FileCheck (CHECK + NOWALL) | `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q5-0-q8-0.mlir` | PASS |

Build: forced clean `ninja tcrv-opt` (ODS `.td` touched → `RVVOps.cpp.inc` regen,
forced relink) — **exit 0**.

## The q5_0 ABI + the qh 5th-bit handling (the only non-q4_0 part)

**Plain `block_q5_0`** (22 B): `d` fp16 @0, `qh[4]` @2, `qs[16]` @6.
ggml reconstruct (`dequantize_row_q5_0` / `ggml_vec_dot_q5_0_q8_0`): for element
`j∈[0,16)` low-half weight `= ((qs[j]&0x0F) | (qh_bit_j << 4)) − 16`; high-half
element `j+16` weight `= ((qs[j]>>4) | (qh_bit_{j+16} << 4)) − 16`. So **element e ↔
qh bit e** exactly (verified from ggml source).

**The x16 repack `block_q5_0x16` (stride 352 = 16×22):**
- `d[16]` @0 — 16 inline fp16 scales (copy).
- `qs[256]` @32 — the SAME 16-way interleave as `block_q4_0x16`, stored **RAW**
  (no `^0x88` bake; the bias lives in the assembled 5-bit field). Nibble-load path
  byte-identical to the q4_0 GEVM.
- `qhmask[32]` @288 — the qh field **TRANSPOSED + bit-packed**: because lanes are
  the 16 blocks (not the 32 elements), `qhmask[e]` bit `b` = block `b`'s qh bit for
  element `e`. 32 masks × 2 B = **64 B**. Bit is **non-inverted** (matches the
  proven block-dot `(nibble | (bit<<4)) − 16` reconstruct).

**Why bit-packed, not byte-packed:** 32+256+64 = 352 = exactly 16×22, byte-exactly
the plain q5_0 bytes for the same 512 weights — **no byte blow-up**. The Win-B
premise was streaming-out, not fewer bytes; byte-packing qh (512 B → stride 800)
would forfeit it by construction.

**The emitted lane decode** (per nibble step `i`, per 8-lane half `h`): scalar-read
the 16-bit `qhmask[i]` (low) and `qhmask[16+i]` (high) → splat to u16 lanes → `vsrl`
by `(vid + h*8)` so lane `l` of strip `h` reads bit `(l + h*8)` → `vand 1` → `vsll
4` → `vncvt` u16→u8 = the `{0,16}` 5th-bit term → `vor` into the unsigned nibble →
`vreinterpret` u8→i8 → `vsub 16`. This is the **proven block-dot reconstruct**
(`fifthBitLane` + `reinterpretBias`), grounded in the validated in-tree path.

## GATE 1 — byte-exact oracle: **PASS, norm = 0.000e+00**

`verify_emitted_gemv_q5_0.cpp` drives the compiler-emitted GEVM (reads MY repacked
`vx`) vs a **CANONICAL ggml q5_0 reference** that reconstructs from the **PLAIN
pre-repacked `block_q5_0`** (independent of `make_block_q5_0x16`). A transpose/invert
bug CANNOT be mirrored on both sides — agreement proves BOTH packer and emitter
against canonical ggml.

```
NC=16  (1 grp)  max_abs_err=0.000e+00  norm=0.000e+00  PASS
NC=32  (2 grp)  max_abs_err=0.000e+00  norm=0.000e+00  PASS
NC=64  (4 grp)  max_abs_err=0.000e+00  norm=0.000e+00  PASS
NC=336 (21 grp) max_abs_err=0.000e+00  norm=0.000e+00  PASS
VERDICT: PASS
```
`norm=0` (better than the ~e-7 target) because both sides do the identical integer
dot; only the fp32 scale-fold rounds — identically. **Both LMUL anchors byte-exact:**
mf2 (half_lanes=8) AND m1 (half_lanes=16, i8m1→i16m2→i32m4→f32m4) both norm=0.

**CHECK-AGENT RE-VERIFIED (2026-06-25, forced clean rebuild):** the `.td` op
description + emitter header comment + `RVVToEmitCInternal.h` decl previously
described an INVERTED qh scheme (`mask=qh^1`, `A=nibble|(invbit<<4)`, `(A<<3)>>3`)
that did NOT match the shipped code. The ACTUAL packer + emitter + oracle are all
consistently **NON-inverted** (`mask=qh`, `A=nibble|(qh_bit<<4)`, reinterpret u8→i8,
`vsub 16`) — mathematically equivalent, but the stale prose was a live ABI trap
(no producer op enforces the layout). Docs fixed to match code. After a forced
clean `ninja tcrv-opt` (ODS regen), the fresh binary emits a BYTE-IDENTICAL callee
stream (8× vor/vreinterpret/vsub/vncvt, ZERO vxor) and the oracle reproduces
`norm=0.000e+00` at all NC on real `ssh rvv` hardware. **BOTH anchors reproduced on
the clean build:** mf2 (`verify_q5_0.log`) AND m1 (`verify_q5_0_m1.log`, fresh m1
emit also byte-identical to the saved m1 artifact) — both norm=0 at all NC. The mf2
adversarial i16-edge (previously only saved for m1) also reproduced on hardware
(`adversarial_mf2.log`, maxrel 7.3e-8 / 4.9e-7). The byte-exact oracle is confirmed
REAL and the qh 5th-bit reconstruct is confirmed correct.

**Baseline-selection confirmed from PRIMARY SOURCE (not just the FINDING):** grepped
`/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/{repack.cpp,repack.h,arch/riscv/repack.cpp}`
— the repacked block families that ship are q4_0x{4,8,16} / q8_0x{4,8,16} /
iq4_nlx{4,8,16} / mxfp4x{4,8}; **NO block_q5_0x*, no q5_0 gemv**. The baseline
`ggml_ref.cpp` is byte-for-byte transcription-faithful to the real shipped
`ggml_vec_dot_q5_0_q8_0` at `arch/riscv/quants.c:328` (the `__riscv_v` vlenb==16
body: vle8/vand/vsrl reinterpret → vcreate → vlm_v_b4 qh → vmnand → vsub_vx_i8m2_mu
masked −16 → vwmul_vv_i16m4 → vwredsum). The Win-B baseline is the REAL vectorized
ggml block-dot, NOT scalar.

**i16-accumulator headroom — adversarially confirmed.** q5_0 doubles the weight
range to [−16,15] vs q4_0's [−8,7], so the cloned q4_0 i16 `vwmacc` lane-accumulator
runs hotter: per-lane half-sum max = 16 × 127 × 16 steps = **32512 ≤ 32767**
(headroom 255), holding ONLY under the q8_0 |q|≤127 invariant. The random oracle
(q ← `lroundf` ∈ [−127,127]) cannot generate the boundary, so `adversarial_q5_0.cpp`
forces it directly: q8 = +127 with all weights −16 (per-lane sum −32512, the i16
edge) and q8 = −127 with all weights +15 — **both PASS** (`maxrel` 7.3e-8 / 4.9e-7)
at BOTH anchors. Byte-exactness is thus airtight for q5_0×q8_0. (A wider activation
or any quant reusing this emitter outside the |q|≤127 invariant would need an i32
accumulator.)

## GATE 2 — lit: **PASS**

- `repack-gemv-q5-0-q8-0-dataflow.mlir` — round-trip accept (h8, h16, m1+h16) + 7
  fail-closed rejects (kind / stride 352 / qh-offset 288 / act-stride 34 / scale
  model / half_lanes∉{8,16} / m1+h8). `--verify-diagnostics` exit 0.
- `rvv-to-emitc-repack-gemv-q5-0-q8-0.mlir` — emit FileCheck: op gone, no unrealized
  cast, the qh-decode intrinsic sequence pinned, NOWALL `redsum`-free. PASS.
- Full suite: **714/717** (the 3 failures are pre-existing
  `Scripts/rvv-generated-bundle-abi-e2e-*-strided-widening-dot-reduce` tests — the
  vlse16 strided-dot path, ZERO references to block-quant/q5_0/my files; orthogonal).

## GATE 3 — Win-B micro @VLEN128: **LOSS (honest)**

Baseline = ggml's REAL `ggml_vec_dot_q5_0_q8_0` block-dot run once per column
(16 columns) vs OUR repack GEVM (16 columns at once). SANITY `max_rel=0.000e+00`
(our GEVM = ggml per-column, bit-exact).

| anchor | ours (ns) | ggml block-dot ×16col (ns) | **ratio ggml/ours** | verdict |
|---|---|---|---|---|
| mf2 (half_lanes=8, two 8-lane strips) | 17595 | 13528 | **0.769** | LOSS |
| m1 (half_lanes=16, one 16-lane strip) | 20692 | 13503 | **0.653** | LOSS (worse) |

**Regime named:** VLEN128, SG2044, clang-18 -O3, taskset -c 2, baseline =
`ggml_vec_dot_q5_0_q8_0` per-column. Best anchor (mf2) = 0.769×.

### Why it LOSES (causal, both anchors tested)

q5_0 is **compute-bound, not gather-heavy** — the opposite of the Win-B hypothesis.
ggml's block-dot handles the qh in ONE `vsub_vx_i8m2_mu` masked −16 over the full
32-element i8m2 (wide LMUL) + a single `vwmul`/`vwredsum`. Our repack GEVM must, per
nibble step, **expand the transposed qh mask lane-wise** (vid/vsrl/vand/vsll/vncvt ×
{lo,hi} × strips) and do scalar-broadcast `vwmacc` for a SINGLE activation column.
The block-as-lane reduction-wall removal (no `vredsum`) is real but is **likely**
dominated by the qh-expansion op-count (the added ALU q4_0 does not have). The m1
wide-LMUL anchor is WORSE (0.653×) **as expected** for wide LMUL on VLEN128 (per the
`repack-winA-always-mf2` record: m1 = more uops on this microarch regardless of
kernel) — so m1<mf2 is the baseline, NOT an independent isolation of the qh cost.
The qh-expansion op-count is the likely cause; it is not independently isolated. Net:
this matches the recorded pattern — **gather-heavy WINS (our split-gather),
compute-bound LOSES (ggml wide LMUL)**; q5_0 is compute-bound.

**The micro is GENEROUS to ggml**, which makes the loss MORE robust: our GEVM reads
the q8_0 activation ONCE per 16-column group, while ggml's per-column block-dot
re-streams it 16× — and we STILL lose. The LOSS is therefore real on the COMPUTE
axis, not a memory-traffic artifact of the comparison. (Ratio measured at n=2048;
both sides are O(nb)-linear so it does not flip at decode n; 4096/11008 not
separately timed.)

This is the **§8b honest frame**: declining = matching ggml. The kernel is a correct,
byte-exact addition to RVV coverage; it is NOT a perf win at VLEN128 vs ggml's
heavy-but-efficient block-dot.

## e2e (SEPARATE axis — NOT measured here)

Per the micro/e2e discipline: the micro LOSS does NOT settle e2e. Reasoned-NULL
likely for decode (memory-bound), but the repack changes weight memory layout
(stride 352 GROUP-major vs plain 22 row-major), so e2e is **genuinely unmeasured**.
No e2e claim is made.

## Blockers / residue

None blocking. The op + verifier + emitter + both lit tests are durable and green;
the byte-exact oracle is the strongest correctness gate (norm=0 at both anchors).
The only "negative" is the honest micro LOSS, which is a sanctioned outcome.

**CONCERN (non-blocking residue):** a `make_block_q5_0x16` PRODUCER op (the q5_0
analog of `GgmlPackQ40ToX16Op`, with the transposed bit-packed qh) was NOT authored
— a separate materialization-capability step, e2e-redundant (a producer packs at
load), and the consumer GEVM is fully validated against the packer authored in the
harness. BUT: with no producer op in-tree, the `.td` op description is the ONLY
in-tree contract for the transposed bit-packed qh layout a future producer would
follow — which is exactly why the stale INVERTED prose was a live trap and the
check-agent doc fix (now NON-inverted, matching the shipped emitter + harness
packer) was load-bearing, not cosmetic. A future q5_0 producer must pack
NON-inverted: `qhmask[e] bit b = block b's qh bit e`.

## Artifacts

`.trellis/tasks/06-22-n1n2n3-complete-multiprofile/artifacts/kernel-coverage/q5_0-emit/`:
`input-repack-gemv-q5_0.mlir`, `input-repack-gemv-q5_0-m1.mlir`, `emitted.emitc.mlir`,
`emitted-repack-gemv-q5_0.cpp` (+ `-m1`), adapters, `verify_emitted_gemv_q5_0.cpp`
(oracle), `adversarial_q5_0.cpp` + `adversarial_m1.log` (i16-headroom edge),
`winB_micro_q5_0.cpp` + `winB_micro.log` / `winB_micro_m1.log`,
`ggml_ref.cpp` (ggml's real RVV q5_0 block-dot).

Source: `RVVOps.td`, `RVVDialectWideningOps.cpp`, `RVVToEmitCBlockQuantLinear.cpp`,
`RVVToEmitC.cpp`, `RVVToEmitCInternal.h`; lit:
`test/Dialect/RVV/repack-gemv-q5-0-q8-0-dataflow.mlir`,
`test/Conversion/RVV/rvv-to-emitc-repack-gemv-q5-0-q8-0.mlir`.
