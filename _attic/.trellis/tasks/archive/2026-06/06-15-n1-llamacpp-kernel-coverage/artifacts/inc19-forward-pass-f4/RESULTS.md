# INC-19 F4 — ggml `quantize_row_q8_0` (the f32 → block_q8_0 ACTIVATION QUANTIZER, the f32 → quant BRIDGE)

Date: 2026-06-16. Board: `ssh rvv` (riscv64, rv64gcv, VLEN=128, `__riscv_v_min_vlen
128`, `__riscv_zvl128b`, `__riscv_zfhmin`, `__riscv_v_intrinsic 12000`, clang 18.1.3).

## What landed

A STRUCTURED byte-exact compiler-emitted drop-in for ggml's RVV-path
`quantize_row_q8_0` (the f32 → block_q8_0 ACTIVATION QUANTIZER;
riscv/quants.c:32-71). F4 is the f32 → QUANT BRIDGE: it produces the block_q8_0
activation that our COMMITTED q4_0_q8_0 / q8_0_q8_0 block-dot kernels CONSUME, so
a real forward pass can feed our dot kernels. It is structurally NEW vs F1/F3/F5/
F5b (a per-block max-abs REDUCTION + a scalar scale + an f32→i16→i8 NARROWING
CONVERT + a structured scalar branch + an AoS block-quant store).

- New op `tcrv_rvv.quantize_row_q8_0` (x `const float *` in, the block_q8_0 byte
  buffer `uint8_t *` out, n; fail-closed verifier I7; `kind` bounded to
  `"ggml_quantize_row_q8_0"`; the AoS block-format facts qk=32 / block_stride=34 /
  scale@0 / quant@2 as bounded I4 mirror attrs; no SEW/LMUL/policy attr at the I5
  boundary; pinned at the m8 anchor ggml uses — no strip_lmul knob, matching
  silu's precedent because the m8 reduction fold gives no byte-exact LMUL freedom).
- Lowering (`emitGgmlQuantizeRowQ80`, lib/Conversion/RVV/RVVToEmitC.cpp): the AoS
  block loop (`nb = n/32`); per block, in ONE e32m8 strip (`vl = QK8_0 = 32`,
  relying on Zvl128b ⇒ VLEN≥128 — the SAME capability the q4_0 mb4-elided shape
  uses): `vfabs` + `vfredmax`(amax) → scalar `d = amax/127` → the STRUCTURED
  `id = d ? 1/d : 0` (emitc.cmp + emitc.if) → the native `(_Float16)d` AoS store
  at byte 0 → `vfmul_vf(id)` scale → `vfncvt_x_f_w_i16m4` (round-to-nearest-even)
  → `vncvt_x_x_w_i8m2` (truncate) → `vse8` the 32 int8 qs at byte 2.

## The op + lowering (amax / scale / round / narrow / fp16) — files

- include/TianChenRV/Dialect/RVV/IR/RVVOps.td — `GgmlQuantizeRowQ80Op` def.
- lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp — `GgmlQuantizeRowQ80Op::verify()`
  (fail-closed: kind, the AoS block-format facts, the `const float *` x / `uint8_t
  *` vy ABI, the m1 result, the with_vl policy).
- lib/Dialect/RVV/IR/RVVDialect.cpp — additive `uint8_t *` for the `OutputBuffer`
  role's allowed C type (the ggml block-quantizer writes a `void *vy` byte buffer;
  the fp16 d + int8 qs stores address it as a mutable byte cursor).
- lib/Conversion/RVV/RVVToEmitC.cpp — `isGgmlQuantizeRowQ80Body` recognizer, the
  dispatch wiring, `emitGgmlQuantizeRowQ80` emitter.
- test/Conversion/RVV/rvv-to-emitc-ggml-quantize-row-q8-0.mlir — the F4 lit test.

## raw() = 0 + structured proof

The whole kernel is STRUCTURED emitc nodes — the block loop is `emitc.for`, the
`id = d ? 1/d : 0` conditional is `emitc.cmp` + `emitc.if` (two assigns to an `id`
lvalue), the fp16 d store is `cast`→`subscript[0]`→`cast`→`assign` (a native
`_Float16` store), the intrinsics are `emitc.call_opaque`, the narrowing convert a
`call_opaque` chain. Of the lowered IR: **`emitc.raw` / `raw(` = 0**, and the
non-comment `emitc.verbatim` count = **0** (every verbatim is a `// `-prefixed
provenance comment line). Every value (the block `for`, the scalar `div`/`cmp`,
the `if`, the pointer `add`/`cast`, the `subscript`, every intrinsic `call_opaque`,
every `literal`) is a node in the IR graph.

## Byte-exact HW evidence (ssh rvv) — `board_validation_stdout.txt`

The UNMODIFIED compiler-emitted `quantize_row_q8_0_kernel.cpp` compiled `-O2
-march=rv64gcv` and run vs a VERBATIM transcription of ggml's RVV
`quantize_row_q8_0` (riscv/quants.c:43-65, the `__riscv_v` branch DEPLOYED on the
board), the deployment oracle:

```
INC-19 F4 quantize_row_q8_0: blocks 115584/115584 d bit-exact, 115584/115584 qs bit-exact, 115584/115584 FULL block bit-exact (vs ggml RVV quantizer)
NC-round (roundf, round-half-AWAY): 28902/115584 blocks qs correctly DIFFER; on engineered-tie blocks 28896/28896 DIFFER (rne vs roundf)
NC-fp16 (off-by-one-ULP d): 101136/115584 blocks d correctly DIFFER
CLOSE-THE-LOOP (f32->quantize->q4_0_q8_0 dot *s): 1920/1920 bit-exact (our_quantize->dot vs ggml_quantize->dot)
DISCRIMINATION: NC-round SHARP (28896/28896 tie blocks differ); NC-fp16 SHARP (101136/115584 differ)
RESULT: PASS (FULL block_q8_0 d+qs bit-exact AND close-the-loop *s bit-exact vs ggml RVV quantizer; NC-round pins rne, NC-fp16 pins the (_Float16) cast)
```

- **EXACT, not near-exact.** The FULL block_q8_0 bytes — BOTH the fp16 `d`
  (memcmp of the 2 ggml_half bytes) AND the 32 int8 `qs` (memcmp of the 32 bytes)
  — are BIT-IDENTICAL over 10 row sizes (32 … 4096 + 11008, incl. single-block
  and ffn dims) × 8 value distributions × 24 rows = **115584/115584 on the full
  block**. The distributions include small/large/tiny uniform, all-zero blocks
  (amax=0 ⇒ d=0 ⇒ every q=0), single-spiky-lane (amax from one element), realistic
  mixed-magnitude activations, AND two ENGINEERED-TIE distributions (amax forced
  to 254 ⇒ d=2, id=0.5, with odd-integer lanes so x0 = k+0.5 lands on half-integer
  rounding ties — the rne-vs-roundf discriminator).

## The rounding / fp16 crux + how resolved (return-item 6)

The single hardest part of F4 is matching ggml's EXACT rounding + fp16 conversion.
The task note ("ggml uses roundf or nearbyint") was pre-source uncertainty; the
primary-source read resolves it decisively:

1. **The ROUNDING crux.** The scalar `_ref` (ggml-quants.c:258) rounds with
   `roundf(x0)` = round-half-**AWAY**. But on the board `__riscv_v` is defined, so
   the DEPLOYED kernel is the RVV path (riscv/quants.c:60), which rounds with
   `vfncvt_x_f_w_i16m4` = the dynamic frm = **round-to-nearest-EVEN**. They DIVERGE
   at half-integer ties (2.5 → 2 rne vs 3 roundf). **Resolution:** the oracle is
   the RVV path, and the lowering REPLICATES the exact `vfncvt_x_f_w_i16m4`
   intrinsic node-for-node — so it inherits rne + the i8 saturating clamp + every
   NaN/inf/edge case for free (no rounding is reasoned about; the same instruction
   is emitted). **NC-round** (the SAME kernel but `roundf`, the `_ref` rounding)
   is SHARP: **28896/28896** engineered-tie blocks correctly DIFFER — so the qs
   byte-compare DISCRIMINATES ggml's rne, proving we matched ggml's METHOD, not a
   tolerance.
2. **The FP16 crux.** The board is `__riscv_zfhmin`, so `GGML_CPU_FP32_TO_FP16(d)`
   is the native `(_Float16)d` cast (`fcvt.h.s`, rne) — NOT a software fp16 pack.
   **Resolution:** the lowering emits a STRUCTURAL `_Float16` store
   (`*(_Float16 *)(yb) = (_Float16)d`). **NC-fp16** (an off-by-one-ULP d on every
   nonzero-d block) is SHARP: **101136/115584** blocks' d correctly DIFFER (the
   14448 non-differing are exactly the all-zero blocks whose d=0 is correctly
   preserved) — so the d byte-compare pins ggml's EXACT `(_Float16)` cast.
3. **The `d ? 1/d : 0` conditional** is load-bearing: the all-zero block (amax=0 ⇒
   d=0) must give id=0 so every q=0; a bare `1/d` would give inf, then 0·inf=NaN,
   and vfncvt(NaN) would not match ggml's q=0. It is emitted as a STRUCTURED
   emitc.cmp + emitc.if, validated by the all-zero distribution (dist 3).

The negative controls are DISCRIMINATION EVIDENCE; the kernel verdict gates on
`block_exact == total && close_loop == total`, not on a control being sharp.

## The CLOSE-THE-LOOP test (return-item 3) — the INTEGRATION demo

The critical closing-the-loop test composes TWO separately-proven COMPILER kernels
end-to-end:
- **pipeline A (ours):** our F4 `quantize_row_q8_0(x)` → the COMMITTED compiler
  `q4_0_q8_0` block-dot kernel (INC-2a, proven byte-exact + live token-identical)
  → `s_ours`.
- **pipeline B (ggml):** ggml's RVV `quantize_row_q8_0(x)` → ggml's generic
  `vec_dot_q4_0_q8_0` → `s_ref`.

`s_ours` == `s_ref` BIT-EXACTLY over **1920/1920** cases (the fp32 `*s` memcmp).
This proves the full f32 → quantize → dot pipeline is byte-exact through DISTINCT
compiler kernels. **Honest framing:** this result is CONFIRMATORY *given*
block-exactness (out_ours == out_ref byte-for-byte ⇒ identical dot inputs) plus
INC-2a's proven dot — it is the integration demonstration the task called
critical (two compiler kernels composing), not an independent discriminator on top
of the block-exact result.

## lit + reds

- Full clean rebuild GREEN.
- `check-tianchenrv`: **628 total, 625 pass, 3 fail** — the 3 failures are the
  SAME pre-existing DOCUMENTED reds (`rvv-generated-bundle-abi-e2e-self-test` +
  the two `computed-masked-strided-input-widening-dot-reduce-add` dry-run tests);
  none reference F4. The F4 change is purely ADDITIVE (the new F4 lit test is the
  +1 green: 628 = 627 prior + 1).
- **Additivity proven (byte-identical, not just lit-pass)**: F1 scale / F3
  rms_norm / F5 silu / F5b soft_max each re-render **BYTE-IDENTICAL** to their
  committed kernel artifacts (`diff` of the `tcrv-opt … | mlir-translate` output
  vs the committed `.cpp` = empty for all four); the q4_0 + q8_0 block-dot lit
  tests PASS. My change touches only 4 files (the F4 op def, the F4 verifier, the
  F4 emitter + recognizer/dispatch, the OutputBuffer ctype allowlist) — none touch
  any sibling's emission path, and F4 leaves the shared function-type/header region
  byte-identical (it is a void, 3-header kernel like every non-soft_max sibling).
- **The q4_0 stale-artifact note**: re-rendering the q4_0 dot vs the OLD hand-saved
  `inc2a-block-kernel-structured/tcrv_emitted_kernel.cpp` shows a diff — but that
  is PRE-EXISTING artifact drift (a stale `.cpp` predating INC-2a's FMA-expression
  fix), NOT a regression: the current q4_0 emission matches the PASSING q4_0 lit
  test, AND the freshly-regenerated dot kernel re-validated 1920/1920 vs ggml's
  generic dot in the close-the-loop here. Out of F4's scope (INC-2a territory).
- **The `uint8_t *` OutputBuffer allowlist addition broke no test**: the full suite
  passes with only the 3 documented reds — there was no negative test asserting
  `OutputBuffer` rejects `uint8_t *`.

## What's left (the forward-pass family)

With F1 (scale) + F3 (rms_norm) + F5 (silu) + F5b (soft_max) + F4
(quantize_row_q8_0) landed byte-exact, the remaining ops to COMPLETE the
forward-pass op set are:
- **F6** `rope`: f32 elementwise rotation (reuses F1/F2 vfmul/vfmsub/vfmadd lanes)
  + a verbatim scalar libm `cosf`/`sinf` angle cache (a DIFFERENT exactness axis —
  libm-linked, not a vectorized polynomial). The COMPOSITION rung; with it the
  forward-pass op set is complete.
- **q8_1 quantizer sibling** (`quantize_row_q8_1`): the SAME amax/scale/narrow as
  F4 + an extra `vfredusum` block sum stored as `block_q8_1.s` (and stride 36, the
  q8_1 AoS layout). A separate op (NOT folded into the q8_0 kind), a small
  increment that INHERITS F4's reduction + narrowing-convert machinery. Noted as
  the next quant-bridge rung when q4_1/q5_1 dots need their q8_1 activation.

## Files

- include/TianChenRV/Dialect/RVV/IR/RVVOps.td
- lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp
- lib/Dialect/RVV/IR/RVVDialect.cpp
- lib/Conversion/RVV/RVVToEmitC.cpp
- test/Conversion/RVV/rvv-to-emitc-ggml-quantize-row-q8-0.mlir
- artifacts/inc19-forward-pass-f4/{rvv-to-emitc-ggml-quantize-row-q8-0.mlir,
  quantize_row_q8_0_kernel.cpp, q4_0_q8_0_dot_kernel.cpp, inc19_validate.cpp,
  board_validation_stdout.txt, RESULTS.md}.
