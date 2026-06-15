# N1 coverage: really replace llama.cpp ggml RVV kernels (Q4_0×Q8_0 first)

## Goal

Push N1 覆盖广度 from "synthetic kernels we invented" to **"can really stand in for the
specific RVV kernels llama.cpp (ggml) actually ships"** — the hard, externally-defined
coverage standard the user set. First concrete milestone: emit a **numerically-exact,
deployable replacement for `ggml_vec_dot_q4_0_q8_0`** (the dominant weight-matmul kernel for
a Q4_0 model — the board runs `llama-2-7b-chat.Q4_0.gguf`), validated byte-exact against
ggml's own `_generic` reference on real `ssh rvv`. Then widen along the kernel family and
(N3) measure against ggml's hand-written RVV kernel.

## What I already know (from research/, 2026-06-15)

Three research files (all primary-source, ssh-rvv board = VLEN=128, rv64imafdcv+zvfh+zfhmin):
- [`research/q4_0_q8_0-kernel-map.md`](research/q4_0_q8_0-kernel-map.md) — exact semantics + the precise gap list.
- [`research/ggml-rvv-kernel-taxonomy.md`](research/ggml-rvv-kernel-taxonomy.md) — 30 `ggml_vec_dot_*` kernels, 5 capability families, the priority-1 set for Q4_0 inference.
- [`research/our-current-coverage-inventory.md`](research/our-current-coverage-inventory.md) — what we deploy today; biggest gap = no block loop / no per-block scale / wrong nibble convention.

**Exact kernel** (`block_q4_0` = {fp16 d; 16 packed bytes}, `block_q8_0` = {fp16 d; 32 int8}, QK=32):
```
s = Σ_blocks  d_x·d_y · Σ_{i=0..31} ((nibble_i − 8) · q8_i)         // fp32 accumulate
```
low nibble of byte j pairs with q8[j]; high nibble with q8[j+16].

**Gap (LARGE, but decomposed):**
- **g1** asymmetric i4×i8 — only the q4 weights unpack; q8 stays plain int8 (our packed_i4 is symmetric i4×i4).
- **g2** offset-binary decode `nibble−8` (`15→+7`) vs our two's-complement (`15→−1`). Fix: XOR `0x88` → reuse existing sign-extend path, OR a structural `−8` bias op.
- **g3** per-block dual fp16 scale `d_x·d_y` — scalar `fcvt.s.h`×2 (NO vector zvfh needed), our dequant has a single runtime f32 scale.
- **g4** AoS QK=32 block loop, fp32 scalar accumulation of `scale·reduction` across blocks (largest piece — an outer structure, not an op). g4b: ggml anchors i8m1→i16m2→immediate `vwredsum`; we anchor i32m1.
- **g5** low/high nibble ↔ q8[0..15]/q8[16..31] split pairing.
- **g6** ABI: `void ggml_vec_dot_q4_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc)`, writes `*s`.

**Reusable today (no gap):** `vwredsum` i16→i32 reduce; signed i8×i8→i16 `vwmul`/`vwmacc`; unit-stride `vle8` packed load; scalar lane-0 extract.

## Architecture decision (ADR-lite)

**Context**: the Q4_0 kernel is block-structured (AoS + per-block fp16 scales + fp32
accumulation) — a real departure from our flat-SoA single-scale bodies.

**Decision**: build a **new block-quantized dot-product selected-body capability**, realized
**Q4_0-faithful first** (byte-exact, deployable, the externally-defined kernel), then
generalize across Family A (q5_0/q8_0/iq4_nl single-scale) → Family B (q4_1/q5_1 scale+min)
→ K-quants. Honor I5 (executable facts structural in the typed body, not strings), I4
(metadata mirror), I7 (fail-closed), I8 (real ssh-rvv evidence is the only HW proof).

**Consequences**: a genuinely new IR shape (block loop + per-block scale extraction + the
asymmetric one-sided unpack). Bigger than the deferred-wide work, but each increment is
HW-validated byte-exact and the first milestone is a REAL ggml kernel, not a synthetic one.

## Plan (incremental, each ssh-rvv-validated)

- **INC-1 — integer core (closes g1, g2, g5).** A one-sided nibble-unpack op (ggml
  offset-binary decode: `&0x0F`/`>>4` then `−8`, or XOR `0x88`→existing sign-extend) → i8
  lanes → existing signed widening product against a **plain-i8** q8 stream → `vwredsum` →
  **byte-exact single-block i4×i8 integer dot** (`sumi`), validated vs `_generic` at n=32.
- **INC-2 — block loop + scales + ABI (closes g3, g4, g6) → deployable.** AoS QK=32 block
  iteration, per-block fp16 `d_x`,`d_y` scalar `fcvt.s.h` + fp32 accumulation, `*s` store,
  the ggml call signature. Produces a deployable `.o/.h` that is a drop-in for
  `ggml_vec_dot_q4_0_q8_0`, byte-exact vs `_generic` over many n (incl. non-block-multiple
  tails handled exactly as ggml does — ggml requires n%32==0; honor that contract).
- **INC-3 — family widen.** q5_0/q8_0 (single-scale Family A), then q4_1/q5_1 (scale+min).
- **N3 — measure.** Bench the deployed Q4_0 kernel vs ggml's hand-written RVV kernel on the
  board (fair: both -march=rv64gcv, same data). Honest win/parity/loss reported.

## Acceptance Criteria

- [ ] INC-1: a one-sided i4(offset-binary)×i8 single-block integer dot, byte-exact vs `ggml_vec_dot_q4_0_q8_0_generic` integer partial at n=32, on ssh rvv.
- [ ] INC-2: a deployable `.o/.h` whose output equals `ggml_vec_dot_q4_0_q8_0` (`*s`) to fp32 tolerance over n ∈ {32, 64, 256, 4096, …} on ssh rvv, with real fp16 scales from real block data.
- [ ] Full clean rebuild green; all existing narrow/byte/deferred-wide paths byte-identical; the documented environmental reds unchanged.
- [ ] N3: a fair ssh-rvv measurement of our Q4_0 kernel vs ggml's RVV `ggml_vec_dot_q4_0_q8_0`, reported honestly.

## Out of Scope (this task)
- K-quant super-blocks (q4_K/q6_K) and the IQ/ternary tail — later rungs.
- Full llama.cpp model integration / end-to-end token generation — kernel-level parity first.
- Hardware capability probe ingestion (I6) — still `--march`-selection-live.

## Technical Notes
- llama.cpp shallow clone: `/home/kingdom/phdworks/llama.cpp`; kernel at
  `ggml/src/ggml-cpu/arch/riscv/quants.c:222`; generic ref at `.../quants.c:174`; block
  structs in `ggml/src/ggml-common.h`.
- fp16→fp32 on the board is **scalar** `fcvt.s.h` (`simd-mappings.h:95`) — no vector zvfh required for INC-2.
- Our adjacent path: `tcrv_rvv.packed_i4_nibble_unpack_product` (+ `standalone_reduce` + `dequantize`), lowered in `lib/Conversion/RVV/RVVToEmitC.cpp` (~3560); ABI emission + e2e harness as for the deferred-wide work.
