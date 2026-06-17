# Research: ggml RVV kernel surface taxonomy + Q4_0 coverage ladder

- **Query**: Enumerate and classify the full ggml RVV kernel surface (vec_dot + non-dot inference ops), grouped by structural family, with a prioritized coverage ladder for LLM inference (starting from real Q4_0 llama-2-7b forward pass).
- **Scope**: external (reading the fresh llama.cpp clone at `/home/kingdom/phdworks/llama.cpp`)
- **Date**: 2026-06-15

All line numbers below are in the llama.cpp checkout as cloned. Primary files:
- `ggml/src/ggml-cpu/arch/riscv/quants.c` (6596 lines) — explicit `__riscv_v` quant vec_dot + quantize_row kernels.
- `ggml/src/ggml-common.h` — block struct definitions (lines 89, 177-446).
- `ggml/src/ggml-cpu/simd-mappings.h` — `GGML_SIMD` / `GGML_F32_VEC_*` RVV macro layer (line 1265+).
- `ggml/src/ggml-cpu/vec.h`, `vec.cpp` — elementwise/reduction vec ops (consume GGML_SIMD; plus explicit `__riscv_v` silu/expf helpers).
- `ggml/src/ggml-cpu/ops.cpp`, `ggml-cpu.c` — graph ops (softmax/rmsnorm/rope/norm) and dtype conversion.

---

## Findings

### Part 1 — All quant `ggml_vec_dot_*` kernels in arch/riscv/quants.c

30 vec_dot function definitions total. 13 are "leaf" public kernels exposed in the type-traits table; the rest are micro-arch (`_vlNNN` / `_xtheadvector`) variants dispatched by VLEN inside the public q2_K/q3_K wrappers, or the public dispatcher wrappers themselves. Each public kernel has an `#if defined(__riscv_v)` RVV body with a `*_generic(...)` C fallback on the `#else` branch.

| Kernel (def line) | Quant pair | Block (QK / struct) | Scale model | Bits | Simple-dot vs super-block | Family |
|---|---|---|---|---|---|---|
| `ggml_vec_dot_q4_0_q8_0` (222) | q4_0 × q8_0 | QK=32; `block_q4_0` = `half d` + `qs[16]` nibbles | single fp16 `d` per 32; offset −8 | 4 | simple dot-reduce-dequant | **A** |
| `ggml_vec_dot_q5_0_q8_0` (328) | q5_0 × q8_0 | QK=32; `half d` + `qh[4]` (5th bits) + `qs[16]` | single fp16 `d`; 5th bit from qh | 5 | simple + bit-reassembly | **A** |
| `ggml_vec_dot_q8_0_q8_0` (435) | q8_0 × q8_0 | QK=32; `half d` + `qs[32]` int8 | single fp16 `d`; no unpack | 8 | simplest dot-reduce | **A** |
| `ggml_vec_dot_iq4_nl_q8_0` (5592; vl128 5479 / vl256 5534) | iq4_nl × q8_0 | QK=32; `half d` + `qs[16]` | single fp16 `d`; 4-bit index → 16-entry non-linear LUT | 4 (NL) | simple dot + table lookup | **A** |
| `ggml_vec_dot_mxfp4_q8_0` (6583; vl128 6470 / vl256 6525) | mxfp4 × q8_0 | QK=32; `uint8 e` (E8M0) + `qs[16]` | single E8M0 power-of-two exp `e`; 4-bit E2M1 mantissa → LUT | 4 (FP4) | simple dot + LUT + exp scale | **A** |
| `ggml_vec_dot_q1_0_q8_0` (563; vl256 484 / vl128 523) | q1_0 × q8_0 | QK1_0=128; `half d` + `qs[16]` (1 bit/elem) | single fp16 `d`; ternary/binary bits | 1 | simple dot + bit-expand | **A** |
| `ggml_vec_dot_q4_1_q8_1` (277) | q4_1 × q8_1 | QK=32; union `{d,m}` + `qs[16]` | fp16 `d` (delta) + fp16 `m` (min); y side carries `s = d·Σq` | 4 | dot + min-correction | **B** |
| `ggml_vec_dot_q5_1_q8_1` (382) | q5_1 × q8_1 | QK=32; `{d,m}` + `qh[4]` + `qs[16]` | fp16 `d` + fp16 `m`; 5th bit | 5 | dot + min + bit-reassembly | **B** |
| `ggml_vec_dot_q2_K_q8_K` (944; dispatch → xtheadvector 582 / vl128 692 / vl256 847) | q2_K × q8_K | QK_K=256; `block_q2_K`: `scales[16]` (4-bit d+min pairs) + `qs[64]` + `{d, dmin}` | super-block fp16 `d`/`dmin`; 16 sub-blocks × 4-bit scale & 4-bit min | 2 | super-block hierarchical | **C** |
| `ggml_vec_dot_q3_K_q8_K` (1607; → xtheadvector 962 / vl128 1107 / vl256 1257 / vl512 1370 / vl1024 1488) | q3_K × q8_K | QK_K=256; `hmask[32]` + `qs[64]` + `scales[12]` (6-bit) + `half d` | super-block fp16 `d`; 16 sub-block scales packed 6-bit; high bit in hmask | 3 | super-block hierarchical | **C** |
| `ggml_vec_dot_q4_K_q8_K` (2064) | q4_K × q8_K | QK_K=256; `{d,dmin}` + `scales[12]` (K_SCALE_SIZE, 6-bit) + `qs[128]` | super-block fp16 `d`/`dmin`; 8 sub-blocks × 6-bit scale+min | 4 | super-block + min | **C** |
| `ggml_vec_dot_q5_K_q8_K` (2081) | q5_K × q8_K | QK_K=256; `{d,dmin}` + `scales[12]` + `qh[32]` + `qs[128]` | super-block `d`/`dmin`; 8× 6-bit scale+min; 5th bit in qh | 5 | super-block + min + bit | **C** |
| `ggml_vec_dot_q6_K_q8_K` (2720) | q6_K × q8_K | QK_K=256; `ql[128]` + `qh[64]` + `scales[16]` (int8) + `half d` | super-block fp16 `d`; 16 sub-blocks × 8-bit signed scale | 6 | super-block hierarchical | **C** |

Additional K-super-block IQ / ternary kernels present (same family C machinery; out of scope for Q4_0 but part of "all kernels"):

| Kernel (def line) | Pair | Notes |
|---|---|---|
| `ggml_vec_dot_iq1_s_q8_K` (3136) | iq1_s × q8_K | 1.5625 bpw, grid-codebook + super-block scale |
| `ggml_vec_dot_iq1_m_q8_K` (3719) | iq1_m × q8_K | 1.75 bpw, grid + per-sub scales |
| `ggml_vec_dot_iq2_xxs_q8_K` (4781) | iq2_xxs × q8_K | 2.0625 bpw, codebook grid |
| `ggml_vec_dot_iq2_xs_q8_K` (4504) | iq2_xs × q8_K | 2.3125 bpw, grid + 4-bit scales |
| `ggml_vec_dot_iq2_s_q8_K` (4213) | iq2_s × q8_K | 2.5625 bpw, grid + signs |
| `ggml_vec_dot_iq3_xxs_q8_K` (5457) | iq3_xxs × q8_K | 3.0625 bpw, grid |
| `ggml_vec_dot_iq3_s_q8_K` (5080) | iq3_s × q8_K | 3.4375 bpw, grid + signs |
| `ggml_vec_dot_iq4_xs_q8_K` (5950) | iq4_xs × q8_K | 4-bit NL on super-block (`half d` + `scales_h` + `scales_l[4]`) |
| `ggml_vec_dot_tq1_0_q8_K` (6281) | tq1_0 × q8_K | ternary 1.6875 bpw (5 elems/byte) |
| `ggml_vec_dot_tq2_0_q8_K` (6454) | tq2_0 × q8_K | ternary 2.0625 bpw (2 bits/elem) |

Note: K-quant publics dispatch to a VLEN-specific inner kernel (`_vl128/_vl256/_vl512/_vl1024` and `_xtheadvector`). The reference RVV hardware (`ssh rvv`) VLEN selects which inner path runs — so "covering q3_K" means matching the one inner variant the target VLEN uses, not all five.

### Part 2 — Non-dot RVV ops needed for inference

Two RVV delivery mechanisms coexist:
1. **`GGML_SIMD` macro layer** (`simd-mappings.h:1265+`): when `__riscv_v_intrinsic` is defined, `GGML_SIMD` is defined and `GGML_F32_VEC*` map to `vfloat32m1_t` + `__riscv_vfmacc/vfadd/vfmul/vle32/vse32` (GGML_F32_STEP=16, EPR=4). Every vec op written against `GGML_F32_VEC_*` gets RVV automatically.
2. **Explicit `__riscv_v` / `__riscv_v_intrinsic` blocks** hand-written for cases the macro layer can't express (variable-length reductions, fp16/bf16 widening, activation transcendentals).

| Op | File:loc | RVV path | Mechanism |
|---|---|---|---|
| `ggml_vec_dot_f32` (f32 activation·activation) | vec.cpp:11, RVV 87-102 | yes | explicit `__riscv_v` (m8 vfmacc + vfredusum) |
| `ggml_vec_dot_f16` (KQ/attention f16 dot) | vec.cpp:264, RVV 321+ | yes (`__riscv_zvfh`) | explicit `__riscv_v_intrinsic` |
| `ggml_vec_dot_bf16` | vec.cpp:139, RVV 198+ | yes (`__riscv_zvfbfwma`) | explicit (bf16 widening FMA) |
| `ggml_vec_dot_f16_unroll` | vec.h:142, RVV 210+ | yes | explicit `__riscv_v_intrinsic` |
| `ggml_vec_mad_f32` / `_unroll` (axpy in rope, etc.) | vec.h:319 / 585 | yes | GGML_SIMD (`GGML_F32_VEC_FMA`) |
| `ggml_vec_scale_f32` | vec.h:703 | yes | GGML_SIMD (`GGML_F32_VEC_MUL`) |
| `ggml_vec_sum_f32_ggf` / reductions | vec.h:1495+ | partial | GGML_SIMD where applicable, else scalar |
| `ggml_vec_silu_f32` (SwiGLU FFN gate) | vec.cpp:380, RVV 405-410 | yes | explicit `__riscv_v_intrinsic` → `ggml_v_silu_m2` (vec.h:1363) using `ggml_v_expf_m2` (vec.h:1324) |
| `ggml_vec_swiglu_f32` | vec.cpp:417 | yes | explicit, same silu helper |
| `ggml_vec_gelu_f32` | vec.h:988/1003 | LUT/scalar (no dedicated RVV `ggml_v_gelu`; SVE/x86 only) | scalar table on RVV |
| fp32→fp16 / fp16→fp32 row conversion | ggml-cpu.c:3417 (`__riscv_zvfh`), 3451 (`__riscv_zvfhmin`) | yes | explicit `__riscv_v_intrinsic` |
| `ggml_compute_forward_soft_max` | ops.cpp:5429 | via GGML_SIMD vec helpers (`ggml_vec_max_f32`, `ggml_vec_soft_max_f32` exp loop) | macro layer + scalar `expf` tail |
| `ggml_compute_forward_rms_norm` | ops.cpp:3823 | via `ggml_vec_dot_f32` (sum of squares) + `ggml_vec_scale_f32` | GGML_SIMD-backed helpers |
| `ggml_compute_forward_norm` | ops.cpp:3730 | scalar mean/var + GGML_SIMD scale | mostly scalar |
| `ggml_compute_forward_rope` | ops.cpp:5961 | scalar per-pair rotate (no dedicated RVV); some mad helpers SIMD | mostly scalar |
| SSM-scan inner | ops.cpp:9580 | **`// todo: RVV implementation` → np=0 → scalar fallback** | none (Mamba only) |
| RWKV WKV (wkv6/7) | ops.cpp:10834 | **`// Route to scalar implementation //TODO RVV` ** | none (RWKV only) |
| `quantize_row_q8_0` (activation-side quant) | quants.c:32, RVV 39-64 | yes | explicit `__riscv_v` (vfabs+vfredmax+vfncvt) |
| `quantize_row_q8_1` | quants.c:73, RVV 79+ | yes | explicit `__riscv_v` (adds vwredsum for `s`) |
| `quantize_row_q8_K` | quants.c:122, RVV 126+ | yes | explicit `__riscv_v` (K-quant activation path) |

Dispatch wiring (`ggml-cpu.c` type_traits table, ~line 213+): each weight quant type binds `.from_float = quantize_row_*` and `.vec_dot = ggml_vec_dot_*_*`. `mul_mat` quantizes the activation row once via `from_float` (→ q8_0/q8_1/q8_K) then loops `vec_dot` over weight rows. So the matmul kernel = (`quantize_row_q8_x` on activations) + (the weight type's `vec_dot`).

### Part 3 — The 5 structural families (by capability needed)

- **Family A — symmetric block dot-reduce-dequant (single scale).** q4_0, q5_0, q8_0, iq4_nl, mxfp4, q1_0. Pattern: load 8-bit y block, unpack/LUT the weight nibbles to int8, widening int8×int8 multiply (`vwmul`/`vwmacc`), `vwredsum` to int32, multiply by `x.d * y.d`. iq4_nl/mxfp4 add a 16-entry value LUT; mxfp4 adds an E8M0 exponent scale; q5_0/q1_0 add bit-reassembly. **No min, no super-block.** Capability: int8 widening MAC + int reduction + fp16 scalar scale.
- **Family B — asymmetric block (scale + min).** q4_1, q5_1. Same int8 MAC core as A, but dequant is `x = d·q + m`; the y side is q8_1 carrying `s = d·Σq` so the min term folds in as `m·s`. Capability = A + per-block min accumulation against the activation sum.
- **Family C — super-block K-quants (hierarchical scales).** q2_K, q3_K, q4_K, q5_K, q6_K, plus iq1_s/iq1_m/iq2_*/iq3_*/iq4_xs/tq1_0/tq2_0. QK_K=256 super-block with one fp16 super-scale (`d`, and `dmin` for the asymmetric ones) plus 8 or 16 sub-block scales packed at 4/6/8 bits (and high-bit planes qh/hmask). Activation is q8_K (carries `bsums` per group of 16 for the min correction). Capability = A/B core + sub-block scale unpacking + two-level (super × sub) scaling + grid/codebook lookups for IQ. These are the kernels with VLEN-specialized inner variants.
- **Family D — elementwise / reduction activation ops.** softmax, rms_norm, norm, rope, silu/swiglu/gelu, scale, mad, the f32/f16/bf16 vec_dot. Mostly delivered through the `GGML_F32_VEC_*` macro layer; silu and the q8 conversions have hand-written `__riscv_v` paths; gelu/rope/norm lean scalar; SSM-scan and RWKV are explicit RVV TODOs (scalar). Capability = f32/f16 vector FMA + horizontal reduce + `expf` (silu/softmax).
- **Family E — activation-side quantize_row.** quantize_row_q8_0 / q8_1 / q8_K. The `from_float` half of every quantized matmul: vabs → vredmax (amax) → reciprocal scale → vfncvt to int8 (q8_1 also `vwredsum` for `s`; q8_K computes `bsums`). Capability = f32 abs + max-reduce + narrowing fp→int convert.

### Part 4 — Prioritized coverage ladder

A real **Q4_0 llama-2-7b forward pass** invokes only a tiny slice. The matmuls (attention QKV/O projections + FFN gate/up/down) are all `mul_mat` over **Q4_0 weights**, and they dominate runtime. Each such matmul = `quantize_row_q8_0(activations)` then `ggml_vec_dot_q4_0_q8_0` per row. Around them: rms_norm (×2 per layer), softmax (attention), rope (Q/K), silu+mul (SwiGLU FFN), residual adds, and the f16 KQ/KQV attention dots. Embedding/output and KV cache are f16/f32.

**Rung 0 — Priority-1 set (gets Q4_0 inference numerically correct):**
1. `ggml_vec_dot_q4_0_q8_0` (the weight matmul kernel) — **Family A**.
2. `quantize_row_q8_0` (activation quantization feeding every matmul) — **Family E**.
3. `ggml_vec_dot_f32` and `ggml_vec_dot_f16` (rms_norm sum-of-squares; f16 attention QK/KQV dots) — **Family D**.
4. `ggml_vec_silu_f32` / `ggml_vec_swiglu_f32` (FFN gate) — **Family D**.
5. `ggml_vec_scale_f32` + `ggml_vec_mad_f32` + softmax exp + rope rotate — **Family D** (macro/scalar; correctness first, RVV speed later).
These 5 cover a correct + mostly-vectorized Q4_0 forward. Items 1, 2, 3(f32/f16), 4 have real RVV intrinsic bodies; rope/softmax-exp ride the macro layer / scalar tail.

**Rung 1 — round out the classic legacy quants (one model family swap each):**
- Family A breadth: `q8_0_q8_0`, `q5_0_q8_0`, `iq4_nl_q8_0`, `mxfp4_q8_0`. (q8_0 is the simplest validation kernel — do it alongside Rung 0 as a smoke test.)
- Family B: `q4_1_q8_1`, `q5_1_q8_1` (+ `quantize_row_q8_1`).

**Rung 2 — K-quants (the common real-world quant for 7B+, e.g. Q4_K_M):**
- `quantize_row_q8_K` (Family E for super-blocks), then Family C: `q4_K`, `q6_K`, `q5_K`, `q3_K`, `q2_K`. Note VLEN-specific inner variants — target the one the `ssh rvv` VLEN selects. Q4_K + Q6_K together cover the dominant "K_M" mixed-quant layouts.

**Rung 3 — full surface (IQ / ternary + remaining activation RVV):**
- Family C IQ/ternary: iq1_s, iq1_m, iq2_xxs/xs/s, iq3_xxs/s, iq4_xs, tq1_0, tq2_0.
- Family D long tail: dedicated RVV for gelu, rope, norm, and the explicit-TODO SSM-scan (ops.cpp:9580) and RWKV WKV (ops.cpp:10834) paths (only needed for Mamba/RWKV architectures, not llama).

## Caveats / Not Found

- Line numbers are from the current `/home/kingdom/phdworks/llama.cpp` clone; they will drift across upstream versions — anchor on function names, not lines.
- I did not benchmark; "matmuls dominate" is the standard transformer-inference fact, not a measured TianChen-RV claim. Any speedup assertion still needs real `ssh rvv` evidence (per project discipline / I-rules).
- `block_nvfp4` (ggml-common.h:213) and a possible `q8_K`-pairing for nvfp4 exist as structs but I found **no `ggml_vec_dot_nvfp4_*` RVV kernel** in arch/riscv/quants.c — nvfp4 has no RISC-V vec_dot yet (excluded from the ladder).
- K-quant public kernels select an inner `_vlNNN` / `_xtheadvector` variant by runtime VLEN; this taxonomy counts them as one logical kernel per quant type. Confirm the exact inner path on the target hardware before claiming a match.
- The `GGML_SIMD` RVV mapping uses fixed `f32m1` / EPR=4 (vlen>=128 compatible) for the macro layer, while hand-written quant kernels use larger LMUL (m1–m8) directly. Replacement work must reproduce both styles.
