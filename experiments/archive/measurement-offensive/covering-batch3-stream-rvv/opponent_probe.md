# opponent probe -- streaming-format opponent identity + vectorization (covering-batch③)

## opponent identity (per-datum binding)
- **dequant** opponent = ggml `dequantize_row_<fmt>` exported (T, global) from the board's own
  `libggml-base.so.0` (md5 `1b4580c4…`, tag b9692/f3e1828, gcc-15). This IS the dispatched path:
  `libggml-cpu.so` exports NO `dequantize_row_*` override (nm empty) → ggml's forward dequant uses the
  base type-traits `to_float` = exactly the symbol we time against. All 25 formats present in base.
- **quant** opponent = ggml `quantize_row_q8_0` / `quantize_row_q8_K` exported from `libggml-cpu.so.0`
  (md5 `d1adc634…`, the arch-dispatched non-`_ref`/non-`_generic` symbol at 0x914cc / 0x9165e), gcc-15.
- **forward** opponent = **transcribed** `ggml_vec_add_f32` / `ggml_vec_mul_f32` (`z[i]=x[i]±y[i]`) and
  the accurate `ggml_gelu_f32` (tanh) reference. Rationale: ggml's `ggml_vec_*` are inline/static in
  `vec.h` (NOT exported from any .so — nm confirms absent), so the fair opponent is the identical scalar
  source, compiled by the neutral link compiler (g++-15) which autovectorizes it. This is disclosed as a
  transcribed-reference opponent (tier below linked-symbol), symmetric across BOTH ledgers.
  **gelu caveat**: ggml's DEPLOYED `ggml_compute_forward_gelu` uses the f16 lookup table
  `ggml_table_gelu_f16` (a different, cache-cheap, lossy path). We time vs the accurate tanh reference
  (byte-exact to our emit's formula, maxrel 1.3e-6); the LUT deploy path is NOT what we compare and is
  noted as an out-of-scope faster-but-lossy alternative.

## compiler-symmetry ledger [CASE-COMPILER-ASYMMETRY]
Only OUR kernel .o codegen varies (clang-18.1.8 deploy / gcc-15.2.0 kernel-symmetric); the driver + final
link are ALWAYS g++-15 (neutral = the compiler that built the opponent .so and the transcribed refs). So
the gcc ledger is a valid compiler-SYMMETRIC kernel account; the clang ledger is the SYSTEM/DEPLOY account
(we ship clang .o). For memory-bound streams the two ledgers largely converge at the bandwidth wall —
EXCEPT iq4_nl, where they split hard (see below).

## ★iq4_nl codebook: the vluxei indexed-gather trap (objdump-verified, rock-stable 3x)
The iq4_nl decode is a per-nibble codebook lookup `y=d*kvalues_iq4nl[nibble]`. objdump of the timed binaries:
| binary                         | codebook lowering        | GB/s  |
|--------------------------------|--------------------------|-------|
| OUR gcc iq4_nl .o              | **0 RVV insns, scalar** (32 lb + 16 lbu table loads) | **2.79** |
| OUR clang iq4_nl .o           | `vluxei32.v` indexed gather | 0.545 |
| ggml `dequantize_row_iq4_nl` (gcc) | `vluxei64.v` indexed gather | 0.578 |
- **Indexed-gather (`vluxei`) is a performance trap on this SG2044**: both clang-ours AND the ggml
  opponent lower the codebook to a vector indexed-gather and both run ~0.55 GB/s. gcc SCALARIZES our
  source into a tight unrolled scalar table lookup → 2.79 GB/s = **4.83× vs the gcc-compiled ggml opponent
  (compiler-SYMMETRIC, valid)** but only **0.94× on the shipped clang deploy path**.
- This is the MIRROR of the retracted K-quant S6 asymmetry: there clang-ours *inflated* vs gcc-opp; here
  clang-ours *deflates* (our deploy compiler picks the slow gather). The symmetric-gcc win is real but
  latent — blocked by our own clang codegen. Named **[GAP-CLANG-GATHER-TRAP]**.

## disclosure
Single-core kernel-axis streaming (one `dequantize_row`/`quantize_row`/elementwise call per sweep over an
oversized cold pool > board LLC, so weights stream cold from DRAM — cache-hygienic). Throughput timing;
correctness established BEFORE timing by an in-driver gate (ours vs opponent on identical input, byte-exact
for dequant/quant, maxrel<5e-4 for forward). [NG-4]: candidates + named gaps, NOT eight-gate beats.
