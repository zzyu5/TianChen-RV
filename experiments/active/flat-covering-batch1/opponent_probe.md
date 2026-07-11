# opponent probe — FLAT repack@VLEN128 absence + opponent identity/vectorization

## opponent-absence (why block-dot IS the fair opponent for prefill GEMM @ VLEN128)
Source: board's own `/home/ubuntu/llama.cpp-upstream-native/ggml/src/ggml-cpu/repack.cpp` (tag b9692/f3e1828,
the exact ggml the linked `libggml-cpu.so` md5 d1adc634… was built from). RISC-V repack trait selector:

    // q4_0  (:4589)                          // q8_0  (:4710)
    if (ggml_cpu_has_riscv_v()) {             if (ggml_cpu_has_riscv_v()) {
      switch (__riscv_vlenb() * 8) {            switch (__riscv_vlenb() * 8) {
        case 128: { break; } // TODO   <===       case 128: { break; } // TODO   <===
        case 256: return &q4_0_16x1_q8_0;         case 256: return &q8_0_16x1_q8_0;
      }                                          }
    }

- q4_0, q8_0: riscv repack exists ONLY at **case 256** (k1/VLEN256). **@VLEN128 → `break` → nullptr → block-dot
  fallback.** (Same TODO pattern as q4_K/q2_K/iq4_nl in the same file.)
- q4_1, q5_0, q5_1: **NO `ggml_cpu_has_riscv_v()` repack branch at all** → nullptr at every VLEN → block-dot always.
- ⇒ At VLEN128, ggml's prefill `mul_mat` for ALL 5 FLAT formats dispatches per-(row,col)
  `ggml_vec_dot_qX_qY`. That is exactly the opponent this cell times against (linked from the board's own .so).
  **Structural L1 path-win basis CONFIRMED for all 5.**

## opponent identity (per-datum binding)
- symbol = arch-dispatched `ggml_vec_dot_q{4_0,4_1,5_0,5_1,8_0}_q8_{0,1}` (public `T`, NOT the `_generic` twin)
- built by **gcc-15.2.0** (board shipped ggml compiler) ; opponent side is compiler-insensitive per
  [CASE-COMPILER-ASYMMETRY] (block-dot ~1.002× clang-vs-gcc) — so it is a valid fixed baseline for BOTH ledgers.
- vectorization (objdump rvv-insn count inside each symbol): q4_0 ≈ 6, q8_0 ≈ 6 (light intrinsic block-dot);
  q5_0 ≈ 17 (better vectorized). This asymmetry EXPLAINS the ratio spread: light-vectorized q4_0/q8_0 opponents
  → our repack wins 3.5–7×; well-vectorized q5 opponent → our repack wins only ~1.2–1.4×.
- wording tier (per repack-campaign-terrain discipline): opponent = ggml **real dispatched intrinsic block-dot**
  (NOT hand-tuned inline-asm like the K-quant vec_dots; NOT `_generic` scalar). So the win is
  repack-vs-dispatched-intrinsic-block-dot — a legitimate path-candidate, tier below "vs hand-tuned asm".

## disclosure
Single-core kernel-axis (opponent = single-thread per-(row,col) block-dot loop, a prefill-GEMM proxy, not the
threaded `mul_mat`). Throughput timing; correctness established by the in-driver ZERO-MODEL numeric gate +
construction byte-exact oracle lit. [NG-4]: candidate, not a beat.
