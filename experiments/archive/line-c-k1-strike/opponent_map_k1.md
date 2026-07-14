# opponent_map(k1_fp, ggml_commit) — Line C · k1 精准打击 1a

> **口径**: 承 `covering-batch3-stream-rvv/opponent_probe.md` §2.4 体例. Per-datum opponent identity +
> as-shipped + vectorization, ON `ssh k1` (SpacemiT X60 / VLEN256 / 8 harts / stock ggml). **Read-only**
> board probing (objdump/nm/readelf/CMakeCache) — board CLEAN, nothing written, restore trivially clean.
> Raw evidence = `raw_probe_evidence.txt` (this dir). **[NG-4]**: kernel-axis opponent-identity map, NOT beats.

## board / opponent fingerprint
- `k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / 8 harts / Bianbu 6.6.63 / gov=performance 1.6GHz / IME ext present`
- STOCK opponent lib = `/data/k1build-stock/bin/libggml-cpu.so.0` (→ `.so.0.15.1`), ggml tag 0.15.1.
- **★compiler identity (DECISIVE — CMakeCache proof)**: stock `CMAKE_C_COMPILER=/usr/bin/clang-18`,
  `CMAKE_CXX_COMPILER=/usr/bin/clang++-18`. **k1 STOCK ggml opponent = clang-18**; OUR deploy = clang-18.
  ⇒ **kernel-axis is COMPILER-SYMMETRIC on k1** (both clang-18), **UNLIKE rvv** (stock=gcc-15, the
  [CASE-COMPILER-ASYMMETRY] withdrawal). The t4a MANIFEST's "asymmetry identical in kind to board A"
  assumption is **CORRECTED** by CMakeCache. **kernel-axis vs system-axis CONVERGE on k1** (no asymmetry
  artifact) — the k1 kernel-axis ratios (q4_K 3.106× / q5_K 1.916× from t4a) are compiler-symmetric-VALID,
  NOT clang-ours-vs-gcc inflation.  *(disclosure still required per 口径: kernel=clang-18 / system=clang-18.)*
- **★SpacemiT/IME vendor path in stock = ABSENT** (`nm | grep -ic spacemit = 0`). Opponents are pure upstream
  ggml (riscv-16x1 tuned repack OR generic block-dot vec_dot). No vendor-IME kernel in the shipped opponent.

## dispatch map — `ggml_repack_get_optimal_repack_type` (repack.cpp:4528), riscv/VLEN256
riscv repack is `#if __riscv_zvfh` + `switch(vlenb*8){case 256: if(ne1%16==0) return &X_16x1;}`. Formats WITHOUT a
riscv branch (or whose riscv branch is `nullptr`) fall to per-(row,col) **block-dot `ggml_vec_dot_*`** (the factory
matmul fallback). Two opponent tiers on k1:

| tier | formats (as-shipped VLEN256) | opponent identity | verdict meaning |
|------|------------------------------|-------------------|-----------------|
| **A · riscv-16x1 tuned repack** | **q4_0, q4_K, q2_K, iq4_nl, q8_0** | `ggml_gemm/gemv_X_16x1_*` (clang-18, `vsetivli 16`=VLMAX@256, vectorized) | **硬对拼格** — opponent is a runnable tuned repack; **parity = good result** |
| **B · block-dot vec_dot fallback** | q5_K, q6_K, q3_K, mxfp4, nvfp4, q4_1, q5_0, q5_1, iq1_s/iq1_m/iq2_*/iq3_*/iq4_xs, tq1_0/tq2_0, q1_0 | `ggml_vec_dot_X_*` (clang-18, vectorization varies — see below) | opponent = factory block-dot; net-new-scaffold **candidate** axis (kernel-axis) |

### tier-B vec_dot vectorization (dispatched opponents, objdump rvv-insn count)
- `ggml_vec_dot_q5_K_q8_K` = **13 rvv insns / 15 vsetvli → VECTORIZED** intrinsic block-dot (real tuned-ish opponent).
- `ggml_vec_dot_q6_K_q8_K` = **0 rvv insns → SCALAR** (weak opponent; notable — q6_K matmul opponent is un-vectorized).
- `ggml_vec_dot_q3_K_q8_K` = 6 rvv insns / 14 vsetvli → lightly vectorized.
⇒ the q5_K opponent (the [X-0] target) is a **vectorized clang-18 block-dot** = a legitimate compiler-symmetric
opponent (matches t4a "factory block-dot" framing; the k1 kernel-axis 1.916× is a real symmetric ratio, not asymmetry).

## ★ q8_0-k1 VLEN256 correctness — DOUBLE-EXIT (task-registered, both T8-bound)
Stock `ggml_gemm/gemv_q8_0_16x1_q8_0` head = `vsetivli zero,16,e32,m2,ta,ma`; gemv rvv_insns≈65 (well-vectorized).
At VLEN256: VLMAX(e32,m2)=256·2/32=**16** ⇒ hardcoded AVL=16 **== VLMAX ⇒ runs FULL WIDTH = HEALTHY**.

- **EXIT-1 · 健康 (RESOLVED)**: `q8_0-k1 = 硬对拼格`. The upstream q8_0 repack kernel is a correct, vectorized,
  as-shipped tuned kernel at VLEN256 (`vsetivli 16` native). Parity against it = a good result (opponent is a
  real tuned kernel, not a weak fallback). Whole 16x1 family (q4_K/q2_K/iq4_nl) shares the same healthy `vsetivli 16` head.
- **EXIT-2 · 碎片化第二证据 (SIMULTANEOUSLY TRUE)**: the SAME hardcoded-AVL=16 body is **VLEN-NON-PORTABLE** —
  at VLEN128 VLMAX(e32,m2)=8, AVL=16 clamps to 8 → half the 16-interleave columns skipped → **garbage** (PROVEN in
  G5-M1b: 裸翻 gate = garbage RED; our emitted vl=8 carrier = coherent GREEN). ⇒ **[GAP-Q8_0-VLEN128-KERNEL]** =
  手写库 VLEN 不可移植 = **C1 对手侧碎片化供弹**. Healthy@256 ∧ broken@128 = the same non-portable body = the
  fragmentation thesis (a hand-written library kernel that is correct on exactly ONE VLEN, not capability-keyed).

## honest scope / [NG-4]
Kernel-axis opponent-identity map only. No timing re-run this session (t4a kernel-axis ratios reused, now re-labeled
compiler-symmetric-VALID by CMakeCache). No beats, no e2e claims. Board read-only; restore trivially clean.

## durable files (this cell)
- `opponent_map_k1.md` — this analysis (dispatch map + q8_0 double-exit + compiler-symmetry resolution).
- `opponent_map_k1.csv` — per-format opponent identity / as-shipped / vectorization grid.
- `raw_probe_evidence.txt` — raw objdump/nm/CMakeCache board-probe outputs (read-only; board clean).
