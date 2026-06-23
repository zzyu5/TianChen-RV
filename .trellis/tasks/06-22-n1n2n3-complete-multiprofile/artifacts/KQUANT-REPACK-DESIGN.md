# K-quant (+ q8_0) repack — design + honest feasibility verdict (2026-06-24)

**VERDICT: TRACTABLE, multi-day/structural for q4_K, and — decisively — NOT upstream-blocked (this separates
q4_K from q4_1). SEQUENCING PIVOT (advisor-endorsed): q8_0 repack FIRST (least effort, closes gap 6), then
q4_K (dominant-quant headline), q6_K last (simpler kernel but harder e2e).**

## Q1 — ggml baseline + e2e routing (make-or-break): q4_K e2e leg EXISTS
ggml ships the FULL RVV q4_K repack (unlike q4_1):
- `block_q4_Kx8`/`block_q4_Kx16` weight types (`repack.h:43-58,86-94`); `ggml_gemv_q4_K_16x1_q8_K` RVV SIMD
  impl at `arch/riscv/repack.cpp:260`, `ggml_gemm_…:983` (guarded `#if __riscv_zvfh`).
- `block_q8_Kx4` activation mat-quantizer + `ggml_quantize_mat_q8_K_4x1` (`repack.cpp:349-352`).
- mul_mat ROUTES q4_K via repack **at VLEN256 only** (`repack.cpp:4620` → `q4_K_16x1_q8_K`); **VLEN128 =
  `// TODO`→break→nullptr** (`:4619`) → falls back to plain `ggml_vec_dot_q4_K_q8_K` (a BEATABLE baseline).
- **Win-B baseline**: @K1 VLEN256 = ggml's OWN hand-tuned RVV q4_K repack (strong bar); @rvv VLEN128 = the
  fallback block-dot (beatable — the q4_0 5.68× exploit regime).
- **Engagement @VLEN128** uses the q4_0 template (NOT native routing): patch `repack.cpp` case 128 (q4_K
  `:4619`, q8_0 `:4713`) to return the repacked path when `ne[1]%16==0`, + `.inc`-body-swap
  (`arch/riscv/tcrv_emitted_repack_{gemm,gemv}.inc`) — per `winA-e2e/FINDING.md:60-86`. llama tree currently
  clean (upstream `6eab471`); those were per-experiment edits.
- **q6_K is closer to q4_1's gap**: NO riscv_v routing branch, NO RVV SIMD arch impl (only `_generic`/NEON).
  Simpler kernel (8-bit scales, no 6-bit dance, no min/dmin) BUT harder e2e engagement + no native RVV baseline.

## Q2 — our mechanism: "16-blocks-as-lanes"
`block_q4_0x16` interleaves 16 weight columns; one fp32 vector accumulator per 16 cols; `vwmacc`-into-vector,
**NO per-block vredsum** (lane-wise). VLEN divergence = strip count only (`numHalves = weight_interleave/
half_lanes`, `RVVToEmitCBlockQuantLinear.cpp:2193-2203`). ggml's `q4_K_16x1` uses the IDENTICAL idea
(`ncols_interleaved=16`, one `vfloat32m2_t sumf`, `arch/riscv/repack.cpp:281-388`) — confirms the approach maps.
**Win-A knob NOT free**: repack carries `integer_core_lmul`+`half_lanes` (`RVVOps.td:4252-4253,4363-4364`,
`RVVRepackStripWidthMaterialization.cpp:78`), BUT the K-quant 6-bit unpack is hardcoded m2
(`RVVToEmitCKQuant.cpp`) — must be made LMUL-parametric to inherit Win-A. Design work, not inheritance.

## Q3 — block-as-lane maps to q4_K superblock? Yes structurally, but a NEW kernel (rewrite, not q4_0 variant)
q4_K = 256 elems / 8 sub-blocks of 32 / 4-bit qs / 6-bit scales+mins in `scales[12]` / one (d,dmin) fp16.
Three non-trivial changes vs q4_0:
1. **Weight layout**: `block_q4_Kx16` with a custom RVV-only 6-bit re-pack (`make_block_q4_Kx16`,
   `repack.cpp:2913`, `blck_size_interleave==1`) laying scales/mins into `out.scales[0..191]` for lane-wise
   `vle8_v_u8m2(&scales[j*64])` unpack. Our op consumes THIS layout (or we author our own interleave).
2. **Scale handling**: lane-wise 6-bit unpack at u8m2 across 16 lanes (ggml `arch/riscv/repack.cpp:299-315`).
   Our q4_K block-dot has the 6-bit utmp/kmask dance but SCALAR (`RVVToEmitCKQuant.cpp:919-1006`) — re-express
   lane-wise (new code).
3. **Dual d/dmin fold**: ALREADY proven lane-wise in our q4_1 repack (`RVVToEmitCBlockQuantLinear.cpp:3580-3629`);
   q4_K differs: min term uses bsums × per-sub-block mins, 8 sub-blocks → inner vwmacc loops over sub-blocks
   with per-sub-block scale. Lane-wise accumulator (no vredsum) preserved (memory-locality transplant thesis holds).

## Q4 — cells filled, effort, SEQUENCING
Cells a q4_K repack fills: Win-A micro (once unpack LMUL-parametric); Win-B micro (vs ggml's RVV q4_K repack
@K1 strong bar / vs vec_dot @rvv128 beatable); Win-B e2e (realistic WIN @rvv VLEN128 via `.inc`-swap;
disclosed likely LOSS @K1 like q4_0's 0.74×); Win-C = NONE.

**SEQUENCING:**
1. **q8_0 repack FIRST** — simpler than q4_0 (int8, NO nibble unpack, single scale). ggml ships
   `ggml_gemv/gemm_q8_0_16x1_q8_0` RVV (`arch/riscv/repack.cpp:447,1336`) + VLEN256 routing + `.inc`-swap.
   Gives the q8_0 VLEN-selection flip an e2e home → **closes load-bearing gap 6**. Least effort × highest N1 value.
2. **q4_K repack SECOND** — dominant-quant headline, multi-day/structural rewrite (~3-5d). Not upstream-blocked.
3. **q6_K LAST/noted** — simpler kernel, harder e2e (no native RVV routing, generic-only baseline).
   (q5_0/q5_1 repack: flat-block direct q4_0 extensions, lower e2e value than q8_0's gap-6 closure.)

## Critical files
- `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (q4_0 repack :2122; q4_1 lane-wise dual-fold :3580-3629)
- `lib/Conversion/RVV/RVVToEmitCKQuant.cpp` (q4_K 6-bit unpack scalar today :919-1006/:1571-1594 → vectorize)
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (repack op defs :4155-4583 → new ops + verifier)
- `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp` (Win-A half_lanes/integer_core_lmul :78)
- ggml: `arch/riscv/repack.cpp` (q4_K baseline :260/:983; q8_0 :447/:1336) + `repack.cpp` (routing patch
  sites :4619 q4_K / :4713 q8_0; `make_block_q4_Kx16` :2913)
