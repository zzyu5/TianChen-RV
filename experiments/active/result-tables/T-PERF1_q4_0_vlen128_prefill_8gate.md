# [PERF-1] 8-gate status cell — q4_0 @ VLEN128 e2e prefill (repack GEMM)

- **campaign**: repack · **status**: ACTIVE (gate-by-gate accounting; NOT a fully-sealed 8-gate Win)
- **claim under test (bound)**: on board `ssh rvv` (openEuler, VLEN128), the TianChen-RV
  compiler's capability-keyed path selection engages a front-door-**constructed** q4_0 repack
  GEMM (`be66c917 typed_repack_gemm_loop_body`) on the prefill (pp128) path and delivers
  **e2e prefill 5.92×** [5.91,5.93] vs stock upstream ggml **block-dot** (git `f3e1828`),
  DVFS 0.00%, greedy-token-consistent. This is an **L1 path-selection win** (capability-keyed
  repack **routing** + memory-locality), **NOT** "our GEMM out-codegens their GEMM".
- **authoritative gate definition**: 科研目标总纲 v2 §4.4 [PERF-1]
  (`docs/canon/TianChen-RV_科研目标总纲v2.md:168-170`).
- **primary evidence cell**: `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/`
  (5.92× constructed era). Companion routing-era 5.08× cell:
  `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/`. The two are ONE comparability
  spine, **not additive** (see that cell's MANIFEST §6.1 reconciliation).

## Verdict: 5 / 8 PASS · 3 pending → NOT a sealed universal Win-C; beat wording stays LOCKED ([NG-4])

| # | gate (§4.4) | status | evidence pointer + (local re-verify) |
|---|---|---|---|
| ① | 字节精确 (byte-exact) | **PASS via ULP-upper-bound branch — NOT ULP0** | Repack GEMM uses vector FMA (`__riscv_vfmacc_vv_f32m2`; re-verified locally: `tcrv-opt … --tcrv-rvv-lower-quant-contraction --tcrv-rvv-lower-to-emitc \| FileCheck` EMIT prefix PASS). Silicon verdict = **FMA_FOLD_BOUNDED_ULP**, substantiated as-accurate-or-better than ggml's own scalar vs f64 truth (2/4 shapes strictly closer). `experiments/sealed/silicon/silicon-validation-gemm/results/q4_0_repack_gemm/evidence.json`. E2E: greedy-token 3/3 A==B. Construction golden (CONSTRUCT/EMIT/PLAN/core==prod) re-verified locally GREEN on HEAD `ba0a71f8`. **Honest**: correctness met by bounded-ULP, the wording MUST NOT say "byte-exact vs ggml". |
| ② | VLEN 翻转 lit (128/256 双配置) | **MISSING** (RVV1.0 VLEN256 flip lit) | RVV1.0 **VLEN128** body lit exists + re-verified locally PASS (`test/Conversion/RVV/rvv-to-emitc-typed-repack-gemm-loop-body-vlen128-full-body.mlir`). The second body arm is **RVV0.7.1 / XTheadVector** whole-LMUL (`…-rvv07-full-body.mlir`) — a **family** flip, NOT a VLEN256 flip. No RVV1.0 VLEN256 codegen-flip lit for the repack GEMM. Runtime VLEN256 e2e exists (k1) but that is a measurement, not the lit gate. |
| ③ | 双板各一次 objdump 验封 | **PARTIAL (1/2 boards)** | Board A (rvv VLEN128): SEALED — mattr-decoded fingerprint (vwmacc 512 / vle8 242 / vfwmul.vf 21 = repack mechanism) `…/rvv-vlen128-q4_0-gemm-constructed-sealed/objdump_fingerprint.txt` + `…/rvv-SEALED-…/objdump_seal.txt`. Board B (k1 VLEN256): **NO objdump seal** — `experiments/sealed/repack/k1-vlen256-q4_0-flip/` has no objdump file. → **pending new board batch (k1 objdump)**. |
| ④ | micro AND e2e (llama-bench, prefill/decode 分相, 多 prompt 长度) | **PARTIAL** | e2e PRESENT: prefill pp128 5.92× + decode tg32 1.91×, phase-split, n=10, 95% CI (`…/constructed_evidence.json`). micro PRESENT: isolated-kernel silicon-validated + kernel byte-exact estimate 5.045×. **Pending**: (a) micro↔e2e **Amdahl transduction** accounting (flagged OPEN in the fullmarch NOTES); (b) **multi-prompt-length** sweep — only pp128 measured (single length). |
| ⑤ | 双板都验证 | **PASS (dual-board measured; win is VLEN128-CONDITIONAL)** | Board A rvv/VLEN128 prefill **5.92× WIN**; Board B k1/VLEN256 prefill **1.0022× PARITY** (compute-bound, expected per VLEN-flip). Both boards measured. The win does NOT replicate as a win on board B — it is characterized parity (the VLEN-flip mirror; decode flips to 0.854× GAP at VLEN256). `experiments/sealed/repack/k1-vlen256-q4_0-flip/evidence.json` + T8 row `q4_0-gemm-k1-vlen256-prefill-parity`. Honest: the WIN itself is single-board (VLEN128). |
| ⑥ | 实验纪律 (freq-lock / pin / median+var / same-commit ggml / model+format+ctx) | **PASS** | DVFS 0.00% span; cores 8-11 pinned / 4thr; median + IQR + 95% CI, n=10; both trees git `f3e1828`; model `tinyllama-q4_0.gguf` sha256 `da3087fb14aede55`, q4_0, pp128; T-N floor prefill 0.25% (5.92× is >200× the floor). `…/target_profile.txt`, `…/noisefloor_evidence.json`. **Honest confound**: SPEC bzip2 co-tenant throughout — captured in the T-N floor; paired interleaved protocol cancels common-mode (ratio protected, absolute t/s depressed). |
| ⑦ | 机制合成归因 (selector 日志证明能力键选中获胜变体) | **PASS** | Runtime ENGAGE banner = selector log: `TCRV CONSTRUCTED GEMM(front-door be66c917 typed_repack_gemm_loop_body) ENGAGED` (`…/objdump_fingerprint.txt` §1). Mechanism = capability-keyed path selection: A routes q4_0→repack at VLEN128; stock B stays on block-dot (repack VLEN-gated/TODO there). mattr-objdump fingerprint confirms the repack mechanism. **Upgrade (not blocking)**: sampled PMU hot-symbol profile = 0 samples (RISC-V PMU paranoid=2) — a sampled runtime profile is a remaining nice-to-have; the ENGAGE banner + static objdump satisfy the selector-log requirement. |
| ⑧ | 措辞门 (主张精确到相, [L-1]) | **PASS (conditional on disciplined wording)** | Claim bound to phase=prefill · board=rvv-openEuler-VLEN128 · format=q4_0 · baseline=ggml `f3e1828` · mechanism=capability-keyed repack path-selection. NOTES wording disciplined. **Constraints**: MUST NOT generalize to "faster repack codegen" (it is routing + memory-locality; prefill tiled kernel is byte-identical A/B in the routing era), MUST NOT claim byte-exact (see ①), MUST NOT drop the board/phase binding. |

## Local re-verification performed this session (HEAD `ba0a71f8`, `build/bin/tcrv-opt` + `FileCheck`)
- `rvv-emit-quant-contraction-q4-0-repack-gemm-prefill-vlen128.mlir` — default + NOWHOLE prefixes: **PASS**.
- `rvv-to-emitc-typed-repack-gemm-loop-body-vlen128-full-body.mlir` — **PASS**.
- `q4-0-q8-0-repack-gemm-full-pipeline-export-e2e.mlir` — CONSTRUCT / EMIT / PLAN / (core==prod diff): **PASS**. OBJECT/SYMBOL are `REQUIRES: tianchenrv-local-rvv-object-clang` (correctly skipped without an on-host RISC-V object clang).
- EMIT confirms the emitted kernel uses `__riscv_vfmacc_vv_f32m2` → the correctness verdict is legitimately **FMA-fold bounded-ULP**, not byte-exact (gate ① nuance above).

## Missing / pending items → next board batch
1. **Gate ② — RVV1.0 VLEN256 flip lit** for the repack GEMM (codegen adapts across VLEN128/256; lib-side, out of this cell's touch-set).
2. **Gate ③ — board-B (k1 VLEN256) objdump seal** for the deployed repack GEMM/GEVM (new k1 board run).
3. **Gate ④ — micro↔e2e Amdahl transduction accounting + multi-prompt-length e2e sweep** (pp256/pp512 …), single pp128 today.
4. (nice-to-have, gate ⑦) sampled PMU hot-symbol profile once a board relaxes `perf_event_paranoid`.

Until ②③④ close, this is a **characterized board-A-conditional L1 path-win (5/8)** — beat/​outperform wording remains **locked** per [NG-4]; the only quotable sentence is the fully-bound one above.
