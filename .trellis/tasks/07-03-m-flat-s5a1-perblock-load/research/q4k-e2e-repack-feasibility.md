# Research: q4_K e2e repack extension feasibility (repack campaign cell #2)

- **Query**: read-only recon — can we extend the repack e2e harness to q4_K? (models on-board, A-tree q4_K repack routing status, opponent confirm, setup cost). No code/build/commit/hardware-measure.
- **Scope**: mixed (internal compiler + A-tree llama.cpp fork on `ssh rvv`/`ssh k1`, read-only probe)
- **Date**: 2026-07-06

## TL;DR verdict

**NEEDS SETUP — not directly measurable today, but NO hard structural wall.**
- In-compiler q4_K repack (op → verifier → emitter → lit test) is **BUILT**.
- rvv board already has a q4_K model (`DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`, 4.6G) → the **VLEN128 Win-B leg has a model ready, no quantize needed**.
- The blocker is **A-tree wiring**: q4_K repack is **NOT wired** into the llama.cpp fork (q4_0 IS, and is reusable as a template). Plus a rebuild + A/B measurement.
- k1 (VLEN256 parity leg) additionally has **no q4_K model** on-board.

---

## Findings

### 1. Models on board (read-only `find`)

| board | q4_K model present? | path | size |
|---|---|---|---|
| **rvv** (VLEN128, user `ubuntu`) | **YES** | `/home/ubuntu/workspace/workspace3/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf` | 4.6G |
| rvv (also present) | Q2_K / Q2_K_L | same dir | 3.0G / 3.1G |
| rvv (q4_0 baseline) | — | `~/llama-2-7b-chat.Q4_0.gguf`, `~/tcrv-llamacpp/models/tinyllama-q4_0.gguf` | 3.6G / 609M |
| **k1** (VLEN256, user `bianbu`) | **NO** | only `~/tcrv-k1-llama/models/tinyllama-q4_0.gguf` | 609M |

- **VLEN128 leg**: model READY on rvv (no quantize). Note it's NOT in the A-tree `models/` dir; llama-bench can point at the absolute workspace3 path.
- **VLEN256 parity leg**: k1 needs a q4_K model → either `scp` the 4.6G DeepSeek from rvv, or quantize a smaller one (cost: one `llama-quantize` run + transfer).

### 2. In-compiler q4_K repack — BUILT (op → verifier → emitter → test)

| Layer | Location | Status |
|---|---|---|
| ODS op (GEMV) | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:5113` `repack_gemv_q4_K_q8_K` | present |
| ODS op (GEMM) | `RVVOps.td:5245` `repack_gemm_q4_K_q8_K` | present |
| Verifier GEMV/GEMM | `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:2896` / `:3103` | present |
| Emitter GEMV | `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:3734` `emitRepackGemvQ4KQ8K` | real body (reads op, lane-wise `vwmacc`, two f32m2 accumulators) |
| Emitter GEMM | `RVVToEmitCBlockQuantLinear.cpp:4845` `emitRepackGemmQ4KQ8K` | real body |
| Emitter dispatch | `lib/Conversion/RVV/RVVToEmitC.cpp:352-357` (`isRepackGem{m,v}Q4KQ8KBody`) | wired |
| Lit tests | `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-K-q8-K.mlir` (101 lines, 36 CHECK, real kernel body), `...-repack-gemm-q4-K-q8-K.mlir` | present |

So the in-compiler "op → emitted C `.inc`" path for q4_K repack exists and is testable (`tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate` → `.inc`, mirroring the q4_0 M2 flow). Progress since the 06-24 design doc, which was pre-implementation.

### 3. In-compiler gaps for q4_K repack ROUTING / e2e divergence

- **Path selection DECLINES q4_K repack at every VLEN ≥ 128.** `lib/Plugin/RVV/RVVContractionPathSelection.cpp:79-81` — fact 1 (`ggmlHandTunedVLENNativeExists`) returns `true` for `Q4_K` (VLEN-native inline-asm exists), so `selectContractionAlgorithm` returns `BlockDot` with reason `"block-dot-decline-q4_K-vlen-native-exists"` (`:135-137`). To have the compiler *select* repack for q4_K, fact-1 for Q4_K would have to flip. **Caveat: this selection is INERT (audit attrs only; emitted C byte-identical). A hand-applied `.inc`-swap e2e experiment — like the q4_0 M2 SEAL — does NOT require the selector to flip.**
- **Stage-C strip-width materialization does NOT handle q4_K repack ops.** `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp` stamps `half_lanes`/`integer_core_lmul` on q4_0 (`:122/:134`), q4_1 (`:150/:166`), q8_0 (`:183`) repack ops — **no `GgmlRepackGem{v,m}Q4KQ8KOp` branch**. So q4_K repack gets no capability-driven VLEN128↔256 divergence stamp. Per design doc (below) the K-quant 6-bit unpack is hardcoded `m2` → the Win-A LMUL knob is NOT inherited without extra work. (Not needed for a fixed-VLEN128 Win-B e2e; needed for a Win-A knob sweep.)

### 4. A-tree (llama.cpp fork) q4_K repack routing — NOT WIRED

A-trees located: **rvv = `~/tcrv-llamacpp`**, **k1 = `~/tcrv-k1-llama`** (both on upstream ggml `master`, rvv @ `f3e1828`, ggml checkout `6eab471`). Integration is HAND-applied (C3 producer / C4 harness automation OPEN, per M2 SEAL).

| item | q4_0 (reusable template) | q4_K repack (target) |
|---|---|---|
| emitted `.inc` staged | `arch/riscv/tcrv_emitted_repack_{gemv,gemm}.inc` (Jun 23) | **absent** |
| repack.cpp injection | `ggml_gemv_q4_0_16x1_q8_0` (:210) calls `tcrv_emitc_…gemv_q4_0…` (:241); GEMM at :968/:1001; engagement print `TCRV EMITTED GEMV … ENGAGED` | `ggml_gemv_q4_K_16x1_q8_K` (:331) / `ggml_gemm_q4_K_16x1_q8_K` (:1073) are **UNUSED stubs** (all params `UNUSED(...)`, no tcrv call) |
| Patch A (`case 128` override forcing repack on VLEN128) | applied for q4_0 | **absent for q4_K** (dispatch `disp:4619` still `break`→`nullptr`) |

**Different axis already wired (do not confuse):** rvv A-tree has `arch/riscv/tcrv_q4k_swap_active.inc` (Jun 30) `#include`d at `quants.c:1771` and called at `:1774` (`tcrv_q4k_vl256_swap`). Its header says *"AUTO-GENERATED q4_K **vec_dot** swap"*, source op `tcrv_rvv.q4_k_q8_k_block_dot`. **This is a block-dot-vs-block-dot swap (flat block-dot campaign / N1 coverage), NOT the repack Win-B.** k1 A-tree has NO q4k swap and NO q4_K repack wiring.

### 5. Opponent confirmation (matches task premise) — `routing-audit.md`, ggml @ `6eab471`

| cell | ggml dispatched kernel | our repack vs |
|---|---|---|
| **q4_K rvv VLEN128** (decode + prefill) | block-dot `vec_dot_q4_K_q8_K_vl128` — **inline RVV ASM** `quants.c:1770` (dispatcher `:2064`); repack gated OFF `disp:4619 break→nullptr` (no ggml repack on rvv) | **Win-B candidate**: our q4_K repack routing vs hand inline-asm block-dot (harder opponent than q4_0's plain intrinsic body) |
| **q4_K k1 VLEN256** (decode / prefill) | ggml's OWN repack `gemv_q4_K_16x1_q8_K` (`rp:260`) / `gemm_q4_K_16x1_q8_K` (`rp:983`), `#if __riscv_zvfh` | **parity expected** (both repack, like q4_0's k1 0.997× TIE) |

✓ Confirms task points 3. Also `KQUANT-REPACK-DESIGN.md:12` independently confirms VLEN256-only ggml routing + VLEN128 `//TODO`→break.

### 6. Setup cost estimate (reuse of q4_0 template)

**Reusable from q4_0 (already proven, M2 SEAL):**
- `.inc` emit flow (`tcrv-opt … | mlir-translate` → `.inc`).
- repack.cpp injection pattern (call emitted symbol through ABI adapter; ggml 7-arg → emitted 5-arg adapter).
- Patch A `case 128` override mechanism (retarget to q4_K dispatch line `disp:4619`).
- Engagement two-sided gate (`TCRV EMITTED … ENGAGED` print + objdump tail-jump verify).
- llama-bench A/B harness (ON/OFF same thermal state, `.so` timestamp check) + coherence seal ("Paris") + numeric norm check.

**New q4_K-specific steps:**
1. Emit q4_K repack gemv/gemm `.inc` (emitters + tests exist → low effort; needs a build).
2. Fill the stubbed `ggml_gemv/gemm_q4_K_16x1_q8_K` bodies with the tcrv call + adapter, and add Patch A `case 128 → q4_K repack` at `disp:4619` (`ne[1]%16==0` gate). Heavier ABI than q4_0: activation is `block_q8_Kx4` / `ggml_quantize_mat_q8_K_4x1` (super-block q8_K, not q8_0) — that packer already ships in ggml (`repack.cpp:349-352`).
3. Rebuild `libggml-cpu.so` on rvv; A/B with the DeepSeek Q4_K_M model (reuse harness).
4. **VLEN256 parity leg (k1)**: transfer/quantize a q4_K model; wire the same `.inc`-swap (on k1 q4_K repack is already native, so Patch A not needed — the swap replaces ggml's own repack body; comparison is parity).

Design-doc effort anchor (`KQUANT-REPACK-DESIGN.md:4-5,54`): q4_K repack "TRACTABLE, **multi-day/structural (~3-5d)**, NOT upstream-blocked" — but that estimate was for building the *emitter* (now DONE). Remaining = the A-tree wiring + build + measure, which is closer to the q4_0 template effort (a few steps) plus the heavier q8_K activation ABI.

## Caveats / Not Found

- **WIN is NOT established — historical micro was a LOSS.** `routing-audit.md:113` records the doc's q4_K repack "Win-B 0.55×/0.74×" was measured **vs the rvv block-dot** — i.e. prior *micro* numbers showed q4_K repack **losing** (0.55×/0.74× < 1.0). This is exactly the basis for the in-compiler decline policy (§3). The design doc (`:47`) nonetheless predicts a *realistic e2e WIN @ VLEN128 via memory-locality* (the q4_0 decode 2.6× mechanism) and a "disclosed likely LOSS @ k1 like q4_0's 0.74×". These two data points are in tension; per project memory the historical numbers are **STALE**. **This recon does NOT adjudicate — the e2e Win-B for q4_K@128 is UNVERIFIED and is the key open risk (could be a loss, not a Win-B).**
- **No hardware measurement performed** (read-only recon; only `ls`/`find`/`grep` on-board).
- **Bit-exactness on real q4_K weights not re-sealed here.** Emitter has lit tests (byte/shape), but board numeric correctness against real q4_K weights (analogous to q4_0's "Paris" coherence + norm 1.9e-5) would need to be established during actual wiring.
- Line numbers cited against ggml checkout `6eab471` and current TianChenRV branch `refactor/full-refactor-m1`; A-tree files re-confirm before edit (M2 SEAL notes prior per-experiment edits).
- The q4_K repack functions in the A-tree repack.cpp (`:331`/`:1073`) show only the non-zvfh UNUSED-stub prologue in the read; the real ggml zvfh body is under `#if __riscv_zvfh`. Not fully read (out of read-only scope); does not change the "no tcrv injection" finding.
