# G6-B · [GAP-EMIT-UNROLL] · rolled-loop emission mode — Phase-2 rvv/gcc-15 kernel-axis seal

- **campaign**: G6-B emit-quality lever ([K-7] 5th first-emission lever). Extends Phase-1 (clang-17 rvv
  byte-exact + FORM) to the **rvv deployment compiler gcc-15** (VLEN128, gcc/gcc SYMMETRIC).
- **board**: rvv / openEuler / riscv64 / **VLEN128 (VLENB=16, probed)** / 64c
- **compiler**: **gcc-15.2.0** `riscv64-unknown-linux-gnu` @ `/opt/tcrv-toolchains/gcc-15.2.0` (native on board;
  the *system* gcc is 12.3.1 which lacks riscv_vector.h — that is why Phase-1 deferred; this toolchain gcc-15
  is exactly the shipped-ggml compiler, so the seal is now real).
- **march**: `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on -O2` (identical to Phase-1)
- **date**: 2026-07-13 · **status**: kernel-axis seal COMPLETE · byte-exact GREEN · NOT committed (main session)
- **provenance**: emitted `.c` are the EXACT Phase-1 leaves, live on board `/tmp/g6b-emit-unroll` (md5
  unrolled=`2dd964fe`, rolled=`4e1f8089`, gevm=`b6513f6b` — match REGEN_RECIPE.md). No emitter re-run, no
  source-tree touch. gcc-15 build in a fresh scratch `/tmp/g6b-gcc15`.

## [1] BYTE-EXACT HARD GATE — GREEN (gcc-15, rvv, VLEN128)
Two independent proofs (ZERO-MODEL), identical to Phase-1 clang-17 **to the digit**:
- **INDEPENDENT scalar oracle** (decodes pre-repack blocks, isum−summs fold): ROLLED **INT_mismatch_total=0**
  all 8 shapes, NORM worst_ulp=3402 worst_norm=6.618e-07 — bit-identical to UNROLLED and to clang-17.
- **direct A/B identity** (both schedules, same random input, memcmp full fp32): **43008/43008 bytes, 0
  mismatch, 4 seeds** (0xC0FFEE / 0xBADF00D / 0x12345 / 42). INT + NORM (full fp16 scale/min fold) paths.

→ The rolled schedule is byte-exact under gcc-15 as well as clang-17. Correctness is compiler-invariant.

## [2] KERNEL-AXIS FORM SEAL — gcc-15 vs clang-17 (uniform objdump methodology)
Methodology reproduces Phase-1's clang-17 headline numbers **exactly** (textB 26968/12522, vwmacc 2304/384,
vsetvli 86/82 — validates the counting). spill/reload here = sp/s0-relative vector store/load (a defined
uniform heuristic; differs from Phase-1's scalar-spill count but applied identically to both compilers).

| form | textB | vsetvli | vsetivli | vwmacc | spill | reload | maxVreg |
|---|---:|---:|---:|---:|---:|---:|---:|
| clang UNROLLED | 26968 | 86 | 1 | 2304 | 32 | 32 | v30 |
| clang ROLLED   | 12522 | 82 | 1 | 384  | 14 | 13 | v31 |
| **gcc UNROLLED** | **104656** | **929** | **504** | 2304 | 15 | 22 | v31 |
| **gcc ROLLED**   | **17880**  | **101** | 1   | 384  | 4  | 4  | v31 |

**Answer to "does gcc-15 also narrow vwmacc/text?" — YES, and far MORE dramatically than clang:**
- **textB**: gcc 104656→17880 = **−82.9%** (vs clang −53.6%). gcc's full-unroll text is **3.88× larger than
  clang's** (104656 vs 26968) — the full-static-unroll is a codegen disaster under the deployment compiler.
- **vset ceremony** (vsetvli+vsetivli): gcc **1433→102 = −92.9%** (vs clang ~87→83, negligible). The
  full-unroll makes gcc emit 1433 vset instructions on the mixed-SEW core; rolling collapses it to 102.
- **vwmacc**: 2304→384 = **−83.3%** (both compilers — algorithm-determined, compiler-invariant).

→ **[CASE-KQUANT-GCC-CODEGEN] a-priori CONFIRMED**: "gcc explodes on the mixed-SEW full-unroll core"
(the q6_K 820-vsetvli / 2995-vsetivli pathology). Here q2_K full-unroll under gcc = **104KB text + 1433
vset**; the rolled emission mode BOUNDS the regalloc/vset code and heals it. The lever helps MORE under
gcc-15 (the deployment compiler) than under clang-17 — the hypothesis was right.

## [3] KERNEL-AXIS THROUGHPUT — gcc-15 SYMMETRIC (valid; NOT the clang-ours-vs-gcc-shipped artifact)
Opponent = real dispatched `ggml_vec_dot_q2_K_q8_K` block-dot from the board's own
`libggml-cpu.so` (`build-gcc15-rv64gcv`) → **gcc-15 on BOTH sides** (dispatches to `_vl128` at VLEN128;
ggml's q2_K repack selector returns nullptr@128, so block-dot IS the real fallback). Driver =
`kquant_gemm_paired_q2k_driver.c` (cold-flush 224MiB, best/median GMAC/s). Prefill shape K=2048 nr=64 nc=512,
15 reps × 15 rounds, load-gated (loadavg 1-min 2.4→3.0 throughout, vLLM EngineCore idle ~2%, relIQR ≈3%).

| form | OURS GMAC/s | vs block-dot | note |
|---|---:|---:|---|
| UNROLLED (shipped S6 form) | 1.763 | **0.390×** | reproduces historical q2_K rvv S6 ~0.386× LOSS |
| ROLLED | 0.480 | **0.106×** | |
| block-dot (opponent) | 4.55 | 1.0 | |

- **rolled / unrolled throughput = 0.106/0.390 = 0.272×** → on the hot kernel-axis microbench the rolled form
  **REGRESSES 3.68×**. Consistent across shapes (small K512/8/64: rolled 0.169× vs unrolled 0.293×). Tight,
  no pollution.
- **Both forms LOSE to block-dot** (q2_K has no repack@VLEN128 on rvv; the repack-GEMM approach without
  I-cache/threading leverage loses to the tuned per-block dot on rvv). rolled loses much more.

**Answer to "is rolled ≥ unrolled / ≥ parity vs block-dot on rvv/gcc-15? does it improve the S6 0.386× LOSS?"
— NO on the kernel-axis microbench.** This is a textbook **[CASE-MICRO-E2E]**: the rolled mode's win is
**code volume** (text −82.9%, vset −92.9% under gcc), an **I-cache-pressure** lever. A hot microbench keeps
the full-unroll fully I-cache-resident, so the compactness pays nothing while the rolled form pays runtime
loop overhead + stack-panel i16-partial spill/reload traffic (vse16/vle16 to s0) on every iteration — pure
overhead with no compensating benefit **in this regime**. The lever's payoff can only manifest in e2e where
the leaf competes with the rest of the model for L1I. Kernel-axis is not the arena that tests this lever.

## [4] e2e feasibility —門 OPEN on rvv (routing), lever eval belongs to k1 Phase-2
- The kernel-axis "rolled ≫ unrolled → evaluate e2e" trigger is **NOT met** (rolled regresses on kernel-axis).
- **rvv e2e routing 門 is OPEN**: ggml's q2_K repack trait selector returns **nullptr @ VLEN128** — our repack
  GEMM is not even on the rvv dispatch path (rvv falls back to block-dot). Deploying rolled q2_K on rvv e2e
  would require wiring q2_K repack routing @128 (the analog of the q4_0 VLEN128 gate), which does not exist
  for q2_K on rvv. Not a kernel problem; a routing gap.
- The **lever's e2e test is on k1** (VLEN256, gcc-15/clang-18 deployment) where `q2_K@k1 e2e prefill 0.873×`
  was measured (G5 `M2-q2_K-k1-e2e`) and where the repack path IS dispatched — that is the P1-designated
  Phase-2 e2e concern (`experiments/active/g6-b-emit-unroll/P2-q2k-k1-e2e/`). This cell settles the **rvv
  gcc-15 kernel-axis** obligation only.

## VERDICT
gcc-15 (rvv deployment compiler) seal COMPLETE. **byte-exact GREEN** (oracle + A/B, gcc-15 = clang-17 to the
digit) ∧ **FORM seal**: rolled narrows text **−82.9%** and vset ceremony **−92.9%** under gcc-15 — MUCH more
than clang (−53.6% / negligible), confirming [CASE-KQUANT-GCC-CODEGEN] (gcc explodes on the full-unroll:
104KB/1433-vset, healed to 17.9KB/102-vset by rolling). **Kernel-axis throughput**: rolled **0.272× unrolled**
(regresses 3.68×), both lose to block-dot (unrolled 0.390× reproduces S6). Per **[CASE-MICRO-E2E]** the
code-volume lever cannot pay on a hot microbench; its arena is e2e I-cache pressure. **[NG-4]: FORM seal +
kernel-axis, NOT a beat, NOT e2e.** e2e門 open on rvv (no q2_K repack@128 routing); lever e2e eval = k1 Phase-2.

## touched (this cell) — reversible, NO git action
- board: NEW ephemeral scratch `/tmp/g6b-gcc15` (gcc-15 objects/binaries + build/measure scripts). P1 dir
  `/tmp/g6b-emit-unroll` UNTOUCHED (md5 re-verified). No board system file / shared ggml / governor changed.
- local: NEW casefile `experiments/active/g6-b-emit-unroll/rvv-gcc15-kernel-axis/` only. NO tracked source
  (emitter/schema/ODS/verifier/ROADMAP/T8) modified — Phase-1's committed rolled path used read-only.
- raw: `raw/board_seal_gcc15.txt` (authentic board output: byte-exact + form seal + throughput).
