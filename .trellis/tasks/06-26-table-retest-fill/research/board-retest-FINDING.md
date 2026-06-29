# Board re-test FINDING — rvv (SG2044, VLEN128, 64 harts)

Date: 2026-06-29. Board: **rvv** UP (load ~0.32, 118 GB free, nproc=64). Live `build/bin/libggml-cpu.so.0` md5 == `304d1128b0c4c58ad6093fb8abe2015e` (m1) re-asserted at session start.

> DEVIATION (documented): the plan references `/data` for the decoupled arms + snapshots, but **`/data` does not exist on the board** (root `/` has 658 GB free — the recon conflated `/` with `/data`). Used `/home/ubuntu/q4k-retest-scratch/` as the functionally-equivalent decoupled path (same root fs). All `/data/q4k_*.so` references below map to `/home/ubuntu/q4k-retest-scratch/q4k_*.so`.

> DEVIATION (documented): the plan's objdump discriminator says "mf2 == 0 over the whole `.so`". The whole-`.so` `e32,m4|vslidedown` count is **163** on m1 (dominated by other kernels: q6_K/q3_K/IQ etc.) — it can NOT be 0 for mf2. The correct **scoped** discriminator disassembles exactly the q4_K swap symbol `tcrv_emitc_q4k_kernel` (via `nm -S` addr+size): m1 = 4× `e32,m4|vslidedown` (3 e32,m4 + 1 vslidedown), mf2 expected 0. Plus the source-`.inc` grep (m1 inc = 12× `i32m4|vslidedown`; mf2 inc = 0×) as a redundant pre-build gate.

---

## Phase-0 (MANDATORY guard) — DONE

- P0.1 live `.so` md5 == 304d1128 (asserted).
- P0.2 decoupled m1: `cp build/bin/libggml-cpu.so.0 → /home/ubuntu/q4k-retest-scratch/q4k_m1.so`, md5 == 304d1128 (asserted).
- P0.3 dirty-tree snapshot → `/home/ubuntu/q4k-retest-scratch/tree-snapshot.tgz` (md5 `3686dcd3af5d687d985a70c9e351e2d0`, 7 files: 4 tracked M [ggml-cpu.c, repack.cpp, arch/riscv/quants.c, arch/riscv/repack.cpp] + 3 untracked .inc [tcrv_q4k_swap_active.inc, tcrv_emitted_repack_gemm.inc, tcrv_emitted_repack_gemv.inc]).
- Source-inc pre-gate: active inc == m1 (md5 8da97c54, 12× `i32m4|vslidedown`).

---

## TIER-1 Cell-group C — q4_0 repack PREFILL GEMM 5.68× reseal + DECODE GEVM sibling — MEASURED + SEALED

- **Board/regime**: rvv VLEN128, 64c, `-t 16` (winB_e2e.sh harness, no taskset — the canonical reseal harness; pp256 prefill is compute-bound so core-pinning is not load-bearing here).
- **ggml baseline (named)**: BLOCK-DOT `ggml_vec_dot_q4_0_q8_0` (OFF arm = case128 reverted → break → nullptr → block-dot; md5 1f2727b5, objdump-confirmed 10 real RVV ops per recon). rvv ships NO q4_0 repack at VLEN128 (routing-audit P3). NOT vs ggml's own repack.
- **Gates passed**: ON .so md5 == cabcd588 / OFF .so md5 == 1f2727b5 per arm (harness prints); body-compare seal `tcrv_emitted_repack_gemm.inc` md5 == `4e21a79c` (== archived emitted-repack-gemm.cpp) + `nm -D ON.so` shows `tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel...` defined T (compiler-emitted, NOT hand kernel); ENGAGED "TCRV EMITTED GEMM(q4_0_16x1 VLEN128 compiler-emitted) ENGAGED nr=256 nc=64 nb=128" fired ON=3/7, OFF=0/0; numeric byte-exact already board-PASS (norm 1.34e-5 vs _generic, cited). Coherence: q4_0 path already certified (ON .so is the live coherent build).
- **Measured (2 ON/OFF pairs, -r3 each)**:

| arm | pair-1 pp256 | pair-2 pp256 | pair-1 tg64 | pair-2 tg64 |
|---|---|---|---|---|
| ON (our compiler-emitted repack, mf2 auto-form) | 17.95 ± 0.01 | 18.00 ± 0.02 | 6.78 ± 0.12 | 8.06 ± 0.56 |
| OFF (ggml block-dot) | 3.15 ± 0.00 | 3.14 ± 0.00 | 2.63 ± 0.05 | 2.49 ± 0.01 |

- **PRIMARY — prefill GEMM reseal (A3-analog for q4_0)**: pp256 ratio = 17.95/3.15 = **5.70×**, 18.00/3.14 = **5.73×** → **~5.70–5.73×** (consistent with documented 5.68×). **SEALED as compiler-emitted mf2 auto-form** (md5 cabcd588 = NARROW = mf2, the only RVV1.0 auto-emitted repack form). Do NOT claim WIDE/m1 (separate manual stamp ba8be266, not staged), do NOT conflate with orthogonal Win-A·rvv 1.30× knob.
- **SIBLING — decode GEVM**: tg64 ratio = 6.78/2.63 = **2.58×**, 8.06/2.49 = **3.24×** → central **~2.6×** (matches documented 2.60×; pair-2 tg ON has high variance ±0.56 → conservative read 2.58×). STORED-x16 memory-locality decode win.
- **Evidence-status**: MEASURED + SEALED. Closes 待重测 #5 (未盖章 → sealed).

---

## TIER-2 Cell-group A — q4_K block-dot Win-A m1-vs-mf2 8B memory-wall judge

STATUS: in-progress (q4_0 C banked durably first; q4_K mf2 build + A/B follows). See updated section below.

---

## RESTORE

(pending — final pristine restore to be recorded here.)
