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
- **Caveat (user wrap-up framing, 2026-06-30)**: the user flagged perf as a premature 收尾 step. C is preserved as MEASURED because (a) it ran on the **consistent 7B** baseline (llama-2-7b-chat.Q4_0, matching the historical 06-18/06-22 runs — unlike the q4_K 8B model), and (b) it was the explicitly-named "reseal" deliverable (待重测 #5) consuming a FROZEN compiler-emitted body. The main session should still weigh the user's "perf is 收尾, not now" stance before folding into the table.

---

## TIER-2 Cell-group A — q4_K block-dot Win-A m1-vs-mf2 8B memory-wall judge

- **Model**: DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf (4.92 GB, read-only, mixed-quant → m1/mf2 ratio is e2e-with-dilution; q6_K/q5_K tensors carry no knob).
- **Board/regime**: rvv VLEN128, 64c. A1 = decode tg64 @ `taskset -c 0-15 ... -t 16`; A3 = prefill pp256 @ same `-t 16` runs (FREE, same invocations). A2 (-t1) = NOT RUN (board-crash precedent), board-pending.
- **ggml baseline (named)**: NOT vs-ggml — this is the m1/mf2 axis, BOTH our emit, swapped INTO the shipping rvv q4_K decode path `ggml_vec_dot_q4_K_q8_K_vl128` block-dot (routing-audit P3: no VLEN128 q4_K repack; quants.c:1771 includes active inc, :1774 calls the ABI shim). The number = m1/mf2 ratio.

### Arm build + GATES (all PASSED before perf)
- mf2 build: `cp mf2.inc → active.inc; touch; cmake --build build --target ggml-cpu -j4` (BOARD ggml-cpu library, NOT tcrv-opt/compiler) → built target ggml-cpu OK (shadow warnings only).
- **md5 distinctness**: q4k_m1.so = `304d1128...`, q4k_mf2.so = `9192b923...` → DISTINCT (proves build changed the binary, not a stale-object no-op).
- **source-inc gate**: m1 active inc 12× `i32m4|vslidedown` (md5 8da97c54) vs mf2 active inc 0× `i32m4|vslidedown` + 10× i32m2 (md5 b22e3fc0).
- **scoped objdump discriminator** (exact symbol `tcrv_emitc_q4k_kernel` via `nm -S` addr+size):
  - m1 kernel (size 0x512): **4× `e32,m4|vslidedown`** (3 e32,m4 + 1 vslidedown), disasm-md5 `dd099500`.
  - mf2 kernel (size 0x786): **0× `e32,m4|vslidedown`**, disasm-md5 `6b86cb64` (DISTINCT body — kernel genuinely recompiled).
- **coherence (m1)**: engagement print "TCRV Q4_K BLOCK-DOT ENGAGED (LMUL=m1 compiler-emitted)" FIRED during an 8B forward pass (R-state, ran to harness-timeout, no crash). Note: literal "Paris" not asserted — DeepSeek-R1-Distill is a reasoning model (`<think>` preamble) and this llama-cli build rejects `-no-cnv` (conversation mode); engagement-fired + forward-progress is the load-bearing coherence signal. mf2 coherence rides body byte-identity to oracle-verified ours-mf2.cpp (bit-exact 2012/0 both VLENs) + shared ABI shim, per plan; the objdump discriminator stands in (mf2 inc engagement print "LMUL=mf2 compiler-emitted" present at line 420).

### A/B perf — **HALTED / NOT ADOPTED AS A RESULT** (user directive, 2026-06-30)

**STATUS: q4_K memory-wall judge HALTED.** Mid-session the user (relayed via coordinator) directed: performance testing is **premature** — it is the wrap-up / 收尾 step, to be done AFTER the implementation is mature, and a lot of implementation is still undone. Additional reasons the q4_K perf is **not cleanly comparable / not adopted**:
- **Model inconsistency**: the only 8B Q4_K_M present on rvv is `DeepSeek-R1-Distill-Llama-8B` — inconsistent with the project's historical **7B/3B** baselines, so these ratios are not cleanly comparable.
- **Board-identity unverified** (per directive): treat the board provenance as unconfirmed for the purpose of adopting these numbers.
- **Mixed-quant dilution**: DeepSeek-8B-Q4_K_M carries q6_K/q5_K tensors with no m1/mf2 knob → any ratio is magnitude-capped.

The 3-round A/B happened to complete before the stop reached the board. Per directive the numbers are **NOT presented as a result** and **NO headline ratio / memory-wall verdict is drawn**. Recorded below **for provenance only — DO NOT CITE, DO NOT fold into the table.** Gate validation (md5/source-inc/objdump discriminator/engagement-print, above) all passed; engagement fired per arm with matching LMUL. Raw logs: `rvv:/home/ubuntu/q4k-retest-scratch/ab_{m1,mf2}_r{1,2,3}.{out,err}`, `ab_all.log`.

> PROVENANCE-ONLY (NOT A RESULT — premature/收尾, 8B-DeepSeek≠7B/3B baseline, board-identity unverified):
> | round | arm | pp256 | tg64 | engaged |
> |---|---|---|---|---|
> | 1 | m1 | 7.02 | 2.03 | 15× LMUL=m1 |
> | 1 | mf2 | 4.66 | 1.79 | 1× LMUL=mf2 |
> | 2 | m1 | 7.01 | 2.01 | 7× LMUL=m1 |
> | 2 | mf2 | 4.66 | 1.65 | 1× LMUL=mf2 |
> | 3 | m1 | 6.96 | 1.87 | 4× LMUL=m1 |
> | 3 | mf2 | 4.63 | 1.64 | 4× LMUL=mf2 |
>
> A2 (-t1 decode) = NOT RUN (board-crash precedent + halt). No verdict claimed.

---

## RESTORE — DONE, board pristine

Restored after the q4_K halt:
1. `tar xzf tree-snapshot.tgz` → 4 tracked M files + 3 untracked .inc back to Phase-0 state.
2. `rm` the extra `tcrv_q4k_swap_mf2.inc` I scp-added (not in the original tree).
3. `cp q4k_m1.so → build/bin/libggml-cpu.so.0.15.1` (live `.so` restored from the mf2-last-arm state back to m1).
4. `touch` active.inc (=m1) so any future rebuild regenerates m1, not the stale mf2 `.o` cache.

**MD5-asserts (all PASS)**: live `.so.0` == `304d1128b0c4c58ad6093fb8abe2015e` (m1); active.inc == `8da97c54` (m1); emitted gemm.inc == `4e21a79c`; gemv.inc present. `git status --porcelain` == original dirty state (M quants.c / arch/riscv/repack.cpp / ggml-cpu.c / repack.cpp + untracked tcrv_emitted_repack_{gemm,gemv}.inc + tcrv_q4k_swap_active.inc + ggml-cpu.c.orig); no stray mf2 inc; q4_0 `TCRV-WINB-ON-TOGGLE` preserved at repack.cpp:4592. No processes of others killed.

> Build-cache note: `build/*.o` for ggml-cpu reflect the mf2 compile, but the SOURCE tree + live `.so` are both restored to m1; the `touch` on active.inc ensures a future `cmake --build` recompiles to m1. Scratch dir `/home/ubuntu/q4k-retest-scratch/` (m1/mf2 `.so` backups + raw logs + tree-snapshot.tgz) left in place for provenance.
