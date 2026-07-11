# T6 batch-k1 — whole-model e2e transduction (k1 board · IME 3 + q4_K vl16)

> Board casefile for **T6 batch-k1** (agent acbabb2d-line). Scope = verify whether
> the kernel-axis candidates on k1 — **IME 3 格** (q4_0/q8_0/q4_K @ime, batched
> leaf ~2.09× compute-account, `[GAP-IME-LEAF-PIPELINE]` closed) and **q4_K vl16**
> (Win-K1-VLEN, q4_K's only full-eight-gate green micro cell, 1.085× vs hand-brick)
> — **transduce to whole-model phased e2e**.
>
> **HEAD unchanged at `d078fba3`** (no git in this line; main session commits).
> Board dir isolated; zero stock-tree modification; read-only against existing builds.
> **[NG-4] methodology framing only** — honest transduction accounting; no false-green.

## ★★★ HEADLINE SCOUT RESULT — `[GAP-IME-E2E-INTEGRATION]` CONFIRMED (high-value negative)

**The tcrv IME GEMM is NOT wired into any llama.cpp forward path.** As of HEAD
`d078fba3`, the tcrv IME matmul_tile kernels (`tcrv.ime.q4_0_matmul_tile` /
`q8_0` / `q4_K` + `tcrv_ime_vmadot_mac_kloop` batched leaf) are **standalone
certified constructs** (host int32 oracle + k1 silicon `vmadot 0xe210312b`
int32 0-diff, g4-m1b-reseal-batched, ~2.09× compute-account) — they are consumed
**only** by standalone test harnesses (`test/Target/IME/*.c`) and lit
materialization tests (`test/Conversion/EmitC/ime-*.mlir`). **No tcrv→llama IME
bridge exists.** Therefore there is **no tcrv IME e2e to measure** — the IME 3
cells stay kernel-axis-only yellow.

### Scout evidence (code + board, verified against current tree, not memory)
| probe | result | meaning |
|---|---|---|
| `grep -rl 'q4_0_matmul_tile\|vmadot_mac_kloop' lib` | only `lib/Plugin/IME/IME*.cpp` (emitter) | tcrv IME op has an emitter |
| consumers of the emitted IME kernel | only `test/Target/IME/*.c` + `test/Conversion/EmitC/ime-*.mlir` | standalone seal/lit only — **no forward path** |
| `grep -riE 'ggml.*forward\|mul_mat' lib/Plugin/IME lib/Dialect/IME` | 0 (only ggml **block-layout** comments) | IME code knows ggml's byte layout, is **not wired into** ggml forward |
| board `grep -rl 'q4_0_matmul_tile\|vmadot_mac_kloop' ~/tcrv-k1-llama/ggml/src` | **0 hits** | tcrv IME kernel **absent** from the board's llama.cpp ggml |
| board tcrv-deployed kernels in ggml | `tcrv_emitted_repack_gemv/gemm.inc` (RVV q4_0 repack, VLEN256/128) | the ONLY tcrv kernel wired into forward is **RVV q4_0 repack** — not IME, not q4_K |
| board `build-ime` libggml-cpu | 32 `vmadot` | vendor ggml-spacemit IME backend (`ime1::gemm_kernel_i8i4`), **not tcrv** |
| board `build-off` libggml-cpu | 0 `vmadot` | RVV fallback |

**Contrast that sharpens the gap:** tcrv's RVV **q4_0 repack** GEVM/GEMM *is*
deployed into the board's ggml forward (`ggml_gemv_q4_0_16x1_q8_0` →
`tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel`, "TCRV EMITTED GEVM … ENGAGED").
So the project *has* a tcrv→forward integration mechanism — it was simply never
built for the IME matrix path (or the q4_K vl16 lane-width path). The absence is
an **integration gap**, not a fundamental barrier.

## Board fingerprint
| field | value |
|---|---|
| board | k1 (SpacemiT X60), VLEN=256, 8 harts, `ime` on harts 0–3, `taskset -c 0-3` |
| toolchain (deployed llama) | vendor build — `build-ime` = clang-18 `-fno-integrated-as` `GGML_CPU_RISCV64_SPACEMIT=ON`; `build-off` = same flags, `=OFF` |
| model | `~/tcrv-k1-llama/models/tinyllama-q4_0.gguf` (637 MB, tinyllama-1B Q4_0) — **only Q4_0 gguf on board; no q4_K / q8_0 model, no llama-quantize** |
| ledger | SYSTEM (absolute tok/s) — vendor backend, tcrv not in loop |
| opponent-identity | **N/A for tcrv** (tcrv IME not in forward). Reference measured = vendor SPACEMIT ON vs same-toolchain RVV OFF |
| eight-gate | **NOT invoked** (no tcrv kernel under test in forward) |

## IME transduction reference (VENDOR backend — NOT tcrv · best-case ceiling)
Runner: `tools/e2e-harness/board/t6_k1_ime_vendor_toggle.sh`. This is the vendor
ggml-spacemit IME backend (whose own `ime1::gemm_kernel_i8i4` IS wired into
forward) — the strongest available proxy for "if a wired IME GEMM existed, would
~2× compute transduce e2e?". **This is context for the GAP, explicitly NOT a
tcrv perf-covered result.**

Fresh run 2026-07-11 (`vendor_toggle_run_raw.txt` / `vendor_ime_toggle.csv`),
tinyllama-1B-Q4_0, `taskset -c 0-3`, r=3, threads=4:

| regime | OFF (RVV fallback) | ON (vendor SPACEMIT IME) | ON/OFF | excess over decode floor |
|---|--:|--:|--:|--:|
| pp256 (prefill) | 24.90 ± 0.02 | 40.06 ± 2.93 | **1.61×** | +0.15 |
| pp512 (prefill) | 24.26 ± 0.04 | 36.18 ± 0.47 | **1.49×** | +0.03 |
| **tg64 (decode, M=1, CONTROL — IME can't run)** | 5.66 ± 0.00 | 8.26 ± 0.02 | **1.46×** | — (floor) |

Reproduces prior 2026-06-25 clean toggle (pp256 1.62× / decode 1.47×) within noise.
The decode falsifier (the load-bearing point) is tight (±0.00/±0.02); ON pp256 ±2.93
(7.3%) is the known vendor-arm variance. `vmadot` engagement re-verified live:
build-ime lib = 32 vmadot, build-off = 0.

**Reading (fresh + prior 2026-06-25 `IME-E2E-SPACEMIT-TOGGLE-FINDING`):** the
prior clean toggle found pp256 1.62× BUT the **decode control (tg, M=1) = 1.47×**
— i.e. ~1.47× of the "IME" speedup survives in the M=1 GEVM regime where the IME
`vmadot` array physically cannot run ⇒ it is the **SpacemiT RVV kernel-family
swap** (227 SpacemiT symbols), NOT the IME matrix unit. The IME-unit-specific
increment (~+0.18) is confounded with GEMM-vs-GEVM kernel-family differences and
**cannot be cleanly isolated**. Even the vendor's fully-wired IME GEMM yields **no
cleanly-isolated IME-unit e2e win** on this memory-bound 1B model ⇒ the tcrv IME
~2.09× compute-account would **not** transduce e2e here even if it were wired.
Consistent with [[kernel-wins-dont-transplant-to-e2e]] (IME 5.51× kernel → 0.86×
decode e2e; micro↛e2e law).

## q4_K vl16 — whole-model e2e: `[GAP-KQUANT-E2E-INTEGRATION]` (integration + model gap)
- The q4_K vl16 lever (Win-K1-VLEN, 1.085× vs hand-brick) is a **micro / kernel-axis**
  construct. It is **not deployed** into the board ggml forward: the only tcrv
  kernel wired into forward is RVV **q4_0** repack; there is **no q4_K repack path**
  (per [[q4-0-e2e-is-routing-not-kernel]]: K-quant `case128 = //TODO`, no repack chain).
- There is **no q4_K gguf on the board and no `llama-quantize` binary** to make one.
- ⇒ whole-model phased q4_K e2e on k1 is **not runnable** with the current board
  state without (a) deploying the vl16 q4_K kernel into ggml forward and (b)
  provisioning a q4_K model. Recorded as a named gap, not fabricated.

## Transduction accounting (four-column · honest · per ROADMAP T6 row)
| cell | kernel-axis (compute-account) | deployed in forward? | e2e phased (tcrv) | classification |
|---|---|---|---|---|
| q4_0@ime | ~2.09× (M≥4) / 1.33× (M=1), SELF, silicon-sealed | **NO** (standalone cert only) | **N/A — not wired** | kernel-axis-only · `[GAP-IME-E2E-INTEGRATION]` |
| q8_0@ime | ~2.09× (M≥4), SELF, silicon-sealed | **NO** | **N/A — not wired** | kernel-axis-only · `[GAP-IME-E2E-INTEGRATION]` |
| q4_K@ime | ~2.09× (M≥4), SELF, silicon-sealed | **NO** | **N/A — not wired** | kernel-axis-only · `[GAP-IME-E2E-INTEGRATION]` |
| q4_K vl16 | 1.085× vs hand-brick (Win-K1-VLEN, full 8-gate micro) | **NO** (RVV q4_0 repack only; no q4_K path) | **N/A — not wired + no model** | kernel-axis-only · `[GAP-KQUANT-E2E-INTEGRATION]` |
| _reference:_ vendor IME | (vendor `ime1::gemm_kernel_i8i4`) | vendor: YES | _see table above; unisolable, decode-floor kernel-family_ | NOT tcrv — ceiling proxy |

## perf-covered contribution of this batch: **0 cells turned green** (as expected · honest)
Neither the IME 3 cells nor q4_K vl16 can be measured e2e because the tcrv kernels
are not wired into forward. No cell flips from kernel-axis-candidate (yellow) to
perf-covered (green). The **honest, high-value** deliverable is the named gap
`[GAP-IME-E2E-INTEGRATION]` (+ `[GAP-KQUANT-E2E-INTEGRATION]`): structural cert
(m4_class=certified) + kernel-axis ~2× are real achievements; e2e integration is
a disclosed boundary, not a failure. The perf-covered pipeline for these cells
routes through **building the tcrv-IME→ggml-forward bridge** (analogous to the
existing RVV q4_0 repack deployment), then re-measuring — a future battle, not
this board line.
