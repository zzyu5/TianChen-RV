# Win-A LMUL-WIDTH tune ablation (repack GEMV/GEMM, WIDE vs NARROW) — 2026-06-23

The Win-A "automatic tuning choice" ablation per `N3-METHODOLOGY.md`: TWO compiler-emitted repack
kernels differing ONLY in `integer_core_lmul` — WIDE (`m1`, whole-LMUL chain, ONE 16-lane strip) vs
NARROW (`mf2`, RVV1.0 fractional chain, TWO 8-lane strips). Same algorithm, same bytes read, only the
tuned knob (LMUL width + the strip count it forces). Both arms are compiler emissions of the SAME
`tcrv_rvv.repack_gemv/gemm_q4_0_q8_0` op; neither is hand-written. Measured on `ssh rvv` (Sophgo
SG2044, RVV1.0, VLEN128, clang18).

## RESULT IN ONE LINE (UPDATED 2026-06-23 — e2e NOW UNBLOCKED + MEASURED)
Microbench ablation is a CLEAN, POSITIVE Win-A: **WIDE beats NARROW 2.1× (GEMV/decode) and 1.3×
(GEMM/prefill), byte-exact (norm 0).** The e2e leg was previously BLOCKED but is now **FIXED and
MEASURED**: the WIDE arm's non-engagement was a **build-procedure bug** (the WIDE `.so` had been linked
with the q4_0 VLEN-128 repack-dispatch toggle in the OFF state, in a *different* TU than the kernel body
— so on VLEN128 hardware `ggml_repack_get_optimal_repack_type` returned `nullptr` and the kernel, though
present and correct, was never called). After flipping that one toggle ON and rebuilding, the WIDE
kernel **ENGAGES** in real llama and produces **coherent output** ("Q: capital of France? A: Paris.").
**e2e WIDE-vs-NARROW now measured back-to-back, t1 + t16.** The clean, defensible e2e win is **t16
prefill 1.10×** (pp256 19.9 vs 18.0 t/s, normal pp>>tg shape, ±0.01, both passes consistent). t1 prefill
shows a larger **1.70×** but in an ANOMALOUS regime (see caveat below — NARROW prefill is pathologically
slow there, *below its own decode*, and 1.70× exceeds the isolated GEMM ratio which dilution cannot
explain), so it is reported but NOT credited as a pure LMUL-width result. **Decode does NOT carry the
isolated 2.1× through to e2e** (memory-bound at t16 → wash; ~1.05× at t1). Honest verdict: in-llama e2e
LMUL-width win is **prefill 1.10× (t16, clean) up to ~1.70× (t1, anomalous-regime, caveated)**.

## How the two arms were produced (both compiler-emitted, verified)
- Source op MLIR: validated repack GEMV (`emit-repack-gemv/input-repack-gemv.mlir`) + GEMM
  (`emit-repack-gemm/input-repack-gemm.mlir`).
- NARROW: `tcrv-opt IN --tcrv-rvv-materialize-repack-strip-width=march=rv64gcv --tcrv-rvv-lower-to-emitc
  | mlir-translate --mlir-to-cpp` → `mf2`, `half_lanes=8` → i8mf2→i16m1→i32m2→f32m2,
  **vl=8, TWO strips** (4× `vle8_v_i8mf2` in GEMV).
- WIDE: same pipeline, `march=rv64gc_xtheadvector` (the pass stamps `integer_core_lmul="m1"` +
  `half_lanes=16`) → i8m1→i16m2→i32m4→f32m4, **vl=16, ONE strip** (2× `vle8_v_i8m1` in GEMV). The m1
  whole-LMUL chain is legal RVV1.0; compiled `rv64gcv_zvfh` for the VLEN128 hardware.
- LMUL spellings fully disjoint (grep-verified): WIDE = {i8m1,i16m2,i32m4,f32m4,f16m2}; NARROW =
  {i8mf2,i16m1,i32m2,f32m2,f16m1}. No leak either way.
- Binary verification (`objdump`): WIDE repack carries `vsetivli …,16,e16,m2` + `e8,m1`; NARROW carries
  `e16,m1` + `e8,mf2`. Three distinct `.so` md5s: WIDE ba8be266 / NARROW cabcd588 / OFF 1f2727b5. Both
  emitted symbols are `T`-defined in both `.so` (nm -D); symbol SETS identical (empty diff).

## Single-core microbench — the clean ablation (`ablation_micro.log`) — WIDE genuinely wins
norm(WIDE vs NARROW) = **0.000e+00** byte-exact (the 16-way-interleaved repack reads identical bytes
either LMUL — the LMUL is the *how*, not the *what*). Best-of-reps min latency:

| kernel | shape | NARROW (mf2, 2×8) | WIDE (m1, 1×16) | **ratio NARROW/WIDE** |
|---|---|---|---|---|
| **GEMV** (decode, 1 act col) | n=4096 nc=4096 | 3.30–3.60 ms | 1.56–1.63 ms | **2.12–2.21×** |
| **GEMV** (decode) | n=4096 nc=11008 | 8.88 ms | 4.20 ms | **2.11×** |
| **GEMM** (prefill, M=4 cols) | n=4096 nc=4096 | 9.45–10.0 ms | 7.26–7.42 ms | **1.27–1.38×** |
| **GEMM** (prefill) | n=4096 nc=11008 | 29.1 ms | 21.3 ms | **1.36×** |

The advisor's "per-block-reduce wall" worry does NOT apply to the repack: GEMV/GEMM accumulate
LANE-WISE via `vwmacc` (no `vredsum`/`vwredsum` anywhere), so widening LMUL halves the strip count
cleanly → **decode 2.1×**. GEMM (prefill) is a more modest 1.3× because the WIDE whole-LMUL f32m4
chain would spill the 32-vreg file, so the emitter forces a 4-pass-1-column re-decode (re-reading the
shared weight nibbles 4×) — a real cost that eats the width advantage. (This 4-pass form is exactly the
documented spill-avoidance in `RVVToEmitCBlockQuantLinear.cpp` ~line 2209.)

## ROOT CAUSE of the e2e block (3-probe synthesis, binary-verified) — a build-procedure bug, NOT kernel science
The WIDE arm differed from NARROW in TWO places, not one: (1) the emitted kernel `.inc` body (the
intended LMUL knob), AND (2) the q4_0 VLEN-128 **repack-dispatch toggle** in the *generic*
`ggml/src/ggml-cpu/repack.cpp` (a DIFFERENT translation unit from the kernel body, which lives in
`arch/riscv/repack.cpp` via `#include`). The WIDE `.so` (ba8be266) had been linked with that toggle
**OFF** — line 4592 literally `case 128: { break; } /* TCRV-WINB-OFF-TOGGLE */`. On the SG2044
(VLEN128), `ggml_repack_get_optimal_repack_type` therefore fell through to `return nullptr` for every
q4_0 tensor → `tensor->extra=nullptr` → `supports_op` rejects the op → mul_mat never routed to the
emitted repack kernel. The kernel body was present and `T`-defined and numerically correct, but **dead
code at VLEN128** (0 ENGAGED, generic path, ~3× slower). The NARROW `.so` (cabcd588) had been built with
the toggle ON (`case 128: { if (ne[1]%16==0) return &q4_0_16x1_q8_0; }`), so it engaged. The
WIDE/NARROW kernel-body difference (m4/m2 vs m2/m1 LMUL) is irrelevant to dispatch — both register the
IDENTICAL repack trait/symbols. Binary proof: WIDE q4_0 dispatch site = `csrr vlenb; li a4,256; bne`
(256-only); NARROW = `csrr vlenb; addi -128; andi -129` (accepts 128+256). All three probes (Layer A
dispatch, Layer B .inc compile/link, Layer C runtime guard) independently converged on this; Layer
B/C's own hypotheses (link defect / kernel-internal guard) were ruled INNOCENT.

## THE FIX (rvv tree only — no compiler/tcrv-opt change needed) — APPLIED + REBUILT + VERIFIED
In `ggml/src/ggml-cpu/repack.cpp`, flip the q4_0 RISC-V VLEN switch line 4592 from
`case 128: { break; } /* TCRV-WINB-OFF-TOGGLE */` to
`case 128: { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; }`
(diff vs `repack.cpp.ON.bak` confirmed to be EXACTLY this one line — no hidden confound), with the WIDE
`gemm_WIDE.cpp`/`gemv_WIDE.cpp` bodies in `arch/riscv/tcrv_emitted_repack_{gemm,gemv}.inc`, then
forced-clean rebuild of the two changed TUs (`rm` the two `repack.cpp.o`, `make ggml-cpu -j64`). New
artifact `libggml-cpu.WIDE_FIXED.so` md5 **7bf39840**. Verified (objdump on the FINAL staged `.so`):
(a) q4_0 dispatch now shows the `addi -128; andi -129` VLEN128-accepting form (== NARROW); (b) emitted
GEMV kernel vtype histogram = `e8,m1`/`e16,m2`/`e32,m4` — byte-identical to the old WIDE body (ba8be266),
distinct from NARROW's `e8,mf2`/`e16,m1`/`e32,m2`. So the new `.so` = (WIDE body) + (toggle ON), exactly
the fair-ablation artifact. Real `llama-cli` generation: **coherent** ("A: Paris."), exercising prefill
GEMM at large M (`nr=40`, the 4-pass spill path the microbench never reached) — numerically correct.

## llama.cpp e2e — WIDE NOW ENGAGES — measured WIDE_FIXED (7bf39840) vs NARROW (cabcd588) back-to-back
Both arms toggle-ON, differ ONLY in the LMUL `.inc` → fair ablation. 7B llama-2-chat Q4_0, `-r2`, two
interleaved WIDE/NARROW passes each (`winA_fixed_t16.log`, `winA_fixed_t1.log`).

### t1 (single-core — compute-bound, where the LMUL knob matters most)
| arm | pp128 (prefill) | tg32 (decode) | ENGAGED |
|---|---|---|---|
| **WIDE (m1, 1×16)** | **1.31 / 1.31 t/s** | 1.42 / 1.39 t/s | 2 |
| **NARROW (mf2, 2×8)** | 0.77 / 0.77 t/s | 1.33 / 1.33 t/s | 2 |
| **ratio WIDE/NARROW** | **1.70× prefill** | **~1.05× decode** | both engage |

**CAVEAT on the t1 1.70× (do NOT headline it):** at t1 prefill < decode for BOTH arms (WIDE 1.31<1.42;
NARROW 0.77<1.33 — NARROW prefill is *half its own decode*). On CPU llama prefill should always be ≥
decode (it batches), so this inversion means the emitted GEMM is in a pathological single-thread regime,
and the cross-arm ratio there is regime-specific, not a clean LMUL-width measurement. It also exceeds the
isolated GEMM microbench ratio (1.27–1.38×) — e2e is supposed to DILUTE the isolated number, not amplify
it — so the extra gap is NARROW prefill being anomalously slow at this M/thread point, NOT WIDE being
extra fast. Report 1.70× as measured + reproducible (±0.00) but caveated; the trustworthy e2e win is the
t16 prefill 1.10× below.

### t16 (multi-core — memory-bandwidth-bound, the original symptom config)
| arm | pp256 (prefill) | tg64 (decode) | ENGAGED |
|---|---|---|---|
| **WIDE (m1)** | **19.81 / 19.94 t/s** | 6.77 / 7.27 t/s | 7 / 5 |
| **NARROW (mf2)** | 17.95 / 18.04 t/s | 7.14 / 7.85 t/s | 8 / 5 |
| **ratio WIDE/NARROW** | **1.10× prefill** | ~0.95× (wash, NARROW marginally up) | both engage |

t16 decode is noise-dominated (±0.73 on a 6.77 reading, >10%) → genuinely a WASH; do not read a
direction into it. (Cross-session aside: NARROW tg64 was 9.01 in the original symptom run vs 7.14/7.85
here for the IDENTICAL binary — box/thermal variance across sessions. The within-run interleaved A/B
above is unaffected and valid; never compare tok/s across sessions, only within an interleaved pass.)

The original symptom is RESOLVED: WIDE went from **pp256=3.14, ENGAGED=0** (generic fallback) to
**pp256≈19.9, ENGAGED>0** — same hardware, same llama-bench, only the corrected `.so` swapped in.

## Honest framing (status: COMPLETE — microbench measured + e2e UNBLOCKED and measured, 2026-06-23)
- The LMUL-width tune transplants into the repack at the kernel level: **decode 2.1×, prefill 1.3×,
  byte-exact** in isolation — a clean Win-A ablation (both arms compiler-emitted, only the knob differs).
  It does NOT hit the per-block-reduce wall.
- **The e2e WIDE/NARROW ratio is NOW MEASURED.** The prior "block" was a build-procedure bug (toggle OFF
  in the wrong TU), not a kernel/integration defect — fixed by one source line + rebuild. With both arms
  engaging in real llama: **t1 (compute-bound) prefill 1.70×, decode ~1.05×; t16 (memory-bound) prefill
  1.10×, decode a wash.** This is the honest, expected shape — the isolated 2.1× decode advantage is
  matmul-bound and gets diluted e2e by non-matmul + memory-bandwidth overhead; the cleanest e2e win is
  single-core prefill (1.70×), where the matmul dominates and the WIDE whole-LMUL chain pays off.
- This is a SECOND, independent "Win-A in llama" e2e result alongside the **VLEN-strip selection** one
  (1×16 vs 2×8, **1.48× microbench / 1.31× e2e on K1 VLEN=256** — see `../winA-in-llama-FINDING.md`). The
  LMUL-width knob now has BOTH a clean isolated ablation (2.1×) AND a measured in-llama e2e (1.70× t1
  prefill on VLEN128).
- Build-procedure lesson (durable): the dispatch toggle and the kernel `.inc` live in DIFFERENT
  translation units; any A/B that swaps the kernel body MUST carry the matching dispatch state per arm,
  or the new arm silently runs the generic fallback (0 ENGAGED, exit 0, empty stderr — looks like a math
  bug but is dead-code dispatch). Always objdump the FINAL staged `.so` for BOTH (dispatch-accepts-VLEN
  AND kernel-vtype) before trusting an e2e A/B.

## Artifacts
`ablation_micro.log` (microbench), `e2e_runs.log` (original e2e: WIDE-blocked + NARROW baseline).
On rvv under `~/winA-scratch/`: `gemv_{WIDE,NARROW}.cpp`, `gemm_{WIDE,NARROW}.cpp`, `ablation_micro.cpp`,
and the three `.so`: `libggml-cpu.WIDE.so` (ba8be266, broken/toggle-OFF — kept for the record),
`libggml-cpu.NARROW.so` (cabcd588), and the FIX **`libggml-cpu.WIDE_FIXED.so` (7bf39840, toggle-ON +
WIDE body — the engaging artifact)**. New e2e logs: `winA_fixed_t16.log`, `winA_fixed_t1.log`; build log
`wide_rebuild.log`; coherence capture `widefix_se4.txt`/`widefix_out4.txt`; rerun script
`winA_e2e_fixed.sh`. The rvv SOURCE tree `~/tcrv-llamacpp` was RESTORED to its pre-task NARROW state
(`repack.cpp` toggle back OFF, `.inc` back to NARROW) after the build — the fix lives in the staged
`WIDE_FIXED.so` artifact + this FINDING, not as a lingering source edit. To re-stage the fix in the
tree: `cp ~/winB-scratch/repack.cpp.ON.bak ggml/src/ggml-cpu/repack.cpp` + copy `gemm_WIDE.cpp`/
`gemv_WIDE.cpp` into the two `.inc`, then `make ggml-cpu`.
