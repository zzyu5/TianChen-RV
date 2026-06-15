# P-B7 Stage 2a — the 灯 is ON for the COMPILER-EMITTED 2nd-family body

This is the N3 evidence step that distinguishes "a win is possible" (STEP-1 hand
variant) from "the compiler's OUTPUT wins" (this). Mirrors the byte kernel's P-B4
("灯 ON on emitted body") vs P-B3 ("compiler can emit"). Under I8, the emitted
body's SPEED must be measured, not inferred from an equivalent hand variant.

## What was timed (I8: real ssh rvv, real compiler output)

- **wide-deferred column = the compiler-emitted body** (`emitted_body.c`),
  produced by `tcrv-opt rvv-to-emitc-widening-dot-reduce-wide-lmul.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`. NOT hand-written.
  PROVENANCE (verified): the saved `compiler_emitted_deferred_wide_dot_reduce.cpp.txt`
  is BYTE-IDENTICAL to a fresh regen from the current rebuilt compiler; the
  `.c` TU (`emitted_body.c`) differs ONLY by stripped `//` comments + the dropped
  `extern "C"` wrapper — its 14-intrinsic code sequence is identical to the
  compiler output (`diff` clean). So the timed body IS the compiler's output.
  NOTE on what "compiler-emitted" means here: the INPUT to the conversion is a
  HAND-AUTHORED deferred-wide typed body; what is proven is typed-body -> C ->
  win (conversion + lowering). The kernel -> body SELECTION (the autotuner) is
  the Stage-2b wall, NOT exercised in this measure.
- per-iter-reduce = the compiler's CURRENT dot-reduce emission (the regression
  default the tune must beat). narrow-deferred = the competent narrow naive.
  genuine-scalar = `-march=rv64gc`, objdump-verified ZERO vector ops on the board.
- All four linked into ONE binary, timed interleaved, best-of-11×16,
  correctness-guarded (== scalar oracle, EXACT integer) before any timing.
- Board: real riscv64 `ssh rvv`, clang 18.1.3. Raw: `pb7_lamp_3way_ssh_rvv_stdout.txt`.

## The 灯 (emitted body; best_per_iter_ns ratios, >1.0 = emitted is faster)

| n | emitted vs genuine-scalar | emitted vs per-iter (compiler default) | emitted vs narrow-deferred (competent naive) |
|---|---|---|---|
| 257 | 4.26 | 6.63 | 3.06 |
| 256 | 5.64 | 8.24 | 3.28 |
| 1024 | 6.88 | 10.17 | 3.59 |
| 4096 | 7.52 | 11.14 | 3.78 |
| 16384 | 4.37 | 6.24 | 2.37 |
| 65536 | 3.87 | 5.71 | 2.22 |

**→ The 灯 is ON for the 2nd-family emitted body.** The compiler's actual output
beats genuine scalar 3.87–7.52×, beats its OWN current dot-reduce emission
(per-iter-reduce) 5.71–11.14× (fixing a 0.64–0.70× regression), and beats the
competent narrow naive 2.22–3.78×, at every n. The emitted body matches/slightly
beats the STEP-1 hand variant (the emitted `for(i+=vlmax)` scaffold is tighter
than the hand `while(rem>0)` — the P-B1 scaffold footnote runs in our favor here),
confirming no emission-quality loss.

## The two-lever attribution (honest "why", same as STEP1_CEILING.md)

- The compiler's current dot-reduce emission (per-iter-reduce) is itself a
  REGRESSION vs scalar (0.64–0.70×): a `vredsum` on the critical path every
  iteration (latency-bound). This is a DIFFERENT pathology than the byte kernel's
  under-LMUL grouped-u2.
- Lever 1 (deferred single-reduce): narrow-deferred beats per-iter ~3× — hoists
  the reduce out of the loop. Lever 2 (max-legal LMUL): wide beats narrow ~2.2–3.8×
  — the LMUL-width effect. Headline win = vs the compiler's current emission
  (5.7–11.1×) = the product of both levers.
- Vs a competent *wide* naive: parity by construction (the emitted body IS that
  naive). The N3 value is SELECTING the deferred + wide schedule the compiler does
  not currently emit.

## 灯 status precisely (mirrors P-B4 vs P-B5)

- **灯 ON for the emitted 2nd-family body** — DONE (this file). The compiler emits a
  body that is ssh-rvv numeric-correct (`ssh_rvv_numeric_result.txt`,
  DEFERRED_WIDE_DOT_REDUCE_ALL_PASS) AND wins the 3-way.
- **Selector-driven end-to-end** (kernel -> selector -> realization owner produces
  the deferred-wide dot-reduce body automatically) — see PB7_VERDICT for the 2b
  status (the realization-owner wall).

"若干 kernel" status (graded, not flat): autotuner-driven e2e win = byte ONLY
(P-B5/6); emitted-body measured win (selection is the wall) = i16 dot-reduce
(this); hand ceiling only = u8. STRICT N3 "若干" (autotuner wins on several
kernels) is NOT yet met (byte alone); "a measured win exists on several kernels"
IS met. The byte family already proves the selector mechanism end-to-end.
