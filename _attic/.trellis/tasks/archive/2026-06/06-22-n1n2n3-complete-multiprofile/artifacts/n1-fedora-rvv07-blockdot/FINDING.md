# N1 Fedora RVV0.7 block-dot SELECTION divergence — MEASURED on C920 (2026-06-23)

## Status: MEASURED (real C920 tuner number)

The one missing N1 profile is filled. The q8_0 block-dot selection is now measured on
**all three real profiles**: rvv (VLEN128/RVV1.0), K1 (VLEN256/RVV1.0), and now
**C920 (VLEN128/RVV0.7.1)** — a second ISA generation.

## What was asked vs what was done
The task asked whether a q8_0/q4_0 BLOCK-DOT can be emitted for RVV0.7 at its legal LMUL
candidates and tuned on the C920, to see if the winner diverges from rvv (m2@128) and
K1 (m1@256). **Answer: YES — it emits, compiles, runs byte-exact, and was tuned. The
winner is m2/2/elided (m2 family), which diverges from K1 and matches rvv.**

## Method (identical to the rvv/K1 q8_0 paired runs — `tune_block_dot.py` model)
- `tcrv-opt --dump-candidates` (capability-march `rv64gcv`) = single source of truth for
  the q8_0 legal candidate set: **12 candidates** = 3×mf4 (fractional) + 3×m1 + 6×m2 (whole).
- **RVV0.7 legal subset** = the **9 whole-LMUL candidates** (m1×3, m2×6). The 3 mf4
  candidates are illegal on RVV0.7 (no fractional LMUL) and were pruned — exactly the
  hardware-fact prune. All 9 whole-LMUL candidates were emitted by `tcrv-opt`
  (`--tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`), **zero fractional symbols**.
- Vocabulary across all 9: `i8m1/i8m2 → i16m2/i16m4 (vwmul/vwmacc) → i32m1 (vwredsum reduce)`.
  All whole-LMUL; the widening reduce collapses each strip to a scalar `int32`, so coreLmul
  is free (the property that makes block-dot RVV0.7-runnable, per the prior block-dot proof).
- Shipped 3 TUs (harness/candidates/ggml_ref) to the C920, **byte-identical after scp**
  (sha256 verified). Compiled with XuanTie/PLCT g++ 14.1.1 `-march=rv64gc_xtheadvector
  -mabi=lp64d -ffp-contract=fast` → **RC=0, empty stderr** (zero fractional-LMUL rejection).
- FILTER-then-RANK: gate byte-exact vs ggml's real q8_0 kernel (separate TU) over 9 sizes
  × 200 trials → **9/9 EXACT**; then time best-of-200 min, `taskset -c 3`, n=4096.

## Result (verbatim, two back-to-back runs)
| shape | RUN1 ns | RUN2 ns | family |
|---|---|---|---|
| **m2/2/elided** | **8189.3** | **8145.5** | **m2 ← WINNER** |
| m2/4/elided | 8221.5 | 8195.4 | m2 |
| m2/4/robust | 8434.7 | 8403.4 | m2 |
| m2/2/robust | 8578.2 | 8535.2 | m2 |
| m2/1/elided | 8754.3 | 8713.4 | m2 |
| m2/1/robust | 8885.9 | 8843.7 | m2 |
| m1/4/robust | 9701.4 | 9669.3 | m1 ← best m1 (losing family) |
| m1/2/robust | 9785.6 | 9752.3 | m1 |
| m1/1/robust | 9993.4 | 9964.7 | m1 |

- Winner **m2/2/elided**, reproduces to <0.6% across two runs, identical full ordering.
- **m2 sweeps the top 6 slots; m2 beats best-m1 by +18.5% / +18.7%** — decisive, much larger
  than the ~7% rvv/K1 flip margins.
- objdump (winner symbol, XuanTie objdump): genuine `th.v*` 0.7.1 — `th.vsetvli` ×23,
  `th.vwmul.vv` ×4, `th.vwredsum.vs` ×4, `th.vmv.x.s` ×4, `th.vle.v` ×12, etc.
  **C1 fractional vtype (eN,mf*) = 0 · C2 whole-register vmvNr.v (1.0-only) = 0 ·
  C3 standard-V leak = 0** → real RVV0.7 silicon vector execution, not scalar/1.0.

## Cross-profile q8_0 block-dot winner table (all three profiles measured)
| chip | VLEN | ISA-gen | winner | family |
|---|---|---|---|---|
| rvv (SG2044) | 128 | RVV1.0 | m2/4/elided | **m2** |
| **C920 (SG2042)** | **128** | **RVV0.7** | **m2/2/elided** | **m2** ← this probe |
| K1 (X60) | 256 | RVV1.0 | m1/2/robust | **m1** |

**FINDING: the q8_0 block-dot selection tracks VLEN, not ISA generation.** The C920 — a
**second ISA generation (RVV0.7)** — lands on the **m2 family, same as the same-VLEN
RVV1.0 chip (rvv)**, and diverges from K1 (m1 @ VLEN256). The N1 capability axis that
moves the selection is VLEN (128→m2, 256→m1); the ISA-generation axis (1.0→0.7) does NOT
move the q8_0 block-dot winner. This is the **first block-dot selection number measured on
real RVV0.7 silicon**, completing the previously-UNMEASURED Fedora profile.

A static cost-model pick (m2/2/elided, cost 1050) is family-correct here AND on rvv, but is
wrong-family on K1 (m1) — so the per-capability board measurement remains load-bearing: the
same static argmin is right on two profiles and wrong on the third.

## Relation to the prior "DEAD-END" conclusion (no contradiction)
The prior `fedora-rvv07/FINDING.md` proved an **RVV0.7-vs-RVV1.0 selection FLIP caused by
*pruning fractional LMUL* is structurally impossible** (every winner is a whole multiplier,
so removing the absent fractional LMUL cannot promote a winner). **This probe asks a
different question** — the cross-silicon **board RANKING of the whole-LMUL candidates** (the
direct analog of the q8_0 rvv-vs-K1 paired run). That ranking is genuinely VLEN-sensitive
(m2@128 vs m1@256, both already measured on RVV1.0 silicon). The C920 number shows the
m2-family@VLEN128 selection **holds across the ISA-generation boundary** — a measured fact,
not a fabricated flip. Both statements are true and consistent.

## Artifacts (NO git commit)
- `probe.log` — full checkpoint log (method, raw ladders, objdump, verdict).
- `build/harness.cpp`, `build/candidates.cpp` (9 whole-LMUL TUs), `build/ggml_ref.cpp`.
- `build/board_run.txt` (RUN1 full), `build/board_run2.txt` (RUN2 RESULT lines).
- C920 scratch `/tmp/tcrv-rvv07-q8-blockdot-sel` (cleanup optional; left for re-run).
- Toolchain: `~/xuantie/RuyiSDK-...-plctxthead-.../bin/riscv64-plctxthead-linux-gnu-g++`
  14.1.1, `-march=rv64gc_xtheadvector`, native riscv64 host (compiles + runs on the C920).
