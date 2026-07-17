# K1 (Spacemit X60, VLEN=256, RVV1.0) — Win-A cross-profile data

**2026-06-22.** Ran the SAME compiler-emitted wide-deferred i16 dot-reduce lamp harness
(the Win-A artifact) on K1 via gcc13 -march=rv64gcv. Self-contained, exits clean. Raw:
`k1-lamp-3way-vlen256-stdout.txt`. (wide-deferred body is VLEN-robust — same emitted C
adapts via vsetvl; no re-emission needed to RUN, but selection is identical → see caveat.)

## Cross-profile table (same compiler-emitted kernel, two real chips)
| n | rvv VLEN=128 wide_vs_scalar | K1 VLEN=256 wide_vs_scalar | rvv periter_vs_scalar | K1 periter_vs_scalar | K1 wide_vs_narrow |
|---|---|---|---|---|---|
| 256 | 5.53 | **9.42** | 0.69 | **1.98** | 2.38 |
| 1024 | 6.87 | **13.61** | 0.68 | **1.99** | 3.25 |
| 4096 | 7.52 | **15.32** | 0.67 | **2.00** | 3.61 |
| 65536 | 4.02 | **8.90** | 0.68 | **2.05** | 2.09 |

## Two findings
1. **N3 — the compiler-automatic tune wins on BOTH real chips.** wide-deferred beats genuine
   scalar 8.4–15.3× on K1 (vs 4.0–7.5× on rvv) and competent naive RVV 1.8–3.6× on K1 (vs
   2.1–3.8× on rvv). The resource-aware max-LMUL realization measurably wins across two
   distinct real RVV profiles — not a single-chip artifact.
2. **N1 — the per-iter-reduce baseline INVERTS across capability vectors.** The same
   per-iteration-vredsum kernel runs at **0.65× scalar on rvv (VLEN=128)** but **~2.0× scalar
   on K1 (VLEN=256)**. Same kernel, opposite verdict-vs-scalar on two chips — a concrete
   capability-divergence data point (the cross-lane reduce's relative cost depends on
   VLEN/microarch). (Also: K1's scalar core is weaker — scalar n=257 1305ns vs rvv 807ns —
   which amplifies wide_vs_scalar; the vs-naive number controls for that and stays ~2-3.6×.)

## Honest caveat (this is a RATIO divergence, not yet a SELECTION divergence)
The wide-deferred body is VLEN-robust: the SAME emitted C ran on both chips (vsetvl adapts
AVL at runtime). So the gearbox SELECTION (which rung) was identical on 128 and 256 — what
diverged is the measured ratios. A genuine N1 *selection* divergence (same kernel → DIFFERENT
emitted code per profile) needs the VLEN-aware strip decision (the reframed PRIZE: repack
strip 2×8 @128 vs 1×16 @256) or the per-profile measured>static block-dot tuner. This K1 run
is the cross-profile *measurement* leg; the selection-divergence leg is the next build.
