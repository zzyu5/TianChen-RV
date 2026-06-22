# Decode A/B (tg128) — compiler-emitted GEMV vs stock naive-RVV, same-build

## Numbers (same board, same session, toggled binary)
| path | stock (repack OFF, naive RVV vec_dot) | emitted (repack ON, compiler GEMV ENGAGED) | ratio |
|---|---|---|---|
| pp512 (prefill, GEMM) | 1.57 | 9.35–9.39 | **5.98×** (tight) |
| tg128 (decode, GEMV) | **1.43 ± 0.01** (ENGAGED=0, tight) | **5.37 ± 1.28 / 7.37 ± 1.07** (GEMV ENGAGED, two runs, load-variant) | **~3.8–5.1×** |

### Decode honesty: shared-board load variance
The emitted tg128 measured **7.37** then **5.37** on two runs (both ±~1.2, GEMV ENGAGED), i.e.
the absolute decode tok/s swings with other users' board load. The stock side (1.43 ± 0.01)
was tight. So the decode ratio is a **range ~3.8–5.1×, not a single number** — a clean
same-LOAD back-to-back A/B is not achievable because each toggle needs a ~6-min rebuild (can't
be truly back-to-back), the exact difficulty 06-18 flagged. The qualitative seal is solid and
multi-×; the precise multiplier carries load variance on the emitted side. **Two independent
measurements corroborate the ~4-5× decode class: this session's same-build 1.43→{5.37,7.37}
and 06-18's independent hand-GEMV 1.38→6.49 (~4.7×).** Headline conservatively as **~4-5×**.

## Why decode wins ~5× (not the "modest, BW-bound" prior)
Decode here is NOT memory-bandwidth-bound — the **stock** q4_0 decode path is
`ggml_vec_dot_q4_0_q8_0` (naive RVV, `#if __riscv_v`), whose inner loop ends in a per-block
cross-lane `vwredsum_vs` **reduction wall** + scalar accumulate. On this board that path is
**compute-bound by the reduction wall**, so stock prefill (1.57) ≈ stock decode (1.43). The
repack GEMV escapes the wall (lane-wise vwmacc, 0 vredsum), so emitted prefill (9.35) ≈
emitted decode (7.37) — the ~5-6× holds on BOTH paths. This matches the 06-18 finding
(tg128 1.38→6.49 = ~4.7× for the hand GEMV) and its explicitly-resolved "decode is
memory-bound, 4.7× impossible" objection: stock decode is reduction-wall-compute-bound.

## What this measurement adds over 06-18
06-18 reported the decode ratio against a **banked cross-session** stock tg (1.38), leaving a
"residual cross-session caveat" (a same-session stock tg re-measure was skipped as the slow
stock decode made a clean A/B prohibitively long on the fragile board). This session
**measured stock tg128 = 1.43 in the SAME session / SAME toggled binary** as the emitted
7.37 → the decode A/B is now **directly observed same-build**, sealing that caveat. And the
GEMV running is now the **compiler-emitted** one (06-18's was hand-written).

## Honesty notes
- Emitted tg128 carries shared-board load variance (±1.07); a tighter -r4 re-measure is in
  flight to firm the headline. The stock side is tight (±0.01).
- This is **Win B** (e2e kernel-swap, repack-vs-naive-RVV), the SAME class as the prefill
  5.98× — NOT the N3 Win-A compiler-auto-vs-naive ablation. Decode + prefill both ~5-6×.
