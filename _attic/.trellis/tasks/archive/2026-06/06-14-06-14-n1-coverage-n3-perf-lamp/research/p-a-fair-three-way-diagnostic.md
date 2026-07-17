# P-A: Fair 3-way RVV performance diagnostic (the honest current 性能灯 state)

- **What**: fix the unfair baseline (clang-autovectorized "scalar") and measure a
  controlled 4-column comparison — genuine-scalar / autovec-scalar / naive-RVV /
  tuned-RVV — on the two no-autovectorizable Gearbox low-precision kernels, on real
  `ssh rvv` riscv64 hardware. Measurement only; NO tune change.
- **Date**: 2026-06-14
- **Tool**: `scripts/rvv_fair_three_way_measure.py` (new). Reuses the existing
  generation (`generate_verified_bundle` → sha-identity-contracted tuned bundle) and
  the proven SSH/scp/parse plumbing (`rvv_generated_bundle_abi_e2e`). Does NOT rebuild
  plumbing and does NOT touch the contract-laden gate4 win-claim machinery.
- **Raw evidence**: `artifacts/p-a-fair-three-way/` (git-trackable, NOT tmp):
  `evidence.json`, `three_way_table.md`, per-kernel `remote_run_stdout.txt`,
  `remote_target_profile.txt`, `scalar_objdump.txt`, `rdcycle_probe.json`.

## Headline verdict (one line)

**With FAIR baselines, the 性能灯 is OFF: on both no-autovec kernels the tuned Gearbox
output beats a *genuine* scalar (1.5–4.3×) but does NOT beat a *competent* naive-RVV
loop — packed-i4 tuned/naive ≈ 0.93–1.01 (loses by ~3%), grouped-u2 tuned/naive ≈
0.81–1.13 (loses by ~18% at realistic sizes). The prior "regression vs scalar" was an
autovectorized-baseline artifact; the real gap is "tuned does not yet beat naive RVV."**

## Hardware / method

- Board: real riscv64 (`ssh rvv` → 192.168.8.72 via rvv-jump), Ubuntu clang 18.1.3,
  64 CPUs. Reachable; NOT dry-run / NOT local / NOT QEMU.
- Timer: `clock_gettime(CLOCK_MONOTONIC_RAW)`, warmups=3, best-of-(repeats=9 × iters=16).
- **Fairness control (the key fix)**: all 4 variants are linked into ONE binary and
  timed *interleaved* in the same repeat loop on the same input buffers (board is
  shared / jump-hosted → high variance; cross-run timing would be apples-to-oranges).
- **Correctness before timing**: scalar oracle, all 4 variants, tol 1e-05. PASSED for
  both kernels (`PASS three-way measurement`). A fast-but-wrong kernel is not a win (I8).
- **Genuine-scalar verified, not asserted**: the scalar TU is compiled `-march=rv64gc`
  (no `v` ISA). `objdump -d` of that object ON THE BOARD shows **ZERO vector ops** — the
  disasm is a textbook scalar reduction (`lb`/`lb`/`mul`/`add`, then
  `fcvt.s.w`/`fmul.s`/`fsw`/`ret`; see `*/scalar_objdump.txt`). The detector was
  sanity-checked both ways (empty on scalar, fires on a known-vector disasm).
- **Naive-RVV is a *competent* naive (the fairness control that matters most)**: it is
  the textbook RVV dot-product pattern — `vwmul` then **accumulate element-wise into a
  persistent `vint32m1` vector accumulator (`vwadd.wv`) and reduce (`vredsum`) ONCE
  after the loop**. It deliberately does NOT do a per-iteration cross-lane reduction
  (the latency-bound anti-pattern), and does NOT unroll / group / fuse / region-tune
  (those are the tuned version's job). A strawman naive that reduces every iteration
  made tuned look like a winner; against this competent naive it does not. This is the
  honest bar for "beat naive RVV."
- **rdcycle / rdtime probe**: on THIS board both are U-mode-readable (rdcycle delta ≈
  800k over a 100k spin; rdtime coarse, delta=1 → low-Hz fixed clock, not a cycle
  proxy). Cycles ARE available here but wall-time clock_gettime is the reported timer
  (rdtime too coarse; rdcycle non-portable and adds little over the stable wall-time).

## The 3-way table (real `ssh rvv`, best-per-iter ns; competent accumulate-once naive)

### grouped-u2 (one-element-per-byte, u2-grouped — the auto-selected schedule)

| n | genuine-scalar | autovec-scalar | naive-RVV | tuned-RVV |
|---|---|---|---|---|
| 257   | ~621    | ~244    | ~473    | ~419    |
| 4096  | ~9485   | ~2830   | ~4776   | ~5821   |
| 65536 | ~151806 | ~45478  | ~75888  | ~93399  |

| n | tuned/scalar | tuned/naive | naive/scalar | autovec/scalar |
|---|---|---|---|---|
| 257   | 1.48 | **1.13** | 1.32 | 2.55 |
| 4096  | 1.63 | **0.82** | 1.99 | 3.35 |
| 65536 | 1.63 | **0.81** | 2.00 | 3.34 |

**Verdict: `partial`.** Tuned beats genuine-scalar (~1.5–1.6×) but **LOSES to naive-RVV
at n≥4096 (~0.81×, ~18–19% slower than a basic accumulate-once strip-loop)**. The
Gearbox's grouped (u2) emission — which the selector picks as the DEFAULT — is *worse*
than an untuned competent loop on the exact kernel it auto-selects. Only at n=257 does
it edge naive (1.13×).

### packed-i4 (two-signed-i4-per-byte, u1-unpack-required — high-nibble vwmacc, single-reduce)

| n | genuine-scalar | autovec-scalar | naive-RVV | tuned-RVV |
|---|---|---|---|---|
| 257   | ~3561   | ~956    | ~983    | ~1058   |
| 4096  | ~56348  | ~13868  | ~13129  | ~12996  |
| 65536 | ~850198 | ~207963 | ~198076 | ~204425 |

| n | tuned/scalar | tuned/naive | naive/scalar | autovec/scalar |
|---|---|---|---|---|
| 257   | 3.37 | **0.93** | 3.63 | 3.72 |
| 4096  | 4.34 | **1.01** | 4.29 | 4.06 |
| 65536 | 4.16 | **0.97** | 4.29 | 4.09 |

**Verdict: `partial`.** Tuned beats genuine-scalar (~3.4–4.3×) but does **NOT** beat the
competent naive-RVV: tuned/naive ≈ 0.93–1.01 (median ~0.97), i.e. roughly parity, ~3%
behind. **The apparent "win" in the first measurement was reduction placement, not
tuning** — the tuned packed-i4 kernel reduces once (`...scalar-epilogue-single-reduce`)
and the strawman naive reduced every iteration; once the naive also reduces once, the
edge disappears. Honest result: the tuned packed-i4 is competitive with, but not faster
than, a competent naive.

## Why the prior recorded number was a "regression" (the fairness fix in one column)

The `autovec/scalar` column is the same scalar C source compiled `-march=rv64gcv`
(one flag) — clang **autovectorizes it 2.5–4.1×**. The old gate4 harness timed
tuned-RVV against *that* autovectorized binary and called the resulting 0.76–0.80× a
"regression." It was comparing the compiler's tuned RVV against clang's own
autovectorized RVV, not against a true scalar. So the prior "regression vs scalar" was
a baseline artifact. The *real* gap, revealed by fair baselines, is different and more
useful: **tuned beats genuine scalar comfortably, but is at best parity with a competent
naive RVV loop.** That is the gap P-B must close.

## Precise verdict for the 性能灯 sequencing (does P-B need real tune work? — YES)

- **The 灯 is NOT achievable with the current emission + fair baselines alone.** Fair
  baselines moved the number from "regression vs autovec-scalar" to "wins vs genuine
  scalar," but the spec N3 bar is "实测赢 scalar **且赢 naive RVV**," and the
  beat-naive-RVV half is **unmet on both kernels**:
  - packed-i4: ~parity (0.93–1.01×) — the nearest, but still not a win.
  - grouped-u2: a real loss (0.81×) at realistic sizes — and it is the auto-selected
    default, so the selector is picking the weaker schedule.
- **P-B is genuine tune/emission work, not measurement or attribute plumbing.** To lit
  the 灯 the generated body must get *faster than a competent naive*, which needs:
  - a real cost model replacing the max-unroll tiebreak (today it defaults to grouped,
    which loses to naive — see `gearbox-current-state.md`);
  - resource-fact-derived candidates (today hardcoded constants) so the search can find
    a schedule that beats naive (e.g. multi-accumulator unroll to hide vwmul/vwmacc
    latency, which the single-accumulator naive does not do);
  - packed-i4 is the closest-to-win starting point (already ~parity, already single-
    reduce); grouped-u2 is the clearest deficiency (loses to a trivial loop).
- **The sharpest P-B lead — fusion is NOT where the win is.** The naive here is
  deliberately *conservative*: `vwmul` + `vwadd.wv` + single `vredsum`, with **no
  `vwmacc` fusion and no multiple accumulators**. The tuned packed-i4 *does* use
  `vwmacc` fusion + single-reduce (its own realization label says
  `...high-nibble-vwmacc-scalar-epilogue-single-reduce...`) and still only TIES this
  non-fused naive. So (a) the result is not a strawman — a more natural naive using the
  single-instruction `vwmacc` would be *stronger*, making tuned lose by more, not less;
  and (b) the diagnostic for P-B is concrete: **the vwmacc fusion buys ~nothing
  measurable here.** The win will not come from fusion; it has to come from
  latency-hiding (multiple accumulators / unroll) that neither the single-accumulator
  naive nor the current tuned body does — and note grouped-u2 *attempts* unroll=2 yet
  still loses, so a real cost model (not max-unroll) is the lever.
- **What P-A delivered**: (1) the fairness fix (genuine `rv64gc` scalar, verified
  zero-vector on the board); (2) the missing *competent* naive-RVV baseline; (3) a
  controlled 4-column 3-way table on real hardware, reproducible; (4) the honest
  diagnostic — the 灯 is OFF vs naive RVV on both kernels, so the headline novelty (a
  measured win) requires real P-B tune work, and packed-i4 is where it is closest.

## Caveats

- Reproducibility: ran 4 independent `ssh rvv` invocations across the session; ratios
  are tight (packed-i4 tuned/naive stayed 0.93–1.01, grouped tuned/naive 0.81–1.13).
  One run hit a transient SSH drop (shared/jump-hosted board) and exited non-zero with
  no numbers — re-running succeeded; no fabricated numbers.
- naive-RVV is hand-written in this tool (not compiler-emitted) but is the competent
  accumulate-once pattern, compiles clean on the board, and is correctness-guarded
  against the scalar oracle.
- Only the dequant op family was measured (the two selected variants of
  `widening_product_reduce_dequantize_f32`). The dequant-clamp packed-i4 input MLIRs
  exist under `artifacts/gate4-packed-i4-...-clamp-ssh/` and could be enrolled cheaply.
- Absolute ns drift run-to-run with board load; the interleaved-in-one-binary design
  makes the *ratios* (the verdict-bearing quantity) stable despite that drift.
- Strength (not a caveat): the correctness guard runs before *every* timed run and
  validates all 4 variants against the scalar oracle (incl. the `vwadd.wv`-tail-agnostic
  accumulate-once naive). It passed all sizes both kernels, so the timed computation is
  the correct one — a tail/agnostic bug would fail loudly, not silently skew a number.
