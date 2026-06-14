# P-B step 1: accumulator × LMUL sweep — the winning config (N3 性能灯 GO)

- **What**: ran the already-written `scripts/rvv_accumulator_sweep_measure.py` on real
  `ssh rvv` riscv64 (VLEN=128, Ubuntu clang 18.1.3) over the int8-dot low-precision
  contraction (`byte`) and the packed-i4 contraction (`packed-i4`). Sweep grid:
  A ∈ {1,2,4,8} independent i32 accumulators × LMUL rungs {mf4, mf2, m1, m2} ×
  n ∈ {257(prime tail), 256, 1024, 4096, 16384, 65536}. Each variant correctness-guarded
  (== genuine scalar, tol 1e-05) before timing; all variants interleaved in one binary;
  warmup + best-of-(11×16) `rdcycle` (primary) + `clock_gettime`.
- **Date**: 2026-06-15. **Board**: real riscv64 192.168.8.72 via `ssh rvv` (NOT dry-run / NOT
  local / NOT QEMU). genuine-scalar TU compiled `-march=rv64gc`; `objdump` on the board shows
  **ZERO vector ops** (`scalar_is_genuinely_scalar: True`) → the scalar bar is verified.
- **Raw evidence** (durable, git-trackable):
  `artifacts/p-b1-accumulator-sweep/` (byte) and `artifacts/p-b1-packed-i4/` (packed-i4):
  `evidence.json`, `sweep_table.md`, per-kernel `remote_run_stdout.txt`,
  `remote_target_profile.txt`, `scalar_objdump.txt`, the variant `.c` sources incl. the
  winning `var_v_m2_a1.c`.

## Headline verdict — read all THREE bars (don't conflate them)

The fastest clean config is `L=m2_A=1` (single i32m8 accumulator, i8m2 strip-mine, one
`vredsum`). Its three honest comparisons:

1. **vs genuine (non-autovectorized) scalar — a CLEAN WIN. The scalar 性能灯 is ON.** byte
   **4.52–4.90×**, packed-i4 **12.3–15.9×**, every n, both kernels, real `ssh rvv`.
2. **vs the mf4 naive anchor — 3.3–4.9×, but that number is the LMUL-WIDTH effect, NOT a
   tuning win.** The anchor uses i8mf4 = 4 bytes/strip on a 128-bit VLEN board → it is
   *under-vectorized* (the P-A confound). Beating it is the easy bar.
3. **vs a competent *wide-LMUL* naive — PARITY, because the winner IS that naive.** `L=m2_A=1`
   is algorithmically identical to the anchor (single accumulator, `vwmul`+`vwadd.wv` in-loop,
   one `vredsum`) — only at m2 instead of mf4. A naive at the same max-legal LMUL would match it
   (for packed-i4 it does; for the compute-light byte kernel a clean wide-naive would slightly
   *beat* it — see scaffold footnote). **There is NO latency-hiding / accumulator tuning headroom
   on this board.**

**→ P-B step 2 is GO, but for the RIGHT objective: resource-aware LMUL selection that beats the
current broken Gearbox default, NOT an accumulator/latency-hiding win.** P-A showed the Gearbox's
auto-selected grouped-u2 schedule *loses* to even the mf4 naive (≈0.81×). The defensible,
paper-grade claim is: **"the Gearbox today emits a losing schedule; budget-derived max-legal-LMUL
selection makes it ~4× faster than the mf4-naive and fixes that regression."** The claim is NOT
"tuning beats a competent naive via multiple accumulators" — this sweep disproves that headroom.

## The win, attributed to the knob (the question the task asked)

**The lever is LMUL WIDTH, not multiple accumulators — and "wide-LMUL" is just the naive done
right.** Across every LMUL rung, A>1 is never faster than A=1 at the same rung
(`beats_best_a1 = False` for all A>1; the per-iteration `vwadd.wv` accumulate chain is NOT the
bottleneck on this board → the multiple-accumulator knob the tool was built to test buys nothing).
The speedup is monotone in LMUL width and saturates at the widest rung that fits the vreg budget:

| config (A=1) | i32 acc LMUL | vregs/acc | byte vs-naive (median) | packed-i4 vs-naive (median) |
|---|---|---|---|---|
| L=mf4_A=1 (the anchor itself) | m1 | 1 | ~0.80 (under-vectorized) | ~1.02 |
| L=mf2_A=1 | m2 | 2 | 1.54 | 1.59 |
| L=m1_A=1  | m4 | 4 | 2.97 | 3.78 |
| **L=m2_A=1 (winner)** | **m8** | **8** | **4.07** | **4.34** |

So "beat the mf4 naive" is the *easy* bar (mf4 is under-vectorized — 4 i8 bytes/strip on a
128-bit VLEN board, the P-A confound). The honest, discriminating result is: **wider LMUL is the
entire win; the multiple-accumulator knob is a red herring on this microarchitecture.** A>1
costs vregs (A·acc_regs) for no latency-hiding benefit, so the resource-optimal config is the
*single* accumulator at the *widest LMUL the 32-vreg file allows*.

**Scaffold-overhead footnote (measurement fairness):** the tool's own `L=mf4_A=1` reproduces the
anchor *algorithm* but uses a `while(rem>0){off+=vl; rem=...}` strip loop vs the anchor's tighter
`for(i; i<n; i+=vl)`. For the compute-light **byte** kernel this scaffold costs ~25% at every n
(tool-mf4_A=1 ≈ 1.25× the anchor cycles), so a *clean* wide-LMUL naive (anchor scaffold at m2)
would slightly BEAT the tool's `L=m2_A=1` — i.e. the tuned-vs-competent-wide-naive position is
parity-to-slightly-negative, not positive. For the compute-heavy **packed-i4** kernel the scaffold
is negligible (~2%, sometimes faster), so there it is genuine parity. Either way this reinforces
bar (3): there is no win to be had over a competent *wide* naive — the value is selecting the wide
LMUL in the first place (which the current Gearbox default fails to do).

## The winning config as resource facts (the P-B step 2 target — resource-AWARE, not magic)

The Gearbox enumeration in step 2 should derive the accumulator LMUL from the vreg budget, NOT
hardcode it:

- The widening chain couples LMUL: `i8 mLoad → i16 (2×) → i32 acc (4×)`. One i32 accumulator at
  acc-LMUL m8 occupies **8 vregs**; the live product temp (i16 m4) is **4 vregs**; loads/temps
  ≈ the `reserve` budget (6). Total `8 + 6 = 14 ≤ 32` → **fits**; the next rung up (i8 m4 → i32
  m16) does not exist (LMUL caps at 8), so **m2/m8 is the widest legal rung** → it is the
  resource-derived choice, not a constant.
- Resource fact for the enumerator: **pick the largest acc-LMUL whose `acc_regs + product_regs +
  reserve ≤ 32`, with A=1**, because on this board the accumulate chain latency is hidden by the
  vector length alone (A>1 buys nothing). The budget prune the tool already models is the same
  fact that *rejects* the wasteful combos: `L=m1_A=8`, `L=m2_A=4`, `L=m2_A=8` all exceed 32 vregs
  and are pruned — and crucially `L=m2_A=2` (16 vregs, fits) is measurably WORSE or jittery vs
  `L=m2_A=1` (8 vregs), confirming the extra accumulators are pure waste here.
- **Therefore the Gearbox candidate space P-B step 2 enumerates is**: `{acc-LMUL ∈ rungs : fits
  32-vreg budget}` × `{A=1}` (A>1 pruned by the measured no-benefit fact), and the cost model
  selects max-legal-LMUL. This replaces the current hardcoded-constant / inert-prune / max-unroll
  placeholder (see `gearbox-current-state.md`) with a budget-derived rule that has hardware
  backing.

## Per-n behaviour (n-dependence)

`L=m2_A=1` wins at **every** n on both kernels (including the prime-tail n=257 and the largest
n=65536) — not n-gated. The kernel is compute/throughput-bound at these sizes (the win grows
with n then plateaus, i.e. NOT memory-bandwidth-saturated at 64KB on this board — wider LMUL
still helps, so it is not yet bandwidth-limited). One outlier cell: `L=m2_A=2` at n=65536 spiked
(101033 cyc vs m2_A=1's 57397) — board jitter on a single interleaved cell, not a property of
the config; the m2_A=1 column is stable across the size range. (Shared/jump-hosted board; the
interleaved-one-binary design keeps the verdict-bearing *ratios* stable despite absolute drift.)

## Tool fixes made (the deliverable)

- **Headline-winner ranking** (`analyze().rank_key`): the original ranked winners by *smallest*
  config (fewer accs, then smaller LMUL), so it announced `L=mf2_A=1` (a 1.5× winner) as the
  headline while its own data showed `L=m2_A=1` winning by ~4×. Changed the primary sort key to
  **fastest median-vs-naive** (ties → fewer accumulators → smaller LMUL, the resource-fact
  preference). The headline winner is now the genuinely fastest clean config on both kernels. The
  `beats_best_a1` attribution field is untouched and still reported, so the LMUL-vs-accumulator
  story stays visible. No measurement code touched; re-derived the persisted
  `evidence.json`/`sweep_table.md` from the saved raw `remote_run_stdout.txt` (the primary
  evidence), no re-measurement / no fabricated numbers.
- No other fixes needed: the tool compiled clean on the board first try, `rdcycle` did not SIGILL
  (U-mode readable, as P-A found), the correctness guard passed for every variant at every n
  (incl. the i8m2→i32m8 widening tail at the prime n=257), and the genuine-scalar objdump proof
  fired correctly.

## What P-B step 2 builds (the corrected objective)

Build **resource-aware max-legal-LMUL selection** into the Gearbox: enumerate
`{acc-LMUL ∈ rungs : acc_regs + product_regs + reserve ≤ 32}` with A=1, and a cost model that
selects the widest legal rung (`L=m2_A=1` = i8m2→i32m8, single accumulator). The win to claim is
**over the current Gearbox default**: P-A showed the auto-selected grouped-u2 schedule LOSES to
even the mf4 naive (≈0.81×); budget-derived LMUL selection makes the emitted body ~4× faster than
mf4-naive and ~4.5–15× faster than genuine scalar, fixing that regression. Do NOT claim a
multiple-accumulator / latency-hiding win — this sweep shows that headroom does not exist on this
board (A>1 never beats A=1). The winning C source (the emission target) is
`artifacts/p-b1-accumulator-sweep/byte/var_v_m2_a1.c` and
`artifacts/p-b1-packed-i4/packed-i4/var_v_m2_a1.c`.
