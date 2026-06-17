# WS-B — ssh rvv roofline of the ggml block-dot kernels

**Status**: measured on real `ssh rvv` hardware (the measurement agent built the harness +
ran it; this synthesis is written from its raw logs under
`../artifacts/wsB-roofline/`). Pinned to HEAD `ef7179e7`.

## Host facts (measured, not assumed)
- Board: the project `ssh rvv` target. **VLEN = 128 bits** (probe: `vsetvlmax_e8m1(AVL=32)=16`
  ⇒ VLMAX_e8m1 = 16 ⇒ VLEN = 16×8 = 128; `vsetvlmax_e8m2(AVL=64)=32`). Consistent with
  rv64gcv / Zvl128b.
- Frequency is DVFS/`ondemand`-governed and drifted between runs (sustained 1.49–2.40 GHz
  measured *during* the hot kernel). **All conclusions below use cyc/MAC, which is
  frequency-independent**; GMAC/s figures scale with the (variable) clock and are reported
  only for context.
- Two independent runs (`kernel_timer_run1.log`, `kernel_timer_run2.log`,
  `micro_ceiling_run.log` RUN1/RUN2) agree on cyc/MAC to <1% (cache) / <2% (streaming).

## Roofline anchors (microbenchmarked, `micro_ceiling.c`)
- **Compute ceiling** — the emitted byte dot core (i8→i32 `vwmul`+`vwadd.wv`), tight loop:
  - m2 anchor: **3.86 MAC/cyc** (mac_m2_a1=3.861, a2=3.855 — 1 vs 2 accumulators identical:
    the microbench's vwadd.wv chain already saturates the unit).
  - m1 anchor: 3.04 (a1) → 3.25 (a4) MAC/cyc.
  - ⇒ peak compute ≈ **3.86 MAC/cyc** at the m2 anchor.
- **Memory-bandwidth ceiling** — streaming `vle8m8` over 256 MB (≫ LLC): **2.3–2.5 B/cyc**
  (4.6–5.6 GB/s at the measured clocks).

## Per-kernel placement (cyc/MAC, % of the relevant ceiling)

| kernel    | cache cyc/MAC | MAC/cyc | % of 3.86 compute ceiling | streaming B/cyc(wt) | % of 2.4 mem ceiling |
|-----------|---------------|---------|---------------------------|---------------------|----------------------|
| q8_0_q8_0 | 3.69          | 0.271   | **7%**                    | 0.201               | ~8%                  |
| q4_0_q8_0 | 6.58          | 0.152   | **4%**                    | 0.083               | ~3%                  |
| q4_K_q8_K | 1.98          | 0.505   | **13%**                   | 0.137               | ~6%                  |

(streaming weights-from-DRAM cyc/MAC: q8_0 5.29, q4_0 6.78, q4_K 4.11 — q8_0/q4_K degrade
vs cache, q4_0 barely moves; even streaming, B/cyc(wt) is ~10× below the memory ceiling.)

## Conclusion: the kernels are LATENCY/OVERHEAD-bound, not roofline-bound
Every kernel runs at **4–13% of the compute ceiling AND ~3–8% of the memory ceiling** —
an order of magnitude below *both* roofline walls. The bottleneck is therefore neither
vector-ALU throughput nor load bandwidth; it is **per-element latency / overhead**:
- the per-block reduction (`vwredsum`) → scalar → fp32 accumulate dependency chain,
- per-block scalar address arithmetic + fp16 scale reads,
- nibble/super-block decode serialized in front of the multiply (worst for q4_0: 4% — the
  decode tax; q4_K's super-block amortizes it best at 13%),
- `vsetvl` per strip.

The m2_a1-vs-m2_a2 ceiling tie shows extra accumulators don't help a *saturated* unit — but
the real kernels are 14× below saturation, so they are starved on the latency of the
reduction/scale chain, exactly what independent accumulators / software pipelining hide.

## Knob recommendation (feeds WS-B's tune knob + validates the WS-C interface)
Add **one** microarch lever as a **tune knob** (a `NamedKnob`, NOT a new emit branch),
with the most headroom where the gap to the compute ceiling is largest:
- **`accumulator_count` / software-pipeline depth across blocks**: keep N independent
  per-block partial accumulators in flight so the `vwredsum`→scalar→fp32 latency of block
  *i* is hidden by the integer core of block *i+1*. Expected to lift q8_0 (7%→) and q4_0
  (4%→) most; q4_K already amortizes best.
- This is **just another `NamedKnob`** in the Tier-0 `GenericScheduleCandidate{cost,isLegal,
  knobs[]}` space — it needs **no new method** on `TunableScheduleOpInterface`. ⇒ WS-C
  Tier-1's interface *shape* is independent of WS-B (confirmed). WS-B's role is to be the
  **second consumer** that validates the enumerate→prune→**measure**→stamp *flow* on a new
  axis, and to supply the measured win (authority order: measured > static).
- **Honesty gate**: the knob only counts as N3 evidence once it MEASURABLY beats the
  current schedule on `ssh rvv` (byte-exact first, then ranked). Headroom ≠ win; the win
  must be measured. Not yet done — this roofline only establishes that the headroom EXISTS
  and which lever addresses it.

## Artifacts
`../artifacts/wsB-roofline/`: `probe_host.c` (VLEN/freq), `micro_ceiling.c` (+run log,
the two roofline anchors), `kernel_{q8_0_q8_0,q4_0_q8_0,q4_k_q8_k}.cpp` (the measured
kernels), `kernel_timer.cpp` (+run1/run2 logs, cache & streaming regimes).
