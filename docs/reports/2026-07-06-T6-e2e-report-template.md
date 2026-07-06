# T6 — End-to-End Phase-Split Report (TEMPLATE)

> Report template for the **T6 end-to-end (llama-bench) phase-split** table
> (the `beat`-if-any axis). This is a TEMPLATE: fill new rows following the
> certification gate below. Perf **magnitudes** are quotable only from SEALED
> cells in the body; OPEN / PARK / STALE cells are quarantined to the appendix
> and MUST NOT appear in the body / paper正文.
>
> A/B injection is a clean symmetric two-tree diff: tree A (tcrv-llamacpp emits
> repack) vs tree B (upstream stock block-dot), SAME compiler/flags, the only
> delta being the surgical patch. Correctness = greedy-token A==B (GREEN; NOT a
> bit-exact-vs-ggml claim).
>
> Companion data cells (auto-generated, line-E visibility pack):
> `experiments/visibility/T0-sixstate.md`, `T7-burndown.md`,
> `T2-ledger-anchor.md`. Raw T6 rows: `experiments/T6_e2e_phase_split.csv`;
> sealed cell: `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/`.

## Certification legend (the hard gate)

| status | meaning | may appear in 正文/body? |
|---|---|---|
| **sealed** | preflight 4/4 (march-complete · libcall-free · same-compiler · board-VLEN==target), DVFS locked, CI + between-pass floor, greedy-token GREEN | **YES** |
| **open**   | measured, stable, but a mechanism/attribution question is live | NO — appendix only |
| **park**   | blocked on a parked問題包 item (ledger P1/P3: board-recovery, same-session paired) | NO — appendix only |
| **stale**  | superseded by a clean remeasure, or fingerprint changed | NO — appendix only |

**Rule:** a perf number enters the body ONLY when its cell is `sealed`.
Everything else lives under `## Appendix (quarantine — 禁入正文)`.

**[PERF-1] eight gates:** a phase cell being `sealed` on a single board is NOT a
"Win" claim. The full Win-B needs the remaining gates (dual-board, micro↔e2e
Amdahl transduction, M-shape sweep, PMU sampled attribution, …). Report sealed
single-board cells as sealed, and do NOT upgrade wording to "Win" until the gate
count is met.

**ns / tok/s absolute values are cross-machine INCOMPARABLE** — always carry the
board identity; only same-board same-session A/B ratios are comparable.

---

## Board identity (fill per run)

| field | value |
|---|---|
| board id     | (e.g. rvv-openeuler-vlen128 / gcc-15.2.0 FULLMARCH) |
| VLEN         | (128 / 256) |
| model        | (e.g. tinyllama-q4_0.gguf, sha) |
| march (A==B) | (full-cap; both trees identical) |
| freq / DVFS  | (fixed kHz; span %) |
| correctness  | greedy-token A==B (GREEN / RED) |

---

## BODY — SEALED cells only

### q4_0 @ VLEN128 (rvv single board, tinyllama-q4_0, full-cap march)

| phase | kernel path | ours tok/s | stock tok/s | ratio [CI95] | verdict | mechanism |
|---|---|---:|---:|---|---|---|
| **prefill** (pp128, GEMM) | our path-selection routes q4_0 → repack @VLEN128; stock stays block-dot (repack VLEN-gated) | 19.819 | 3.903 | **5.077× [5.039, 5.111]** | DIFFERENCE | routing-into-capability-gap + memory locality (the tiled GEMM compute is byte-identical between trees; the win is ROUTING, not codegen) |
| **decode** (tg32, GEVM) | our emitted repack GEVM vs stock block-dot | 3.08–3.13 | ~2.00 | **1.50–1.56×** (honest range, 3 pressure snapshots) | DIFFERENCE | repack-vs-block-dot memory behavior; range widens under shared-board load (light 1.564× → heavy 1.50×) |

> Mechanism honesty: the sealed prefill 5.077× is a **path-selection / routing**
> win (our compiler engages repack where stock cannot), NOT a "faster repack
> kernel" claim. The prefill tiled GEMM object is byte-identical across trees.

### q4_0 @ VLEN256 (k1 dual-board, VLEN-flip confirmation)

| phase | ours tok/s | stock tok/s | ratio [CI95] | verdict | reading |
|---|---:|---:|---|---|---|
| **prefill** (pp128, GEMM) | 24.773 | 24.667 | **1.004× [1.002, 1.006]** | **PARITY** | confirms the VLEN128 5× is **VLEN-capability-keyed, not universal**: on VLEN256 ggml's own repack GEMM engages (GEMM object byte-identical), so no gap to route into. |

> The dual-board pair (VLEN128 5.077× DIFFERENCE + VLEN256 1.004× PARITY) is
> itself the mechanism evidence: the win is keyed on the VLEN128 capability gap
> where stock's repack is VLEN-gated off.

### TODO (new sealed rows)

| phase | board | ratio [CI] | status |
|---|---|---|---|
| _q8_0 T6 first batch / prefill M-shape sweep_ | | | sealed |

---

## Appendix (quarantine — 禁入正文)

> Everything below is OPEN / PARK / STALE. Magnitudes here are **not citable** in
> the body or the paper. Kept for provenance and to schedule closure.

### OPEN / PARK

| item | number | why not in body |
|---|---|---|
| q4_0 **decode** @ VLEN256 (k1) | **0.857× [0.849, 0.865]** DIFFERENCE (our emitted GEVM ~14% SLOWER than ggml-native repack) | **PARK P1**: stable paired reversal (not measurement noise), needs adversarial verify — is the emitted VLEN256 GEVM scaffold under-optimized vs ggml hand-tuned native? vs early hand-written repack? Source: `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/`. |
| "generated repack delivers 5× prefill" (constructed-GEMM attribution) | 5.045× (interim) | **PARK P3**: the constructed GEMM was deployed on-board (objdump-fingerprinted as ours) but the clean **paired** 5× remeasure (both sides PASSES=2, same session) was interrupted by board-recovery; only ours-pass1 landed = cross-session confound. The declaration+construction+deploy are real; the paired 5×句 is pending clean remeasure. |
| decode kernel-attributable delta (VLEN128) | — | shared-board bandwidth drifts cross-session (byte-identical prefill drifted 5.045→5.530); needs same-session paired A/B to attribute a decode delta. |

### STALE

| item | number | why stale |
|---|---|---|
| q4_0 prefill (historical) | **5.68× (STALE)** | pre-clean-remeasure figure on the older board/protocol; superseded by the sealed **5.077× [5.039, 5.111]** clean-remeasure. Do not cite 5.68×. |
| any `ssh rvv-old` e2e numbers | — | measured on the retired board; new `ssh rvv` is a different board → cross-board incomparable → stale. |

### Gates remaining to a "Win-B" wording (not yet met)

1. dual-board sealed (VLEN128 DIFFERENCE + VLEN256 PARITY) — prefill **done**; decode P1 open.
2. micro↔e2e Amdahl transduction accounting (gate ④).
3. prefill M-shape sweep + q8_0 T6 first batch.
4. PMU sampled attribution upgrade (RISC-V PMU currently paranoid=2 → 0 samples).
5. clean paired constructed-GEMM 5× remeasure (P3).
