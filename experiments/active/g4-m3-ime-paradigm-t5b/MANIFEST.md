# T5b — G4-M3 IME paradigm-lever measurement (q4_0@ime tracer cell)

> Casefile for the **测量总攻第一战 (G4-M3)** T5b skeleton: the paradigm-lever
> (matrix-range vmadot vs vector-range RVV) × layout × M-sweep crossover study on
> real K1 silicon. **[NG-4] methodology framing only** — no "IME X-times" headline,
> no e2e beat claim, no eight-gate release (this is a **range-vs-range ablation**,
> opponent = OUR OWN RVV-vector construction, not a vendor baseline).

## Artifacts
- `T5b_ime_paradigm_lever.csv` — the measurement grid (6 cells × 10 M-values), run1.
- `t5b_run1_full_raw.txt` — full board stdout (fingerprint header + ZERO-MODEL gate + self-check + grid).
- `t5b_run2_keyrows_raw.txt` / `t5b_run3_keyrows_raw.txt` — two reproducibility reruns (key rows).
- Harness source (in-tree, committed by main session): `test/Target/IME/q4-0-paradigm-lever-t5b-k1.c`
  - board md5 = `c0565b6a55091de3a454064d2b6febf2` (matches in-tree).

## Environment fingerprint (记录项 — 相×板×格式×对手×账本×编译器)
| field | value |
|---|---|
| phase | matmul (prefill M-sweep) + GEVM (M=1); **compute-isolated** (weights pre-decoded, L2-resident) |
| board | k1 (SpacemiT X60), VLEN=256bit, 8 harts, governor=performance @1.6GHz |
| harts | pinned 0–3 via `taskset -c 0-3` (IME present harts 0–3) |
| format | q4_0@ime (structural-sealed, M1 seal `q4-0-matmul-tile-int32-k1seal.c`, 0-perf until now) |
| shape | N=256, K=256; M ∈ {1,2,4,8,16,32,64,128,256,512} |
| opponent-identity | **SELF** (our RVV-vector construction) — range-vs-range ablation, NOT vs-ggml |
| eight-gate | **not invoked** (no beat claim; NG-4 methodology framing) |
| ledger — KERNEL | **compiler-symmetric**: all cells in ONE binary, ONE compiler ⇒ paradigm/maturity lever is the only free variable |
| ledger — SYSTEM | absolute ns; both ranges = k1 SpacemiT gcc 13.2.0 `-O2 -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d` |
| both-ranges-compiler | k1 SpacemiT gcc 13.2.0 (assembles vmadot 0xe210312b) — symmetric |
| k1-shipped-baseline-compiler | clang-18 (NOT exercised here; no vs-shipped claim) |
| vmadot encoding | 0xe210312b at both leaf sites (objdump-verified = M1 seal encoding) |
| board state | pre loadavg 2.61 / post 2.28; temp 48–49 °C stable; neighbor = system daemons (avahi), no heavy hart-0-3 contender |

## Cells (2×2 paradigm×layout collapses to the 3 measured columns per 实验总纲 §3; IME@vector-layout = N/A, documented not a gap)
| cell | paradigm | layout | note |
|---|---|---|---|
| Cell1  | IME-matrix | tiled | **emitter-verbatim** un-pipelined leaf (per-fragment vsetvli e8↔e32 toggle + zero + store + scalar acc[]) |
| Cell1b | IME-matrix | tiled | **register-resident batched** leaf (accumulate v2/v3 across K/8, single vsetvli, store once) = **one maturity lever** |
| Cell2  | RVV-vector | tiled (matrix-opt) | vector 8-elem widening-reduce on the matrix-tiled layout (layout-penalty control) |
| Cell3  | RVV-vector | vector-opt (per-row) | col-outer vwmacc, 1 row at a time (re-reads weights per row) |
| Cell3b | RVV-vector | vector-opt (MR=4, i16 weights) | col-outer + 4-row register blocking = **vector paradigm's FAIR best** |
| Cell3c | RVV-vector | vector-opt (MR=4, i8 weights) | Cell3b but int8 weights (vsext on load) ⇒ **matches IME's exact byte traffic** (airtight compute-isolation control) |

## Measurement hygiene (SOP)
- **ZERO-MODEL correctness gate (before any timing)**: every cell × every M is int32-EXACT (0-diff) vs an
  independent plain-triple-loop GEMM re-derived from the logical matrices. PASS.
- **Self-check** (3× repeat, M=64, Cell3): spreads 3.22% / 2.60% / 3.18% / 5.84% / 7.46% / 4.46% across runs —
  **borderline** (elevated by the avahi neighbor on absolute times). Mitigation = **paired alternation**
  (all cells interleaved per rep) + **N=15 median**; the reported **paired ratios reproduce to ≈1–2% across
  3 independent full runs** (see below), i.e. the neighbor jitter cancels in the ratio. Recorded, not gated out.
- taskset pin 0–3; adaptive inner-repeat so each timed sample ≥ 300 µs.

## Results (run1; ratio = ns/Cell1b, the fair register-resident matrix baseline; >1 ⇒ slower than batched-IME)
| M | Cell1 leaf | Cell1b batched(=1) | Cell2 tiled-vec | Cell3 vec-perrow | **Cell3b vec-fair-i16** | Cell3c vec-fair-i8 |
|--:|--:|--:|--:|--:|--:|--:|
| 1   | 8.49 | 1.00 | 38.1 | 1.30 | **1.30** | 1.55 |
| 2   | 8.53 | 1.00 | 38.2 | 2.61 | **2.62** | 3.09 |
| 4   | 8.51 | 1.00 | 38.1 | 5.23 | **1.92** | 2.34 |
| 8   | 8.60 | 1.00 | 38.3 | 5.26 | **1.90** | 2.34 |
| 16  | 8.61 | 1.00 | 38.5 | 5.29 | **1.90** | 2.33 |
| 32  | 8.55 | 1.00 | 38.3 | 5.30 | **1.90** | 2.32 |
| 64  | 8.57 | 1.00 | 38.4 | 5.34 | **1.92** | 2.38 |
| 128 | 8.56 | 1.00 | 38.4 | 5.38 | **1.99** | 2.44 |
| 256 | 8.51 | 1.00 | 38.1 | 5.49 | **2.05** | 2.50 |
| 512 | 8.40 | 1.00 | 38.0 | 5.51 | **2.09** | 2.51 |

Cross-run reproducibility (Cell3b/Cell1b): M=1 {1.33,1.34,1.31}; M=4 {1.97,1.94,1.91}; M=64 {2.00,1.99,2.00}; M=256 {2.10,2.08,2.06}; M=512 {2.15,2.12,2.10}.

## Readings (pre-registered ⑦ — executed as declared)

**1. Emitter-maturity GAP (the headline mechanism finding).**
The **as-emitted IME leaf (Cell1) LOSES** to the fair vector path (Cell3b) by ≈4× at all M
(Cell1 = 8.5× Cell1b, Cell3b = 1.9–2.1× Cell1b ⇒ Cell1 ≈ 4× Cell3b). The 8.5× Cell1/Cell1b gap is
**pure per-fragment overhead** (e8↔e32 vsetvli toggle + zeroing + store + scalar acc[] around every single
vmadot), NOT vmadot throughput. **One maturity lever** — register-resident int32 accumulation in v2/v3 across
the K/8 fragment loop, a single vsetvli, one store (Cell1b) — flips IME from a 4× loss to a **~2× win** vs the
vector paradigm's best. This is **directly analogous** to the K-quant repack "un-pipelined first-construction
→ tiling lever" story (perf constitution L1). → **Yellow-cell exit = named GAP `[GAP-IME-LEAF-PIPELINE]`**:
the IME backend emitter (`lib/Plugin/IME/IMEBackendEmissionDriver.cpp` `macHelperBody`) currently emits the
per-fragment leaf; it must emit the register-resident batched leaf. Cell1b is the board-proven demonstration
that closing this gap is worth ~4× and yields the matrix-range crossover.

**2. Crossover M\* (compute-isolated kernel account).**
With the fair register-resident baseline, **the matrix range wins at ALL M ∈ {1..512}** (M\* ≤ 1 in the
compute-isolated regime), and the **advantage GROWS with M**: ≈1.33× at M=1 (GEVM) → ≈1.9× by M=4 →
≈2.1× at M=512 (prefill), saturating by M≥4. The advantage is **smallest at M=1**, consistent with the
roofline direction (GEVM is the matrix range's weakest regime).

**3. M=1 roofline reconciliation (micro↛e2e discipline — critical caveat).**
The pre-registered "M=1 ≈ parity (roofline)" prediction is about the **memory-bound** GEVM/decode regime
(weight streaming + dequant dominate, matrix unit can't help). **This microbench deliberately EXCLUDES that**
(weights pre-decoded + L2-resident ⇒ compute-bound), so the M=1 compute-micro win (1.33×) **does NOT
contradict** the roofline prediction and **must NOT be read as an IME decode-e2e win**
([[kernel-wins-dont-transplant-to-e2e]]: IME 5.51× kernel → 0.86× decode e2e). The compute-micro M=1 result
says only that the matrix MAC leaf is not intrinsically slower at M=1; the memory-bound decode verdict is
still parity/open pending a with-decode measurement.

**4. Paradigm advantage is real compute, not a data-type/traffic artifact (airtight control).**
Cell3c gives the vector path IME's **exact int8 weight-byte traffic** (half of Cell3b's int16). It comes out
**SLOWER** than Cell3b (2.3–2.5× vs 1.9–2.1×), not faster — reducing weight bytes (int8) did not help; adding
the vsext widening op hurt. ⇒ both paradigms are **compute/issue-bound** at these sizes, so the ~2× IME
advantage is a genuine **matrix-MAC-throughput** win (vmadot 128 MAC/instr vs vwmacc 32 MAC/instr), not an
int8-vs-int16 memory artifact. The fair vector opponent is therefore Cell3b (its own best).

**5. Layout lever (Cell2 vs Cell3b).** The vector paradigm forced onto the matrix-tiled layout (Cell2,
8-element reductions) is ~38× Cell1b ≈ ~19× worse than its own vector-optimal layout — confirms the 实验总纲 §3
prediction that IME@vector-layout is N/A and that layout choice is first-order for the vector range.

## T5c — M\* writeback to P7 prior (proposal — CANON-level, needs user ratification per 决策权限卡 必问②)
Do **NOT** encode a constant paradigm preference. Proposed capability-keyed cost-model prior:
- **Prefill regime (M ≥ 4, compute-bound)**: prefer matrix-range (IME) when `ime.present` — board-measured
  ~1.9–2.1× over the vector range's best, and the regime is compute-bound so the micro advantage is the more
  likely to hold. **Contingent on `[GAP-IME-LEAF-PIPELINE]` being closed** (as-emitted leaf loses ~4×).
- **Decode/GEVM regime (M = 1, memory-bound)**: treat as **parity / roofline-dominated**; do NOT key selection
  on the compute-micro 1.33× — it does not transplant to memory-bound decode. Needs a with-decode measurement
  before the selector trusts a matrix win at M=1.
- This writeback touches selector cost-model priors (canon per 权限卡) → **staged as proposal, not applied**;
  the harness + data are the T5c input.

## T5d — vendor path (methodology control): NOT RUN this session (honest). Deferred; would be M=1 gain-not-collapsing-to-1.0 external reference only (family-level, not attributable to the matrix unit; per 实验总纲 §3 = 现状唯一能报的 IME e2e).

## perf-covered contribution
q4_0@ime moves from **0 perf data → measured**, with a **contingent** ≥parity/win verdict: as-emitted (Cell1)
it is a LOSS (~4×) ⇒ **not yet perf-covered**; with the named lever (Cell1b) it is a ~2× prefill win ⇒
perf-covered **pending `[GAP-IME-LEAF-PIPELINE]`**. Reported as a **yellow cell with a named fixable GAP**, not
a green cell — the honest state is "matrix-range advantage board-demonstrated, blocked on emitter maturity."

## durable files (git-tracked + untracked-not-ignored in this cell)

> Machine-registry of the durable evidence (supersedes the human-readable `## Artifacts` list above for the
> dir-lint; the in-tree harness source `test/Target/IME/q4-0-paradigm-lever-t5b-k1.c` lives under `test/`, not here).

- `T5b_ime_paradigm_lever.csv` — the measurement grid (6 cells × 10 M-values), run1.
- `t5b_run1_full_raw.txt` — full board stdout (fingerprint header + ZERO-MODEL gate + self-check + grid).
- `t5b_run2_keyrows_raw.txt` — reproducibility rerun #2 (key rows).
- `t5b_run3_keyrows_raw.txt` — reproducibility rerun #3 (key rows).
