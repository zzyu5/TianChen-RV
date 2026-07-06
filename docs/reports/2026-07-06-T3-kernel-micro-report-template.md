# T3 — Kernel Micro-Benchmark Report (TEMPLATE)

> Report template for the **T3 double-board kernel micro-benchmark** table
> (C3′ pattern library + compiler-maturity axis). This is a TEMPLATE: fill new
> rows following the certification gate below. Perf **magnitudes** are quotable
> only from SEALED cells in the body; OPEN / PARK / STALE cells are quarantined
> to the appendix and MUST NOT appear in the body / paper正文.
>
> Companion data cells (auto-generated, line-E visibility pack):
> `experiments/visibility/T0-sixstate.md` (coverage), `T7-burndown.md`
> (maturity), `T2-ledger-anchor.md` (marginal cost). Raw T3 measurement rows:
> `experiments/T3_A_board_A_rvv1.0_vlen128.csv`,
> `experiments/T3_B_board_B_rvv1.0_vlen256.csv`.

## Certification legend (the hard gate)

| status | meaning | may appear in 正文/body? |
|---|---|---|
| **sealed**   | preflight 4/4 (march-complete · libcall-free · same-compiler · fp-match), fixed-freq, CI + between-pass floor, objdump-sealed mechanism | **YES** |
| **open**     | measured but a confound is live (e.g. board lacks clang-20 / zfh); magnitude **not citable** | NO — appendix only |
| **park**     | blocked on hardware / a parked問題包 item (see ledger P1–P3) | NO — appendix only |
| **stale**    | fingerprint changed (实验宪法第1条: 指纹变 → 同指纹格自动 stale) | NO — appendix only |

**Rule:** a perf number enters the body ONLY when its cell is `sealed`. Everything
else lives under `## Appendix (quarantine — 禁入正文)`.

**ns absolute values are cross-machine INCOMPARABLE** — always carry the board
identity; never compare an ns figure from one board to another. Only same-board,
same-session A/B ratios are comparable.

---

## Board identity (fill per run)

| field | board A | board B |
|---|---|---|
| board id       | rvv-openeuler VLEN128 / 64c | k1 SpacemiT X60 VLEN256 / 8c |
| VLEN           | 128 | 256 |
| toolchain      | (clang/gcc + version) | (clang/gcc + version) |
| march          | rv64gcv_zfh_zvfh_zb… (full-cap) | rv64gcv_zfh_zvfh_zicbop… |
| governor/freq  | fixed (record kHz) | performance (record kHz) |
| DVFS span      | (record %) | (record %) |

---

## BODY — SEALED cells only

### Correctness / capability shape (sealed)

These kernel-level facts are SEALED today (byte-exact / bounded-ULP silicon +
capability-shape flips). They are the sealed backbone of T3; perf **magnitude**
cells for the same kernels are pending a clean remeasure (see appendix).

| kernel (op/format) | board | correctness gate | capability shape | provenance |
|---|---|---|---|---|
| vec_dot / iq4_nl | rvv VLEN128 | **BIT-EXACT ULP=0**, 256/256 ×3 seeds (vs no-FMA left-assoc ggml-generic oracle) | flat codebook, SEW32/m1 | `experiments/silicon-validation-batch-1` |
| vec_dot / iq1_s  | rvv VLEN128 | **BIT-EXACT ULP=0**, 256/256 ×3 seeds | super-block ternary grid, vluxei16 | `experiments/silicon-validation-batch-1` |
| vec_dot / iq1_m  | rvv VLEN128 | **BIT-EXACT ULP=0**, 256/256 ×3 seeds | super-block ternary grid (packed scale) | `experiments/silicon-validation-batch-1` |
| gemm_tile / q4_0 (decode GEVM) | rvv VLEN128 | **FMA-fold correct**, rel < 8.3e-4 to fp32 (structurally NOT no-FMA; NOT bit-exact vs no-FMA oracle by construction) | repack lane-wise, mf2 | `experiments/silicon-validation-batch-1` |

> Wording lock: the q4_0 repack GEVM is "FMA-fold correct" (bounded ULP to fp32),
> never "bit-exact vs no-FMA" (it uses a vfmacc fold by construction).

### Sealed micro perf magnitudes

_None sealed yet on the current toolchain policy._ The clean two-axis kernel win
(dequantize vs scalar / vs clang-autovec-naive-RVV, objdump-verified) and the
q8_0/q5_K micro rows are KERNEL-only and were measured under boards lacking
clang-20; per实验宪法 they must be remeasured (clang-20 + board full-cap march)
before a magnitude is sealed. Until then their magnitudes live in the appendix.

| kernel (op/format) | board | ours ns | factory ns | ratio [CI] | mechanism (objdump) | status |
|---|---|---|---|---|---|---|
| _TODO: fill when a micro cell passes preflight 4/4 on clang-20 full-cap_ | | | | | | sealed |

---

## Appendix (quarantine — 禁入正文)

> Everything below is OPEN / PARK / STALE. Magnitudes here are **not citable** in
> the body or the paper. Kept for provenance and to schedule the remeasure.

### OPEN / CONTAMINATED

| kernel | board | number | why not sealed |
|---|---|---|---|
| q8_0 micro (VLEN256 m1 vs factory) | k1 VLEN256 | ~1.004× (parity, directional only) | **fp16-libcall confound**: step-3 both sides lacked zfh → both fp16 paths went `__extendhfsf2`, compressing the ratio toward 1. Magnitude uncitable; only the qualitative parity direction survives. Remeasure on clang-20 + zfh march. Source: `experiments/T3_B_board_B_rvv1.0_vlen256.csv` (top-of-file STALE banner). |
| winA / no-FMA / vsetvli-sched decomposition (+11.0% / −8.4% / −2.0%) | k1 VLEN256 | contaminated | same fp16-libcall confound; amplitude not citable, direction only. |

### STALE

| item | number | why stale |
|---|---|---|
| T-N noise floor (4-row absolute ns anchor) | ns values | `experiments/T-N_noise_floor.csv` top banner: measured → stale after the 2026-07-05 P2c fingerprint change. Absolute ns are board-bound anyway (cross-machine incomparable). |
| rvv-old micro numbers | — | measured on the retired `ssh rvv-old` board; new `ssh rvv` (2026-06-29+) is a different board → cross-board incomparable → stale. |

### Remeasure checklist (to move an appendix cell into the body)

1. board full-capability march (zfh/zvfh present) — kills the fp16-libcall confound.
2. clang-20 (or record the board-clang bound as a caveat).
3. preflight 4/4 + fixed freq + between-pass floor + CI.
4. objdump-seal the mechanism (confirm the emitted instructions).
5. flip the cell `status` to `sealed`, move the row up into the body.
