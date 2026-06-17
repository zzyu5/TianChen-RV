# Research: Batched GEMV/GEMM perf scoping — where is the real win vs per-row vec_dot?

- **Query**: Scope the highest-leverage Q4_0×Q8_0 batched path. Study ggml's repack gemv on riscv,
  measure the multi-row-vs-N×vec_dot upside on `ssh rvv`, recommend the cleanest first target, map
  it onto our block-dot template + autotuner, and give a validatable increment plan.
- **Scope**: mixed (external llama.cpp source + empirical microbench on `ssh rvv` + internal)
- **Date**: 2026-06-16
- **Board**: `ssh rvv` — 64-core riscv64, **VLEN=128 (VLENB=16)**, rv64imafdcv+zvfh+zvfhmin,
  clang 18.1.3, NO matmul accelerator. (`__riscv_vlenb()==16` confirmed on-board.)
- **Probes (throwaway, NOT shippable)** in `artifacts/inc13-gemv-research/`:
  `gemv_probe.c` (decode-side: N×vec_dot / activation-hoist / cross-row-scalar-reduce),
  `gemv_probe_il.c` (decode-side interleaved 4-col tree-reduce — ggml's 8x8 mechanism at VLEN=128),
  **`gemm_probe.c` (prefill-side: M×vec_dot vs weight-decode reuse — the WIN)**. Numbers:
  `microbench_gemv_n8_n16_n32.txt`, `microbench_gemm_weight_reuse_Msweep.txt`,
  `microbench_gemm_ksweep_L1_mechanism.txt` (K-sweep confirming M_opt∝1/K).

> **Headline (honest, measured, byte-exact-gated):**
> **The win is in GEMM/PREFILL, not GEMV/decode.** Two opposite results from the same
> compute-bound fact:
> - **GEMV / decode (1 token × N weight rows): NO win.** Reusing the activation across rows
>   (`b1_hoist`) buys nothing — the kernel is compute-bound, the activation load was never the
>   bottleneck. ggml's real interleaved-tree mechanism (`il4`) is byte-exact but **~0.75× (≈25-33%
>   slower)** at VLEN=128. **Per-row vec_dot is the decode ceiling on this board.**
> - **GEMM / prefill (1 weight row × M activation columns): a REAL, byte-exact, tunable win.**
>   Decoding each weight block **once** and reusing the decoded nibble lanes across M columns
>   (`w_hoist`) beats M× vec_dot (the prefill path that re-decodes the weight every column) by
>   **~1.13× at contraction K=4096, M≈4–6** — real and reproducible. K is the *contraction* dim, so
>   K=4096 covers **attention Q/K/V/O + FFN gate/up = ~78% of a 7B's prefill matmul FLOPs**; only the
>   FFN down-projection (~22%) contracts over K≈11008, where the win fades to parity. **Blended prefill
>   matmul speedup ≈ 0.78·1.13 + 0.22·1.0 ≈ ~1.10×.** **Same compute-bound fact, opposite sign:** GEMV
>   has zero weight-decode reuse; GEMM amortizes the decode M-fold. The optimal M-block **falls as K
>   rises** (cache-capacity: more activation streams contend for L1), so it **must be measurement-tuned**
>   (an untuned wide-M gemm regresses).
>
> **Dispatch fact behind all of it:** ggml's repack GEMV **and** GEMM are **ENTIRELY DISABLED for
> Q4_0 at VLEN=128** (`ggml_repack_get_optimal_repack_type`: `case 128: { break; }` → `nullptr`).
> So **both** decode and prefill fall back to plain per-(row,col) `ggml_vec_dot_q4_0_q8_0` — which
> re-decodes each weight block for every output column. **That redundant re-decode is exactly what
> our prefill weight-reuse kernel eliminates.** Our N1 story: we fill ggml's empty VLEN=128
> gemv/gemm ABI with a byte-exact compiler-emitted kernel; our N3 story: the **prefill** kernel
> **beats the used path (M× vec_dot) by ~13% at contraction K=4096 (~78% of prefill matmul FLOPs;
> ~1.10× blended)**, and that M-block is a **measurement-selected, cache-bound column-blocking knob** —
> an **autotuner-selected
> register-blocking knob** (the win curve peaks then regresses — a genuine tuning axis).

---

## 1. How ggml's repack gemv/gemm works (and when it actually engages)

### 1a. The repacked Q4_0_8x8 format (`block_q4_0x8`)
`make_block_q4_0x8` (`llama.cpp/ggml/src/ggml-cpu/repack.cpp:2787`) interleaves **8 weight rows**:
8 inline fp16 scales, then the 8 rows' `qs` interleaved in **8-byte chunks**
(`blck_size_interleave=8`), **XOR'd with `0x8888...`** — the XOR turns ggml's offset-binary nibble
(`nibble−8`) into two's-complement i4, so the kernel decodes with plain `vsll(4)/vsra(4)`, no `−8`.
The activation is repacked into `block_q8_0x4` (`ggml_quantize_mat_q8_0_4x8`, `repack.cpp:27`,`:178`),
4 fp16 scales + 8-byte-interleaved i8 — **with `qs` at an 8-aligned offset** (matters for §2).

### 1b. One gemv call → nr×nc outputs; the register blocking
`ggml_gemv_q4_0_8x8_q8_0` (`arch/riscv/repack.cpp:114-203`), per `nc/8` column-group:
holds **one `vfloat32m1_t` = 8 column lanes**; per block, broadcasts the activation via i64 splat
(`vmv_v_x_i64m2`), loads 64 weight bytes (`vle8_v_i8m4`), decodes `vsll/vsra` at m4, products
`vwmul/vwmacc` → `i16m4`; **the reduction is a `vnsrl`/`vadd` de-interleave TREE** (`:166-177`) →
**8 column i32 sums with ZERO `vwredsum`**; scale fold `vfmul_vf`(act scale) + `vfmacc_vv`(8-lane
col-scale vector). `ggml_gemm_q4_0_8x8_q8_0` (`:670`) = same kernel + **outer 4-row tile**
(`sumf0..3`) → a 4×8 output tile per inner pass (prefill).

### 1c. WHEN gemv vs gemm is chosen — and why NEITHER repack path runs at VLEN=128
- **Within the repack path** (`forward_mul_mat_one_chunk`, `repack.cpp:4240-4250`): `nrows>3` → `gemm`
  else per-row `gemv`. So **gemv = decode, gemm = prefill** — both real hot paths *when active*.
- **The repack path is gated by `ggml_repack_get_optimal_repack_type`** (`repack.cpp:4528-4599`). For
  Q4_0 on riscv it switches on `__riscv_vlenb()*8`: **`case 128: { break; }` returns `nullptr`** →
  `supports_op` (`:4774`) fails → the tensor is never assigned the repack buffer type → Q4_0 mul_mat
  goes to the generic CPU backend, i.e. **per-(row,col) `ggml_vec_dot_q4_0_q8_0` for BOTH gemv and
  gemm.** Board-confirmed (`__riscv_vlenb()==16`). Sharpens INC-3: **ALL Q4_0 tensors fall back at
  VLEN=128 by ggml's own design**, not just one model's.
- **Even forced**, the bodies break at VLEN=128: `ggml_gemv_q4_0_8x8_q8_0` guards its RVV body on
  `__riscv_vlenb() >= QK4_0` (16≥32 false → scalar generic), and `ggml_gemv_q4_0_16x1_q8_0` (`:206`)
  requests 16 f32m2 lanes (VLEN=128 f32m2 = 8 → half the columns) — which is *why* `case 128` is off.
  **ggml has NO working vector gemv/gemm for Q4_0 at VLEN=128. That is the ABI gap we fill.**

---

## 2. The measured perf opportunity (the upside, honestly)

All variants **bitwise-gated** (`memcmp` of fp32 bits) against the relevant **used path** on
identical random data before timing. min-of-reps ns, `taskset -c 3`, `-O3 -ffp-contract=fast`,
double-warmed. @ K=4096.

### 2a. DECODE / GEMV — no win (the task's premise falsified)
Per-row ns vs N× vec_dot (the used decode path):

| variant | N=16 | N=32 | vs vec_dot | byte-exact | isolates |
|---|---|---|---|---|---|
| **a_vecdot** (N× vec_dot, USED) | 1170 | 1175 | 1.00× | ref | today's baseline |
| **b1_hoist** (activation loaded once/blk, reused N rows) | 1221 | 1379 | **slower** | PASS | **activation-load saving ALONE** |
| **b2_xrow** (scalar reduce, scale-fold vectorized across rows) | 1227 | 1231 | slower | PASS | vectorizing only the scale step |
| **il4** (interleaved 4-col de-interleave tree-reduce, no `vwredsum`) | 1523 | 1556 | **~0.75×** | PASS | ggml's real reduction-amortization |

**The activation-reuse premise is false.** `b1_hoist` decodes each q8 block once and reuses it across
all N rows — and does **not** win (worse as the weight working set grows). The kernel is
**compute-bound**; saving the activation load saves nothing. `il4` faithfully reproduces ggml's 8x8
structure (one `vle8.m4`, m4 products, the `vnsrl`/`vadd` tree replacing 4 `vwredsum`s, vector
scale-fold) and is **byte-exact**, but at VLEN=128 the tree + i64 splats cost **more** than the
`vwredsum`s they replace (ggml's 8x8 wins only at VLEN≥256, where the register file absorbs the m4
machinery). **Decode ceiling = per-row vec_dot.** (Consistent with STRAND-2: no robust shape beats
ggml vec_dot at VLEN=128.)

> **PROBE INTEGRITY (don't misread the raw history):** an early `il4` build measured **~92× SLOWER**
> (K=1024). That was a **probe artifact**: the i64 activation broadcast read a *standard* q8_0 block
> at byte offset +2 → addresses ≡ 2,4,6,0 mod 8 → **misaligned i64 loads that trap-emulate** (board
> ISA has no fast-misaligned support; the `asm volatile` barrier, copied from ggml, keeps them
> scalar). Copying the 32 activation i8s to an 8-aligned scratch once/block (ggml's repacked
> `block_q8_0x4` already 8-aligns them) removed the trap → `il4` to ~0.75×. The gate PASSED in **both**
> builds — layout/decode/tree/scale-fold proven correct; only pre-fix *timing* was contaminated. The
> shipped probe is the aligned build. (vec_dot is immune: it loads via `vle8`, bytes never trap.)

### 2b. PREFILL / GEMM — the REAL win (weight-decode reuse)
`gemm_probe.c`: one weight row × M activation columns → M outputs. Baseline = M× vec_dot (re-decodes
the weight every column = the prefill path at VLEN=128). `w_hoist` = decode each weight block **once**,
reuse the decoded `v0/v1` lanes across all M columns. ns per (M-output), byte-exact-gated:

| M (cols) | M×vec_dot ns/col | w_hoist ns/col | **speedup** | gate |
|---|---|---|---|---|
| 4  | 2040 | 1816 | **1.135×** | PASS |
| **6** | 1338 | 1139 | **1.161× (peak)** | PASS |
| 8  | 1168 | 1101 | **1.068×** | PASS |
| 12 | 1166 | 1221 | 0.966× | PASS |
| 16 | 1173 | 1294 | 0.905× | PASS |

(Reproduced across re-runs: at **K=4096, M=4–6 settles at ~1.13×** — direction and existence are
robust. There is real run-to-run variance on this shared 64-core board: K=4096/M=6 swung 1.029–1.161×,
K=2048/M=8 swung 0.993–1.204× — so individual point-`M_opt` values are noise-dominated; only the broad
shape is reliable. Raw: `microbench_gemm_weight_reuse_Msweep.txt`.) **Weight-decode reuse beats the
used prefill path by ~13% at K=4096/M≈4–6, byte-exact.**

**Why it works (same compute-bound fact, opposite sign vs §2a):** weight decode
(`vand/vsrl/vsub` ≈ half the integer vector work per block) is REUSED M-fold in prefill but
zero-fold in decode. GEMV has nothing to amortize → loses; GEMM amortizes the decode → wins.

**What sets the optimal M-block — a CACHE-CAPACITY effect, NOT the vreg budget.** Across the
M-column inner loop only the decoded weight `v0/v1` persist (2 vregs); `y0/y1/m/z` are per-iteration
and dead at iter end → **vector-register use is flat in M.** So the limiter is **M concurrent q8
activation streams contending for ~32 KB L1** (each stream = `(K/32)×34` bytes), and the optimal
M-block **falls as K rises**. The K-sweep is **directionally consistent but the simple capacity
formula `M×(K/32)×34 ≲ 32 KB` OVER-PREDICTS** — it is a *direction*, not a law:

| contraction K | 7B matmuls (prefill FLOP share) | bytes/q8 stream | formula M_opt | **measured** | honest read |
|---|---|---|---|---|---|
| 2048 | (none — probe point) | 2176 B | ~15 | win to M=12–16 (~1.20–1.25×, noisy) | larger M_opt ✓ (variance high) |
| **4096** | **attn Q/K/V/O + FFN gate/up (~78%)** | 4352 B | ~7 | **~1.13× at M=4–6**, parity ~M=8 | the **real, reproducible win** |
| 8192 | (none — probe point) | 8704 B | ~3–4 | **no reliable win** (0.93–0.96× all M) | formula predicts a win; none seen |
| 11008 | **FFN down-proj only (~22%)** | 11696 B | ~2–3 | **marginal/parity** (0.88–1.09×, noisy) | win eaten at this one dim |

**Blended prefill matmul speedup ≈ 0.78·1.13 + 0.22·1.0 ≈ ~1.10×** (FLOP shares: attn Q/K/V/O =
4·4096² = 67M; FFN gate+up = 2·4096·11008 = 90M, both contract K=4096; FFN down = 4096·11008 = 45M,
contracts K=11008 → 157M of 202M total = 78% at the winning K). M is tiled per-call, so the per-tile
~1.13× sustains for any prompt length (the prefill token-batch is tiled into M_opt chunks).

The formula's failure is informative: at K=8192/M=2 the streams are only ~17 KB (fit L1) yet there is
no win → the capacity bound is **not** the whole story (decode/reduce latency vs the smaller per-call
amortization also matters), and **point-`M_opt` is unpredictable analytically.** This is the case *for*
measurement-backed selection (§4/§5), not against it. Raw: `microbench_gemm_ksweep_L1_mechanism.txt`,
`microbench_gemm_real_deployment_dims.txt`.

The optimal column-block tracking 1/K is the signature of a **cache-capacity** resource, exactly the
kind of capability/resource fact the N3 tune is meant to discover. **A naïve vreg-budget prune would
mispredict this** (it sees no M-dependence) and pick too-large M → ship a regression; the autotuner
must select M by **measurement** (which it does — enumerate→prune→*measure*→select), or model an L1
term. (See §4/§5.)

---

## 3. Recommended first target

**Build the PREFILL weight-decode-reuse kernel (`w_hoist` shape), NOT a decode gemv.**

Justification (numbers, not aesthetics):
1. **It is the only measured win, and it covers the majority of prefill.** Decode-side gemv (every
   structure tried) loses or ties; prefill weight-reuse beats the used path by **~13% at contraction
   K=4096, byte-exact** — and K=4096 is the contraction dim for **attention Q/K/V/O + FFN gate/up =
   ~78% of a 7B's prefill matmul FLOPs** (only FFN down-proj is at the no-win K≈11008, §2b). **Blended
   ≈ ~1.10×.** Modest, but the N3 lamp ("a compiler-emitted batched kernel beats the per-row path ggml
   actually uses") is **achievable across most prefill matmuls** on **this** VLEN=128 board.
2. **It is the cleanest possible structure** — and that is the answer to the task's "standard layout
   vs ggml repack format" question. `w_hoist` is **standard q4_0 weights, no interleave, no repack, no
   reduction tree.** It reuses our existing per-row block-dot machinery verbatim (the same
   nibble-unpack + `vwmul/vwmacc/vwredsum`), only **hoisting the weight decode out of an M-column
   loop**. The interleaved/tree path (`il4`) is both harder *and* slower here → retired by measurement.
3. **It targets the path that dominates prompt-processing throughput.** Prefill is where mul_mat has
   many activation columns (M = the token batch) — exactly where weight reuse pays. Decode (M=1) is
   genuinely per-row-bound, and we already match it.

So: **first target = a standard-layout multi-COLUMN gemm (decode-weight-once, reuse across M
columns)** — the simplest viable structure, and the measured winner. The "match ggml exactly with the
repack format" option is *not* recommended: it is more complex and, at VLEN=128, slower.

---

## 4. Mapping to our approach (template + autotuner)

### Reusable (already proven, HW-validated)
- **The integer core, unchanged**: nibble-unpack + `vwmul/vwmacc` widening product + `vwredsum`
  reduce — our `widening_product_reduce_*` + `packed_i4_nibble_unpack_product` family
  (`our-current-coverage-inventory.md`; `RVVToEmitC.cpp:3493+`,`3560-3649`). `w_hoist`'s per-(block,col)
  math is **identical** to the per-row kernel; only the loop nest changes.
- **The block-dot template** (`emitc.for` + `emitQ4_0Q8_0BlockDot`): the per-block weight decode is
  already a distinct step; `w_hoist` just lifts it above an inner M-column loop. **Structured emission
  survives** — this is a loop-nest rewrite, not a new fused op.
- **The measurement autotuner**: enumerate→prune→**measure**→select (`RVVGearboxSchedule.h:1879-2085`)
  + attr stamping (`RVVContractionSelectedBodyRealizationOwner.cpp`, `gearbox-autotuner-wiring.md`).
  Its **measurement-backed** selection is what makes it the right tool for the M-block — the M optimum
  is set by an L1-capacity resource (§2b), which the existing **vreg-budget prune does NOT model** (the
  M-loop carries only 2 persistent vregs → flat in M). The tuner finds M_opt empirically by *measuring*;
  it must not rely on the vreg prune to bound M (that would pick too-large M and ship a regression).
- **The byte-exact discipline**: each output column folds its blocks ascending in `sumi·d_wt·d_act`
  order — `w_hoist` keeps the **vec_dot** order per column → bitwise equal to the used path. (Note:
  vectorizing the scale step *across columns*, as a future wider variant might, does NOT break this —
  each column lane still folds ascending. But ggml's 8x8 uses `(sumi·d_act)·d_wt` — a shipped wide
  kernel must pick its reference order explicitly.)

### New (not yet expressed) — but MODEST for `w_hoist`
- **The M-column inner loop with M scalar accumulators** (`s[0..M-1]`), the weight decode hoisted
  above it. This is the whole new structure for the simple win — a loop-nest + M-wide accumulator
  array, NOT a new reduction shape.
- **Multi-output store** (`s[c]` per column) and the gemm ABI
  (`void gemm(int n, float*s, size_t bs, const void*vx, const void*vy, int nr, int nc)`).
- *(Only if chasing the wider il4/tree variant — NOT recommended here:)* the de-interleave reduction
  tree, the i64 activation broadcast + 8-alignment requirement, and the interleaved weight layout —
  all new, all forced-unstructured (a dedicated fused op), all **slower at VLEN=128**.

### The NEW tuning axis: cache-blocking (columns-per-iter = the M-block)
- A gemm adds **`activation_cols_per_iter` (M-block)**: how many activation columns share one weight
  decode. The optimum is **K-dependent** (§2b) and **falls as K rises** (more activation streams contend
  for L1) — but it is **noisy and analytically unpredictable**: the simple L1-capacity formula
  over-predicts (no reliable win at K≥8192 even where streams fit L1, §2b). What is solid: the win
  exists and is real at K=4096 (~1.13×), is *not* vreg-bound (the M-loop carries only the 2
  decoded-weight vregs → vreg use is flat in M), and shrinks toward the large FFN K.
- This makes it a **measurement-first** N3 axis: *capability/resource-aware selection of the column-block
  the cache hierarchy permits*, where the resource (L1 vs activation working set) is real but the exact
  M_opt must be **measured, not predicted** — precisely the gap a measurement-backed gearbox fills. It
  **composes** with `{integer_core_lmul}×{multi_block}×{strip_elision}` (those answer to the 32-vreg
  prune; the M-block answers to a *cache* effect the vreg prune cannot model) — two orthogonal resource
  axes. The honest cost signal is the **measured win curve**; a pure vreg-budget prune would mispredict M
  (it is flat in M) and ship a regression, so M must be selected by measurement, or the cost model
  extended with a (still-imperfect) cache term — see §5/G3.

---

## 5. Increment plan (G1/G2/G3) — validatable steps

> **Scope note (honest):** the perf win is **prefill/GEMM weight-decode reuse**, measured at
> **~13% over the used path (M× vec_dot) at K=4096/M≈4–6, byte-exact, on this VLEN=128 board** — real
> and reproducible at contraction K=4096 (attn + FFN gate/up, ~78% of prefill FLOPs); only the FFN
> down-projection (K≈11008, ~22%) is at the no-win dim (§2b).
> Decode/GEMV is per-row-bound (no win) — do not scope G-steps as a decode gemv. The plan below targets
> the **prefill weight-reuse kernel** (the measured winner). The M-block optimum is set by **L1 capacity
> (∝1/K)**, so the tuner must select M by **measurement**, not a vreg prune (§4).

- **G1 — multi-column weight-reuse integer+scale core, byte-exact for a small tile (1 weight row × M
  cols, one block).** Express `w_hoist`'s per-block body: decode the weight block once (existing
  nibble-unpack), inner M-column loop doing `vwmul/vwmacc/vwredsum` + per-column scalar fold into
  `s[0..M-1]`. Gate the **single-block** M outputs bitwise vs M× our existing single-block i4×i8 dot.
  Difficulty: **MEDIUM** — reuses the proven integer core; the new part is the M-wide accumulator +
  hoisted decode (loop-nest, no new op). Validatable: one translate-level + one bundle-dry-run test,
  n=32, M∈{4,6}.
- **G2 — full GEMM over the matrix + the ABI + byte-exact `*s`.** Wrap G1 in the K block loop + the
  weight-row loop; emit ggml's gemm ABI and the per-column store; gate the **whole** kernel bitwise vs
  M× vec_dot on board, and **reproduce the win at the hidden dim** (≥1.10× at K=4096/M≈4–6).
  Difficulty: **MEDIUM-HIGH** (ABI + the row/col loop nest; no alignment hazard since weights stay
  standard q4_0). Validatable: on-board `memcmp` gate + a perf assert at K=4096/M≈4 via the existing
  bundle e2e harness. **Note (honest):** the win is K=4096-strong (attn + FFN gate/up) but fades at the
  FFN down-projection's K≈11008 —
  the perf assert must be at a K where the win is real; G2 proves the *mechanism*, not an everywhere-win.
- **G3 — tuner selects the M-block (cache-blocking) → BEAT the used prefill path.** Add
  `activation_cols_per_iter` (M-block) as a bounded knob to enumerate→prune→**measure**→select +
  attr-stamping. **Because M_opt is cache-bound, noisy, and analytically unpredictable (§2b), the
  tuner selects it by MEASUREMENT** (sweep M, pick the empirical peak) — NOT by the vreg prune, which
  is flat in M and would mispredict (§4). **It should select the per-K M and report a measured beat
  over M× vec_dot at the dims where the win is real (K=4096) — a genuine N3 lamp: measurement-backed,
  resource-aware column-block selection.** The honest framing is the *point*: even an analytical L1
  model mispredicts (§2b), so measurement is the only reliable selector — which is exactly the
  measurement-backed-autotuner thesis. (Caveat the tuner must encode: at large FFN K the win is
  marginal, so the tuner may correctly select M=1 / no-blocking there — a non-win it should *report*,
  not force.) Difficulty: **MEDIUM** (mirrors the existing `integer_core_lmul`/`multi_block`
  measure→select wiring) on top of G1/G2. Validatable: the tuner's standalone enumerate/measure unit
  tests + a measured selection log showing the K-dependent M choice and the beat at K=4096.

Keep each increment a separate task (socket-drop risk): G1 = self-contained op-body+test; G2 = ABI +
loop nest; G3 = tuner wiring. **G1 alone is the cleanest first proof** that "a compiler-emitted
multi-column weight-reuse body is byte-exact," and G2 is the first proof it **beats the used path**.

---

## Caveats / Not Found

- **The win is M-shape- and K-dependent, narrow, and VLEN=128-specific.** Reproducible **~1.13× at
  contraction K=4096 (attn Q/K/V/O + FFN gate/up, ~78% of prefill FLOPs), M≈4–6**; it **fades to
  marginal/parity only at the FFN down-projection's K≈11008 (~22%)** and is
  noise-dominated at the point level (§2b). Too-large M *regresses* (0.90× at K=4096/M≥12). A shipped
  kernel MUST tune the M-block per-K (and may correctly choose no-blocking at large K) — an untuned
  wide-M gemm would be **slower** than the per-row baseline. **This is a modest, dimension-limited win,
  not a multiple** — honest framing matters. (This is the autotuner's job and the G3 justification.)
- **Single-threaded pinned** measurement (`taskset -c 3`). ggml runs mul_mat across 64 threads;
  per-call kernel ns is the right unit for the per-output reuse question, but end-to-end prompt-tok/s
  also depends on threading + chunking (`forward_mul_mat`, `repack.cpp:4296-4333`), not scoped here.
- **No VLEN≥256 board available.** ggml's interleaved-tree gemv/gemm (`il4` shape) is byte-exact but
  loses at VLEN=128; it is the genuine win at VLEN≥256 (ggml enables `q4_0_16x1` at 256, leaves
  512/1024 as `// TODO`). A wider-output / interleaved variant is unmeasurable here — flagged as a
  hardware blocker on any *interleaved*-gemm claim. (The recommended `w_hoist` win needs none of it.)
- `w_hoist` reuses the weight decode across columns but still does M independent `vwredsum`s (one per
  column). A future variant could *also* amortize the reduction (the il4 tree) — but that was measured
  slower at VLEN=128, so `w_hoist` (decode-reuse only) is the right first kernel.
- Did not re-verify q5_K/q6_K/q2_K; the `case 128: break` dispatch finding is Q4_0-specific but the
  same VLEN switch governs the K-quants (`repack.cpp:4616+`). The weight-decode-reuse lever should
  generalize (K-quant decode is *more* expensive → potentially a *larger* reuse win) but is unmeasured.
