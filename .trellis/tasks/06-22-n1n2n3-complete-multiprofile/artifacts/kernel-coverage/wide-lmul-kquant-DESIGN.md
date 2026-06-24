# DESIGN — Wide-LMUL K-quant block-dot emitter (named emitter-maturity target #2)

READ-ONLY scoping doc. No `lib/` edits. Build-incremental plan to make the K-quant
block-dot integer core LMUL-parametric so the gearbox can sweep the proven Win-A
LMUL-width knob, narrowing the compute-bound losses to ggml's wider-LMUL block-dots.

Reference (the proven, already-correct pattern to MIRROR): the repack emitter
`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` threads ONE
`coreLmul = getIntegerCoreLmul()` and derives an `l8/l16/l32` chain
(e.g. `:2157-2162`, `:2706-2711`); the op carries
`OptionalAttr<StrAttr>:$integer_core_lmul` (RVVOps.td `:4253,:4364`); the verifier
pins the legal set `{"mf2","m1"}` with the cross-constraint
`m1 => half_lanes==16` (`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:1477-1488`);
the gearbox stamp derives the width from VLEN in `deriveRepackHalfLanes`
(`lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp:78`).

---

## 0. THE LOAD-BEARING STRUCTURAL FACT (read this first)

The K-quant integer core is NOT the repack "block-as-lane" shape. It has TWO
regions with DIFFERENT LMUL semantics, and the split is the spine of this design:

- **Region A — the 6-bit unpack** (`vsetvl_e8m2(32)` 32-wide chunks: vand/vsrl
  on `u8m2`, vreinterpret to `i8m2`, vse8 into `aux8[256]` in memory). This is
  pure throughput over an element-ordered byte array; the store order is fixed
  and order-independent. **Widening LMUL here is byte-exact trivially** — a clean
  knob. Do it first.

- **Region C — the integer MAC into the 8-lane aux32** (per sub-block:
  `vle8 i8mf2(quarter=8)` → `vwmul_vv i16m1` → `vwmacc_vx i32m2` accumulating into
  a `vint32m2_t aux32` seeded `vmv_v_x_i32m2(0, 8)` — **exactly 8 lanes**, one per
  the 8 sub-block-pair accumulators). The `m2` here is a **structural reduction-lane
  count**, NOT free throughput.

**Why the 8 matters (the byte-exact contract):** the deferred fp fold is an
**8-lane** `vfloat32m2_t sums` seeded `vfmv_v_f_f32m2(0.0f, 8)`, materialized
`vse32_v_f32m2(&sums8[0], sums, 8)` (`numLanes = numSubBlocks/2 = 8`,
`RVVToEmitCKQuant.cpp:494,553,706,1454,1462`), then summed in ASCENDING lane order
`l=0..7` (fp-add non-associative — this IS the byte-exactness target, pinned in the
op description and the I-rules). The aux32 lane count (8) DETERMINES that grouping.

**Consequence for Region C:** naively widening the integer strip to 16/32 lanes
regroups the integer partials → the 8 fp partials change → **byte-exact oracle
FAILS.** Region C is still usable, but ONLY under the **fold-back-to-8** discipline:
widen the *integer* MAC strips (integer add is associative / order-free), then
**fold the wide lanes back to the canonical 8 with integer adds BEFORE `vfcvt`**.

**The fold-back identity is provably exact (derived, not hoped-for).** aux32 lane
`l` holds `Σ_js scale_js · Σ_q prod[js, q·8 + l]` — the 8 lanes are the
element-position-mod-8 residues, accumulated across all sub-blocks with the
per-sub-block scalar `scale` folded in *during* the vwmacc (constant within a
sub-block). For a 16-wide widen:
`aux32_16[l] + aux32_16[l+8] = Σ_js scale_js·(prod[l]+prod[8+l]+prod[16+l]+prod[24+l])
= aux32_8[l]`. Integer-associative + scale-constant-within-sub-block ⇒ the regroup
`aux32_8[l] = Σ_k aux32_wide[l + 8k]` is exact at every legal LMUL, so the 8 fp
partials are byte-identical. **This fold-back is THE correctness risk the byte-exact
oracle gates** (Part 3) — gated, but derivably sound.

Correction to the task's stated hypothesis: the "m4 breaks the u8 scale view
register-group" concern is a **red herring** — the 6-bit scale/min bit-dance
(`:919-1006`) is **scalar `emitc.bitwise` on u32** (kmask1/2/3, w0/w1/w2,
storeUtmp), NOT vector-LMUL'd. It is untouched by any LMUL knob. The real ceiling
is the MAC widening chain `i8 → i16 → i32`: base ∈ {mf2 (today), m1, m2} before the
i32 product hits `i32m8`. That bounds the verifier legal set.

---

## 1. EVERY LMUL HARDCODE SITE in RVVToEmitCKQuant.cpp (file:line + emitted-C purpose)

All token strings are hardcoded `__riscv_*_<lmul>` callees / `v*<lmul>_t` type
decls. Region tag: **A** = unpack (clean widen), **C** = MAC into 8-lane aux32
(needs fold-back), **F** = fp fold / final-sum (8-lane — MUST stay 8, it is the
contract; not a knob).

### q4_K shared core `emitQ4_KSuperBlockAux32Core` (`:775`) — covers q4_K + q5_K
- `:888,908`  **A** — `vreinterpret_v_u8m2_i8m2`, `vse8_v_i8m2`: nibble unpack store.
- `:805,807,816,821,827,830,810`  **A** — `vsetvl_e8m2`, `vle8_v_u8m2`, `vand/vsrl/vadd _u8m2`: the 32-wide nibble unpack strip (mirror q6_K `:83`).
- `:1049`  **C** — `vmv_v_x_i32m2(0, 8)`: aux32 seed (8 lanes — the contract anchor).
- `:1097` (`quarterSetvl`) **C** — `vsetvl_e8mf2(quarter=8)`: the quarter-strip vl.
- `:1117` (`loadCallee`) **C** — `vle8_v_i8mf2`: load q8 + aux8 quarter (8-wide).
- `:1132` (`mulCallee`) **C** — `vwmul_vv_i16m1`: i8×i8→i16 product.
- `:1142` (`maccCallee`) **C** — `vwmacc_vx_i32m2`: aux32 += scale·i16 (8-lane MAC).
- Type decls `:1391-1396` (and the Aux32Partial decls `:1214-1218`): `u8m2/i8m2/i8mf2/i16m1/i32m2/f32m2`.
- `:1469,1478` **F** — `vfmv_v_f_f32m2(0.0f, 8)`: sums seed (8-lane). KEEP 8.
- `:1658,1668,1678` **F** — `vfcvt_f_x_v_f32m2` / `vfmul_vf_f32m2` / `vfadd_vv_f32m2`: the fold. KEEP 8.
- `:1724` **F** — `vse32_v_f32m2(&sums8, sums, 8)`: materialize 8 lanes. KEEP 8.

### q6_K shared core `emitQ6_KSuperBlockAux32Core` (`:29`) — covers q6_K only
- `:48,83`  **A** — `vsetvl_e8m2(32)` + `u8m2`/`i8m2` 6-bit ql+qh unpack (2×128-elem chunks).
- `:163,169`  **C** — `vmv_v_x_i32m2(0, 8)`: aux32 seed (8 lanes).
- q6_K uses a **half-strip** (`cx.half`, `emitHalf` at `:218-285`): `:223 vsetvl_e8mf2(half)`, `vle8_v_i8mf2`, `vwmul_vv_i16m1`, `vwmacc_vx_i32m2`.
- `:338-342` type decls; `:436` `vse32_v_i32m2` aux32 store (partial path).
- fp fold (BlockDot path `:553,663-697,724`) **F** — `f32m2`, 8-lane. KEEP 8.

### q5_K BlockDot `emitQ5_KQ8_KBlockDot` (`:1811`)
- **Reuses `emitQ4_KSuperBlockAux32Core` at `:2007`** (`hasQh=true` injects the 5th-bit
  plane). NO own integer-core LMUL tokens → parametrizing the q4_K core auto-covers q5_K.
  Its own fp-fold decls `:1860-1864` are **F** (8-lane, keep).

### q2_K BlockDot `emitQ2_KQ8_KBlockDot` (`:2253`) — DIFFERENT reduction shape
- `:2400`  **A** — `vsetvl_e8m2(32)` 2-bit unpack chunk.
- `:2295-2303` type decls incl. `i16m2`.
- `:2581 vwmul_vv_i16m2`, `:2601 vwredsum_vs_i16m2_i32m1` **C'** — q2_K does a per-sub-block
  `e8m1(16)` → `i16m2` → **horizontal `vwredsum` into a scalar i32m1**, NOT the 8-lane
  aux32 vwmacc. Its integer base is already `m1`/`m2`-wide. Separate emit, separate effort.

### q3_K BlockDot `emitQ3_KQ8_KBlockDot` (`:2786`) — q6_K-shaped, but inline (not shared)
- `:2835-2840` type decls (`u8m2/i8m2/i8mf2/i16m1/i32m2/f32m2`).
- `:2997 vsetvl_e8m2(32)` **A** — 2-bit + hmask unpack.
- `:3188 vmv_v_x_i32m2(0,8)` **C**; `:3247 vsetvl_e8mf2(half)`, `:3266 vle8_v_i8mf2`,
  `:3283 vwmul_vv_i16m1`, `:3292 vwmacc_vx_i32m2` **C** (half-strip, q6_K-shaped but a
  private copy in q3_K's body — NOT a call to `emitQ6_KSuperBlockAux32Core`).
- `:2948 vfmv_v_f_f32m2`, `:2940` 8-lane fold **F**. KEEP 8.

---

## 2. DO THE K-QUANT BLOCK-DOT OPS CARRY AN LMUL ATTR? — NO. MUST BE ADDED.

`GgmlBlockDotQ4KQ8KOp` (RVVOps.td `:6093`) `arguments` end at
`activation_bsums_byte_offset` — **no `integer_core_lmul`, no `half_lanes`**. The op
description even states: *"No resource shape knobs yet... This K4b op carries only the
structural facts."* Same for `GgmlBlockDotQ6KQ8KOp` (`:5891`),
`...Q5KQ8KOp` (`:6207`), `...Q2KQ8KOp` (`:6302`), `...Q3KQ8KOp` (`:6418`), and the
`...Aux32Op` partials (`:5797,:5987`). Contrast the repack ops which DO carry
`OptionalAttr<StrAttr>:$integer_core_lmul` (`:4253,:4364`).

**=> An attr MUST be added** to each K-quant block-dot op def + verifier.

**Add ONE knob (mirror the single-knob repack pattern, NOT two):**
`OptionalAttr<StrAttr>:$integer_core_lmul`. Do NOT add a `half_lanes` analogue — the
K-quant strip width is structurally pinned (quarter=8 / half=8 from QK_K=256), not a
free strip-tiling axis like the repack's 16-block-as-lane group. The single
`integer_core_lmul` selects the integer-core base LMUL; the emit derives the wide
chain from it.

**Verifier rule** (mirror `RVVDialectWideningOps.cpp:1477-1488`, in the per-op
`::verify()` in the same file — q4_K/q5_K/q6_K/q3_K share the 8-lane-aux32 chain):
```
legal set: integer_core_lmul ∈ {"mf2", "m1", "m2"}   // base of i8→i16→i32 chain
                                                       // unset => "mf2" (today's emit)
```
Rationale for the ceiling at `m2` — bounded on TWO independent grounds:
1. **Register-group:** base `i8mf2` → product `i16m1` → MAC `i32m2` (today); base
   `i8m1` → `i16m2` → `i32m4`; base `i8m2` → `i16m4` → `i32m8`. `m4` base would need
   `i32m16` (illegal).
2. **Sub-block-scale boundary (the REAL semantic limit):** `i8m2` = 32 elements =
   **exactly one sub-block = one scalar `scale`**. Going wider (m4 = 64-wide) would
   put TWO sub-blocks under ONE scalar `scale` in a single vwmacc → wrong
   *independent of* the register-group limit. This same boundary is precisely why
   the fold-back is safe at every legal LMUL: each vwmacc stays within one sub-block,
   so `scale` is constant across the lanes being folded.
Also include the per-quant attr-name allow-list update (mirror the repack
`name == "integer_core_lmul"` clause at `:897,1376,1576`) so the op's attribute
verifier does not reject the new attr. q2_K's verifier (`...Q2KQ8KOp::verify`) uses
the SAME legal-set string but the emit maps it to the `vwredsum` base (Part 4).

---

## 3. THE PRECISE ORDERED BUILD-INCREMENTAL PLAN (each step COMPILES)

First K-quant: **q4_K** — it owns the shared `emitQ4_KSuperBlockAux32Core` (auto-covers
q5_K), already has both the block-dot AND the repack Win-A precedent, and its oracle +
micro harness already exist (`q4_K-repack-oracle.log`, `q4_K-winB-micro.log`,
`kquant-winA-knob-FINDING.md`).

- **Step 1 — add the attr + verifier (op def only, emit unread).** Add
  `OptionalAttr<StrAttr>:$integer_core_lmul` to `GgmlBlockDotQ4KQ8KOp` (+ `...Q4KQ8KAux32Op`),
  the verifier legal-set check `{"mf2","m1","m2"}`, and the attr-name allow-list clause.
  Emit ignores it. **Compiles; byte-exact unchanged** (attr unread = no-op). Run the existing
  q4_K oracle to prove no drift (clean rebuild per the byte-exact-fingerprint memory).

- **Step 2 — thread the knob, default = identity.** In `emitQ4_KSuperBlockAux32Core`,
  read `coreLmul = blockDot.getIntegerCoreLmul().value_or("mf2")` and derive the chain
  `l8 = coreLmul; l16 = (mf2→m1, m1→m2, m2→m4); l32 = (mf2→m2, m1→m4, m2→m8)` (mirror
  `RVVToEmitCBlockQuantLinear.cpp:2160-2162`). Replace the Region-A and Region-C hardcoded
  tokens with `("__riscv_..._i8"+l8).str()` etc. **With unset/`mf2`, the emitted C is
  byte-identical to today.** Compiles; oracle byte-exact.

- **Step 3 — Region A widen (the clean knob).** Drive the unpack `vsetvl_e8m2(32)` and
  `u8m2/i8m2` tokens off `l8`/`l16`, with the unpack `vl` scaled to the wider VLMAX (or
  kept at 32 with the strip-count halved). Order-independent store → **oracle byte-exact at
  every `coreLmul`**. Compiles; oracle at mf2/m1/m2.

- **Step 4 — Region C widen WITH fold-back-to-8 (the gated step).** Widen the MAC strip
  (`vle8 i8<l8>` over 16/32-wide, `vwmul i16<l16>`, `vwmacc i32<l32>` into a wide aux32),
  then **emit the integer fold-back to the canonical 8 lanes** (`aux32_8[l] =
  Σ aux32_wide[l + 8k]`, integer vadd, order-free) BEFORE the existing
  `vfcvt_f_x_v_f32m2(…, 8)`. The fp fold (Region F) STAYS 8-lane, untouched. **Oracle
  byte-exact at every `coreLmul` is the gate** — if it fails, the fold-back grouping is
  wrong (NOT a perf regression — a correctness break). Compiles; oracle at mf2/m1/m2.

- **Step 5 — gearbox stamp + Win-A ablate.** Add a K-quant analogue of the repack
  materialization (`deriveRepackHalfLanes` → a `deriveKQuantCoreLmul(vlenBits)`): VLEN128 →
  default `mf2` (or `m1`); VLEN256 → wider (`m2`). Stamp `setIntegerCoreLmul` on the K-quant
  block-dot ops in a `module.walk` (mirror `RVVRepackStripWidthMaterialization.cpp:120-200`).
  Then micro-ablate.

### Oracle-verify correctness (Step 1-4, the gate)
Per `coreLmul ∈ {unset, mf2, m1, m2}`: clean/forced rebuild (incremental builds are
unreliable per project memory — the ODS `.inc` regenerates and `tcrv-opt` may not relink;
byte-exact fingerprint gate MUST use forced rebuild), then run the existing q4_K block-dot
oracle harness (the `_generic` byte-exact reference, the same one in
`q4_K-repack-oracle-FINDING.md`). **PASS criterion: fp32 `*s` byte-identical to `_generic`
at EVERY LMUL setting.** Any divergence at m1/m2 = the Region-C fold-back grouping bug.

### Win-A micro-ablate (Step 5)
On `ssh rvv` (VLEN128) and `ssh k1` (VLEN256), micro-bench the q4_K block-dot at
`coreLmul = m1 vs m2 vs m4`-equivalent (= base mf2/m1/m2) and knob ON (stamped) vs OFF
(unset default), mirroring the repack Win-A ablation logs (`q4_1-winA-gevm-ablation.log`,
`q8_0-winA-gevm-ablation.log`). **Report kernel-micro and e2e SEPARATELY** (the campaign
finding: compute-bound microbench wins do not transport to memory-bound decode e2e). The
target is the compute-bound K-quants where we currently LOSE to ggml's wider-LMUL fused
block-dots — narrow/close THAT gap; do not over-claim e2e.

---

## 4. RISK + EFFORT (shared-helper map — the effort driver)

### Which K-quants share the emit helper (does q4_K auto-cover the others?)
- **q4_K + q5_K — SHARED.** Both call `emitQ4_KSuperBlockAux32Core` (`:775`); q5_K at
  `:2007` (`hasQh=true`). **Parametrizing the q4_K core auto-covers q5_K — ONE change, TWO
  quants.**
- **q6_K — SEPARATE helper, SAME shape.** `emitQ6_KSuperBlockAux32Core` (`:29`) is the
  half-strip (`cx.half`) twin of the q4_K quarter-strip. Needs the SAME parametrization
  applied to a second helper — mechanical copy of Steps 2-4, low risk (identical 8-lane
  aux32 + fp-fold contract). ~0.5× the q4_K effort.
- **q3_K — SEPARATE inline body, q6_K-shaped.** `emitQ3_KQ8_KBlockDot` (`:2786`) carries a
  PRIVATE copy of the half-strip MAC (`:3247-3301`), not a call to the q6_K helper. Needs the
  knob threaded into its inline body. Same 8-lane contract → same fold-back discipline.
- **q2_K — SEPARATE + DIFFERENT reduction.** `emitQ2_KQ8_KBlockDot` (`:2253`) uses
  `vwmul_vv_i16m2` + `vwredsum_vs_i16m2_i32m1` (per-sub-block horizontal reduce into a
  scalar), NOT the 8-lane aux32 vwmacc. Its integer base is ALREADY `m1`/`i16m2`-wide; the
  knob maps to the `vwredsum` width, and the fold-back discipline does NOT apply (no 8-lane
  aux32 to regroup — the reduce is already a scalar per sub-block). **Highest divergence,
  separate verifier semantics; do LAST or scope out of the first increment.**

### LMUL that breaks the lane math
- **`m4` base is the hard ceiling** — illegal (i32 product → `i32m16`). Verifier rejects.
- **NOT the u8 scale view** (task's hypothesis): the 6-bit scale/min bit-dance is scalar
  u32 `emitc.bitwise` (`:919-1006`), LMUL-free. No register-group constraint there.
- **The real break is the missing fold-back** at m1/m2 in Region C: widening without the
  integer fold-to-8 silently changes the 8 fp partials → byte-exact oracle catches it
  (Step 4 gate). This is documented as THE correctness risk.
- Region A widen is risk-free (order-independent memory store).
- VLEN128 caveat: at VLEN128 a wider base may not increase VLMAX usefully for the small
  per-sub-block strip (8/16 elems) — the Win-A win is expected on `ssh k1` VLEN256, matching
  the prior VLEN-shape-match discipline (a VLEN256-shaped widen can LOSE at VLEN128; report
  per-VLEN-regime, not aggregate).

### Effort estimate
- q4_K (+ q5_K free): attr+verifier + thread + Region-A + Region-C-fold-back + oracle ≈
  **1.5-2 days** (the fold-back is the only novel correctness piece; the threading mirrors
  the proven repack `coreLmul` pattern line-for-line).
- q6_K: **+0.5 day** (second helper, mechanical).
- q3_K: **+0.5 day** (inline copy, same contract).
- q2_K: **+1 day** (different `vwredsum` shape + separate verifier semantics) — scope as a
  follow-on increment, NOT the first.
- Gearbox stamp + Win-A ablation on rvv/k1: **+0.5-1 day**.
- **Total first-increment (q4_K+q5_K, the compute-bound headline pair): ~2-2.5 days.**
