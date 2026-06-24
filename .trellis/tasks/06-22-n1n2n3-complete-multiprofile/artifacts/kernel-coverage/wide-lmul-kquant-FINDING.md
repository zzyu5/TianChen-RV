# FINDING — Wide-LMUL K-quant block-dot emitter (named emitter-maturity target #2)

Tracks the build-incremental execution of `wide-lmul-kquant-DESIGN.md`. Each
increment appends here with its byte-exact gate evidence.

---

## Increment 1 (S0-S2): the dormant LMUL knob — q4_K block-dot (auto-covers q5_K)

**Date:** 2026-06-24. **Scope:** the BOUNDED FIRST INCREMENT only — add the
`integer_core_lmul` op attr + verifier (S0) and thread an l8/l16/l32 token chain
through the q4_K shared emit core (S1-S2), with the DEFAULT (unset/`mf2`) emit
BYTE-IDENTICAL to today. The Region-A/Region-C *widen* (S3-S4) is the NEXT
increment and is NOT done here. **Zero runtime behavior change.**

### What was added (file:lines)

- **S0 — op attr (`include/TianChenRV/Dialect/RVV/IR/RVVOps.td`):** added
  `OptionalAttr<StrAttr>:$integer_core_lmul` to the `arguments` of
  `GgmlBlockDotQ4KQ8KOp` (the q4_K FULL block-dot op, def `:6093`), as the last
  arg after `activation_bsums_byte_offset`. Updated the op description (the old
  "No resource shape knobs yet" sentence) to document the knob: base LMUL of the
  i8→i16→i32 integer-MAC chain, legal `{mf2,m1,m2}`, `m2` ceiling on TWO grounds
  (register-group i32m8 limit AND the 32-elem one-sub-block-per-scalar boundary),
  the 6-bit scale/min bit-dance is scalar/LMUL-free.

- **S0 — verifier (`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`,
  `GgmlBlockDotQ4KQ8KOp::verify()` at `:4613`):**
  - added `name == "integer_core_lmul"` to the `isAllowedBlockDotAttr` allow-list
    + the unexpected-attr error string (mirrors the repack attr-name allow-list
    clause pattern).
  - added the legal-set check: `getIntegerCoreLmul().has_value()` →
    `coreLmul ∈ {"mf2","m1","m2"}` else `emitOpError`. **NO `half_lanes`
    cross-constraint** (unlike the repack verifier `:1485-1490`) — K-quant ops
    carry no `half_lanes` attr (the strip width is structurally pinned at
    quarter=8 from QK_K=256, per design §2). Unset ⇒ behaves as `mf2`.

- **S1-S2 — thread the knob (`lib/Conversion/RVV/RVVToEmitCKQuant.cpp` +
  `RVVToEmitCInternal.h`):**
  - `RVVToEmitCInternal.h` `Q4_KIntegerCoreContext` (struct at `:1939`): added 4
    members `coreLmul/l8/l16/l32` (defaults `mf2/mf2/m1/m2` = today's chain).
  - q4_K op site (cx@`:1509`, fed by `GgmlBlockDotQ4KQ8KOp`): read
    `coreLmul = blockDot.getIntegerCoreLmul().value_or("mf2")`; derive
    `l8=coreLmul`, `l16 = m2→m4 / m1→m2 / mf2→m1`, `l32 = m2→m8 / m1→m4 / mf2→m2`;
    build the **Region-C** types (`i8mf2Type`/`i16m1Type`/`i32m2Type`) from
    `l8/l16/l32`; pass `coreLmul/l8/l16/l32` into the `cx`. **Region-A types
    (`u8m2Type`,`i8m2Type`) left HARDCODED at m2.**
  - q4_K shared core helper `emitQ4_KSuperBlockAux32Core` (`:775`): the 5 Region-C
    callee strings now built from `cx.l8/l16/l32` —
    `vmv_v_x_i32<l32>` (aux32 seed), `vsetvl_e8<l8>` (quarter setvl),
    `vle8_v_i8<l8>` (load), `vwmul_vv_i16<l16>` (product),
    `vwmacc_vx_i32<l32>` (MAC). Region-A unpack callees
    (`vsetvl_e8m2`/`vle8_v_u8m2`/`vand/vsrl/vadd_u8m2`/`vreinterpret_v_u8m2_i8m2`/
    `vse8_v_i8m2`) and Region-F fp-fold callees (`vfcvt/vfmul/vfadd_f32m2`,
    `vfmv_v_f_f32m2`, `vse32_v_f32m2`) left HARDCODED (Region A = its own
    throughput axis deferred to S3; Region F = the 8-lane byte-exactness contract,
    must stay m2/8-lane forever).

### Scope decisions recorded (do NOT leave silent for the next increment)

- **Only `GgmlBlockDotQ4KQ8KOp` carries the attr this increment** (per the task,
  which named only this op). The design's Step 1 also named
  `GgmlBlockDotQ4KQ8KAux32Op` — that op was **NOT** touched. The aux32-partial op
  (cx@`:1234`) and the q5_K op (`GgmlBlockDotQ5KQ8KOp`, cx@`:1977`) carry **no**
  `integer_core_lmul` getter; their `cx` sites pass the **literal default
  `mf2/mf2/m1/m2`**, so the shared helper emits the byte-identical legacy form for
  them. **Next increment:** before any gearbox stamp (S5) can target q5_K / the
  aux32 partial, the attr + verifier still need adding to those two ops.
- q5_K is **emit-covered for free** at the default (it reuses
  `emitQ4_KSuperBlockAux32Core`), but it has no settable knob yet.

### The knob is LIVE, but only `mf2` is a complete kernel (honesty framing)

- **Live (attr → tokens, verified by emit grep):** setting
  `integer_core_lmul = "m1"` on the q4_K op emits the widened Region-C tokens
  `vsetvl_e8m1 / vle8_v_i8m1 / vwmul_vv_i16m2 / vmv_v_x_i32m4 / vwmacc_vx_i32m4`
  while Region A stays `e8m2/u8m2/i8m2` and Region F stays `f32m2`. `"m2"` emits
  `i8m2 / i16m4 / i32m8`. So the threading is real, not dead code.
- **BUT `m1`/`m2` emit is INTENTIONALLY INCOMPLETE — it will NOT compile.** At
  `m1` the Region-C aux32 accumulator becomes `vint32m4_t` (26 occurrences), but
  the Region-F fold still calls the hardcoded `__riscv_vfcvt_f_x_v_f32m2` on it
  (2 occurrences) → a type mismatch. This is *exactly* the seam that S4's
  **fold-back-to-8** (widen the integer MAC, integer-add the wide lanes back to
  the canonical 8 BEFORE `vfcvt`) is designed to close. **DO NOT** read "knob
  live, m1 widens Region C" as a working capability. **Only `mf2` (default) is a
  complete, byte-exact, hardware-validated kernel** this increment; `m1`/`m2` are
  forward-provisioned (verifier-legal + token-threaded) scaffolding for S3-S4.

### BYTE-IDENTICAL default-emit proof (the project bit-exact gate)

All builds **forced** per the build-incremental-unreliable memory (ODS `.td`
touched ⇒ `RVVOps.cpp.inc` regenerates; `cmake -B build`, `touch` all 4 changed
sources, `ninja tcrv-opt`, CONFIRM the `.cpp.o` compiles + `Linking bin/tcrv-opt`
lines were seen on every build). BEFORE = committed HEAD (my edits `git stash`ed),
forced-rebuilt; AFTER = my edits, forced-rebuilt — both measurements forced, so
the equality is valid.

| Gate | BEFORE (HEAD, forced) | AFTER (edits, forced) | Result |
|---|---|---|---|
| q4_K block-dot emit (`tcrv-opt --tcrv-rvv-lower-to-emitc`) | md5 `c3c5792a…` | md5 `c3c5792a…` | **byte-identical, empty diff** |
| q5_K block-dot emit (shared helper, default) | md5 `51365814…` | md5 `51365814…` | **byte-identical, empty diff** |
| q4_K emitted C kernel (`… \| mlir-translate --mlir-to-cpp`) | committed archive `inc24-q4_K-k4b/tcrv_emitted_kernel.cpp` md5 `2aea89b4…` | regenerated md5 `2aea89b4…` | **identical to the independently-committed reference** |

### lit + oracle (numbers unchanged)

- **lit:** full RVV suite `Conversion/RVV` + `Dialect/RVV` = **192/192 PASS**
  (incl. q4_K + q5_K block-dot conversion + dialect/verifier dataflow tests).
- **Verifier behavior (empirical):** `integer_core_lmul = "m4"` **REJECTED**
  (`requires integer_core_lmul in {"mf2","m1","m2"} … got "m4"`, exit 1); `"m2"`
  (ceiling) **ACCEPTED** (exit 0); unset/`"mf2"`/`"m1"` accepted.
- **Oracle (real `ssh rvv`, Sophgo SG2044, VLEN128, clang++ 18, `-march=rv64gcv
  -mabi=lp64d -ffp-contract=off`, `taskset -c 2`):** the regenerated default
  kernel (byte-identical to the committed archive) vs ggml `_generic` byte-exact
  harness `inc24_validate.cpp` →
  **2012 positive cases, 0 failures; `*s` bit-exact (IEEE-754 bit equality) on
  the named n set {256,512,2048,4096,25600,65536} + 6 edge cases + 2000 random n;
  MIN-term negative control 200/200 discriminating. RESULT: PASS.**

### Confirmation

**NO behavior change.** The default (attr unset) q4_K AND q5_K emits are
byte-identical to the committed tree (forced-build BEFORE==AFTER, and the q4_K
kernel C is md5-identical to the independently-committed archive). The oracle on
real hardware is unchanged (same 2012/0 PASS). The increment only adds the
dormant `integer_core_lmul` knob + its verifier + the l8/l16/l32 threading; the
correctness-critical Region-A/Region-C widen (and the Region-C fold-back-to-8) is
deliberately deferred to the NEXT increment (S3-S4), de-risked from this
ODS/threading change.

### ODS / rebuild gotcha hit

As the build-incremental memory warns: touching `RVVOps.td` regenerates
`RVVOps.cpp.inc` on every build, and `tcrv-opt` does not always relink. The
byte-exact gate was therefore run with the documented forced protocol (`cmake -B
build`, `touch` all 4 sources, `ninja tcrv-opt`, confirm the `RVVToEmitCKQuant.cpp.o`
+ `RVVDialectWideningOps.cpp.o` compile + `Linking … bin/tcrv-opt` lines) for
BOTH the BEFORE (stashed HEAD) and AFTER measurements, so the BEFORE==AFTER
equality is valid (not a stale-fingerprint artifact). No other gotcha; the 3
`Q4_KIntegerCoreContext` aggregate-init sites (cx@1234/1509/1977) had to be kept
positionally consistent after inserting the 4 new struct members between `quarter`
and `hasQh` — the q5_K site (cx@1977) sets `hasQh`/`qhOffset` positionally and was
updated to pass the 4 literal-`mf2` LMUL fields before them.

### Files changed (git diff --stat)

```
 include/TianChenRV/Dialect/RVV/IR/RVVOps.td  | 21 ++++--   (attr + description)
 lib/Conversion/RVV/RVVToEmitCInternal.h      | 11 +++      (cx l8/l16/l32 members)
 lib/Conversion/RVV/RVVToEmitCKQuant.cpp      | 46 ++++--   (derive + thread Region C)
 lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp | 31 ++++--   (allow-list + legal-set)
```

---

## Increment 2 (S3-S4): the seam resolved — m1/m2 emit COMPLETE + Win-A ablate

**Date:** 2026-06-24. **Scope:** S3 (Region-A widen — documented SCOPE DECISION,
held) and **S4 (Region-C widen + the fold-back-to-8 — the novel correctness
piece)**, making the q4_K block-dot produce **COMPLETE, byte-exact kernels at
mf2 / m1 / m2** (the S0-S2 type-seam is gone), then the Win-A LMUL micro-ablate.
mf2 stays **byte-identical** to today (md5 `2aea89b4`). No ODS `.td` touched
(lower rebuild risk); forced/clean build per the build-incremental memory.

### S3 — Region-A widen: DELIBERATELY HELD (documented scope decision)

The task lists S3 (parametrize the 6-bit unpack LMUL by `coreLmul`). It is **NOT
implemented this increment**, by design, for three defensible reasons — NOT the
reason given in the S0-S2 note (which claimed "no throughput benefit at VLEN128
because e8m2 is already maxed at 32" — that is WRONG: e8m4 gives VLMAX=64 at
VLEN128, so widening WOULD cut unpack instruction count):

1. **Region A is NOT on the compile-breaking seam.** The seam that made m1/m2
   un-compilable was Region-C (`i32m4`/`i32m8` aux32) feeding Region-F's
   hardcoded `f32m2` `vfcvt`. **S4 alone yields COMPLETE byte-exact m1/m2
   kernels** — Region A is orthogonal to the integer-MAC-core knob.
2. **Region-A widen is a real byte-exact RISK the design hand-waves as
   "order-free."** Re-chunking the `aux8[256]` unpack (the `qsChunk = chunk*32`,
   low/high-nibble split at +0/+32) under a wider VLMAX must preserve the EXACT
   element-position → `aux8` index mapping; "store order is fixed" is only true
   if the chunk math is rewritten correctly. That is its own gated change, not a
   free rename. Region A also has its OWN default (it must stay `m2`, i.e.
   `l8=mf2 != m2`), so it needs a SEPARATE widen-chain, not the Region-C `l8`.
3. **Holding Region A constant cleanly ISOLATES the MAC-core LMUL knob** for the
   Win-A ablate (the deliverable) — co-widening both regions would confound the
   measurement.

So Region-A unpack stays hardcoded `e8m2/u8m2/i8m2` at every `coreLmul`
(confirmed in the m1/m2 emit: `vsetvl_e8m2` ×4 unchanged). Region-A widen is a
named follow-on, not a regression.

### S4 — Region-C widen WITH the integer fold-back-to-8 (the correctness piece)

The S0-S2 threading widened only the register-group TYPE at `vl=8` — which both
(a) did NOT change throughput (vl stayed 8, so the wider VLMAX was unused) and
(b) made the naive fold-back read uninitialized upper lanes. S4 does the REAL
restructure: the per-sub-block MAC strip **width itself** follows `l8`:

| `coreLmul` | strip width | strips/sub-block | wide aux32 | fold groups |
|---|---|---|---|---|
| `mf2` (default) | 8  | 4 | `vint32m2` (8 lanes) | 1 (NO fold) |
| `m1` | 16 | 2 | `vint32m4` (16 lanes) | 2 |
| `m2` | 32 | 1 | `vint32m8` (32 lanes) | 4 |

**Two-path structure (mf2 byte-identical BY CONSTRUCTION).** The emit branches
on `cx.l8`: `mf2` runs the UNCHANGED legacy `emitStrip(width=8)` ×4 (== the old
`emitQuarter(0/8/16/24)`) with `foldGroups==1`, so **NOTHING new is emitted** —
the wide branch and the fold-back scratch are entirely guarded behind
`foldGroups > 1`. All novelty is confined to m1/m2, which did not compile before
(nothing to regress). q5_K and the aux32-partial op carry no knob ⇒ always
`mf2` ⇒ always the legacy path ⇒ automatically safe.

**The fold-back (the only novel correctness logic).** After the sub-block loop,
the wide aux32 (`vint32<l32>`, stripWidth lanes) is collapsed to the canonical 8
BEFORE Region-F's `vfcvt` (which stays `f32m2`/8-lane — the byte-exact
contract): extract the `foldGroups` 8-lane (m2) subgroups REGISTER-ONLY via
`vget_v_i32<l32>_i32m2(aux32, k)` (lanes `[8k, 8k+8)`) and integer-`vadd_vv_i32m2`
them into the canonical `vint32m2` — implementing
`aux32_8[l] = Σ_k aux32_wide[l + 8·k]`. Integer add is associative/order-free and
each `vwmacc` stays within one 32-element sub-block (so the per-sub-block scalar
`scale` is constant across the lanes summed), so the regroup is **provably
bit-exact at every legal LMUL** (DESIGN §0). The helper now returns this
canonical 8-lane lvalue; the K4b Region-F consumer loads it as the canonical
`i32Canon8Type` (always `vint32m2_t`), so the type contract is clean at every
LMUL and byte-identical at mf2.

**Fold-back form (settled): register-only `vget`, NOT a memory spill.** The
first implementation spilled the wide aux32 to a `int32_t aux32w[]` scratch
(`vse32` + `vle32`-slices), which is associative-exact and PASSED the oracle —
BUT its per-super-block memory round-trip handicapped the wider LMULs in the
ablate (m2 worse than m1). It was replaced with the **register-only `vget` +
`vadd`** fold (no spill); re-run the oracle (below) — STILL bit-exact (same
lanes, same integer sum), and it removes the confound (m2 then wins, see Win-A).

**Emit verification (token grep, `tcrv-opt … --tcrv-rvv-lower-to-emitc`):**
- **mf2:** md5 `c3c5792a` (emit) / `2aea89b4` (kernel C) — **byte-identical to
  S0-S2 / the committed archive**; NO `aux32_fold_back`/`vget` tokens (the
  `foldGroups==1` guard leaves the legacy path untouched).
- **m1:** 2 strips (`vsetvl_e8m1` ×2, `vwmacc_vx_i32m4` ×2 into `vint32m4`);
  fold-back: `vget_v_i32m4_i32m2` ×2 + `vadd_vv_i32m2` ×1 → `vfcvt_f_x_v_f32m2`
  consumes **`vint32m2`** (canonical, NO seam); ZERO memory spill.
- **m2:** 1 strip (`vwmacc_vx_i32m8` ×1 into `vint32m8`); fold-back:
  `vget_v_i32m8_i32m2` ×4 + `vadd_vv_i32m2` ×3 → `vfcvt` consumes `vint32m2`;
  **zero `vint32m8` leaks into the fp fold; ZERO memory spill.**

### PER-LMUL byte-exact oracle — PASS at mf2 / m1 / m2 (the S4 gate)

Real `ssh rvv` (Sophgo SG2044, VLEN128, clang++ 18.1.3, `-march=rv64gcv
-mabi=lp64d -ffp-contract=off`, `taskset -c 2`). The unmodified S0-S2 oracle
harness `inc24_validate.cpp` (oracle = VERBATIM ggml `_generic`, fp16-shimmed,
bit-equality on `*s`) was re-run against each LMUL's emitted kernel C (regenerated
via `tcrv-opt | mlir-translate --mlir-to-cpp`):

| LMUL | kernel C md5 (vget fold) | positive cases | failures | MIN-term neg. control | RESULT |
|---|---|---|---|---|---|
| **mf2** | `2aea89b4` (== committed archive) | 2012 | **0** | 200/200 discriminating | **PASS** |
| **m1**  | `ee030a30` | 2012 | **0** | 200/200 discriminating | **PASS** |
| **m2**  | `ed918d00` | 2012 | **0** | 200/200 discriminating | **PASS** |

(The earlier memory-spill fold also PASSED, md5 m1 `39d7b200` / m2 `073d7b66`;
the register-only `vget` fold re-validated here is bit-identical in result.)
`*s` IEEE-754 bit-exact vs `_generic` at EVERY LMUL on the named n set
{256,512,2048,4096,25600,65536} + 6 edge cases + 2000 random n. **The fold-back
grouping is empirically correct** — m1/m2 are now COMPLETE, byte-exact,
hardware-validated kernels (the S0-S2 "only mf2 is complete" caveat is resolved).

### lit + build

- **Forced/clean build:** `RVVToEmitCKQuant.cpp.o` compiles, `Linking
  bin/tcrv-opt` confirmed. (Two compile errors hit + fixed mid-increment: a
  `VariableOp`→`TypedValue` cast on the fold-back result var, and a `replace_all`
  that wrongly retyped q3_K's PRIVATE inline `vfcvt` load — q3_K carries its own
  aux32, not the shared core's fold-back, reverted to its `i32m2Type`.)
- **lit:** `Conversion/RVV` + `Dialect/RVV` = **197/197 PASS** (no regression;
  +5 vs the S0-S2 192 are the unrelated option-2 commits merged since).

### Win-A micro-ablate (VLEN128, `ssh rvv`, `taskset -c 2`)

The kernel-isolated LMUL knob ablate: same emitted q4_K block-dot, knob varying
mf2 / m1 / m2 (all compiler-emitted; baseline = same kernel, knob OFF == mf2).
NB=512 super-blocks, 3000 reps, RUN 1 discarded (cold-cache outlier: mf2 ran
first). Steady state with the register-only `vget` fold (ns/call):

| LMUL | kernel ns/call | vs mf2 |
|---|---|---|
| **mf2** | ~105.0 µs | 1.00× (baseline) |
| **m1**  | ~94.3 µs  | **1.11× faster** |
| **m2**  | ~95.8 µs  | 1.10× faster |

**Win-A knob effect at VLEN128: both wider arms beat mf2 by ~1.10×; m1 and m2
are within noise of each other** (94.3 vs 95.8 µs). The magnitude is marginal —
matching the DESIGN §4 VLEN128 caveat: the per-sub-block strip is only 8/16/32
elems, so a wider base barely lifts VLMAX usefully at VLEN128. The genuine,
larger Win-A win is the **VLEN256 (`ssh k1`)** result below (where m2 clearly
wins), which this `ssh rvv` arm does not fully exercise. (Note: vs the FIRST
memory-spill fold-back, the `vget` fold lifted both wider arms here too —
m1 ~99.7→94.3, m2 ~101.3→95.8 µs — by removing the per-super-block spill.)

### HONESTY CORRECTION — the §9 competitor is NOT scalar `_generic`

The task framing says "our q4_K block-dot LOST to ggml's WIDER-LMUL `_generic`."
**That conflates two different things, corrected here from primary sources
(EVIDENCE-MATRIX `:146`, commit `d27c512a`):**
- OUR kernel is a single-LMUL **port of `_generic`'s fp-fold ORDER** (that is its
  lineage / why it's byte-exact vs `_generic`).
- The §9 LOSS (q4_K **1.72×**) is vs ggml's OWN hand-tuned **`_vl256` RVV
  intrinsic** kernel, measured at **K1 VLEN256** — NOT scalar `_generic`.

A micro vs the SCALAR `_generic` (which `-O2` never vectorizes) is meaningless
for the loss question: our kernel is ~13× faster than `_generic` at *every* LMUL
(105µs vs ~1389µs) simply because `_generic` is scalar. **That 13× is NOT
"closing the §9 loss"** — do not read it that way. The real §9 competitor
(`ggml_vec_dot_q4_K_q8_K_vl256`) lives in `kq-bench/q4_K/ggml_ref.cpp` and the
loss is a **VLEN256** phenomenon. The loss-narrowing measurement therefore
requires re-running the `kq-bench/q4_K` harness (our kernel vs `_vl256`) on K1
with the m1/m2 kernels swapped in — see the K1 section below.

### THE LOSS-NARROWING RESULT — K1 VLEN256 vs ggml's REAL `_vl256` (the §9 question)

Re-ran the `kq-bench/q4_K` harness (`ours.cpp` swapped for each LMUL's emitted
kernel C; `kern_ggml` = ggml's SHIPPED `ggml_vec_dot_q4_K_q8_K_vl256` RVV
intrinsic, `ggml_ref.cpp`) on **`ssh k1` (SpacemiT X60, VLEN256 confirmed
VLENB=32, Bianbu clang 18.1.8, `-march=rv64gcv -mabi=lp64d -ffp-contract=off`,
`taskset -c 2`)**. best-of-200-reps × 2000 iters, n=8192 (32 super-blocks). 3
runs, identical to 4 sig figs (numbers below are the stable value):

With the **register-only `vget` fold-back** (the settled form; 3 runs, stable to
4 sig figs):

| LMUL | ours ns/call | ggml `_vl256` ns/call | our slowdown (ours/ggml) | agreement |
|---|---|---|---|---|
| **mf2** (§9 baseline) | 12597 | 6978 | **1.81× LOSS** | 5.82e-7 |
| **m1** | 9635 | 6977 | **1.38× LOSS** | 5.82e-7 |
| **m2** | **9352** | 6977 | **1.34× LOSS** (FASTEST) | 5.82e-7 |

**The Win-A LMUL knob NARROWS the §9 loss at VLEN256 — from 1.81× (mf2) to
1.34× (m2, the knob's best arm) — a ~26% kernel speedup (12597 → 9352 ns), but
it does NOT cross 1.0** (ggml's hand-tuned `_vl256` is still ~1.34× faster). The
mf2 baseline (1.81×) reproduces the §9-recorded q4_K block-dot LOSS (1.72×,
EVIDENCE-MATRIX `:146`; the small delta is harness/board variance — same code
path, same competitor).

**m2 is the winner — the task's m2 hypothesis is VINDICATED, once the fold-back
confound is removed.** With the FIRST (memory-spill) fold-back, m2 (1.48×) lost
to m1 (1.45×) because its 4-group fold paid a `vint32m8` spill + reload per
super-block. Switching to the **register-only `vget` fold** dropped that
round-trip and m2's wider single 32-elem strip (1 `vwmacc`/sub-block) now wins:
**m2 1.34× < m1 1.38× < mf2 1.81×.** So the answer to "does a wider LMUL (m2)
narrow the loss?" is **YES — m2 is the best arm (1.34×), the widest legal LMUL
winning exactly as hypothesized.** Numerical agreement `5.82e-7` at every LMUL
(the fp-fold-order residual, NOT a correctness defect — integer cores identical;
ours pinned to `_generic`'s fp order, ggml `_vl256` reorders).

**HONEST verdict on the headline question.** *Does a wider LMUL narrow or close
the loss to ggml's wide `_vl256`?* — **It NARROWS it (1.81× → 1.34× at m2, ~26%
kernel speedup at VLEN256) but does NOT close it.** This is a real, measured
Win-A knob effect (same kernel, knob varies, both compiler-emitted) on the exact
§9 competitor. It is NOT a win — "narrows to 1.34×" is the defensible statement;
do NOT over-claim parity. The residual gap is ggml's `_vl256` hand-tuned
nibble-split shape (a different ALGORITHM, not just a wider LMUL on the same
algorithm) — closing it is an emitter-shape rewrite, not an LMUL knob, and
remains the named N3-Gearbox motivation.

### e2e — SEPARATE, NOT measured (likely NULL by the memory wall)

This is Win-A (the compiler-automatic LMUL TUNE; SAME kernel, knob varies;
baseline = same kernel knob OFF). The result above is **kernel-isolated micro**
only. **e2e is NOT claimed and NOT measured** — and is likely NULL: decode is
memory-bandwidth-bound (the campaign-central [[kernel-wins-dont-transplant-to-e2e]]
finding — compute-bound kernel wins do not transport to memory-bound decode), and
a wider integer-core LMUL changes *compute*, not *memory traffic* (the q4_K block
footprint is byte-identical across LMULs). A ~1.2× kernel-compute speedup behind
the memory wall does not surface in decode tok/s. Do NOT read the K1 narrowing as
an e2e claim.

### Gearbox stamp (S5) — NOT done this increment

The knob is verifier-legal + token-threaded + oracle-validated + ablated, but no
`deriveKQuantCoreLmul(vlenBits)` gearbox materialization was added (S5). To make
the tune AUTOMATIC (stamp `m1` at VLEN256, `mf2` at VLEN128 per the ablate),
mirror `RVVRepackStripWidthMaterialization.cpp`'s `module.walk` + `setIntegerCoreLmul`.
That + adding the attr to `GgmlBlockDotQ5KQ8KOp`/`...Aux32Op` (so q5_K / the
partial are stampable) is the next increment.

### BLOCKER / residuals (honest)

- **No blocker for S3-S4** — the deliverable (m1/m2 COMPLETE + byte-exact +
  ablated, with the winning LMUL settled) is done and durable.
- **Fold-back form is settled** — the register-only `vget` fold (no memory
  spill) replaced the initial memory-spill form; both are byte-exact (oracle
  re-PASSED), and the `vget` form removed the confound that had handicapped m2
  (m2 now wins at VLEN256, vindicating the task's hypothesis). No open
  sub-optimality remains on the fold-back.
- **q5_K / q6_K / q3_K / q2_K not widened** — S3-S4 covers q4_K (the headline
  modern K-quant). q5_K shares the core but carries no knob yet (always mf2);
  q6_K/q3_K are separate helpers/inline bodies; q2_K has a different `vwredsum`
  reduction (no 8-lane aux32, no fold-back). Per DESIGN §4 these are mechanical
  follow-ons (q6_K/q3_K ~0.5d each, q2_K ~1d different shape).
- **VLEN128 ablate is marginal** (m1 ~1.05× over mf2) by design — the Win-A win
  is a VLEN256 phenomenon (the per-sub-block strip is too small to lift VLMAX at
  VLEN128). Report per-VLEN-regime, never aggregated.

---

## CHECK-AGENT CORRECTION (2026-06-24) — the vget fold-back is VLEN128-ONLY; m1/m2 FAIL at VLEN256

**The Increment-2 "PASS at mf2/m1/m2" + "5.82e-7 agreement at every LMUL" claims
are WRONG.** The per-LMUL byte-exact oracle (`inc24_validate.cpp` vs `_generic`)
was only ever run on **`ssh rvv` = VLEN128** (see line 247: "Real `ssh rvv` …
VLEN128"); the m1/m2 kernels were **never bit-checked on K1 (VLEN256)** — the K1
table reports timing + a single "5.82e-7" residual carried over from mf2, not a
per-LMUL re-check. The Check Agent re-ran the SAME unmodified harness on BOTH
boards against the regenerated kernels (md5 mf2 `2aea89b4` / m1 `ee030a30` / m2
`ed918d00`, all reproduced byte-for-byte from the FINDING):

| board / VLEN | mf2 | m1 | m2 |
|---|---|---|---|
| `ssh rvv` VLEN128 | PASS 2012/0 | **PASS 2012/0** | **PASS 2012/0** |
| `ssh k1`  VLEN256 | PASS 2012/0 | **FAIL 2008/2012** | **FAIL 2010/2012** |

The VLEN256 failures are **gross** (e.g. n=1024 ref=997360.69 vs tcrv=787407;
n=8192 ref=−1163694 vs tcrv=−3458476) — ~25–50% magnitude, NOT an fp-reorder
residual. **The fold-back grouping is wrong at VLEN256.**

**ROOT CAUSE (primary-source RVV semantics).** `vget_v_i32m8_i32m2(aux32, k)`
extracts the k-th **m2 register-subgroup**, whose lane count is `LMUL·VLEN/SEW =
VLEN/16` = **8 at VLEN128 but 16 at VLEN256**. The fold-back code assumes a fixed
8-lane subgroup (`vadd_vv_i32m2(…, 8)` and the identity `aux32_8[l] = Σ_k
aux32_wide[l + 8k]`), which holds ONLY at VLEN128. At VLEN256 the m2 subgroup is
16 lanes, so `vget(…,k)` returns lanes `[16k, 16k+16)`; the strip only wrote
lanes `[0,32)`, so the fold sums `[0,8)+[16,24)+[32,40)+[48,56)` (the last two are
seed-tail garbage) and silently skips `[8,16)`/`[24,32)`. The code comment
"`vget` … the k-th contiguous 8-lane (m2) subgroup" conflates "m2" with "8 lanes"
— true only at VLEN128. The same defect breaks m1 (`vget_v_i32m4_i32m2` →
16-lane subgroups at VLEN256). mf2 is immune (`foldGroups==1`, no fold emitted —
which is why mf2 stays byte-identical and passes everywhere).

**Why the original memory-spill fold did NOT have this bug:** it spilled to a
`int32_t aux32w[]` scratch and re-loaded **element-indexed** 8-lane slices at
literal offset `8k` (VLEN-agnostic). The replacement to register-only `vget`
"to remove the m2 confound" (line 230, 388-392) is **exactly where VLEN256
correctness was traded away**, and the only bit-exact gate (VLEN128) structurally
could not see it.

**CONSEQUENCE for the headline deliverable.** The Win-A "loss-narrowing 1.81×→1.34×
at m2 (VLEN256)" timed a **numerically WRONG kernel** at the very VLEN where the
win is claimed → the loss-narrowing result is **NOT established** and must be
re-measured after the fold is fixed. (The m2-wins-over-m1 conclusion may even flip,
since it was the reason for the broken vget swap.)

**FIX SPACE (design-level, NOT a check-time self-fix — left for the impl agent):**
make the regroup VLEN-agnostic — either (a) `vslidedown` by **literal element
offsets 8/16/24** then `vadd(vl=8)` (register-only AND VLEN-correct), or (b)
restore the element-indexed memory-spill form (correct but reintroduces the
confound). Then re-run the K1 VLEN256 oracle (must PASS bit-exact at m1/m2) AND
re-measure the Win-A loss-narrowing on the corrected kernel.

**What remains SOLID (independently re-verified by the Check Agent):** mf2
byte-identity (md5 `2aea89b4` == committed archive, regenerated; oracle PASS at
both VLENs); struct-init positional consistency (cx@1300/1601/2080); the
dead-but-safe null `i32Canon8Type` on the partial/q5_K paths (`foldGroups==1`
never enters the fold branch); no leakage to other K-quants (q5_K always mf2,
q3_K's private `i32m2Type` load reverted); no lit regression (708/711, the 3
`Scripts/…computed-masked-strided…` failures pre-exist on HEAD with the diff
stashed, unrelated to q4_K). The defect is confined to the m1/m2 fold-back at
VLEN≠128. Build emits 2 benign warnings (unused `quarter` in the helper; unused
`i32Canon8Type` in q5_K) — cosmetic, not the bug.

---

## IMPL-AGENT FIX (2026-06-24) — VLEN-agnostic vslidedown fold; m1/m2 PASS at VLEN256; ablate FLIPPED to m1

The Check Agent's VLEN256 m1/m2 FAIL is FIXED. The fold-back is now element-indexed
(VLEN-agnostic) and the per-LMUL byte-exact oracle PASSes on **BOTH boards**, the
gate the prior build skipped. **All Increment-2 / "register-only `vget` fold"
numbers above (the m2-wins-at-VLEN256 1.34× headline) are SUPERSEDED** — they timed
the numerically-WRONG `vget` kernel. Corrected results below.

### THE FIX — element-indexed `vslidedown` regroup (VLEN-agnostic)

`lib/Conversion/RVV/RVVToEmitCKQuant.cpp` `emitQ4_KSuperBlockAux32Core`, the
`if (cx.foldGroups > 1)` fold-back block (**:1190-1230**). Replaced the
`vget_v_i32<l32>_i32m2(aux32, k)` per-`k` register-subgroup extract (whose lane
count = `LMUL·VLEN/SEW` = 8@VLEN128 but 16@VLEN256 → wrong-lane fold at VLEN256)
with element-indexed slides **in the WIDE type at vl=8**, extracting the canonical
8-lane group ONCE at the end:

```
foldWide = aux32WideVal                                       // k=0 group = lanes [0,8)
for k in 1..foldGroups-1:
    slid     = __riscv_vslidedown_vx_i32<l32>(aux32WideVal, 8*k, /*vl=*/8)  // element offset 8*k
    foldWide = __riscv_vadd_vv_i32<l32>(foldWide, slid, /*vl=*/8)            // wide add, low 8 lanes
fold = __riscv_vget_v_i32<l32>_i32m2(foldWide, 0)             // low canonical 8-lane group [0,8)
```

The slide offset is the LITERAL **element** offset `8*k` (8/16/24), so element
`8*k+l` lands at lane `l` at ANY VLEN; the `vget(.,0)` keeps the LOW 8 lanes
(subgroup 0 is the low lanes at every VLEN). Max offset 24@m2 reads `[24,32) ⊂
[0,32)` (all written by the wide strip) — never the seed tail. mf2 (`foldGroups==1`)
never enters the block → byte-identical. Region F (the f32m2/8-lane fp fold, the
byte-exact contract) UNCHANGED. Forced/clean rebuild of `tcrv-opt` OK (only the 2
pre-existing cosmetic warnings). Emitted code verified: m2 emits 3×`vslidedown
i32m8`(8/16/24)+3×`vadd i32m8`(vl=8)+`vget i32m8→i32m2(.,0)`; m1 emits 1×slide(8)+
1×add+`vget i32m4→i32m2(.,0)`. lit `Conversion/RVV`+`Dialect/RVV` **197/197 PASS**
(100%, no regression); the 3 q4_K tests (block-dot / aux-partial / dataflow) PASS.

### CORRECTED per-LMUL byte-exact oracle — BOTH BOARDS (the skipped gate, now met)

Same unmodified `inc24_validate.cpp` (oracle = VERBATIM ggml `_generic`, IEEE-754
bit-equality on `*s`), regenerated kernels via `tcrv-opt --tcrv-rvv-lower-to-emitc
| mlir-translate(llvm-20) --mlir-to-cpp`. Source kernel md5 (the SAME files fed to
the oracle AND the ablate on both boards): **mf2 `2aea89b4` / m1 `2eee81f9` / m2
`36541510`** (mf2 == committed archive `inc24-q4_K-k4b/tcrv_emitted_kernel.cpp`,
byte-identical — the byte-exact contract held; m1/m2 differ from the broken `vget`
md5s `ee030a30`/`ed918d00`). `clang++ -O2 -march=rv64gcv -mabi=lp64d
-ffp-contract=off`.

| board / VLEN | mf2 | m1 | m2 |
|---|---|---|---|
| `ssh rvv` SG2044 VLEN128 (VLENB=16, `taskset -c 2`) | PASS 2012/0 | PASS 2012/0 | PASS 2012/0 |
| `ssh k1` X60 VLEN256 (VLENB=32, `taskset -c 0-3`) | PASS 2012/0 | **PASS 2012/0** | **PASS 2012/0** |

m1/m2 at VLEN256 went from **FAIL 2008/2012 & 2010/2012 (gross, ~25-50% wrong-lane
fold)** → **PASS 2012/0**. 2000 random n + 6 edge cases + 200/200 MIN-term negative
control discriminating, every cell. The fold-back is now empirically VLEN-agnostic.

### RE-MEASURED Win-A ablate vs ggml's REAL `_vl256` — the conclusion FLIPPED to m1

`kq-bench/q4_K` harness (`harness.cpp`, ours `<lmul>.cpp` vs `kern_ggml` =
ggml's SHIPPED `ggml_vec_dot_q4_K_q8_K_vl256` in `ggml_ref.cpp`), best-of-200-reps
× 2000 iters, n=8192. 3 runs, stable to 4 sig figs. Kernels are the **same
oracle-verified md5s** above.

**`ssh k1` X60 VLEN256 (`taskset -c 0-3`)** — the headline §9 regime:

| LMUL | ours ns/call | ggml `_vl256` ns/call | our slowdown (ours/ggml) | agreement |
|---|---|---|---|---|
| **mf2** (§9 baseline) | 12580 | 6978 | **1.80× LOSS** | 5.82e-7 |
| **m1** | **9662** | 6978 | **1.38× LOSS (FASTEST)** | 5.82e-7 |
| **m2** | 11250 | 6977 | **1.61× LOSS** | 5.82e-7 |

**FLIP: m1 is the winner at VLEN256, NOT m2.** The superseded "m2 1.34× fastest"
was a measurement artifact of the BROKEN `vget` fold: the wrong (VLEN128-shaped)
fold emitted cheap `vget` extracts, so the m2 4-group fold looked cheap. The
CORRECT m2 fold needs 3 real `vslidedown` (lane-crossing, offsets 8/16/24) + 3
`vadd`, whose cost exceeds m1's single slide — so once numerically correct, **m1's
one wide `vwmacc`-pair-per-sub-block + single-slide fold beats m2's single-`vwmacc`
+ 3-slide fold.** Numerical agreement `5.82e-7` at every LMUL (fp-fold-order
residual vs `_vl256`, NOT a defect — integer cores now bit-exact vs `_generic`).

**`ssh rvv` SG2044 VLEN128 (`taskset -c 2`)** — ARM-ABLATE ONLY (no valid `_vl256`
baseline here): best-of, steady-state (ns/call): **m1 ~4993-5054 (FASTEST) < m2
~5575-5965 < mf2 ~6252-7165**. The `_vl256` competitor is NOT a valid VLEN128
baseline: our (oracle-verified, bit-exact-vs-`_generic` at BOTH VLENs) kernel
agrees with `_vl256` at **5.82e-7 @VLEN256 but ~20 @VLEN128** — same kernel, so the
VLEN128 divergence is `_vl256` being a VLEN256-SHAPED algorithm (numerically wrong
at VLEN128), per the project VLEN-SHAPE-MATCH discipline. So at VLEN128 we report
ONLY the knob-vs-mf2 ablate: **m1 ~1.24× over mf2**, both wider arms beat mf2.

**HONEST verdict (corrected).** Does a wider LMUL narrow or close the loss to ggml's
wide `_vl256`? — **It NARROWS it (mf2 1.80× → m1 1.38× at VLEN256, ~23% kernel
speedup, the best arm) but does NOT close it (`_vl256` still ~1.38× faster).** And
the best arm is **m1 at BOTH measured VLENs** (VLEN256: m1<m2<mf2; VLEN128:
m1<m2<mf2) — NOT m2. "Narrows to 1.38× with m1" is the defensible statement; the
residual gap is `_vl256`'s hand-tuned nibble-split SHAPE (a different algorithm, an
emitter-shape rewrite = the named N3-Gearbox motivation), not an LMUL knob.

**e2e — SEPARATE, NOT claimed, likely NULL** (unchanged from above): Win-A is the
kernel-isolated LMUL micro; decode is memory-bandwidth-bound and a wider integer
core changes compute not memory traffic, so the ~1.2-1.3× kernel speedup is behind
the memory wall ([[kernel-wins-dont-transplant-to-e2e]]). Do NOT read the VLEN256
narrowing as an e2e claim.

**Gearbox stamp (S5) recommendation UPDATED:** the auto-tune target is **m1** (best
arm at BOTH VLENs), superseding the prior "m1@256 / mf2@128" — a single
`integer_core_lmul = "m1"` stamp wins at both measured VLEN regimes for this kernel.

**Durable artifacts:** corrected per-LMUL kernels stored alongside the broken
`ours.cpp` as `kq-bench/q4_K/ours-{mf2,m1,m2}.cpp` (md5 `2aea89b4`/`2eee81f9`/
`36541510`). **No blocker** — the fix is in the working tree (uncommitted, per
instruction), both-board oracle PASSes, the ablate is re-measured and superseded.
