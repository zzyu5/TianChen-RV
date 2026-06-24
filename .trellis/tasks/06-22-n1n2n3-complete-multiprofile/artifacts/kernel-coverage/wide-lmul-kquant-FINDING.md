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
