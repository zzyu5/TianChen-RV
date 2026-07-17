# IQ-family wide-LMUL gather / signs64 op-attr — feasibility landscape (2026-06-30)

**VERDICT: verdict-only, NO code change.** All 5 named wa3/G4 targets (iq1_m, iq2_xs, iq2_s,
iq3_xxs, iq3_s) are either **structurally blocked** (no valid VLEN flip exists) or **body-rewrite +
board-gated** (no clean byte-exact additive op-attr extension). The one clean iq2_xxs-replicable
candidate surfaced by the map — **iq1_s** — is OUT of the named set and is itself not "additive" (needs
an ODS attr add + a board to validate). Documented below as a scoped follow-up, not landed.

**BYTE-EXACT GATE = VACUOUS** (no code change this session; nothing to rebuild/diff). HOST-ONLY; no
board touched (rvv + k1 both busy). No perf claim made; no flip banked.

HEAD `107aff46`. Evidence = file:line in `lib/Conversion/RVV/` + `include/.../RVVOps.td`, cross-checked
vs `doc/KERNEL-优化自查表.md` 注10/注25 + memory [[emitter-maturity-vluxei16-widelmul]].

---

## The discriminating axis (what makes an IQ kernel flippable)

The iq2_xxs flip (the SOLVED pattern, `RVVToEmitCGridCodebook.cpp:84-107`) works because TWO facts hold
at once:
1. **Uniform per-sub-block scale** → the 4 sign-groups (32 elements) reduce into ONE `vwredsum` with a
   single trailing scale → the body is a single **32-lane sub-block**, not split.
2. **i64-based grid, 32-lane = 4 i64 entries** → the gather flips `i64m2` (VLEN128, 4 lanes) ↔ `i64m1`
   (VLEN256, 4 lanes). Legal at both ends.

The flip then rides the EXISTING `integer_core_lmul` op-attr (read at `:85`, default `m2`), and the
signs table is **derived from the existing `ksigns` attr** (`:159-181`, "NO op-attr extension") — so the
"signs64 op-attr" from the old没做清单#8 was already dissolved (not an op-attr gap).

A kernel is **blocked** when either fact fails:
- **per-half scales (ls1/ls2)** force a 16-lane half body, and
- **the grid is i64** → SEW=64 with ELEN=64 makes `i64mf2` ILLEGAL (min LMUL = SEW/ELEN = m1). A 16-lane
  half = 2 i64 entries = `i64m1` at any VLEN, and it cannot narrow (fractional i64 illegal) nor widen
  (merging two halves would share one scale, which the format forbids). → **no valid flip exists.**

---

## Per-kernel classification (the 5 named targets)

| kernel | grid SEW | scale | body | ODS `integer_core_lmul` | class |
|---|---|---|---|---|---|
| iq1_m   | i64 (u64 grid) | **per-half ls1/ls2** | 16-lane half, `i64m1` | YES (dormant) | **STRUCTURAL** |
| iq2_xs  | i64 (u64 grid) | **per-half ls1/ls2** | 16-lane half, `i64m1` | NO | **STRUCTURAL** |
| iq2_s   | i64 (u64 grid) | **per-half ls1/ls2** | 16-lane half, `i64m1` | NO | **STRUCTURAL** |
| iq3_xxs | **i32** (u32 grid) | uniform per-sub-block | per-group-of-8, scalar sign-fold | NO | **body-rewrite + board** |
| iq3_s   | **i32** (u32 grid) | uniform per-sub-block | per-group-of-8, explicit signs region | NO | **body-rewrite + board** |

### iq1_m — STRUCTURALLY BLOCKED
`emitIQ1MQ8KBlockDot` (`RVVToEmitCTernaryBinary.cpp:645`). Already revectorized to a `vluxei16_v_i64m1`
wide gather, but **pinned m1** (`:689` `coreLmul="m1"`, hardcoded). Code comment is explicit
(`:698-707`): the grid dot "must split into TWO halves … because the halves carry DIFFERENT scales
(ls1 vs ls2)" and "a half is ALWAYS 2 u64 = 16 bytes = i64m1, independent of VLEN." This is the
per-half-scale + i64-min-m1 double-block. **No valid flip.**
- **Map trap to flag:** the IQ1M ODS op **declares** `integer_core_lmul` (RVVOps.td def@8078, attr=YES),
  but the emitter never reads it and the structure makes it unusable. The presence of the attr does
  **NOT** mean iq1_m is flip-ready — it was stamped speculatively and is **dormant**. A future agent must
  not read "attr=YES" as "flippable."

### iq2_xs — STRUCTURALLY BLOCKED
`emitIQ2XSQ8KBlockDot` (`RVVToEmitCGridCodebook.cpp:1962`). Hardcoded `wideLmul="m1"`, `i64m1`
(`:2016-2019`); comment `:2013-2014`: "PER-HALF collapse (not 32-lane like iq2_xxs) is forced by the two
distinct per-half scales ls1/ls2." Signs already derived from `ksigns` (`:2075-2079`, no op-attr gap).
**No valid flip** (same i64-min-m1 + per-half block as iq1_m). Matches 注25.

### iq2_s — STRUCTURALLY BLOCKED
`emitIQ2SQ8KBlockDot` (`RVVToEmitCGridCodebook.cpp:2706` region). `numGroupsPerHalf=2`, `halfLanes=16`,
`i64m1` (`:2646-2654`): "2 entries = 16 i8 = ONE half = 2 groups, sharing the half's ls1/ls2 scale …
forced by the two distinct per-half scales ls1/ls2." Signs are an explicit 2048-byte region
(`tcrv_iq2s_signs256`, `:2714`), already gathered. **No valid flip.** Confirms 注25.

### iq3_xxs — body-rewrite + board (NOT a clean additive op-attr extension)
`emitIQ3XXSQ8KBlockDot` (`RVVToEmitCGridCodebook.cpp:663`). Structurally the BEST flip candidate of the
five: **uniform** per-sub-block scale (`int ls = 2*(aux32>>28)+1`, single scale for all 4 groups,
`:1157-1171`) and **i32 grid** (u32[256], `:718-721`) — and i32 CAN go fractional (`i32mf2` legal,
min LMUL = 32/64 = mf2). So a flip is reachable in principle.
**But the current body is NOT in wide-collapse form.** It is the OLD per-sign-group-of-8 path
(`gridOf4Group`, `:913-1053`): a 2-entry `vluxei16_v_i32m1` grid gather (8 lanes used) +
**scalar-broadcast `ksigns`/`kmask` sign-fold** (`vmv/vand/vmsne/vneg/vmerge`, `:992-1031`), KEPT at i32m1
(comment `:907-908` "here KEPT at i32m1"). Enabling a flip therefore requires a **body revectorization
that changes the emit at the default LMUL too** → NOT byte-exact vs the current emit, → needs its own
byte-exact gate vs an independent scalar oracle AND board validation. This is the same multi-step
emitter-maturity work the iq2_xxs revectorization itself was (a prior, separate landing), not an
additive attr read.
**Concrete steps the revectorization needs (so it is scoped, not handwaved):**
1. Collapse the 4 sign-groups → ONE 32-lane sub-block body (single i32 grid gather of 8 entries → i8,
   single reduce), mirroring iq2_xxs's `:399-587` collapse.
2. Replace the scalar-broadcast `ksigns`/`kmask` sign-fold with a **gathered signs plane** — derivable
   from the EXISTING `ksigns` attr exactly as iq2_xxs derives `tcrv_iq2xxs_signs64` (`:161-181`); iq3
   already reuses the same `ksigns_iq2xs` table (`:771-785`).
3. Wire `integer_core_lmul` (read attr, derive `wideLmul`/`idxLmul`) for the i32 chain (`mf2 ↔ m1`),
   and add the IQ3XXS ODS attr + verifier (RVVOps.td def@7688 currently attr=NO).
4. Byte-exact gate vs scalar oracle + a discriminating VLEN-flip lit; THEN board to confirm
   non-regression. Perf is board-pending by construction — no host-side win possible.

### iq3_s — body-rewrite + board (same class as iq3_xxs)
`emitIQ3SQ8KBlockDot` (`RVVToEmitCGridCodebook.cpp:1343`). Same i32 grid (u32[512], `:1385-1387`), same
OLD per-group-of-8 form, `coreLmul="m1"`/`wideLmul="m2"` hardcoded (`:1378-1379`). Delta vs iq3_xxs: no
`ksigns` plane — signs are an **explicit memory region** (`:1437`, "carries no ksigns plane; the signs are
an explicit memory region"). So step 2 above becomes "restructure the explicit signs region into the
32-lane gather" rather than "derive signs64 from ksigns," but the rest is identical: body revectorization
(non-byte-exact) → attr wiring → board. Same class.

---

## Adjacent finding — iq1_s is the structural twin of iq2_xxs (scoped follow-up, NOT landed)

`emitIQ1SQ8KBlockDot` (`RVVToEmitCTernaryBinary.cpp:29`) is ALREADY in the 32-lane wide-collapse form
(`:73-83`: `vluxei16_v_i64m2` gathers 4 grid u64 entries → i8m2 → vwmul i16m4 → **ONE** vwredsum per
sub-block) and has a **uniform** per-sub-block scale — i.e. it satisfies BOTH flip facts and is the exact
pre-flip state iq2_xxs was in. But it is **pinned `i64m2`** (`:81-83`, `479-519` hardcoded; does NOT read
`getIntegerCoreLmul`).

This is the ONE genuinely iq2_xxs-replicable flip the map reveals. **It was deliberately NOT landed
this session**, for honest reasons:
- It is **out of the 5 named wa3/G4 targets** (the task lists iq1_s as "DONE" = revectorized; the flip is
  a separate maturity step).
- It is **not actually "additive"**: the IQ1S ODS op does **NOT** declare `integer_core_lmul`
  (RVVOps.td def@7946, attr=NO) — unlike iq2_xxs which already carried it. So the change is ODS attr +
  verifier (I7 fail-closed) + emitter wiring + a flip lit, i.e. a multi-file structural change, not a
  one-line attr read.
- The **m1 path cannot be validated host-side** (no board). Landing unvalidated flip wiring
  board-pending replicates the over-optimism pattern the campaign keeps correcting
  ([[kernel-wins-dont-transplant-to-e2e]], [[emitter-maturity-vluxei16-widelmul]]). The iq2_xxs flip
  itself landed VLEN256-m1 board-PENDING and was only sealed later on k1 — so this follow-up is
  precedented but must wait for a board.
- **Ceiling is parity, not beat** (note 注10/注25: VLEN256 ggml `_vl256` already uses the m1 shape; the
  flip is regression-removal toward parity, never a headline beat).

**Recommended next bounded step (when a board frees up):** wire the iq1_s flip identically to iq2_xxs
(ODS `integer_core_lmul` add + verifier; emitter reads attr, default `m2`, derives `wideLmul`/`idxLmul`
for the i64 chain `m2↔m1`; byte-exact-at-default so existing fixtures unchanged; discriminating VLEN-flip
lit), then board-validate VLEN256 m1 byte-exact + non-regression. Same recipe applies to iq3_xxs/iq3_s
AFTER their body revectorization (above).

---

## Verdict / what wide-LMUL IQ work actually remains

- **iq1_m / iq2_xs / iq2_s** = STRUCTURALLY BLOCKED (per-half scale + i64-min-m1). **Nothing to do** —
  no valid flip exists; this is a real, useful "stop" finding. (iq1_m's dormant `integer_core_lmul` attr
  is a map trap, not flip-readiness.)
- **iq3_xxs / iq3_s** = reachable in principle (i32 grid, uniform scale) but require a **non-byte-exact
  body revectorization** (per-group-of-8 → 32-lane collapse + gathered signs) + ODS attr + **board**
  validation. Bounded but **multi-step, NOT host-only-byte-exact**.
- **iq1_s** = the one clean iq2_xxs-twin flip, but out-of-scope here, needs an ODS attr add, and needs a
  **board** to validate. Documented as the precise next bounded step, deliberately not forced.

Net: **none of the 5 named targets admits a clean host-only byte-exact additive extension this session.**
The honest deliverable is this structural map — matching the task's anticipated "verdict IS the
deliverable (like G3)" branch. No code changed; byte-exact gate vacuous; no perf claim; no flip banked.
