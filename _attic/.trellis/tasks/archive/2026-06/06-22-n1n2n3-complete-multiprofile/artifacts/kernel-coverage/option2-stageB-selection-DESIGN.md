# Option-2 STAGE B — capability/resource-aware SELECTION lowering (DESIGN, 2026-06-24)

READ-ONLY design. No `lib/` edits. Stage B of "option-2": turn the stage-A pass's
repack ERROR STUB into a real, **fact-driven selection** inside
`RVVLowerQuantContraction` — the step that makes "**the COMPILER itself selects
repack-vs-block-dot from facts**" actually true. Companion to
`option2-stageA-abstract-op-DESIGN.md` (the op + identity pass) and
`path-selection-tune-DESIGN.md` (the measured win/loss matrix + static prior).

---

## 0. AS-BUILT STAGE-A STATE (reconfirmed by inspection — scope B against THIS)

Stage A is being built in the WORKING TREE right now (uncommitted: `RVVOps.td`,
`Passes.td`, `Passes.h`, `RVVDialectWideningOps.cpp`). Reconfirmed against the as-built,
NOT just the design doc:

| Stage-A artifact | As-built? | file:line | Notes that change B |
|---|---|---|---|
| `GgmlQuantContractionOp` ODS | **YES** | `RVVOps.td:3975` | Schema = design: 6 operands (incl. `column_count` nc always; nc is `index`), attrs `quant/scale_model/m_regime/qk/weight_layout/…/min_vlen`. **No x16 facts.** |
| Verifier | **YES** | `RVVDialectWideningOps.cpp:1058` | **Pins `quant=="q4_0"` ONLY** (q8_0/q4_K rejected today); `m_regime ∈ {decode,prefill}`; `weight_layout=="plain"` fail-closed; plain byte pins (18/34/2/16); nc is `index`. |
| Pass DECL `RVVLowerQuantContraction` | **YES** | `Passes.td:876` | `Pass<…,"::mlir::ModuleOp">`. **Carries NO `march`/`isaVectorHints` options** (unlike `MaterializeRVVQ40Schedule` which uses `kRVVBlockDotScheduleOptions`, `Passes.td:387/441`). **← B GAP: VLEN must reach the pass; see §3.** |
| Pass IMPL `.cpp` | **NO** | — | `createRVVLowerQuantContractionPass()` declared (`Passes.h:85`) but unimplemented. Stage A's A2 writes the identity body; **B wires `selectContractionAlgorithm` into THIS file.** |
| Fixture | **NO** | — | `test/Conversion/RVV/rvv-to-emitc-quant-contraction-q4-0-block-dot.mlir` not authored yet. |

**Three things B must reconfirm / flag when A lands:**
1. **`min_vlen` source.** The op carries an OPTIONAL advisory `min_vlen` attr, but the pass
   has NO `march` option. B's `selectContractionAlgorithm` needs a VLEN. **Resolution (§3):
   ADD `kRVVBlockDotScheduleOptions` (`march`+`isa-vector-hints`) to the pass DECL and derive
   VLEN via `plugin::rvv::deriveMinimumVLEN(march, isaVectorHints)` — the SAME authority every
   other capability-gated pass uses. The op's `min_vlen` attr is a secondary cross-check /
   override, never the sole input** (an op authored without `min_vlen` must still select
   correctly from the pass's `-march`).
2. **Quant allow-list is `q4_0`-only today.** The static-prior table (§1) names q8_0 / q4_K
   cells. Those rows are **latent / unreachable until the verifier widens its quant allow-list**
   (a one-line A/B change). B's selector is written for the full table; B's *lit-tested* cells
   are the q4_0 ones (repack vs K1-decline) **plus** q8_0/q4_K rows guarded behind a verifier
   widen — flag as a dependency, do not silently assume q8_0 fixtures verify today.
3. **`m_regime` is an ATTR on the abstract op** (`decode`/`prefill`), so M-regime reaches the
   selector directly from `op.getMRegime()` — NO op-identity inference needed at this layer
   (the abstract op is singular; F2's gemv/gemm op-identity split only applies to the *concrete*
   targets downstream).

---

## 1. `selectContractionAlgorithm` — signature + STATIC PRIOR (branch-free, fact-driven)

A **pure free function** (unit-testable, no MLIR types), placed in a NEW
`lib/Conversion/RVV/RVVContractionPathSelection.{h,cpp}` (beside the pass that calls it; the
path-selection doc put it under `lib/Plugin/RVV/` for the producer framing — in the stage-A
IN-COMPILER framing it belongs in `lib/Conversion/RVV/` next to `RVVLowerQuantContraction`).

```cpp
namespace tianchenrv::transforms {

enum class ContractionAlgorithm { Repack, BlockDot };

// The committed WHAT axes the abstract op carries, lifted to a plain enum so the
// selector is a pure function of capability facts (NOT string-matching on op kind).
enum class QuantType { Q4_0, Q8_0, Q4_K /* widen as the verifier widens */ };
enum class MRegime   { Decode /*M==1, GEVM*/, Prefill /*M>>1, GEMM*/ };

struct ContractionSelection {
  ContractionAlgorithm algorithm;
  llvm::StringRef reason;        // a STABLE audit token (see §3), e.g.
                                 // "repack-kept-q4_0-vlen128-decode" |
                                 // "block-dot-decline-vlen256-decode-k1-loss" |
                                 // "block-dot-decline-q8_0-lean-fallback" | ...
};

// Capability-fact-driven, branch-free over a small static table. NO op-kind string match.
ContractionSelection selectContractionAlgorithm(
    QuantType quant,
    MRegime   mRegime,
    std::int64_t minVLEN);   // deriveMinimumVLEN(march, isaVectorHints) [F3]
} // namespace
```

### Static prior (the measured win/loss matrix → 3 derived capability facts)
Prefer **Repack** iff ALL THREE capability facts hold; else **BlockDot** (= decline = match
ggml native). This is the in-compiler encoding of `path-selection-tune-DESIGN.md` Q3, with the
measured matrix collapsed to per-quant facts (not magic constants):

```
ggmlHandTunedVLENNativeExists(quant, minVLEN):  // fact 1
    q4_K @ VLEN128 -> TRUE  (ggml_vec_dot_q4_K_q8_K_vl128 hand asm) -> repack loses -> decline
    q4_0          -> FALSE
    q8_0          -> FALSE
blockDotIsHeavy(quant):                          // fact 2
    q4_0 -> TRUE   (nibble + per-block vredsum + scattered reads; repack out-streams it)
    q8_0 -> FALSE  (LEAN: 1 vwredsum/block; nothing for repack to remove -> decline)
    q4_K -> (n/a; fact 1 already declines)
vlenOrPrefillFavorsRepack(minVLEN, mRegime):     // fact 3
    (minVLEN == 128) OR (mRegime == Prefill)   // q4_0 @ K1 VLEN256 decode 0.74x LOSS -> FALSE -> decline

selectRepack = !ggmlHandTunedVLENNativeExists && blockDotIsHeavy && vlenOrPrefillFavorsRepack
```

Resulting selection table (the cells the design must prove):

| (quant, VLEN, M-regime) | facts | algorithm | reason token |
|---|---|---|---|
| q4_0, 128, decode | T·T·T | **Repack** | `repack-kept-q4_0-vlen128-decode` |
| q4_0, 128, prefill | T·T·T | **Repack** | `repack-kept-q4_0-prefill` |
| q4_0, 256(K1), decode | T·T·**F** | **BlockDot** | `block-dot-decline-q4_0-vlen256-decode-k1-loss` |
| q8_0, 128, decode | T·**F** | **BlockDot** | `block-dot-decline-q8_0-lean-fallback` |
| q4_K, 128, decode | **F**·… | **BlockDot** | `block-dot-decline-q4_K-vlen-native-exists` |

**Branch-free / I1 / I3 clean:** the function is a pure 3-fact AND; the compiler commits
exactly ONE concrete op per request; there is no runtime `if` in emitted core code. No
op-kind string-matching: the selector reads the op's *committed WHAT attrs* (`quant`,
`m_regime`) + the *derived capability fact* (`minVLEN`), never a `kind=` string.

**Latent-mispick honesty (carry forward from Q3):** fact 3 keeps Repack for ANY `prefill`,
so **q4_0 @ K1-VLEN256 *prefill* is auto-kept though only the *decode* cell was measured.**
Flag as a known latent mispick; do not claim it validated. (Measurement-override hook is
DEFERRED — see §7 "what B does NOT add"; the static prior is sufficient for B.)

---

## 2. THE CRUX — repack-branch / weight-packing boundary (RESOLVED: Option (i))

**The hard fact.** The abstract op carries **PLAIN** weights (`weight_layout="plain"`,
`weight_block_stride=18`, `const uint8_t *` plain `block_q4_0`). The concrete repack target
`GgmlRepackGemvQ40Q80Op` (`RVVOps.td:4266`, verifier `RVVDialectWideningOps.cpp:1718`) **requires
pre-interleaved `block_q4_0x16` bytes**: `weight_block_stride=288`, `weight_interleave=16`,
`weight_quant_byte_offset=32`, `half_lanes∈{8,16}`. Those x16 bytes **do not exist** at this
layer — compiler-driven plain→x16 weight materialization is **deferred to stage C** (the F4
coupling). The repack verifier would also reject anything that does not bind genuine x16
operands. So: **B cannot MATERIALIZE a valid `GgmlRepackGemvQ40Q80Op` without C.**

**RESOLUTION — Option (i): B fully lowers DECLINE cases + emits SELECTION/AUDIT for repack
cases, but DEFERS repack-op materialization to C.** Precisely, the Stage-B pass body does:

```
selection = selectContractionAlgorithm(quant, mRegime, deriveMinimumVLEN(march, hints));
switch (selection.algorithm) {
  case BlockDot:                       // DECLINE — fully lowered, byte-exact, no C needed
    create<GgmlBlockDotQ40Q80Op>(weight_base, activation_base, output,
                                 element_count, vl)        // DROP nc (§1.3 stage-A)
        with attrs verbatim (kind="ggml_q4_0_q8_0_block_dot", scale_model, qk=32,
        weight_block_stride=18, activation_block_stride=34, quant_byte_offset=2,
        activation_high_byte_offset=16);                   // == today, byte-identical
    stamp audit attrs on the new op (§3);
    erase abstract op;
    break;

  case Repack:                         // SELECTED-repack, weights-NOT-yet-packed
    // B does NOT create GgmlRepackGemvQ40Q80Op (it would need x16 bytes that C supplies).
    // B emits the byte-exact block-dot lowering EXACTLY as the DECLINE case ABOVE,
    // PLUS an audit attr recording that REPACK WAS SELECTED but materialization is deferred:
    create<GgmlBlockDotQ40Q80Op>(... same as BlockDot case ...)
        + stamp audit attrs:  contraction_algorithm = "repack"
                              path_selection_reason  = selection.reason
                              path_materialization   = "deferred-stage-c"   (§3)
    erase abstract op;
    break;
}
```

**Why this is the cleanest boundary:**
- **(a) Byte-exact-safe.** EVERY case — decline AND repack-selected — emits the SAME
  byte-identical `GgmlBlockDotQ40Q80Op` body that today's hand-authored IR emits. The repack
  *selection* changes only **audit attributes** (which are EmitC-inert metadata, NOT runtime
  behavior). So the emitted C is byte-identical to today on EVERY path → **zero e2e regression,
  the clean fingerprints `f810ce6b` (block-dot) / `cb04b219` (full) stay intact** on a forced
  rebuild. No silently-wrong repack path can exist because no repack op is emitted from plain
  bytes.
- **(b) Lit-testable WITHOUT hardware or weight-packing.** The selection + audit is observable
  in `tcrv-opt` purely from the audit attr (and an IR-dump CHECK). No x16 bytes, no `ssh rvv`,
  no model repack needed to prove the SELECTOR fires correctly. The repack *materialization* is
  what needs C/hardware; the *selection* does not.
- **(c) Honest.** The audit attr `path_materialization="deferred-stage-c"` makes it explicit
  that the repack DECISION is real but its EFFECT is not yet realized — no over-claim that B
  "runs the repack kernel." B's honest claim: **"the compiler now SELECTS repack-vs-block-dot
  from capability facts in-compiler; B realizes the BlockDot/decline selections fully and audits
  the Repack selection pending C's weight materialization."** This is exactly the
  selection-correctness frame the path-selection doc mandates (NEVER a Win-B).

**What B emits, stated exactly:**
- **block-dot-SELECTED (decline) case** → `GgmlBlockDotQ40Q80Op` (4 operands, nc dropped, plain
  stride-18 attrs verbatim) + audit `contraction_algorithm="block-dot"`,
  `path_selection_reason=<token>`, `path_materialization="realized"`. **Byte-identical C to
  today.**
- **repack-SELECTED case** → the IDENTICAL `GgmlBlockDotQ40Q80Op` body + audit
  `contraction_algorithm="repack"`, `path_selection_reason=<token>`,
  `path_materialization="deferred-stage-c"`. **Byte-identical C to today** (audit attrs are
  metadata; they do not reach the emitter's value-producing path — see §3 for the inert-carrier
  argument). The repack op is materialized by C, not B.

*(Note vs the stage-A doc's repack ERROR STUB: B REPLACES the `emitError("repack lowering is
stage C")` stub with the Option-(i) "select-repack → audited block-dot body, materialization
deferred" branch. The error stub becomes reachable-and-handled instead of fail-closed.)*

---

## 3. The AUDIT field — where, schema, how consumed

**Primary carrier = an ATTRIBUTE the Stage-B pass STAMPS on the lowered concrete op at the
point of decision.** NOT the `low_precision_resource` mirror — confirmed by inspection that the
mirror is the WRONG carrier here:
- `appendRVVLowPrecisionStableResourceCompilerFactMetadata` (`RVVEmitCRouteMetadata.cpp:108`) is
  gated on `selection.hasSelection` and mirrors a **within-algorithm resource SHAPE** selection
  (`RVVLowPrecisionStableResourceCompilerFacts`, `RVVEmitCRouteProvider.h:289` — SEW/LMUL/
  packing/unroll fields; **NO algorithm-path field**, confirmed). It does **NOT fire on the plain
  block-dot vec_dot path** (grep: it is invoked only from `RVVEmitCRouteMetadata.cpp` for the
  packed-i4 primitive selection). So extending that mirror would NOT surface the path decision on
  the block-dot output B emits. `route_family_plan` (`:214`) is a within-algorithm family id, not
  a repack-vs-block-dot discriminator.

**Schema (audit attrs stamped on the concrete `GgmlBlockDotQ40Q80Op` by B):**
```
tcrv_rvv.contraction_algorithm   = "repack" | "block-dot"        (StrAttr)
tcrv_rvv.path_selection_reason   = <stable reason token, §1>     (StrAttr)
tcrv_rvv.path_materialization    = "realized" | "deferred-stage-c" (StrAttr)
```
**Inert-carrier requirement (load-bearing for byte-exactness):** these are stamped under a
`tcrv_rvv.*` namespace **the block-dot verifier's attribute allow-list MUST be widened to
accept** (today the allow-list at `RVVDialectWideningOps.cpp:910` rejects unknown attrs
fail-closed; B adds these three to `isAllowedBlockDotAttr`), AND the EmitC emitter must IGNORE
them (they carry no SEW/LMUL/policy, so `isForbiddenDataflowParameterAttr` does not trip; they
are pure provenance, like the existing `tcrv_rvv.q4_0_schedule.*` audit trail the
`MaterializeRVVQ40Schedule` pass already stamps additively). **Verify in B3 that the emitter
emits byte-identical C with these attrs present** (the schedule-audit precedent says additive
provenance attrs are emitter-inert; confirm with the fingerprint gate).

**Secondary (OPTIONAL) surface:** if a future emit also wants the decision in the artifact
metadata stream alongside `low_precision_resource.*`, add a free-standing
`tcrv_rvv.contraction_path.{algorithm,reason,materialization}` triple to the route-metadata
output — but this is NOT required for B's validation (the op attr is sufficient and lit-CHECKable).

**How consumed:** (1) **lit FileCheck** reads the attr off the `tcrv-opt` IR dump (B's primary
validation). (2) A future producer / e2e harness reads `path_materialization="deferred-stage-c"`
to know which cells C must still materialize. (3) Honesty audit: the attr is the in-IR proof that
the SELECTION happened in-compiler (not in a build-side table), which is the N3 claim B supports.

---

## 4. PRECISE ORDERED BUILD STEPS (each COMPILES)

> Precondition: stage-A A1–A4 landed (op + verifier + identity pass IMPL + byte-exact fixture).
> If A2's pass IMPL is not yet written, B1–B2 fold into writing it directly (the identity body
> IS the BlockDot case; B adds the selector dispatch around it).

**B1 — `selectContractionAlgorithm` (PURE, unit-testable, wired to NOTHING).** Add
`lib/Conversion/RVV/RVVContractionPathSelection.{h,cpp}` with the §1 enum + free function + the
3 capability-fact helpers (`ggmlHandTunedVLENNativeExists`, `blockDotIsHeavy`,
`vlenOrPrefillFavorsRepack`). Add a `unittests/` (or lit-via-a-test-harness) check of all 5
table rows. **COMPILES; wired to nothing; zero behavior change.**

**B2 — WIRE the selector into the pass; decline cases lower to block-dot EXACTLY as today
(byte-exact).** In `RVVLowerQuantContractionPass::runOnOperation` (the A2 IMPL): (i) ADD
`kRVVBlockDotScheduleOptions` (`march`+`isa-vector-hints`) to the pass DECL at `Passes.td:876`
(§0 GAP fix) so `march` is a pass member; (ii) per walked `GgmlQuantContractionOp`, compute
`minVLEN = deriveMinimumVLEN(march, isaVectorHints)`, lift `quant`/`m_regime` attrs to the
enums, call `selectContractionAlgorithm`; (iii) for `BlockDot` AND (for now) `Repack`, emit the
IDENTICAL `GgmlBlockDotQ40Q80Op` body (§2) — i.e. with the static table defaulting q4_0@128 to
Repack but the Repack branch still emitting the block-dot body, the emitted C is byte-identical
to today on ALL cells. **COMPILES; byte-exact (fingerprints unchanged); the SELECTOR runs but
both branches emit the same body.**

**B3 — AUDIT field.** Widen `isAllowedBlockDotAttr` (`RVVDialectWideningOps.cpp:910`) to accept
`tcrv_rvv.contraction_algorithm` / `…path_selection_reason` / `…path_materialization`; stamp them
on the lowered op in B2's branches (§3). **Re-run the fingerprint gate to PROVE the three
provenance attrs are emitter-inert (C byte-identical with them present).** **COMPILES; byte-exact;
the decision is now in-IR and lit-CHECKable.**

**B4 — REPACK-selected handling per the §2 boundary (the honest deferral).** Make the `Repack`
branch stamp `contraction_algorithm="repack"` + `path_materialization="deferred-stage-c"` (vs
the `BlockDot` branch's `"block-dot"`/`"realized"`), still emitting the block-dot body (§2). This
is where B's selector becomes OBSERVABLE: q4_0@128 now audits `repack` (deferred), the declined
cells audit `block-dot` (realized). **COMPILES; byte-exact** (still the same emitted body
everywhere; only audit attrs differ by cell). **This is the first point B's output DIFFERS from
stage-A's pure-identity — but only in inert audit metadata, NOT in emitted C.**

*(Order rationale: B1 pure → B2 wire identity-preserving → B3 audit channel → B4 differentiate
the audit by cell. Every step keeps the emitted C byte-identical; the only thing that changes
across B is which audit token each cell carries. The actual e2e-moving change — emitting a real
repack op + repacking weights — is stage C.)*

---

## 5. VALIDATION — lit (NO hardware needed for B)

New fixtures under `test/Conversion/RVV/`, run `tcrv-opt %s --tcrv-rvv-lower-quant-contraction
--tcrv-rvv-lower-to-emitc | FileCheck %s` with the relevant `--march`:

**(a) q4_0@128 selects repack (audited).** Author the abstract op (`quant="q4_0"`,
`m_regime="decode"`) with `--march=rv64gcv` (→ minVLEN 128). CHECK the lowered op carries
`tcrv_rvv.contraction_algorithm = "repack"`, `path_materialization = "deferred-stage-c"`,
`path_selection_reason = "repack-kept-q4_0-vlen128-decode"`. CHECK the emitted body is still the
block-dot `tcrv_emitc_ggml_vec_dot_q4_0_q8_0` fn (materialization deferred).

**(b) q4_0@K1-VLEN256-decode / q8_0@128 / q4_K@128 select block-dot, byte-identical to today.**
- q4_0 decode with `--march=rv64gcv_zvl256b` (→ minVLEN 256) → CHECK `contraction_algorithm =
  "block-dot"`, reason `…vlen256-decode-k1-loss`, `path_materialization="realized"`; emitted C
  **byte-identical** to `rvv-to-emitc-q4-0-q8-0-block-dot.mlir`'s output.
- q8_0@128 / q4_K@128 (guarded behind the §0-note verifier quant-widen; if the verifier still
  pins q4_0-only when B lands, these two rows are **table-unit-tested in B1 but lit-deferred**
  until the widen — flag explicitly, do not author a fixture the verifier rejects).

**(c) the audit field carries the reason.** Every fixture CHECKs all three audit attrs by exact
token. A negative CHECK-NOT guards that NO `tcrv_rvv.repack_gemv_q4_0_q8_0` op appears (B never
materializes one).

**Byte-exact guard (project MEMORY build-incremental-unreliable):** FORCED/clean rebuild. After
B4, the existing `rvv-to-emitc-q4-0-q8-0-block-dot.mlir` is unchanged and the fingerprints
`f810ce6b` / `cb04b219` are intact (B adds ZERO bytes on the existing hand-authored path; the new
pass only fires on `quant_contraction` ops, and even when it selects "repack" it emits the
block-dot body + inert audit attrs).

**Explicitly NOT in B:** any `ssh rvv` / `ssh k1` micro or e2e run. Those validate C's
materialization (the cell that actually moves e2e — q4_0@K1 0.74x→tie). B is pure lit.

---

## 6. EXPLICIT B / C BOUNDARY (what C must add)

| | Stage B (this doc) | Stage C |
|---|---|---|
| **Adds** | `selectContractionAlgorithm` + static prior (in-compiler, branch-free); the Repack branch as an AUDITED-but-DEFERRED block-dot body; the 3 audit attrs + verifier allow-list widen; `march` option on the pass | compiler-driven **plain→x16 weight materialization** (the F4 coupling): synthesize the `block_q4_0x16` bytes (stride 288, interleave 16, quant offset 32, the `^0x88` repack bias) so the Repack branch can create a VALID `GgmlRepackGemvQ40Q80Op` (`RVVOps.td:4266`); + the real PRODUCER that authors the abstract op in the live pipeline (replacing the hand-authored fixture / `cmake_inject.py`/`transform_repack.py` harness); + `half_lanes` derivation (`deriveRepackHalfLanes`, `RVVRepackStripWidthMaterialization.cpp:78`) wired into the new repack op |
| **Behavior change** | NONE in emitted C (audit attrs only); first IN-IR decision divergence | **Moves e2e** (different kernel runs + model repacked to x16) |

**What C must SUPPLY for the repack op (the exact deferred operands/facts, from the GEMV
verifier `RVVDialectWideningOps.cpp:1718`):**
- weight operand re-bound to the **x16** `const uint8_t *` (interleaved `block_q4_0x16`), NOT the
  plain stride-18 base the abstract op carried;
- attrs: `kind="ggml_repack_gemv_q4_0_q8_0"`, `weight_block_stride=288`, `weight_interleave=16`,
  `weight_quant_byte_offset=32`, `activation_block_stride=34`, `activation_quant_byte_offset=2`,
  `half_lanes∈{8,16}` (VLEN-derived), optional `integer_core_lmul` (`mf2` RVV1.0 / `m1` RVV0.7);
- the nc/`column_count` operand B carried through (the abstract op's `index` nc → the repack op's
  `column_count`) — **this is WHY stage A carries nc always; B preserves it into the audit/defer
  marker so C can wire it without re-deriving.**

C is the dominant effort (the F4 weight-packing coupling) and the ONLY stage that moves e2e.

---

## 7. WHAT B DOES NOT ADD (scope fence)

- **Measurement-override hook.** The path-selection doc's `RepackPathMeasurement` "measured >
  static" override is DEFERRED. B's static prior is sufficient; the override is a later refinement
  (and needs a measurement record plumbed in). Flag, don't build.
- **q8_0 / q4_K verifier widen.** B's TABLE covers them; B's verifier-reachable LIT covers q4_0.
  Widening the abstract op's quant allow-list (`RVVDialectWideningOps.cpp:1113`, the `q4_0`-only
  pin) is a small A/B-adjacent change but is its OWN step — flag as a dependency for the q8_0/q4_K
  lit rows.
- **The real repack op + weight packing.** Stage C (§6). B never emits `GgmlRepackGemvQ40Q80Op`.
- **`low_precision_resource` mirror extension.** Not the carrier (§3); skip it.

---

## 8. RISK / EFFORT / BLOCKERS

- **RISK (audit-attr emitter-inertness).** The byte-exact claim hinges on the 3 audit attrs being
  emitter-inert (C byte-identical with them present). **Mitigation:** the `MaterializeRVVQ40Schedule`
  pass ALREADY stamps additive `tcrv_rvv.q4_0_schedule.*` provenance attrs that are emitter-inert
  (precedent); B3 RE-RUNS the fingerprint gate to PROVE it for the new attrs before relying on it.
  If any attr trips the emitter, fall back to surfacing the decision ONLY in the secondary
  route-metadata stream (§3) and keep the op attr-free.
- **RISK (selection diverging the C silently).** Resolved by Option (i): Repack-selected emits the
  SAME block-dot body as decline, so NO cell can silently emit a wrong/unmaterialized repack. The
  decision is honest metadata, not a half-built kernel.
- **BLOCKER (deferred, not B's):** the repack MATERIALIZATION needs C's plain→x16 (F4). B
  side-steps it entirely by Option (i) — B is unblocked.
- **GAP (must reconfirm when A lands):** (1) add `march` option to the pass DECL (§0/B2); (2) the
  `min_vlen` attr is advisory, NOT the VLEN source — derive from `-march`; (3) the q4_0-only
  verifier quant pin gates the q8_0/q4_K lit rows.
- **NON-blocker (good news):** both selector inputs are compile-time facts — VLEN via
  `deriveMinimumVLEN(march, hints)` (F3), M-regime via the abstract op's `m_regime` ATTR (no
  runtime M, no op-identity inference at this layer).
- **EFFORT:** B1 (pure selector + unit test) ~½ session. B2 (wire + march option, identity-
  preserving) ~½ session. B3 (audit attrs + allow-list widen + fingerprint re-prove) small. B4
  (differentiate audit by cell) small. **Total B ~1 focused session, all lit in `tcrv-opt`, NO
  hardware.** (C is the multi-session weight-packing bulk that moves e2e.)
- **HONESTY GUARD (project MEMORY winc-structural-null / kernel-wins-dont-transplant):** B claims
  ONLY selection-correctness expressed IN-COMPILER (the decision moved from a build-side table into
  a capability-fact-driven pass). B emits byte-identical C → NO perf claim, NO Win-B. The q4_0@K1
  e2e correction lands in C (regression removal, not a new win). Do not over-claim B.

---

## 9. CRITICAL FILES (file:line)
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:3975` `GgmlQuantContractionOp` (as-built; nc
  always, plain facts, `min_vlen` advisory) — the input B's pass consumes.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:4266` `GgmlRepackGemvQ40Q80Op` (288/16/32/
  half_lanes) — the C target; B records its deferred operands, never creates it.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:1058` abstract verifier (q4_0-only pin at the
  quant check; `weight_layout="plain"` fail-closed) — **flag the quant-widen dependency.**
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:910` `isAllowedBlockDotAttr` — **B3 widens for the
  3 audit attrs**; `:1718` repack-gemv verifier — the exact x16 facts C must supply.
- `include/TianChenRV/Transforms/Passes.td:876` `RVVLowerQuantContraction` DECL (as-built; **NO
  march option — B2 adds `kRVVBlockDotScheduleOptions`**); `:387` the options defvar to reuse.
- `include/TianChenRV/Transforms/Passes.h:85` `createRVVLowerQuantContractionPass()` decl (IMPL
  not yet written — B writes the selector dispatch into it).
- `lib/Plugin/RVV/RVVCapabilityProfile.cpp:253` `deriveMinimumVLEN(march, isaVectorHints)` — the
  VLEN authority B's selector reads (NOT the op's `min_vlen` attr alone).
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteMetadata.cpp:108` `appendRVVLowPrecisionStableResource…` —
  gated `hasSelection`, within-algorithm shape mirror, **NO path field, NOT on the block-dot path
  → confirmed the WRONG audit carrier; B stamps an op attr instead.**
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h:289`
  `RVVLowPrecisionStableResourceCompilerFacts` — confirmed no algorithm-path field.
- `lib/Plugin/RVV/RVVQ40ScheduleMaterialization.cpp:54` — the 64-line ModuleOp-pass shape (incl.
  the `march`/`isaVectorHints` option threading) B's pass IMPL clones.
- `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp:78` `deriveRepackHalfLanes` — the
  half_lanes derivation C wires into the new repack op (NOT B).
- `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot.mlir:34` — the byte-exact ORACLE; B's
  decline + repack-deferred fixtures must emit byte-identical C to it.
