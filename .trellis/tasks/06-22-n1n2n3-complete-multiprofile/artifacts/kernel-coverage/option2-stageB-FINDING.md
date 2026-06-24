# Option-2 STAGE B — capability/resource-aware in-compiler SELECTION (FINDING, 2026-06-24)

Stage B IMPLEMENTED. The compiler now AUTONOMOUSLY SELECTS repack-vs-block-dot
from CAPABILITY FACTS in-compiler and RECORDS the decision in-IR. This is the
in-compiler capability-aware SELECTION MECHANISM (the user's chosen N3 novelty) —
**SELECTION-correctness only**. Emitted C is BYTE-IDENTICAL (block-dot) for ALL
cells; the e2e algorithm switch (weight-packing) is stage C. **NOT a Win-B, NOT a
new perf number, NOT an e2e claim.**

Built against committed stage A (27572edd: abstract `GgmlQuantContractionOp` +
the `RVVLowerQuantContraction` identity pass with a q4_0-decode→block-dot branch
and a stage-C error stub for every other cell).

---

## B1 — the pure fact-driven selector (unit-/lit-testable)

`selectContractionAlgorithm(QuantType, MRegime, int64 minVLEN) -> {algorithm,
reason}`, a PURE free function (no MLIR types):
- decl: `include/TianChenRV/Plugin/RVV/RVVContractionPathSelection.h`
- impl: `lib/Plugin/RVV/RVVContractionPathSelection.cpp` (fn at the bottom; the 3
  fact helpers `ggmlHandTunedVLENNativeExists` / `blockDotIsHeavy` /
  `vlenOrPrefillFavorsRepack` in the anon namespace).

**Capability-fact-driven, branch-free** (a 3-fact AND): prefer Repack iff
`!ggmlHandTunedVLENNativeExists(quant,VLEN) && blockDotIsHeavy(quant) &&
(VLEN==128 || Prefill)`; else BlockDot (= decline = match ggml native). Reads ONLY
the lifted `quant`/`m_regime` enums + the derived `minVLEN` — **NO op-kind string
match, NO ABI-string / family-name branch** (I3/N2 discipline). The 5 static-prior
rows and their stable reason tokens:

| (quant, VLEN, M) | facts | algo | reason token |
|---|---|---|---|
| q4_0, 128, decode | T·T·T | Repack | `repack-kept-q4_0-vlen128-decode` |
| q4_0, *, prefill | T·T·T | Repack | `repack-kept-q4_0-prefill` |
| q4_0, 256, decode | T·T·**F** | BlockDot | `block-dot-decline-q4_0-vlen256-decode-k1-loss` |
| q8_0, 128, decode | T·**F** | BlockDot | `block-dot-decline-q8_0-lean-fallback` |
| q4_K, 128, decode | **F**·… | BlockDot | `block-dot-decline-q4_K-vlen-native-exists` |

**PATH DEVIATION from DESIGN §1** (`lib/Conversion/RVV/`): the selector + pass live
in `lib/Plugin/RVV/`, NOT `lib/Conversion/RVV/`. **Forced by a real CMake dependency
cycle:** the pass must call `deriveMinimumVLEN` (in `TianChenRVRVVPlugin`), but
`TianChenRVRVVEmitCRouteProvider → TianChenRVConversionRVV`, so making
`TianChenRVConversionRVV` link the plugin is a configure-time cycle. The move is the
ESTABLISHED pattern (every `Q*ScheduleMaterialization` pass that calls
`deriveMinimumVLEN` already lives in `lib/Plugin/RVV/`) and is an INTRA-lib call
(zero new link edges). `RVVCapabilityProfile.cpp` is in the same lib. The DESIGN's
"beside the pass that calls it" rationale is preserved; the pass just moved with it.

**No unit-test harness exists** (`unittests/` absent; project validates by lit by
design). The selector is validated by lit, NOT googletest — see B-validation below.
Lit reaches the 3 reachable q4_0 rows (fact-3 BOTH ways + repack/decline BOTH ways).
The q8_0 / q4_K rows are **ENCODED in the selector but UNEXERCISED in B** (facts 1
`ggmlHandTunedVLENNativeExists` and 2 `blockDotIsHeavy` are encoded-but-untested);
the verifier blocks them — see CONFLICT 1 below.

---

## B2 — wire the selector + march option + VLEN source + verifier note

`lib/Plugin/RVV/RVVLowerQuantContraction.cpp` (moved from `lib/Conversion/RVV/`):
per walked `GgmlQuantContractionOp` it lifts `quant`/`m_regime` to the selector
enums, derives `minVLEN = deriveMinimumVLEN(march, isaVectorHints)`, calls
`selectContractionAlgorithm`, and lowers via the Option-(i) block-dot body.

- **march option ADDED** to the pass DECL (`include/TianChenRV/Transforms/Passes.td`
  `RVVLowerQuantContraction`): `let options = kRVVBlockDotScheduleOptions;` (the
  established defvar at `Passes.td:386`, reused) — the GAP the DESIGN §0/B2 flagged.
- **VLEN SOURCE = `deriveMinimumVLEN(march, isaVectorHints)`** — the SAME capability
  authority `MaterializeRVVQ40Schedule` consumes (`RVVCapabilityProfile.cpp:253`),
  **NOT** the op's advisory `min_vlen` attr. Default `-march ""` → `deriveMinimumVLEN
  == 0` → fact 3 false → BlockDot (the honest no-capability behavior; keeps the
  existing default-march emit fixture green).
- **Verifier quant-pin NOT widened** (DESIGN §5b/§7 honored; see CONFLICT 1). The
  q4_0-decode AND q4_0-prefill block-dot cells lower to `GgmlBlockDotQ40Q80Op`
  emitting BYTE-IDENTICAL C to today.

The repack-SELECTED branch does NOT create `GgmlRepackGemvQ40Q80Op` (Option-i);
both branches emit the identical block-dot body.

---

## B3 — audit attrs (emitter-inert) + allow-list widen

The pass stamps three INERT provenance attrs on the lowered op:
```
tcrv_rvv.contraction_algorithm  = "repack" | "block-dot"
tcrv_rvv.path_selection_reason  = <stable token, B1>
tcrv_rvv.path_materialization   = "deferred-stage-c" (repack) | "realized" (block-dot)
```
Allow-list widened: `RVVDialectWideningOps.cpp` `isAllowedBlockDotAttr` (the
`GgmlBlockDotQ40Q80Op::verify` lambda) now admits these three names — exactly like
the precedent `tcrv_rvv.q4_0_schedule.*` resource-provenance trail. Confirmed none
of the three trip `isForbiddenDataflowParameterAttr` (they carry no SEW/LMUL/policy;
`RVVDialect.cpp:2094` matchers don't name them) — pure provenance.

**Emitter-inertness PROVEN** (see byte-exact gate).

## B4 — audit differentiated per cell

The audit reflects the actual per-(quant,VLEN,M) selection: q4_0@128-decode and
q4_0@prefill audit `repack`/`deferred-stage-c`; q4_0@VLEN256-decode and the
default-march cell audit `block-dot`/`realized`. (Stage A's pure identity carried
NO audit; B4 is the first point B's output differs from stage A — but ONLY in inert
audit metadata, never in emitted C.)

---

## BYTE-EXACT GATE — before/after EQUALITY (the stale absolutes are DEAD)

Same-binary 3-march emit through `--tcrv-rvv-lower-quant-contraction[=march=…]
--tcrv-rvv-lower-to-emitc` on the stage-B selection fixture (q4_0 decode):

- repack-selected (`march=rv64gcv`, VLEN128) emit  **==  byte-identical  ==**
  block-dot-selected (`march=rv64gcv_zvl256b`, VLEN256) emit  **==**  default emit.
  → **Option-(i) proven: both branches emit the SAME block-dot C; only the audit
  attr differs.**
- stage-A quant-contraction fixture emit (now carrying audit attrs)  **==
  byte-identical ==**  the hand-authored `rvv-to-emitc-q4-0-q8-0-block-dot.mlir`
  oracle emit. → **B3 emitter-inertness proven: emitted C == today.**
- grep of the emitted C for any audit token / `quant_contraction` / `repack_gemv`
  = **0** (no audit token, no repack op survives into emitted C).

This is the decisive test (the audit attrs are emitter-inert so emitted C is
unchanged on EVERY branch; only the IR audit attr differs). The diff-EMPTY is on
post-`--tcrv-rvv-lower-to-emitc` emitted C (the IR before emitc legitimately differs
— it now carries the audit attrs).

**Methodology note (the literal stash/rebuild was ABANDONED, substitute is provably
equivalent).** The task's git-stash-u → forced-clean-rebuild before/after was
ATTEMPTED but abandoned: the `git mv` of the pass file left an orphan
`.cpp` after the stash that broke the clean cmake reconfigure
(`llvm_check_source_file_list`). The SUBSTITUTED proof is **oracle-equality +
full-lit-green**, which is targeted at the exact claim:
- The "before" reference is REAL and pre-validated: `rvv-to-emitc-q4-0-q8-0-block-dot.mlir`
  is the hand-authored canonical block-dot emit that stage A already gates against;
  `stage-B-pass-output == that oracle` is `before == after` for the ONLY emit fixture
  present in BOTH tree states.
- **The MEMORY `build-incremental-unreliable` stale-tcrv-opt risk is EMPIRICALLY
  REFUTED, not assumed:** the binary demonstrably emits march-DIVERGENT selections
  (VLEN128→repack vs VLEN256→block-dot) AND stamps the NEW audit tokens — a stale
  (pre-stage-B) binary would ignore `-march` entirely and stamp NO audit attrs, so
  the observed divergent+audited output IS proof the linked binary is current. (Build
  log also shows `Linking bin/tcrv-opt` after the two new `.cpp.o` compiles.)

---

## LIT VALIDATION (no hardware)

NEW `test/Conversion/RVV/rvv-lower-quant-contraction-stage-b-selection.mlir`: ONE
q4_0-decode module lowered at THREE march tiers (VLEN128/VLEN256/default) proving
fact-3 discrimination + the audit + `CHECK-NOT` no-repack-op / no-quant_contraction
guarantee at the IR level. PASS.

REWRITTEN `test/Conversion/RVV/rvv-lower-quant-contraction-prefill-stub.mlir`
(CONFLICT 2): q4_0-prefill now selects repack (fact-3 prefill-override, VLEN-blind)
→ block-dot body + `repack`/`deferred-stage-c`/`repack-kept-q4_0-prefill` audit
(was a fail-closed error stub in stage A). PASS.

UNCHANGED `test/Conversion/RVV/rvv-to-emitc-quant-contraction-q4-0-block-dot.mlir`
(default march → block-dot; checks emitted C, unaffected). PASS.

Full suite: **707/710 PASS.** The 3 failures
(`rvv-generated-bundle-abi-e2e-{self-test,…strided-input-widening-dot-reduce-add-dry-run}`)
are **PRE-EXISTING and UNRELATED**: pure Python harness `--self-test`
(`scripts/rvv_generated_bundle_abi_e2e.py`, git-CLEAN/untouched by B) failing on
"fake bundle generation lost product-reduction dequant metadata" — runs no
tcrv-opt, references neither the contraction op nor the audit attrs. The full
block-dot + quant-contraction + autotuner blast-radius family (42 fixtures) is
**42/42 PASS** (verifier widen broke nothing).

Build: clean reconfigure + `ninja tcrv-opt` OK; `RVVContractionPathSelection.cpp.o`
+ `RVVLowerQuantContraction.cpp.o` compile in the plugin lib, `Linking bin/tcrv-opt`
relinks — no dependency cycle.

---

## CONFLICTS vs the task prose (surfaced for the lead, resolved per AUTHORITATIVE design)

1. **"widen the verifier's q4_0-only pin to admit q8_0/q4_K + lit those rows"
   (task prose) is INCOHERENT with the as-built abstract op — NOT done.** The
   abstract `GgmlQuantContractionOp` verifier hard-pins `qk==32 /
   weight_block_stride==18 / activation_block_stride==34 / quant_byte_offset==2 /
   activation_high_byte_offset==16` — all q4_0×q8_0-specific. q8_0 and q4_K are
   DISTINCT concrete ops with DISTINCT schemas (`q8_0_q8_0_block_dot`: stride 34/34,
   qk=32; `q4_k_q8_k_block_dot`: qk=256, stride 144/292, K-quant super-block attrs).
   The single `GgmlBlockDotQ40Q80Op` is q4_0×q8_0 ONLY, so a q8_0/q4_K request
   lowered to it would be SEMANTICALLY WRONG, and widening only the quant string
   would let an incoherent op verify. The AUTHORITATIVE DESIGN §5b/§7 already
   resolved this: q8_0/q4_K are "table-unit-tested in B1 but lit-deferred … do not
   author a fixture the verifier rejects." **Resolution: selector ENCODES the full
   3-quant table; B lit-tests ONLY the reachable q4_0 rows; q8_0/q4_K (facts 1 & 2)
   are encoded-but-unexercised in B.** The verifier quant-widen + the q8_0/q4_K
   lowering targets are a separate downstream step (likely folds into / after C).

2. **prefill-stub fixture flipped from error to selection.** Stage A's
   q4_0-prefill error stub is intentionally GONE in B (DESIGN §2:
   "reachable-and-handled instead of fail-closed"); the dataflow VERIFIER remains
   the fail-closed gate. The fixture was the ONLY consumer of the removed stub error
   string — rewritten to assert the repack-deferred selection.

## HONESTY (mandatory)

Stage B is **SELECTION-CORRECTNESS in-compiler ONLY**. The compiler now
autonomously selects + records the measured-best algorithm from capability facts
(VLEN derived from `-march` + the committed quant/m_regime). **Emitted C is
byte-identical (block-dot) for ALL cells; NO e2e algorithm switch, NO perf number,
NO Win-B, NO e2e claim.** The repack DECISION is real and audited
(`path_materialization="deferred-stage-c"`); its EFFECT (plain→x16 weight
materialization + a real `GgmlRepackGemvQ40Q80Op`) is stage C — the only stage that
moves e2e. **Latent mispick (carry-forward):** fact 3 keeps Repack for ANY prefill,
so q4_0@K1-VLEN256 *prefill* is auto-kept though only the *decode* cell was
measured — a known latent mispick, not a validated cell (the measurement-override
hook is DEFERRED per DESIGN §7). **Reason-token wart:** the no-capability default
(`-march ""` → VLEN 0, q4_0 decode) reuses the `block-dot-decline-q4_0-vlen256-
decode-k1-loss` decline token — the string says "vlen256" though VLEN is 0; it is a
decline either way (correct algorithm), but the reason string is imprecise for the
VLEN<128 non-prefill case. Unexercised in any real profile (no shipped target has
VLEN 0 with full-V); the selection fixture deliberately does NOT CHECK the reason on
the DEFAULT row for this reason. A precise low/no-VLEN decline token is a trivial
later refinement, not a code change B needs.

## FILES
- NEW `include/TianChenRV/Plugin/RVV/RVVContractionPathSelection.h`
- NEW `lib/Plugin/RVV/RVVContractionPathSelection.cpp`
- MOVED+REWRITTEN `lib/Plugin/RVV/RVVLowerQuantContraction.cpp` (from `lib/Conversion/RVV/`)
- `include/TianChenRV/Transforms/Passes.td` (RVVLowerQuantContraction: + `let options = kRVVBlockDotScheduleOptions;` + description)
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (isAllowedBlockDotAttr: + 3 audit attrs)
- `lib/Conversion/RVV/CMakeLists.txt` (− RVVLowerQuantContraction.cpp), `lib/Plugin/RVV/CMakeLists.txt` (+ 2 .cpp)
- NEW `test/Conversion/RVV/rvv-lower-quant-contraction-stage-b-selection.mlir`
- REWRITTEN `test/Conversion/RVV/rvv-lower-quant-contraction-prefill-stub.mlir`
