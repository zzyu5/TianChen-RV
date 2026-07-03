# Research: Six-State Ladder [K-4] — HEAD-verified table + E5 dependency boundary

- **Query**: HEAD-verify the six-state placement of the ~24 block-dot ops + 3 decomposed shapes; determine what E6 can auto-read NOW vs what needs the E5 provenance manifest.
- **Scope**: internal (repo)
- **Snapshot**: HEAD `1bbab882695e812a2d334a8d6f5c7860cc2c93b1`, 2026-07-03. (Prior audit `7185a62b` §2 anchors were re-verified; file line numbers shifted — `RVVToEmitCBlockQuantLinear.cpp` was split, see below.)
- **Authority**: [K-4] six-state ladder (`科研目标总纲v2.md:78-80`); `执行总纲v2.md` §2 (lines 119–136); [L-8] strong-vs-weak (`:48`); 实验总纲 line 27/107 (provenance manifest precedes six-state automation).

---

## 1. The six-state ladder (definition, [K-4])

`absent → emittable (can emit + conversion unit test) → dispatch-wired (production dispatch wired, body hand-written) → constructed-weak (descriptor-parameterized assembly, arithmetic core = a SELECTED hand-written helper) → constructed (body built from pattern-library primitives, STRONG sense) → covered (constructed + [K-5] applicable gates green + regression rack on record)`.

Silicon-sealed (per-board objdump golden) is tracked **separately**, not folded into the ladder.

**Discriminator ([L-8]/[A-4]):** weak vs strong = *is the arithmetic core a selected hand-written helper (weak) or built from typed pattern-library primitives (strong)?*

---

## 2. HEAD-verified six-state table (denominator = 24 block-dot ops + 3 decomposed shapes)

Each kernel takes its highest state. Counts re-verified at HEAD `1bbab882` and match the `7185a62b` §2 audit (**17 / 7 / 3**).

| State | Kernels | Count | HEAD anchor / reason |
|---|---|---|---|
| absent / emittable | — | 0 | all 24 ops at least dispatch-wired |
| **dispatch-wired** (body = hand-written ggml `_generic` monolith) | q2_K q3_K q4_K q5_K q6_K, iq4_xs iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s, tq1_0 tq2_0 (**15 super-block**) + **nvfp4** + **q1_0** (2 flat, own monoliths) | **17** | 15 super-block go through `MonolithicBlockDotRouteFamily::SuperBlock`; **nvfp4** has its own emit method (`emitNvfp4Q8_0BlockDot`, `lib/Conversion/RVV/RVVToEmitCCodebookFp4.cpp:~761`, nvfp4-specific layout, NOT the shared `emitFlatBlockDot`); **q1_0** own method `emitQ1_0Q8_0BlockDot` (`RVVToEmitCBlockQuantLinear.cpp:5951`). Neither q1_0 nor nvfp4 is in the `deriveFlatBlockDotDescriptor` whitelist. |
| **constructed-weak** (descriptor SELECTS decode/fold; arithmetic = hand-written helper via VerbatimOp) | q4_0 q8_0 q4_1 q5_0 q5_1 (**flat-plain**) + **iq4_nl, mxfp4** (flat-codebook) | **7** | flat-plain q4_0/q5_0/q5_1/q4_1/q8_0 are **thin shims → `emitFlatBlockDot`** (`RVVToEmitCBlockQuantLinear.cpp:53,90,127,…`, "Thin shim: … instance of the descriptor-driven emitFlatBlockDot"); iq4_nl + mxfp4 are thin shims → `emitFlatBlockDot` via `CodebookGatherNibble`/`hasCodebook` (`RVVToEmitCCodebookFp4.cpp:33,658`). Descriptor lives in `RVVToEmitCInternal.h` (`deriveFlatBlockDotDescriptor`). |
| **constructed** (typed RVV primitives, STRONG) | decomposed N-operand 3 shapes: **q4_0-nibble / offset-binary N=3 / codebook N=3 LUT** | **3 shapes** | typed primitive construction via the 4 decomposed front doors (`RVVDequantDotSourceFrontDoor.cpp`, `RVVCodebookDotSourceFrontDoor.cpp`, `RVVReductionSourceFrontDoor.cpp`, `Construction/RVVContractionRouteIdentity.cpp`); exported both VLEN128/256, mutation-tested e2e lits. |
| **covered** (strong + [K-5] all green + CI) | 0 strict | **0** | [K-5] not all met: no `.github` CI, ULP bound absent, objdump golden not all-board. |

> **q4_0 dual-path** (执行总纲 §2 note, resolved in coverage-denominator.md §4): q4_0 exists as constructed-weak (flat path, production authority) *and* as constructed/strong (decomposed demo). Under the **separate-keys** denominator policy these are two different roster keys, not one key upgrading.

**Burn-down list (still hand-written, awaiting strong construction; auto-generator does NOT exist):**
1. Super-block / K-quant (15): q2_K q3_K q4_K q5_K q6_K iq4_xs iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s tq1_0 tq2_0 (2048-entry grids, `vluxei16` verbatim).
2. Flat remaining (2): nvfp4, q1_0.
3. Flat half-burned but still weak (7): q4_0/q8_0/q4_1/q5_0/q5_1/iq4_nl/mxfp4 — decode/fold are descriptor-*selected* hand-written C helpers inside `emitFlatBlockDot`.

**Forward ops (B类) six-state** (evidence in coverage-denominator.md §3): rms_norm, softmax, rope, silu, scale = **≥dispatch-wired** (emitter `RVVToEmitCForwardElementwise.cpp` + conversion lits); gelu/add/mul/cpy = **absent**. These are separate denominator keys; their six-state is likewise hand-labeled for E6.

---

## 3. E5-dependency boundary: what E6 auto-reads NOW vs what needs the provenance manifest

实验总纲 line 107 is explicit: **"provenance 清单先于六态自动化"** (otherwise T0 is human-judged and [L-8] is unenforced). PRD lines 129/134 confirm E6's six-state auto-readout is **gated on E5**, and E6 should ship "分母 + 四指标 (六态用现有人工标签起步) + ledger". So the boundary:

### Computable NOW by E6 (from structural facts / existing labels)
- **Denominator** — the roster keys (coverage-denominator.md). Pure enumeration, no provenance needed.
- **`C_dispatch`** — "≥ dispatch-wired". A kernel is ≥dispatch-wired iff it has a production front-door / emitter. This is a **structural fact** readable today: block-dot 24 all wired (single table-driven pass `RVVMonolithicBlockDotSourceFrontDoor.cpp`); forward ops wired iff an emitter+lit exists. E6 can compute this from a committed six-state label table; the labels are mechanically defensible.
- **`C_construct+`** (weak or better) — "≥ constructed-weak". The weak set is **structurally identifiable today**: membership in the `deriveFlatBlockDotDescriptor` whitelist (7 flat) is a code fact. Auto-derivable in principle; for MVP read from the label table.
- **`C_attr`** — leveled (^CT / load / ^RT). Today: ^CT partial (in-IR selection attrs only, no JSONL — E4 territory), load = 0, ^RT = 0. E6 reports the level string, not a single ratio ([COV-2]/[D-4]).

### Needs E5 (provenance / pattern-primitive manifest) for AUTO-readout
- **`C_construct` (STRONG)** — the load-bearing headline [L-8]. The *auto* discrimination "constructed-weak vs constructed (strong)" requires E5's provenance manifest: each mechanically-constructed body emits its **pattern-primitive ID list**; strong = manifest present ∧ no opaque hand-written helper; weak = descriptor-selected hand fragment (实验总纲 line 27). Without E5, "is this body strong?" is a human judgment.
  - **BUT the strong count is small and hand-known today** (= the 3 decomposed shapes, a separate bucket). So E6 can still *report* `C_construct` from a **committed hand-labeled six-state table** carrying an explicit `"auto_readout": "pending-E5"` marker per strong claim. E6 is not blocked; it just can't yet *machine-enforce* [L-8].
- **[L-8] enforcement guardrail** — E5 is what stops "weak reported as strong" in CI. E6's script should *consume* the label table but **not** be the enforcement point; add a TODO hook so that when E5's manifest lands, the script cross-checks the label against the manifest.

### Recommendation
E6 ships: denominator + `C_dispatch` + `C_construct+` (structurally derivable) + `C_construct` (strong) **read from a committed hand-labeled six-state table** with a per-row `auto_readout: pending-E5` flag + `C_attr` level string. Six-state **auto-readout** (state machine + [L-8] enforcement) is explicitly **deferred to E5**. This matches the PRD's own sequencing (no over-reach).

## Caveats / Not Found
- No six-state **state machine** exists in code (`six-state`/`dispatch-wired` grep = 0 outside spec/docs) — [K-4] "可自动读出" is unbuilt; the table above is hand-derived from structural anchors.
- The weak/strong boundary for the **super-block** bodies (Fork C) was judged "constructible but low-ROI, deprioritized" (实验总纲 line 45) — they stay dispatch-wired; not a boundary change, just not burned down yet.
- `RVVToEmitCBlockQuantLinear.cpp` was split since the `7185a62b` audit; the fp4/codebook emitters now live in `RVVToEmitCCodebookFp4.cpp` and the descriptor in `RVVToEmitCInternal.h`. Counts unchanged (17/7/3).
