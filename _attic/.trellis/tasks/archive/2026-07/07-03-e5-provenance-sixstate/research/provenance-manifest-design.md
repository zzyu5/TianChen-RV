# Research: E5 — provenance manifest format, attach point, emit + read design

- **Query**: Design the provenance manifest (format, where it attaches, how emitted, how a tool reads it to derive six-state); make the strong/weak rule mechanical; define the minimal `pattern_id` vocabulary (or reuse an existing structure).
- **Scope**: internal (design synthesis over HEAD facts)
- **Date**: 2026-07-03
- **HEAD**: `c6ebe0b1401c7e218ee34b73510e91d47a550aa3`
- **Contract**: 实验总纲v1 §1.6 (line 27) + §6.2 (line 107); 科研目标总纲v2 [K-4] (`:78-80`), [L-8] (`:48`), [PAT-1] (`:118`).

---

## 1. The substrate already exists — E5 serializes, it does not invent

Two facts from `emission-paths-map.md` set the whole design:

1. **The strong primitive-ID list already exists as structured data.** Each strong shape's ordered
   pattern-primitive IDs = the `semanticRoleGraph` field on its `kRetainedSelectedBodySpecializations`
   route (`RVVConstructionProtocol.cpp:538-564`), materialized as an ordered `ExecutableRoleStep[]`
   by `getRVVSelectedBodyExecutableRoleSteps(typedComputeOpName)` (`:4946`). The
   `rvv_construction_protocol = "extension-family-construction-protocol.v1"` attr is ALREADY stamped
   on the strong front-door ops.
2. **The Python consumer cannot read C++/MLIR.** `coverage_metrics.py` is stdlib-only and forbidden
   from touching C++/ODS (`:8-9`). So a **serialized side artifact is mandatory net-new C++**, even
   though the underlying data exists. "Already exists" = the data; the sink does not.

Therefore E5 = **(a) a C++ option-gated sink that writes a provenance JSONL, mirroring E4** +
**(b) a Python reader that folds it into the six-state derivation**.

---

## 2. Manifest format (per-kernel JSONL, canonical)

One JSON object per constructed/weak kernel per line, canonical JSON (sorted keys, no incidental
whitespace) — the same idiom E1/E4 use (`check_schema_gate.py:59-61`;
`coverage_metrics.py:58-63` `canonicalize`). Byte-stable for lit.

```jsonc
{
  "op": "product_reduce",                 // roster/six-state key part 1
  "format": "offset_binary_n3",           // roster/six-state key part 2
  "engine": "",                           // "" except gemm_tile rvv/ime
  "construction": "typed-primitive-body", // or "descriptor-selected-composition"
  "opaque_helper": false,                 // true for the weak path
  "primitives": [                         // ordered pattern-primitive IDs
    "tcrv_rvv.widening_product",
    "tcrv_rvv.standalone_reduce",
    "tcrv_rvv.gearbox_cross_region_handoff",
    "tcrv_rvv.dequantize"
  ],
  "construction_protocol": "extension-family-construction-protocol.v1", // strong only; else omit
  "derived_state": "constructed"          // "constructed" | "constructed-weak"
}
```

Weak example:

```jsonc
{
  "op": "vec_dot", "format": "q4_0", "engine": "",
  "construction": "descriptor-selected-composition",
  "opaque_helper": true,
  "primitives": ["flat-decode:offset_binary_nibble", "flat-fold:left_assoc"],
  "derived_state": "constructed-weak"
}
```

- `derived_state` is written by the emitter, but the Python reader **recomputes it** from
  `construction`/`opaque_helper`/`primitives` and asserts agreement — the JSON value is a convenience,
  the rule (§4) is the authority. This keeps the "六态从此脚本可判、不可辩解" property in the tool,
  not the C++.
- No timestamps, no repo sha inside each line (determinism); the report `$meta` snapshot is added by
  the reader from git HEAD, exactly as `coverage_metrics.py:179-188` does.

---

## 3. Where it attaches + how it is emitted (additive, option-gated — E4 pattern)

E4's landed sink is the exact template to copy (`Passes.td:165-189` options
`attribution-jsonl` / `attribution-jsonl-no-timestamp`; `VariantSelection.cpp:637-644` file open,
`:1095-1099` canonical serialization). Mirror it:

**Recommended: an option-gated sink on the RVV→EmitC conversion pass** (the pass that runs
`emitFlatBlockDot` AND lowers the typed pre-realized bodies), because that is the single point where
BOTH paths are observable in the same walk, before they collapse to indistinguishable `emitc`.

- Add a pass option `provenance-manifest` (`std::string` path; empty ⇒ **disabled**, default off) +
  `provenance-manifest-no-timestamp` if any time field is ever added (currently none — omit).
- Emit **one line per lowered kernel**:
  - **STRONG branch** (the typed pre-realized body is present / a front-door-constructed body):
    read the already-derived `getRVVSelectedBodyExecutableRoleSteps(typedComputeOpName)` →
    `primitives[]`; set `construction="typed-primitive-body"`, `opaque_helper=false`; run the
    existing `rejectMixedPreRealizedContractionBody` allowlist (or the completeness check
    `collectSelectedExecutableRoleSequence(...).complete()`) as the **assertion** that no opaque op
    slipped in — if it fails, `opaque_helper=true` and the row degrades to weak (honest fail).
  - **WEAK branch** (`emitFlatBlockDot` ran from a `FlatBlockDotDescriptor`): serialize
    `descriptor.decodePrimitive` + `descriptor.foldModel` (+ `hasCodebook`) as
    `primitives[]` fragment IDs; `construction="descriptor-selected-composition"`,
    `opaque_helper=true`.

**Alternative attach point (also viable):** stamp the manifest as an **attribute on the emitted
op/module** (like `rvv_construction_protocol` already is on strong front-door ops), then a separate
tiny exporter walks the module and writes the JSONL. This keeps provenance in-IR (auditable via lit
FileCheck) but needs a second walk. The pass-option JSONL sink is smaller and directly mirrors E4 —
**recommend the JSONL sink; escalate only if the reviewer wants the in-IR attribute form.**

**Byte-exact / additive guarantee:** the sink writes to a **file** gated behind an option defaulting
off ⇒ when absent, lowering is byte-identical to today. No existing `.mlir`/`.cpp` expectation
changes. The new JSONL gets its own deterministic lit test (§ `l8-enforcement-design.md`).

---

## 4. The strong/weak decision rule, made mechanical

The reader (or a shared helper) applies exactly (mirrors 实验总纲 line 27 verbatim):

```
constructed         ⟸  construction == "typed-primitive-body"
                        ∧ opaque_helper == false
                        ∧ primitives non-empty and all ∈ typed-primitive registry (§5)

constructed-weak    ⟸  construction == "descriptor-selected-composition"
                        ∨ opaque_helper == true
```

"清单存在 ∧ 无不透明手写 helper" ⟺ `primitives non-empty ∧ opaque_helper == false`. A weak row can
NEVER satisfy the strong branch because it carries `opaque_helper=true`. This is the [L-8] "弱充强 =
违宪" made undebatable.

---

## 5. The `pattern_id` vocabulary — REUSE, don't invent

**[PAT-1] pattern registry does not exist under that name** — `grep PatternRegistry / patternLibrary
= 0` in code (execution-doc `:75`, `:88` confirms). BUT [PAT-1]'s own text (`科研目标总纲v2:118`) says
the **收缩/产品-归约族 is already data-ized** — and it is, as **two existing structures**:

1. **`ContractionRouteIdentity` registry** (`RVVContractionRouteIdentity.cpp:33-374`) — 7 routes,
   each keyed by `(headOpName, isSigned)` with an ordered `ContractionSourceSpec[]`
   (`bodyStepPosition`, `slotName`, `abiRole`, …). This is the per-source route data.
2. **`kRetainedSelectedBodySpecializations[]`** (`RVVConstructionProtocol.cpp:357-750+`) — op_kind →
   `semanticRoleGraph` (the `+`-joined typed-primitive-op chain) + emitc-route IDs. This is the
   **ordered primitive-ID sequence** — the closest thing to a pattern registry already in tree.

**Recommendation:** the strong `pattern_id` vocabulary = **the typed-op mnemonics themselves**
(`tcrv_rvv.widening_product`, `tcrv_rvv.standalone_reduce`, `tcrv_rvv.gearbox_cross_region_handoff`,
`tcrv_rvv.dequantize`, plus the packed-i4 head variants). They are stable ODS mnemonics
(`RVVOps.td`), already the tokens of `semanticRoleGraph`. No new vocabulary needed — E5 serializes
the existing `semanticRoleGraph` split on `+`.

The weak `pattern_id` vocabulary = **the `FlatDecodePrimitive` + `FlatFoldModel` enum names**
(`RVVToEmitCInternal.h:73-108`), namespaced (`flat-decode:*`, `flat-fold:*`) to make clear they are
descriptor-selected fragments, not typed primitives.

A future full [PAT-1] `{pattern_id, requires, transform, mechanism, metrics_hook, status}` schema
(execution-doc `:88` notes `metrics_hook` grep=0) is **out of E5 scope** — E5 needs only the
`pattern_id` sequence, which the two existing structures already supply. Flag the full-schema
promotion as a later [PAT-1] item.

---

## 6. How the tool reads it to derive six-state

`coverage_metrics.py` today reads `schema/coverage-sixstate.v1.json` (hand-labeled) and computes
`C_construct` (STRONG only) / `C_construct_plus` (weak). E5 makes the strong/weak rows
**auto-derived**:

1. E5 emits `schema/provenance-manifest.v1.jsonl` (committed artifact, produced by a forced-clean
   build running the RVV→EmitC pass over the strong e2e inputs + the weak block-dot inputs with
   `--provenance-manifest=...`).
2. `coverage_metrics.py` (or a new `provenance_sixstate.py` that feeds it) reads the JSONL, applies
   the §4 rule per `(op, format, engine)` key, and produces the `constructed` / `constructed-weak`
   state for exactly the rows that have a manifest line. Rows with no manifest line keep their
   hand-label (absent / emittable / dispatch-wired — see `l8-enforcement-design.md` for the split).
3. The `pending-E5` marker is dropped from the auto-derived rows.

Determinism: JSONL is byte-stable (canonical, no timestamps); the reader adds the git-HEAD `$meta`
snapshot exactly like `coverage_metrics.py:179-188`. Same HEAD ⇒ byte-identical report.

## Caveats / Not found

- The single cleanest emit point requires that both the typed pre-realized body AND the
  `emitFlatBlockDot` descriptor are reachable in one pass. Confirm at implement that the RVV→EmitC
  conversion sees both (the strong bodies arrive as pre-realized typed ops; the weak arrives as
  `GgmlBlockDot*` ops lowered by `emitFlatBlockDot`). If they live in different passes, emit two
  JSONL fragments and concatenate (still one committed artifact).
- `getRVVSelectedBodyExecutableRoleSteps` returns `llvm::Expected<...>`; a strong shape whose lookup
  errors must be surfaced (fail-closed) not silently dropped — else a strong row silently vanishes
  from the numerator.
- Whether the manifest is a committed artifact (`schema/provenance-manifest.v1.jsonl`, hashed like
  the roster) vs a CI-only report is a small governance call — recommend committed + hashed so the
  six-state report is reproducible off HEAD without a build. Escalate.
