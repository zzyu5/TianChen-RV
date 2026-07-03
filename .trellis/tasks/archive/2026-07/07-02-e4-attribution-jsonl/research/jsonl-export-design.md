# Research: E4 — compile-time selection attribution JSONL export design

- **Query**: HOW to emit the [D-4]① JSONL; object schema; reason-enum remap; lit-determinism; additivity.
- **Scope**: internal (design synthesis over HEAD facts)
- **Date**: 2026-07-02

Authoritative object shape (`docs/TianChen-RV_科研目标总纲v2.md:90`,
`.trellis/spec/variant-pipeline/generation-selection-tuning.md:65`):
`{kernel, candidates[], keys_evaluated{}, chosen, reason∈{only_feasible, prior,
measured}, declared_instance_hash, ts}`. M1 hard gate = C_attr^CT sampled 100%.

---

## 1. Sink mechanism — pass option, one object per kernel

- **Add a pass option** to `SelectVariants` (`Passes.td:130-165` — the pass has
  **no** options today, this is the first). Two options:
  - `Option<"attributionJsonl", "attribution-jsonl", "std::string", "\"\"">` —
    output file path; empty ⇒ **sink disabled** (default off).
  - `Option<"attributionJsonlNoTimestamp", "attribution-jsonl-no-timestamp", "bool", "false">`
    — omit / fix `ts` for deterministic lit (see §4).
- The pass is constructed via factory `createSelectVariantsPass(plugins)`
  (`tools/tcrv-opt/tcrv-opt.cpp:59-61`). MLIR populates option fields from the CLI
  regardless of constructor; the existing copy-ctor
  (`VariantSelection.cpp:543-546`) inherits the TableGen base's option copy — verify
  the option fields round-trip through it (they are members of the generated base).
- **One JSON object per kernel per line** (JSONL), appended to the file. Emit for
  **every** kernel `runSelection` processes, across all four `VariantSelectionKind`s
  (see §3), so the C_attr^CT denominator is honest.

### Sink placement anchor (exact)

Insert the emit call in `runSelection` (`VariantSelection.cpp:562-593`) **right
after**:

```
VariantSelectionPlan plan = std::move(*planOrError);   // :576
```

**before** the `switch (plan.kind)`. At that point `plan.rankedVariants`
(candidates), `capabilities` (`TargetCapabilitySet`, for keys_evaluated +
declared_instance_hash), and `kernel` are all in scope, and every kind is covered
uniformly. For `NoViableVariant` (early `return` at `:587-588`) the emit still runs
because it precedes the switch.

> A module-level file handle (opened once, appended per kernel) avoids interleave.
> Simplest: the pass opens the path in `runOnOperation` before the walk, writes one
> line per kernel inside the walk, closes after. Keep the writer target-neutral
> (this pass must not branch on RVV/family per its own contract, `Passes.td:135-158`).

---

## 2. JSON object schema (proposed, canonical)

Serialize as **canonical JSON** — sorted keys, no incidental whitespace — mirroring
the E1 idiom (`json.dumps(sort_keys=True, separators=(",",":"))`,
`check_schema_gate.py:59-61`) so lines are byte-stable:

```jsonc
{
  "kernel": "<KernelOp.getSymName()>",
  "candidates": [
    {
      "variant": "<VariantOp.getSymName()>",
      "origin": "<cost.getOriginPlugin()>",
      "score": <cost.getScore()>,            // omit if !hasScore()
      "explicit_preference": <bool>,          // cost.hasExplicitPreference()
      "rank": <index in rankedVariants>,
      "feasible": <genericallyAvailable && conflictFree>,
      "requires_runtime_guard": <requiresRuntimeCapabilityGuard>,
      "fallback_role": "conservative-fallback" // omit if none
    } // ... every plan.rankedVariants entry, in rank order (already stable)
  ],
  "keys_evaluated": {
    "<capability-symbol-or-id>": "<available|unavailable|unknown|conflicting>"
    // union over all candidates' `requires`, re-derived at sink; sorted keys
  },
  "chosen": "<plan.selectedVariant.getSymName()>",  // or null (see §3)
  "reason": "only_feasible" | "prior" | "measured",  // derived, see §5
  "declared_instance_hash": "<sha256 hex over expanded fact set>", // d2a-resolution-record-design.md
  "ts": "<omitted / fixed sentinel in --no-timestamp mode; else ISO-8601>"
}
```

Field sourcing: all `candidates[]` fields come from the live
`VariantSelectionCase` + `VariantCostEstimate` (see `attribution-current-state.md`).
`keys_evaluated{}` re-derived from `variant.requires` ×
`TargetCapabilitySet.lookupBySymbolName` at the sink.

---

## 3. Coverage across all four VariantSelectionKind (denominator honesty)

| `plan.kind` | `chosen` | Notes |
|---|---|---|
| `StaticVariant` | `plan.selectedVariant` | single feasible pick |
| `FallbackOnly` | `plan.selectedVariant` (== fallback) | one-candidate case (`VariantSelection.cpp:726-729`) |
| `RuntimeDispatch` | `plan.selectedVariant` | `plan.dispatchCases` are guarded candidates already inside `candidates[]`; the fallback is a candidate too. `chosen` = the statically-selected variant; guarded cases carry `requires_runtime_guard=true` |
| `NoViableVariant` | `null` | fail-closed (`:587-588`, `:628-631`); all candidates infeasible (or `rankedVariants` empty). Emit a record with `chosen=null`, `reason` omitted/`"none"`, so the C_attr^CT denominator counts fail-closed kernels honestly |

- The **`fallback-coverage-missing`** concern (`plan.missingFallbackCoverage`,
  `VariantSelection.cpp:690`) is a **separate warning-status axis** — do NOT map it
  into the `reason` field. If it needs surfacing, add an optional boolean
  `missing_fallback_coverage` field; keep it out of the three-value enum.

---

## 4. Lit determinism (`ts` + hashes)

- **`ts`**: with `--attribution-jsonl-no-timestamp`, either omit `ts` entirely or
  write a fixed sentinel (`"ts":"0"`). Lit tests always run in that mode →
  deterministic. Production runs emit a real timestamp.
- **`declared_instance_hash`**: SHA256 of the canonical fact-set serialization —
  deterministic for a given input IR. Stable across runs. (The C++ serializer MUST
  sort descriptors by id before hashing — see `d2a-resolution-record-design.md` —
  else the hash is order-dependent and lit-fragile.)
- **`candidates[]` order**: rank order is already stable (the existing tie-break
  `buildPreferenceTieBreakReason` + `preference_rank` guarantees fallback-role →
  original IR order → symbol name, `VariantSelection.cpp:292-324`).
- **`keys_evaluated{}` order**: emit with **sorted keys** (canonical JSON).
- **Lit shape**: `RUN: tcrv-opt %s -tcrv-select-variants
  --attribution-jsonl=%t.jsonl --attribution-jsonl-no-timestamp` then
  `RUN: FileCheck --input-file=%t.jsonl %s`. New test lives under
  `test/Transforms/VariantSelection/`.

---

## 5. reason-enum remap — DERIVED, not a rename (and the honesty escalation)

The three-value `reason` is **orthogonal** to the in-IR `reason` attr
(`variant-selected`/`fallback-coverage-missing`). It must be **derived from the
plan**, per spec (`generation-selection-tuning.md:65`):

- **`only_feasible`** = after legality filter, exactly **one** feasible candidate
  (`count(rankedVariants where genericallyAvailable && conflictFree) == 1`). Fully
  derivable today from the plan. **Clean.**
- **`prior`** = ≥2 feasible candidates and the capability-prior ranking layer
  chose. **⚠ ESCALATION — the prior layer does not exist yet.** Today the order
  is static-cost + `hasExplicitPreference` (`ExtensionPlugin.cpp:1362`); execution
  doc §5 ⑦ (`执行总纲v2.md:218`) says the selector is *capability-blind*. Two
  options (a genuine decision, not researchable):
  1. **Emit `prior` for score/preference-driven multi-candidate picks now**
     (bounded refactor). Risk: `prior` then means "static-cost/preference chose",
     which **overstates** capability-keying vs the spec's intent, and per ⑦ any
     "winning variant" is not actually capability-selected. This is the
     over-optimism failure mode this project's memory repeatedly flags (Win-C NULL,
     N1-substrate).
  2. **Honest `prior` requires [SEL-1] capability-prior layer to land first** →
     E4 then depends on SEL-1 (much larger; SEL-1/SEL-2 are separate work items).
- **`measured`** = memoized measurement chose. **Unreachable at the exec selection
  layer today** — measurement lives at the RVV *schedule* stage
  (`measured_ns`, `RVVScheduleMaterialization.h:211`), not here. Valid enum value,
  but at ① it can only fire once the exec selector consumes a memoized measurement
  record ([SEL-3]). Emit the value in the schema; it will simply not occur at ①
  until SEL-3.

### Additivity of the in-IR reason attr

**Do NOT remap the in-IR `reason` attribute.** It stays byte-identical
(`variant-selected` etc.) so the existing `.mlir` lit tests that assert it
(`ime-mma-*-materialization.mlir`, `plugin-variant-materialization-builtin.mlir`,
etc.) keep passing. `JSONL.reason` is a **newly derived field** written to the side
file, not a rename. This is exactly why E4 is additive **and** why the two `reason`
axes coexist without conflict.

---

## Additive / byte-exact guarantees

- The JSONL sink writes to a **file**, gated behind an option defaulting to off →
  when the option is absent, the pass behaves byte-identically to today. Existing
  in-IR attributes (`addPreferenceMetadata`, the two diagnostic markers) are
  **untouched**.
- The new deterministic JSONL gets its **own** lit test; the existing
  VariantSelection C++ smoke test (`variant-selection.test` →
  `tianchenrv-variant-selection-test`) and the preference-attr lit assertions stay
  green.

## Caveats / Not found

- Whether the sink is a pass option vs a separate `tcrv-opt`-boundary writer is a
  small mechanism choice; pass option is recommended (keeps it inside the pass that
  owns the plan, and lit-drivable). Escalate if the reviewer prefers a
  tool-boundary sink.
- `keys_evaluated{}` value vocabulary (`available/unavailable/unknown/conflicting`)
  is proposed here from the legality logic (`analyzeRequirementLegality`,
  `VariantSelection.cpp:190-204`); confirm the exact token set during implement.
