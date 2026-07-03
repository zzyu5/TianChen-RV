# Research: E4 — scope summary, MVP + land sequence, escalations

- **Query**: recommended MVP; cheap-refactor vs new machinery; decisions to escalate; byte-exact/lit gate; build/test commands.
- **Scope**: internal
- **Date**: 2026-07-02
- **Contract rows**: [D-4①] JSONL (M1 hard gate C_attr^CT=100%), [D-2a] loading-time record, [F-4] attribution completeness. All C1.

---

## The spine: E4 is a bounded refactor + **one real escalation**

The rich in-IR selection attributes already exist and the candidate/legality data
is **not discarded** before materialize time (`plan.rankedVariants` +
`TargetCapabilitySet` both live at the sink — see `attribution-current-state.md`).
So most of E4 is a cheap additive sink. **The one thing that turns E4 from "clean
bounded refactor" into "bounded refactor + a real decision" is the `prior` reason
honesty question** (below). Everything else is mechanical.

---

## Cheap refactor (existing material, additive)

1. **JSONL sink of existing attrs** — read `plan.rankedVariants` +
   `VariantCostEstimate` and serialize `candidates[]`, `chosen`, at the sink point
   after `VariantSelection.cpp:576`. All data already materialized.
2. **`keys_evaluated{}`** — re-derive at the sink from `variant.requires` ×
   `TargetCapabilitySet.lookupBySymbolName`. Inputs live; **no re-plumbing** of
   `RequirementLegality`. Bounded.
3. **`only_feasible` reason** — derivable today (count feasible == 1). Clean.
4. **Additive guarantee** — sink behind an option defaulting off; in-IR `reason`
   attr NOT remapped (stays `variant-selected`), so existing lit stays green.
5. **Deterministic lit** — `--attribution-jsonl-no-timestamp` + stable hashes +
   canonical JSON (sorted keys, rank-ordered candidates).

## Needs new (small, bounded) machinery

6. **`declared_instance_hash` C++ serializer + SHA256** — net-new (grep = 0).
   Canonicalize the expanded fact set (`TargetCapabilitySet.getCapabilities()`,
   **sorted by id**) → `llvm/Support/SHA256.h`. One helper serves both
   `JSONL.declared_instance_hash` and D-2a. (`d2a-resolution-record-design.md`.)
7. **D-2a resolution record** — compute + stamp `declared_instance_hash` (on the
   `dispatch_available` `RuntimeParamOp` or `KernelOp`) + document the one-per-
   process record shape. Runtime hot path unchanged ([NG-3] preserved).

---

## Genuine decisions to ESCALATE (not researchable)

1. **`prior` reason honesty (the spine).** No capability-prior ranking layer
   exists at the exec selection stage; ordering is static-cost + explicit
   preference, and execution doc §5 ⑦ (`执行总纲v2.md:218`) calls the selector
   *capability-blind*. Choose:
   - (a) Emit `prior` for score/preference-driven multi-candidate picks **now**
     (bounded), accepting that `prior` overstates capability-keying until [SEL-1]
     lands; **or**
   - (b) Make honest `prior` depend on [SEL-1] capability-prior layer first (E4
     grows to include SEL-1 — much larger).
   This is the classic over-optimism trap (cf. project memory: Win-C NULL,
   N1-substrate). C_attr^CT=100% is mechanically trivial to hit; the reason values
   are only as honest as the selector. **Recommend (a) with an explicit doc caveat
   that `prior` == static-cost/preference today**, and a note that SEL-1 upgrades
   its meaning — but the parent agent must sign off.

2. **JSONL sink mechanism** — pass option (recommended, lit-drivable, owns the
   plan) vs tcrv-opt tool-boundary writer. Minor.

3. **`ts` handling** — omit vs fixed sentinel in `--no-timestamp` mode. Minor.

4. **D-2a cut** — compile-time stamp + documented shape ONLY (recommended, since
   the runtime chain is D-2b/M2), vs also dropping a live per-process record now.
   Escalate the exact boundary.

5. **`measured` at ①** — unreachable at the exec layer today (measurement is a
   schedule-stage fact, `RVVScheduleMaterialization.h:211`). Keep the enum value;
   it stays unused at ① until [SEL-3]. Confirm this is acceptable for the M1 gate.

---

## Explicit scope BOUNDARY (scope-creep guard)

- **E4 = selection stage ① + D-2a loading-time record only.**
- **OUT of E4:** the [D-4] **scheduling-stage** attribution — the already-rich
  `candidate_count / legal_candidate_count / selected_cost / measured_ns /
  selection_reason` in `RVVScheduleMaterialization.h:195-216`. That is a **separate
  M2 增量** deliverable (`执行总纲v2.md:202`). The machinery exists; do not pull it
  into E4.
- **OUT of E4:** runtime attribution ③ (hwprobe/instance-hash table = D-2b/D-3, M2/M3).

---

## Recommended land sequence (MVP)

1. Add `support::computeDeclaredInstanceHash(const TargetCapabilitySet&)` (sorted
   fact set → canonical JSON → SHA256 hex). Unit/lit anchor for determinism +
   profile-equivalence (profile ≡ explicit list ⇒ equal hash).
2. Add the two `SelectVariants` pass options + the per-kernel JSONL writer at the
   sink point (`VariantSelection.cpp:576`); cover all four `VariantSelectionKind`s
   incl. `NoViableVariant` (`chosen=null`).
3. Derive `reason` (`only_feasible` clean; `prior` per escalation #1; `measured`
   enum-present-but-unused).
4. D-2a: stamp `declared_instance_hash` in DispatchRuntimeGuard path + document the
   one-per-process record shape.
5. New deterministic lit test under `test/Transforms/VariantSelection/`; keep
   existing preference-attr lit + the C++ smoke test green.

---

## Byte-exact / lit gate

- **Existing green must stay green:** in-IR attrs untouched (sink is additive,
  option-gated off by default). Lit tests asserting preference attrs / in-IR
  `reason` (`ime-mma-*-materialization.mlir`,
  `plugin-variant-materialization-builtin.mlir`,
  `variant-selection-pass-invalid.mlir`) unaffected; the C++ smoke test
  `variant-selection.test` (→ `tianchenrv-variant-selection-test`) unaffected.
- **New JSONL** has its **own** deterministic lit test
  (`--attribution-jsonl=%t.jsonl --attribution-jsonl-no-timestamp` +
  `FileCheck --input-file=%t.jsonl`).
- The E4 diff SHOULD change zero existing `.mlir`/`.cpp` test expectations. If it
  does, that signals a non-additive change — investigate.

## Build / test commands

```bash
cmake --build build                 # incremental build (see caveat below)
cmake --build build --target check-tianchenrv   # full lit + unit suite
# scoped: run the VariantSelection lit dir + the new JSONL test
```

> **Build caveat (project memory):** incremental builds in this tree are
> unreliable — ODS `RVVOps.cpp.inc` regenerates and `tcrv-opt` sometimes fails to
> relink. Any byte-exact / attribute-stability claim must use a forced/clean
> rebuild before trusting lit output.

## Caveats / Not found

- `only_feasible`/`prior`/`measured` tokens: grep = 0 in code (docs/spec only) —
  net-new.
- `keys_evaluated` value vocabulary proposed, not yet fixed — confirm at implement.
- `symbolName` inclusion in the instance hash is an open correctness call
  (`d2a-resolution-record-design.md`).
