# Stage 2 Phase-B — CapabilityModel string-retirement plan

> Task: `.trellis/tasks/06-12-stage2-typify-capability-attr` · Phase-B of the typify plan
> (`research/capability-typification-plan.md` §6 already scoped Phase-A = IR string relations
> deleted, DONE). This document plans the *resolution-layer* retirement so I1 (capability
> first-class queryable) stops being cosmetic: typed in IR, typed at resolution.
>
> Inputs: discovery findings d1-descriptor / d2-consumers / d3-endgame, plus five direct
> verifications recorded in §7.

---

## 0. Decision

**Strategy: `descriptor-backed-by-typed-attr-same-API`** (strangler-fig).
**Rejected: `passes-hold-the-attr-directly`** — structurally impossible as a drop-in and
architecturally wrong even later:

1. `CapabilityDescriptor` holds no `Operation*`/`Attribute` handle back to IR
   (`include/TianChenRV/Support/CapabilityModel.h:60-69`); a pass holding a
   `const CapabilityDescriptor*` cannot reach an op's attr.
2. Descriptors have a second, IR-free source: the RVV probe-fact path
   (`lib/Plugin/RVV/RVVCapabilityProfile.cpp:137-148`) builds descriptors from `ssh rvv`
   runtime data with **no backing op and no CapabilityRelationsAttr**. The
   `TargetCapabilitySet` is the deliberate union of IR-declared + runtime-probed
   capabilities; that union is the N1/N3 story, not a mirror to delete.
3. The hottest primitive, `collectAvailableConflictsForCapability`
   (`lib/Support/CapabilityModel.cpp:424-472`, called by exactly 4 passes), is a
   whole-SET graph traversal (`collectProvidersByID` + `satisfiesID` across every
   capability). One op's attr cannot answer "who in this kernel provides/conflicts id X".

So: the **resolution engine in `CapabilityModel.{h,cpp}` is the single chokepoint**. Retype
its relation storage to the typed attr, keep the descriptor/set query API byte-stable, and
all ~13 descriptor-holding consumers + ~20 build sites migrate transitively with **zero pass
edits**. `TargetCapabilitySet` is permanent infrastructure and survives every phase.

---

## 1. Descriptor change (the concrete `CapabilityModel.{h,cpp}` edit)

### 1.1 Storage

Replace the three string relation vectors with the typed attr itself (single source of
truth, maximal provenance — the descriptor holds the very interned attr the IR holds):

```cpp
// CapabilityModel.h — REMOVE (h:67-69):
llvm::SmallVector<std::string, 4> providedIDs;
llvm::SmallVector<std::string, 4> impliedIDs;
llvm::SmallVector<std::string, 4> conflictingIDs;
// ADD:
tcrv::exec::CapabilityRelationsAttr relations;   // null == no relations
```

Constructor (h:29-35): drop the three `llvm::ArrayRef<std::string>` relation params; add a
trailing `tcrv::exec::CapabilityRelationsAttr relations = {}` param. Header already includes
`ExecOps.h`, so no new include cycle.

### 1.2 Accessors (verified zero external consumers — free to retype)

`getProvidedIDs/getImpliedIDs/getConflictingIDs` currently return
`ArrayRef<std::string>` (h:49-53). Grep over `lib/ include/ test/ tools/` shows **no caller
outside CapabilityModel.{h,cpp}** (only `collectAvailableConflictsForCapability` uses
`getConflictingIDs` internally at cpp:455,467). Retype them to
`llvm::ArrayRef<mlir::StringAttr>` (returning `relations ? relations.getX() : ArrayRef<...>{}`),
and add `tcrv::exec::CapabilityRelationsAttr getRelations() const`.

### 1.3 Queries (the actual I1 fix)

`providesID/impliesID/conflictsWithID` (cpp:272-282) stop dispatching to the std::string
`containsID` (cpp:219-223) and instead scan the attr lists with build-time-equivalent
normalization (the attr verifier is hygiene-only and does **not** require trimming —
`ExecOps.td:38-42` description, `ExecOps.cpp:549-564` — while today's
`collectTypedCapabilityIDRelation` trims and drops empties at intern time; query-time must
replicate that for parity):

```cpp
static bool relationListContains(llvm::ArrayRef<mlir::StringAttr> ids,
                                 llvm::StringRef id) {
  return llvm::any_of(ids, [&](mlir::StringAttr entry) {
    return entry && entry.getValue().trim() == id;
  });
}
```

`satisfiesID` (cpp:284-287) keeps its `getID() == id || providesID || impliesID` shape.
The query keys stay `llvm::StringRef`: plugin id constants (`kRVVCapabilityID` etc.) are
compile-time strings; forcing StringAttr-keyed queries would thread MLIRContext through
every plugin `isApplicable` for zero correctness gain. **Do not claim a perf win** — interned
StringAttr value-compare is the provenance/single-source-of-truth fix; identity fast-pathing
is out of scope and any speed claim would need real measurement (project discipline).

### 1.4 Build paths

- **IR path** — `makeDescriptor` (cpp:205-217) passes
  `op->getAttrOfType<tcrv::exec::CapabilityRelationsAttr>(kRelationsAttrName)` **straight
  through**. `collectTypedCapabilityIDRelation` (cpp:101-127) and
  `sourceCapabilityIDRelation` (cpp:135-143) — the restringify — are deleted.
- **Synthetic path (the binding constraint, resolved)** — a `CapabilityRelationsAttr` needs an
  `MLIRContext`. Verified: `buildRVVTargetCapabilitiesFromProbeFacts` /
  `RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts` currently have **zero callers
  anywhere** (lib/include/test/tools), so adding a `mlir::MLIRContext &` parameter is
  zero-call-site churn. `addAvailableCapability` (RVVCapabilityProfile.cpp:137-148, only
  uses `providedIDs`) mints `CapabilityRelationsAttr::get(&ctx, providesAttrs, {}, {})`.
  Unit tests: `test/Support/CapabilityModelTest.cpp` already owns an `MLIRContext` (line
  63, with DialectRegistry) — its 5 synthetic construction sites (384/399/405/429/450) mint
  attrs from that context. Document the new lifetime contract: a `TargetCapabilitySet` /
  `CapabilityDescriptor` with relations must not outlive the `MLIRContext` that minted them
  (pass-scoped in practice; previously the string copies had no such coupling).
- `CapabilityConflict::conflictID` stays `std::string` in Phase-B (diagnostic payload, not a
  resolution key; keeps the 4 conflict passes at zero churn). Optional Phase-C retype.

---

## 2. Consumer migration (ordered, chokepoint-first)

1. **Chokepoint** — `CapabilityModel.{h,cpp}` engine retype (§1). All relation/availability
   consumers migrate transitively; zero pass edits.
2. **CheckCapabilityRequires** (`lib/Transforms/CheckCapabilityRequires.cpp:148-164`) —
   first *verification* target, not an edit: the pure-relation legality decision, lit-covered
   (`test/Transforms/CapabilityRequires/check-capability-requires.mlir`,
   guarded_conflict_dispatch). This is the N1 first-evidence gate.
3. **DispatchRuntimeGuard / VariantSelection / VariantDispatchSynthesis** (the other 3
   `collectAvailableConflictsForCapability` callers) — ride along free; parity lit only.
4. **Plugin isApplicable batch** (RVV/Scalar/Offload/Template/Toy/TensorExtLite via
   `satisfiesID` / `lookupProviderByID` / `isCapabilityAvailableByID`) — no source change;
   assert parity via their lit suites.
5. **RVVCapabilityProfile probe path + CapabilityModelTest** — the only true source edits
   outside the engine (thread `MLIRContext&`, mint typed relations).
6. **Deferred, untouched**: `HartParallelCapabilities` (string `count` property) and
   `RVVEmitCRoutePlanning` property reads (SEW/LMUL/policy/march) — properties stage,
   Phase-C. `lookupBySymbolName` / `variant.requires = [@sym]` resolution — **different
   namespace** (real MLIR symbols, not capability ids); not part of this migration.

---

## 3. What gets deleted, and when

| Item | Where | When |
|---|---|---|
| `SmallVector<std::string,4> providedIDs/impliedIDs/conflictingIDs` + the 3 `ArrayRef<std::string>` ctor params | `CapabilityModel.h:67-69, 29-35` | B.1 |
| `containsID` std::string scan | `CapabilityModel.cpp:219-223` | B.1 |
| `collectTypedCapabilityIDRelation` + `sourceCapabilityIDRelation` (the restringify) | `CapabilityModel.cpp:101-143` | B.1 |
| `ArrayRef<std::string>` relation accessors (retyped to `ArrayRef<mlir::StringAttr>`) | `CapabilityModel.h:49-53` | B.1 |
| Legacy `"availability"` string fallback in `getCapabilityStatus` + its 22 fixture lines | `CapabilityModel.cpp:40-44` | B.2 (status enum step) |
| Open-string `isUnavailableStatus` matching (becomes enum-driven) | `CapabilityModel.cpp:508-521` | B.2 |
| Raw-string `status = "..."` spellings on CapabilityOp/TargetOp fixtures (~712 sites) | `test/**/*.mlir` | B.2, mechanical sed |
| **Never deleted**: `TargetCapabilitySet` (aggregation + cross-set resolution), `CapabilityDescriptor` as union-of-sources value type, `std::map` properties (Phase-C typing, not deletion), `lookupBySymbolName` symbol path | — | — |

---

## 4. kind / status / properties verdicts

- **status — needs ODS typing for full retirement; schedule as Phase-B.2 behind an audit.**
  Fixture survey (targeted at `tcrv.exec.capability`/`tcrv.exec.target` ops): 710
  `"available"`, 1 `"unavailable"`, 1 `"disabled"` — effectively closed, confirming d3.
  CAUTION: the bare attr name `status` appears 1362 times across 357 test files; the other
  ~650 sites (`"supported"`, `"selected-lowering-boundary"`, `"role-op-boundary"`,
  `"selected"`, `"no-active-route"`, ...) belong to OTHER ops (variant/route/front-door).
  The enum must be scoped to CapabilityOp/TargetOp only. Typing also closes a latent hole:
  today any unknown status (incl. `""`) is silently Available
  (`availabilityFromStatus`/`isUnavailableStatus`, cpp:508-521) — the enum makes that a
  verifier error, a deliberate behavior tightening the audit must adjudicate.
- **kind — stays string short-term.** De-facto small, de-jure open and plugin-extensible
  (`offload-runtime`, `toy-template`, `tensorext-lite-fragment`, RVV kinds, `isa-vector`).
  Already ODS-typed as `OptionalAttr<StrAttr>`; an open enum is a Phase-C nicety with low
  I1 value.
- **properties — stay `std::map<std::string,std::string>` this stage; Phase-C =
  DictionaryAttr-shaped typing.** Genuinely open and polymorphic (`count` parsed as int,
  `architecture`/`supported_sew`/`supported_lmul`/policy strings, `runtime_abi`/
  `handoff_kind`); `stringifyCapabilityProperty` (cpp:56-82) intentionally accepts
  String/Bool/Integer/Float/FlatSymbolRef. Largest surface, lowest I1 value — the I1
  relation contract does not depend on it.

---

## 5. Ordered steps (each gate = build green + named lit/unit)

1. **B.1a — chokepoint retype.** Edit `CapabilityModel.{h,cpp}` per §1: store
   `CapabilityRelationsAttr`, retype accessors, rewrite the four predicates +
   `collectAvailableConflictsForCapability`/`lookupProviderByID`/`collectProvidersByID`
   internals, pass the attr through `makeDescriptor`, delete the restringify trio. Update
   the 5 `CapabilityModelTest.cpp` synthetic sites to mint attrs from the test's context;
   add one new unit asserting padded-entry parity (`" rvv.v "` matches query `"rvv.v"`).
   **Gate:** build + CapabilityModelTest + `lit test/Transforms/CapabilityRequires/`
   (guarded_conflict_dispatch = N1 first evidence).
2. **B.1b — probe path typed.** Thread `mlir::MLIRContext &` through
   `buildRVVTargetCapabilitiesFromProbeFacts` (RVVCapabilityProfile.{h,cpp}) and
   `RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts`; `addAvailableCapability`
   mints the provides attr. Document the context-lifetime contract on the builder and on
   `CapabilityDescriptor`. **Gate:** build + RVV plugin unit/lit suites.
3. **B.1c — ride-along parity sweep (zero source change expected).** Run conflict-pass lit
   (DispatchRuntimeGuard, VariantSelection, VariantDispatchSynthesis) + all six plugin
   isApplicable suites + full `check` loop. Any diff here is a regression in B.1a, not a
   migration item.
4. **B.2a — status audit (cheap, decisive).** Inventory `status` spellings on
   CapabilityOp/TargetOp only (fixtures + C++ writers via `kStatusAttrName` grep), plus the
   22 legacy `availability` fixture lines and the 1 empty-status site. Output: closed value
   set + migration sed + adjudication of unknown-status-now-errors.
5. **B.2b — status ODS enum (conditional on 4).** Enum attr on CapabilityOp/TargetOp only
   (`available`/`unavailable`, folding `disabled`/`missing` aliases per audit); re-back
   `getCapabilityStatus`/`availabilityFromStatus`/`isUnavailableStatus`; mechanical fixture
   migration (~712 sites); delete the legacy `availability` fallback. **Gate:** build + full
   lit. If the audit reveals open usage on capability ops, demote to Phase-C and ship B.1
   alone — B.1 is independently the honest "relations no longer compared as std::string"
   deliverable.
6. **B closes — record the Phase-C boundary in the task journal:** properties
   DictionaryAttr typing + property-consumer migration (HartParallel `count`, RVVEmitC
   SEW/LMUL/policy, Offload `runtime_abi`/`handoff_kind`); kind open-enum;
   `CapabilityConflict::conflictID` retype; any passes-hold-attr refactor (rejected as a
   goal — the set is permanent). Stage-2 does NOT do these.

---

## 6. Risks

1. **Attr lifetime coupling (new).** Descriptors now hold context-owned attrs; a
   `TargetCapabilitySet` must not outlive its `MLIRContext`. Today's string copies had no
   such coupling. Mitigation: pass-scoped usage already holds everywhere; document on the
   class; the probe-facts builder takes the context explicitly so the owner is visible.
2. **Trim/normalization parity.** Normalization moves from intern-time to query-time; a
   padded `" rvv.v "` entry must still match and `getConflictingIDs` consumers must trim.
   Mitigation: single `relationListContains`/trim helper + dedicated unit test (step 1).
3. **Status enum scope-creep.** `status` is a shared attr name across many non-capability
   ops (~650 other fixture sites). The enum must bind to CapabilityOp/TargetOp ODS only;
   the audit (B.2a) is mandatory before any sed.
4. **Behavior tightening on unknown status.** `"" `/garbage statuses flip from
   silently-Available to verifier error in B.2 — desirable for I1 honesty but a real
   behavior change; adjudicate per-fixture in the audit.
5. **Dialect registration in synthetic contexts.** `CapabilityRelationsAttr::get` requires
   the TCRV Exec dialect loaded; unit tests and the future probe caller must
   register/load it (CapabilityModelTest already has a registry; assert at builder entry).
6. **Probe path is currently unwired.** Zero callers today means zero churn but also zero
   integration coverage; when the `ssh rvv` probe flow is wired (N3), the caller must
   supply the compilation context — the new signature forces this correctly, but add a
   builder unit test now so B.1b is not dead-on-arrival.
7. **Hidden string-relation readers.** Verified none for the relation accessors, but the
   prior plan flagged I4 mirrors (`formatRVV...Mirror`, `TargetArtifactExport.cpp`) as
   string consumers of other capability fields — re-grep at implementation time before
   claiming the deletion inventory complete.
8. **No performance claims without measurement.** Interned-attr storage is a provenance
   fix; any "faster resolution" claim needs real evidence per project discipline.

---

## 7. Verification log (this planning session, beyond findings JSON)

| Check | Result |
|---|---|
| External consumers of `getProvidedIDs/getImpliedIDs/getConflictingIDs` | **None** outside CapabilityModel.{h,cpp} — accessors free to retype |
| Callers of `buildTargetCapabilitiesFromProbeFacts` / `buildRVVTargetCapabilitiesFromProbeFacts` | **None** in lib/include/test/tools — context param is zero-churn |
| Status values on `tcrv.exec.capability`/`tcrv.exec.target` fixtures | 710 `"available"`, 1 `"unavailable"`, 1 `"disabled"` — closed on these ops |
| Raw `status = "..."` across all test .mlir | 1362 sites / 357 files; majority on non-capability ops — enum must be op-scoped |
| `MLIRContext` in CapabilityModelTest | Present (line 63, with DialectRegistry) — synthetic test sites can mint attrs |
| `CapabilityRelationsAttr` verifier | Hygiene-only, trimming NOT required (ExecOps.td:38-42, ExecOps.cpp:549-564) — query-time trim needed for parity |
| Synthetic `CapabilityDescriptor` ctor sites | RVVCapabilityProfile.cpp:143 + CapabilityModelTest.cpp:384/399/405/429/450 — full inventory |
