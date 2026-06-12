# Stage 2 — N1 Capability Typification Plan

Task: `06-12-stage2-typify-capability-attr`. Reconciles 4 scoping findings (c1 current-model,
c2 consumers, c3 existing-ods, c4 ods-design) into one buildable, strangler-fig plan.

N1 claim being advanced: **RISC-V extension heterogeneity as first-class capability IR**.
I1 reversal we are repairing: cross-capability relations
(`provides`/`implies`/`conflicts`) exist today only as *untyped, ODS-absent, string-keyed*
side attributes, queried by `std::string ==`. The fix is one typed, queryable MLIR Attribute
that C++ passes/plugins resolve in-pass.

---

## 0. The decisive design fork (resolved against c1/c3, in favor of c4)

c1 and c3 recommend modeling the relations as `FlatSymbolRefAttr` arrays. **That is wrong**
and would break every relation fixture. The relations reference capability **IDs** — free
dotted strings in their own namespace — NOT MLIR symbols.

Primary-source proof (`test/Transforms/CapabilityRequires/check-capability-requires.mlir:130-141`):

```mlir
tcrv.exec.capability @inline_asm {
  id = "vendor.inline_asm", kind = "toolchain",
  conflicts = ["build.policy.no_inline_asm"], status = "available" }
tcrv.exec.capability @no_inline_profile {
  id = "build.policy.profile", kind = "build-policy",
  provides = ["build.policy.no_inline_asm"], status = "available" }
```

`build.policy.no_inline_asm` is referenced by `provides`/`conflicts` but is **not** a symbol —
there is no `tcrv.exec.capability @build_policy_no_inline_asm`. Conflict resolution walks IDs,
not the symbol table: `lookupProviderByID` → `collectProvidersByID` → `satisfiesID`
(`lib/Support/CapabilityModel.cpp:371,388,402,262-265`). Contrast `variant.requires = [@inline_asm]`
(`:149-151`) which IS a real `FlatSymbolRefAttr` resolved by `lookupBySymbolName`
(`lib/Transforms/CheckCapabilityRequires.cpp:143-149`). **Two distinct namespaces.**

Therefore relation entries must stay `StringAttr` (capability IDs), wrapped in a typed attr.
This is c4's recommendation, and it is the one consistent with the fixtures.

---

## 1. chosen_ods_design

A single new `AttrDef` on the Exec (`tcrv`) dialect, mirroring the project's only existing
AttrDef house style (`TCRVRVV_PolicyAttr`, `RVVOps.td:68-83`).

```tablegen
// Base class beside CapabilityOp, mirroring TCRVRVV_Attr (RVVOps.td:38-41).
class TCRVExec_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TCRV_Exec_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def TCRVExec_CapabilityRelationsAttr
    : TCRVExec_Attr<"CapabilityRelations", "capability_relations"> {
  let summary = "typed provides/implies/conflicts capability-id relations";
  let description = [{
    Typed structure for a capability's cross-capability relations. Each list holds
    capability *IDs* (free dotted strings in the capability-id namespace, e.g.
    "build.policy.no_inline_asm"), NOT MLIR symbol references. Queried directly by
    C++ capability-aware passes/plugins; no string-keyed getAttrOfType lookup.
  }];
  let parameters = (ins
    OptionalArrayRefParameter<"::mlir::StringAttr">:$provides,
    OptionalArrayRefParameter<"::mlir::StringAttr">:$implies,
    OptionalArrayRefParameter<"::mlir::StringAttr">:$conflicts);
  let assemblyFormat =
    "`<` (`provides` `=` $provides^)? (`implies` `=` $implies^)? "
    "(`conflicts` `=` $conflicts^)? `>`";
  let genVerifyDecl = 1;   // ports the hygiene checks from ExecOps.cpp:177-215
}
```

Justification against I1 (queryable + resolvable in-pass):
- **Queryable by C++ type, not string key.** A pass reads
  `capOp.getRelationsAttr()` (typed `CapabilityRelationsAttr`) and iterates
  `rel.getProvides()` as `ArrayRef<mlir::StringAttr>` — the same C++-type consumption
  precedent as `tcrv::rvv::PolicyAttr` passed by type in plugin signatures
  (`RVVStandaloneReductionSelectedBodyRealizationOwner.cpp:174,187`). Replaces the
  silent-skip string parse `op->getAttrOfType<ArrayAttr>("provides")` +
  `dyn_cast<StringAttr>` (`CapabilityModel.cpp:106-118`).
- **Relations resolvable.** Entries remain capability IDs, so the existing
  by-ID resolution (`satisfiesID`, `lookupProviderByID`,
  `collectAvailableConflictsForCapability`) keeps working unchanged — the typed attr is a
  faithful, lossless structuring of today's semantics, not a namespace change.
- **Verifier folds in.** `genVerifyDecl=1` absorbs the non-empty / no-trim-needed /
  single-line / no-duplicate checks currently hand-rolled in
  `verifyCapabilityIDRelationAttr` (`ExecOps.cpp:177-215`).

Deliberately **out of scope** for Stage 2: typing `kind` (de-facto small enum but open),
`status` (closed available/unavailable, but not part of the relation contract), and
`properties` (genuinely open key/value: rvv SEW/LMUL/march/mabi, hart `count`). Those are
later steps; the I1 contract is the relations.

---

## 2. ods_home

- **Lives in** `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`, in the `tcrv` (Exec)
  dialect, directly beside `CapabilityOp` (`ExecOps.td:39-53`). It becomes the Exec
  dialect's **first** AttrDef (the dialect has none today).
- **Relationship to existing capability structure:** EXTEND `CapabilityOp` additively. Add a
  new `OptionalAttr<TCRVExec_CapabilityRelationsAttr>:$relations` argument alongside the
  existing untyped `provides`/`implies`/`conflicts` side attrs (which stay as legacy
  discardable attrs through Stage 2). The same optional attr is added to `TargetOp`
  (`ExecOps.td:33-34`), which also carries relations today.
- **CapabilityProviderComposition.h: do NOT reuse or mirror.** It is a pure string helper
  (every accessor returns `llvm::StringRef`, no Attr/Type;
  `CapabilityProviderComposition.h:14-25`). It is a *consumer to bridge later*, not a typed
  structure to imitate. No duplication: the only typed structure that exists to mirror is
  `TCRVRVV_PolicyAttr`, and we mirror its TableGen shape only.
- **No new dialect.** Capability structure already lives in `tcrv`; a separate dialect would
  fragment the contract.

---

## 3. additive_first_step (smallest buildable, breaks nothing)

Define the AttrDef + generated bindings + dialect registration so that an optional
`relations = #tcrv.capability_relations<...>` attr can be *attached and round-tripped*, with
**zero** changes to `CapabilityModel.{h,cpp}` or any of its ~25-30 consumers. The legacy
string `provides`/`implies`/`conflicts` path and `verifyCapabilityIDRelationAttr` stay live.

Exact files (4 edits + 1 fixture), mirroring the RVV attrdef wiring:

1. `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
   - add `include "mlir/IR/AttrTypeBase.td"`,
   - add `let useDefaultAttributePrinterParser = 1;` to `TCRV_Exec_Dialect`
     (`ExecOps.td:7-21`; currently absent — see `RVVOps.td:30`),
   - add `class TCRVExec_Attr` + `def TCRVExec_CapabilityRelationsAttr` (sketch above),
   - add `OptionalAttr<TCRVExec_CapabilityRelationsAttr>:$relations` to `CapabilityOp`
     arguments (and to `TargetOp`).

2. `include/TianChenRV/Dialect/Exec/IR/CMakeLists.txt`
   - beside the existing `add_mlir_dialect(ExecOps tcrv)`, add explicit attrdef tablegen
     (mirror `include/.../RVV/IR/CMakeLists.txt:6-7`):
     ```cmake
     set(LLVM_TARGET_DEFINITIONS ExecOps.td)
     mlir_tablegen(ExecAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=tcrv)
     mlir_tablegen(ExecAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tcrv)
     add_public_tablegen_target(MLIRExecAttrsIncGen)   # if add_mlir_dialect's target
                                                       # does not already cover these
     ```
     (Confirm at build whether `add_mlir_dialect`'s generated target already picks the new
     `mlir_tablegen` outputs; if so the explicit `add_public_tablegen_target` is unneeded.)

3. `include/TianChenRV/Dialect/Exec/IR/ExecOps.h`
   - after the op-class include (`ExecOps.h:13-14`), add
     `#define GET_ATTRDEF_CLASSES` + `#include ".../Exec/IR/ExecAttrs.h.inc"`
     (mirror `RVVDialect.h:16-17`).

4. `lib/Dialect/Exec/IR/ExecOps.cpp`
   - add `#define GET_ATTRDEF_CLASSES` + `#include ".../ExecAttrs.cpp.inc"` near the other
     generated includes,
   - in `TCRVExecDialect::initialize()` (`ExecOps.cpp:1188-1193`) add
     ```cpp
     addAttributes<
     #define GET_ATTRDEF_LIST
     #include "TianChenRV/Dialect/Exec/IR/ExecAttrs.cpp.inc"
         >();
     ```
     (mirror `RVVDialect.cpp:14288-14291`),
   - implement the `CapabilityRelationsAttr::verify` body (`genVerifyDecl`) by porting the
     four checks from `verifyCapabilityIDRelationAttr` (`ExecOps.cpp:177-215`).

5. `test/Dialect/Exec/capability-relations-attr.mlir` (NEW, round-trip only)
   - one kernel with `tcrv.exec.capability @c { id = "...",
     relations = #tcrv.capability_relations<conflicts = ["build.policy.no_inline_asm"]> }`,
     `RUN: tcrv-opt %s | tcrv-opt | FileCheck %s` proving parse/print round-trips and the
     verifier accepts a well-formed attr (plus a `-verify-diagnostics` negative for a
     duplicate id).

After this step: `ninja` / `lit` stay green; no consumer reads the new attr yet.

---

## 4. n1_first_evidence (single most convincing decision point)

**`CheckCapabilityRequires` conflict-rejection path** (`lib/Transforms/CheckCapabilityRequires.cpp:162-165`,
via `TargetCapabilitySet::collectAvailableConflictsForCapability`,
`CapabilityModel.cpp:402-435`).

Why this one:
- It is a **real legality decision** (flags unguarded dispatch cases whose required capability
  is unavailable or *conflicting*) — squarely N1, not a mirror/diagnostic.
- It exercises **only the relation contract** (`conflicts`), the pure I1 surface — no
  `properties`/`count` dependency (unlike the HartParallel candidate, which also needs the
  string `count` property we are deferring).
- `collectAvailableConflictsForCapability` is the **single most-reused primitive** (4 passes:
  CheckCapabilityRequires, PluginVariantLegality, VariantSelection, DispatchRuntimeGuard /
  VariantDispatchSynthesis). Migrating it first yields the broadest structural payoff.
- Already **lit-covered** with a conflicts fixture (`guarded_conflict_dispatch`,
  `check-capability-requires.mlir:128-160`).

How wired (still additive, strangler-fig): make
`TargetCapabilitySet::buildFromKernelChecked` / `makeDescriptor`
(`CapabilityModel.cpp:183-195,267`) **prefer the typed** `CapabilityRelationsAttr` when present
on the op, falling back to the legacy string `provides/implies/conflicts` arrays when absent.
This means the conflict decision in CheckCapabilityRequires is driven by the typed attr without
touching the pass or its callers — the descriptor's relation vectors are now sourced from the
typed attr.

How tested:
- **lit:** clone `guarded_conflict_dispatch` into a new case that expresses the conflict via
  `relations = #tcrv.capability_relations<conflicts = ["build.policy.no_inline_asm"]>`
  (and `provides = [...]` on the provider) and assert the same conflict diagnostic /
  guard-required outcome (`test/Transforms/CapabilityRequires/check-capability-requires.mlir`,
  plus `-invalid` pair).
- **unit:** `test/Support/CapabilityModelTest.cpp` — add a case asserting a descriptor built
  from a typed-relations op reports the same `getConflictingIDs()` /
  `collectAvailableConflictsForCapability` result as the legacy string form (parity test).

This is the minimal, testable N1 first structural evidence: one pass making a real legality
decision from a typed capability attribute, with no regression to string-model consumers.

---

## 5. consumer_migration_order

Easiest / highest-evidence first; each step keeps the build green.

1. **`makeDescriptor` / `buildFromKernelChecked`** — the chokepoint
   (`CapabilityModel.cpp:183-195,267`): read typed `CapabilityRelationsAttr` with string
   fallback. This is the bridge that powers everything downstream and is done as part of N1
   first-evidence (§4).
2. **CheckCapabilityRequires** conflict path (N1 first-evidence; §4).
3. **PluginVariantLegality + VariantSelection.analyzeRequirementLegality**
   (`PluginVariantLegality.cpp:56-62`, `VariantSelection.cpp:159-204`) — same
   conflicts-resolution primitive; unit + lit covered.
4. **DispatchRuntimeGuard + VariantDispatchSynthesis**
   (`DispatchRuntimeGuard.cpp:115-148`, `VariantDispatchSynthesis.cpp:86,114`) — same
   `collectAvailableConflictsForCapability`; ride along with step 3's typed primitive.
5. **HartParallelCapabilities** (`HartParallelCapabilities.cpp:44-112`) — `provides`-resolution
   of `target.hart_count` via the typed attr. Defer the `count` property (still string) to a
   later properties-typification step.
6. **Plugin enablement** across RVV / Scalar / Offload / Template / Toy / TensorExtLite
   `isApplicable` (`*ExtensionPlugin.cpp`, uniform `isCapabilityAvailableByID` /
   `lookupProviderByID` + `satisfiesID`) — migrate as one batch behind the typed accessor.
7. **RVVEmitCRoutePlanning** (`RVVEmitC RoutePlanning.cpp:81-90`) — heaviest surface, only
   direct `providesID`/`impliesID` caller plus many string properties; relations first,
   properties later.
8. **Pure-mirror / artifact emitters** (`formatRVV...Mirror`, `TargetArtifactExport.cpp:1215-1216`)
   — I4 mirrors, lowest value, migrate last (or leave on strings until deletion).

After all relation consumers read the typed attr, **drop the legacy string `provides/implies/
conflicts` side attrs from fixtures** and remove the string-array fallback in `makeDescriptor`
and the `verifyCapabilityIDRelationAttr` op-level verifier (now covered by the attr's own
`genVerifyDecl`).

---

## 6. deletion_endgame (dead-mirror removal)

Per the dead-mirror-removal discipline, delete only AFTER every consumer above reads typed
attributes. Two-phase:

- **Phase A — delete the string *relations* (Stage 2/3 boundary):** once steps 2-8 read the
  typed attr, remove (a) `kProvidesAttrName/kImpliesAttrName/kConflictsAttrName` +
  `collectCapabilityIDRelation` (`CapabilityModel.cpp:23-25,102-121`), (b) the string-array
  fallback in `makeDescriptor`, and (c) `verifyCapabilityIDRelationAttr`
  (`ExecOps.cpp:177-215`) and its callers in `CapabilityOp::verify`/`TargetOp::verify`.
  `CapabilityDescriptor` may keep `SmallVector<std::string>` internally if accessors still
  expose IDs, but the IR-side string relations are gone.
- **Phase B — delete the string CapabilityModel itself (later stage, out of this task):** the
  whole stringly-typed `CapabilityDescriptor`/`TargetCapabilitySet` mirror is removed only when
  `id`/`kind`/`status`/`properties` are *also* typed and every consumer queries the attr
  directly. That is a separate Stage (3+) task; Stage 2 does NOT delete the string model.

Stage 2's deletion deliverable is bounded: kill the *string relation representation in IR*,
not the entire model.

---

## 7. Evidence table (file:line)

| Fact | Evidence |
|---|---|
| Relations absent from ODS; CapabilityOp types only sym_name/id/kind | `include/TianChenRV/Dialect/Exec/IR/ExecOps.td:39-53` |
| provides/implies/conflicts read as untyped ArrayAttr<StringAttr> by name, silent-skip | `lib/Support/CapabilityModel.cpp:23-25,102-121` |
| Relation queries are string ==; satisfiesID is string-keyed | `lib/Support/CapabilityModel.cpp:197-201,250-265` |
| Ad-hoc string verifier (hygiene only, no symbol resolution) | `lib/Dialect/Exec/IR/ExecOps.cpp:177-215` |
| Relations reference IDs, NOT symbols (decisive fork) | `test/Transforms/CapabilityRequires/check-capability-requires.mlir:130-141` |
| requires=[@sym] are real symbols, resolved by symbol name (separate namespace) | `.../check-capability-requires.mlir:149-151`; `lib/Transforms/CheckCapabilityRequires.cpp:143-149` |
| By-ID resolution path (lookupProviderByID/collectProvidersByID/conflicts) | `lib/Support/CapabilityModel.cpp:371,388,402-435` |
| Only existing AttrDef = house style to mirror | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:38-41,68-83` |
| RVV attrdef CMake wiring template | `include/TianChenRV/Dialect/RVV/IR/CMakeLists.txt:6-7` |
| RVV addAttributes / GET_ATTRDEF wiring template | `lib/Dialect/RVV/IR/RVVDialect.cpp:14288-14291`; `include/.../RVV/IR/RVVDialect.h:16-17` |
| Exec dialect has no attrdef / no flag / addOperations only | `ExecOps.td:7-21`; `lib/Dialect/Exec/IR/ExecOps.cpp:1188-1193`; `include/.../Exec/IR/CMakeLists.txt:1` |
| Typed-attr C++ consumption precedent (PolicyAttr by type) | `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp:174,187` |
| CapabilityProviderComposition.h is string helper, not typed (do not mirror) | `include/.../Exec/IR/CapabilityProviderComposition.h:14-25` |
| N1 first-evidence conflict path | `lib/Transforms/CheckCapabilityRequires.cpp:162-165` |
| Most-reused primitive = conflicts resolution (4 passes) | `CheckCapabilityRequires.cpp:163`; `VariantSelection.cpp:198`; `VariantDispatchSynthesis.cpp:114`; `DispatchRuntimeGuard.cpp:137` |
| HartParallel candidate also needs string `count` property (why not first) | `lib/Transforms/HartParallelCapabilities.cpp:44-60` |

---

## 8. design_forks_resolved

1. **StringAttr vs FlatSymbolRefAttr for relation entries → StringAttr.** Relations reference
   capability IDs (free dotted strings, own namespace), not symbols; SymbolRefAttr (c1/c3)
   breaks all relation fixtures. Decided by `check-capability-requires.mlir:130-141`.
2. **One CapabilityRelationsAttr wrapper vs three separate array params on the op vs a
   relations-op → one wrapper AttrDef.** Single queryable typed attr keeps the op shape stable,
   matches PolicyAttr house style, and is queried by C++ type in-pass.
3. **New dialect vs extend Exec → extend `tcrv` (Exec).** Capability structure already lives
   there; the AttrDef is the dialect's first.
4. **Reuse CapabilityProviderComposition.h vs not → not.** It is a string helper to bridge
   later, not typed structure to mirror.
5. **First-evidence target: HartParallel (c2) vs CheckCapabilityRequires (c4) →
   CheckCapabilityRequires.** Pure-relation decision (no deferred `count` property),
   migrates the most-reused conflicts primitive, lit-covered.
6. **Additive-first vs in-place swap → additive (strangler-fig).** ~25-30 consumers funnel
   through the descriptor; typed build path + parallel legacy attrs, cut over per
   decision-influence.
7. **Scope of deletion in Stage 2 → string *relations* only, not the whole CapabilityModel.**

---

## 9. open_questions

1. `add_mlir_dialect(ExecOps tcrv)` may already generate an aggregate tablegen target; need to
   confirm at build whether the new `mlir_tablegen(ExecAttrs...)` outputs are auto-picked or
   require an explicit `add_public_tablegen_target` + `add_dependencies` (RVV uses explicit).
2. Mnemonic/printer: `#tcrv.capability_relations<...>` vs default-pretty form — confirm
   `useDefaultAttributePrinterParser=1` yields the intended spelling and that no existing
   fixture/grep accidentally collides on the `capability_relations` mnemonic.
3. Should `genVerifyDecl` reject IDs that also fail to match any provider in the kernel? Today
   the verifier is pure hygiene (no cross-op resolution) and resolution is a pass-time concern;
   keeping it hygiene-only preserves current semantics — confirm we do NOT want symbol-table-
   style resolution at verify time.
4. `TargetOp` also carries relations via `capability_providers`/side attrs — confirm whether
   `TargetOp` needs its own `relations` attr in Stage 2 or only `CapabilityOp` (provider ops
   are the ones declaring provides/conflicts in fixtures).
5. Parity-test granularity: do we want a golden round-trip ensuring typed-attr descriptors are
   byte-identical to string-attr descriptors before deleting the legacy path, or is the
   per-consumer lit/unit coverage sufficient (relevant to Phase A deletion safety)?
