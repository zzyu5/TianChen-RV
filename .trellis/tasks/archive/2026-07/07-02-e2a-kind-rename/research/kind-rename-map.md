# Research: `kind` → `target_kind` / `region_kind` rename map

- **Query**: Scope a bounded, mechanical rename of the op-classification `kind` attribute
  (on `tcrv.exec.target` → `target_kind`; on `tcrv.exec.region` → `region_kind`) while
  KEEPING the capability-fact `kind` (on `tcrv.exec.capability` / `CapabilityDescriptor`).
- **Scope**: internal (code + ODS + lit fixtures)
- **Date**: 2026-07-02
- **Repo HEAD at audit**: `b128e3d5` (NOTE: past the `7d781994` cited in the task; every anchor below re-grepped against `b128e3d5`, not prior audits).
- **Canon decision**: git `7d781994` — "op-attribute kind → target_kind/region_kind (distinct from capability-fact kind); code-only rename."
- **Contract**: `.trellis/spec/core-dialect/tcrv-exec-contract.md:95` (`target_kind` example), `:137` (命名消歧契约), `:159` ([S-1] capability-fact `kind` closed enum stays).

---

## The three meanings of `kind` in this tree (disambiguation)

| # | Meaning | Where | Decision |
|---|---|---|---|
| A | **Capability-fact `kind`** — closed enum `{isa_ext, sub_ext, uarch, policy}` ([S-1]) | `tcrv.exec.capability` op + `CapabilityDescriptor` C++ struct + all `capability.getKind()` | **KEEP `kind`** |
| B | **Op-classification `kind`** — profile / capability-provider classification | `tcrv.exec.target` op attr (discardable, NOT ODS-declared) | **→ `target_kind`** |
| B'| **Region-classification `kind`** — extension-resource region tag | `tcrv.exec.region` op attr (ODS-declared `$kind`) | **→ `region_kind`** |
| C | **Extension op-classification `kind`** — RVV/IME operation identity (`"add"`, `"copy"`, `ggml_*_block_dot`, …) | `tcrv_rvv.*`, RVV plugin/EmitC, RVVDialect* | **OUT OF SCOPE — do NOT touch** (~250+ occurrences; third meaning, unrelated) |

Central complication: `kKindAttrName("kind")` is a **single shared string constant** (declared twice) currently used for meanings A, B, and B'. The rename requires **splitting it by call site**, not a global find/replace.

---

## Master table (ranked by file)

Legend — Meaning: `A`=cap-fact(KEEP) · `B`=target(→target_kind) · `B'`=region(→region_kind) · `C`=extension(OUT).

### ODS — `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`

| line | op / struct | current | meaning | decision | notes |
|---|---|---|---|---|---|
| 112 | `CapabilityOp` (`tcrv.exec.capability`) arg `$kind` | `kind` | A | **KEEP** | `OptionalAttr<StrAttr>`; the [S-1] capability-fact enum. No ODS change. |
| 224 | `RegionOp` (`tcrv.exec.region`) arg `$kind` | `kind` | B' | **→ `region_kind`** | `OptionalAttr<StrAttr>`, no default. **Only ODS edit in the whole task.** Regenerates `ExecOps.{h,cpp}.inc`; generated accessor `getKind()`→`getRegionKind()` and the generated attr-name string `"kind"`→`"region_kind"`. |
| 87–99 | `TargetOp` (`tcrv.exec.target`) | — | B | (see C++) | **TargetOp has NO ODS `kind` arg** (args = `sym_name`, `capability_providers`, `relations`; `assemblyFormat = "$sym_name attr-dict"`). Target's `kind` lives as a **discardable attr-dict attribute**, read only via C++ string literal. So the target rename touches **no ODS and regenerates no `.inc`.** |
| 245 | `DiagnosticOp` `$selection_kind` | `selection_kind` | — | **KEEP** | Compound name, not the bare disambiguated `kind`. |
| 248 | `DiagnosticOp` `$plan_kind` | `plan_kind` | — | **KEEP** | idem |
| 249 | `DiagnosticOp` `$emission_kind` | `emission_kind` | — | **KEEP** | idem |
| 252 | `DiagnosticOp` `$runtime_abi_kind` | `runtime_abi_kind` | — | **KEEP** | idem |
| 256 | `DiagnosticOp` `$artifact_kind` | `artifact_kind` | — | **KEEP** | idem |

### C++ core — `lib/Dialect/Exec/IR/ExecOps.cpp`

| line | context | current | meaning | decision | notes |
|---|---|---|---|---|---|
| 36 | `constexpr llvm::StringLiteral kKindAttrName("kind");` | `kind` | A+B+B' | **SPLIT** | Shared constant. Recommended: keep `kKindAttrName("kind")` for A, add `kTargetKindAttrName("target_kind")` for B and `kRegionKindAttrName("region_kind")` for B', then repoint each call site below. |
| 321 | `!isMissingOrEmptyStringAttr(target..., kKindAttrName)` (id+kind provider gate) | `kind` | B | **→ target** | reads TargetOp's kind. |
| 588–592 | `TargetOp::verify()` reads `kKindAttrName` (l.591); pairs id/kind (l.592) | `kind` | B | **→ target** | |
| 596 | error msg `<< kIdAttrName << "' and '" << kKindAttrName` | `kind` | B | **→ target** | Constant-derived → message text becomes `'target_kind'`. Matches `verify.mlir:161` expected-error. |
| 601–602 | `requireStableSingleLineWhenPresent(op, kKindAttrName)` (target) | `kind` | B | **→ target** | |
| 612 | hardcoded prose `"id and kind capability identity"` | `kind` | B | **→ target (judgment)** | Human-facing prose, NOT constant-derived. Reword to `target_kind`? Matches part of the capability_providers error. See `[Ambiguity/Prose]` in summary. |
| 625–632 | `CapabilityOp::verify()` reads `kKindAttrName` (l.630, 632) | `kind` | A | **KEEP** | capability-fact requiredness check. |
| 676 | hardcoded prose `"non-empty id and kind"` (kernel `target=@profile` ref verify) | `kind` | B | **→ target (judgment)** | Matches `verify.mlir:206` expected-error. Prose reword judgment. |
| 730–732 | provider-composition reads `kKindAttrName` on TargetOp | `kind` | B | **→ target** | |
| 947–950 | `RegionOp::verify()` reads `kKindAttrName` (l.948); error msg (l.950) | `kind` | B' | **→ region** | Constant-derived → message text becomes `'region_kind'`. Matches `verify.mlir:391` expected-error. |

### C++ core — `lib/Dialect/Exec/IR/CapabilityProviderComposition.cpp`

| line | context | current | meaning | decision | notes |
|---|---|---|---|---|---|
| 16 | `constexpr llvm::StringLiteral kKindAttrName("kind");` | `kind` | B | **→ target** | In this file the constant is used **only** for TargetOp reads (204, 229). Rename/point to `kTargetKindAttrName`. |
| 83 | hardcoded prose `"...does not carry non-empty id/kind"` | `kind` | B | **→ target (judgment)** | prose reword judgment. |
| 201–204 | `isCapabilityProviderTarget(TargetOp)` reads `kKindAttrName` (l.204) | `kind` | B | **→ target** | Gate that decides if a target is a capability provider. |
| 225–229 | `getCapabilityProviderKind(op)`: TargetOp branch reads `kKindAttrName` (l.229) | `kind` | B | **→ target** | **COUPLED with export golden** — see l.227 next row + summary §4. |
| 227 | same fn, CapabilityOp branch: `capability.getKind()` (struct accessor) | `kind` | A | **KEEP** | descriptor accessor, not an op attr. |

### C++ core — `lib/Support/CapabilityModel.cpp`

| line | context | current | meaning | decision | notes |
|---|---|---|---|---|---|
| 45 | `isCoreCapabilityAttribute`: `attrName == "kind"` in allow-list | `kind` | A+B | **KEEP + ADD `target_kind`** | Generic filter run over **both** capability and target provider attrs (called at l.87 `collectCapabilityProperties`). Keep `"kind"` (A) **and add `"target_kind"` (B)**, else target's classification attr leaks into the capability *property* map → behavior/byte change. **Required coupled edit.** `region_kind` NOT needed (regions aren't capability providers). |
| 87 | `if (isCoreCapabilityAttribute(attrName) || ...) continue;` | — | A+B | context | confirms the l.45 allow-list gates target-op attrs. |
| 257 | `makeDescriptor(target..., getStringAttr(target, "kind"))` (referenced module profile) | `kind` (literal) | B | **→ `"target_kind"`** | String literal, not the constant. **COUPLED with export golden** (summary §4). |
| 282 | `capability.getKind().value_or("")` (CapabilityOp descriptor) | `kind` | A | **KEEP** | |
| 295 | `makeDescriptor(target..., getStringAttr(target, "kind"))` (kernel-local provider) | `kind` (literal) | B | **→ `"target_kind"`** | **COUPLED with export golden** (summary §4). |
| 439 | `if (capability.getKind() == kind)` (descriptor query) | `kind` | A | **KEEP** | `kind` here is a local param compared against descriptor field. |

### C++ — capability-fact accessors & producer (all KEEP)

| file:line | context | meaning | decision |
|---|---|---|---|
| `include/TianChenRV/Support/CapabilityModel.h:41` | `llvm::StringRef getKind() const { return kind; }` — `CapabilityDescriptor` struct accessor | A | **KEEP** |
| `include/TianChenRV/Plugin/ExtensionPlugin.h:49` | `llvm::StringRef getKind() const { return kind; }` — capability descriptor struct accessor | A | **KEEP** |
| `lib/Plugin/ExtensionPlugin.cpp:320,330,769` | `capability.getKind()` | A | **KEEP** |
| `lib/Plugin/{Offload,IME,TensorExtLite,Toy,Template}/*.cpp` (`getKind() != kXxxCapabilityKind`, ~8 sites) | capability-fact checks | A | **KEEP** |
| `lib/Transforms/CheckCapabilityRequires.cpp:251` | `capability.getKind()` in diagnostic | A | **KEEP** |
| `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp:115,127,218–239` | `capability.getKind()` + hardcoded `";kind="` mirror token; `facts.selectedProviderKind` | A / artifact | **KEEP** | see summary §4 — the `;kind=` mirror string is emitter-hardcoded and independent of the MLIR attr name. |
| `lib/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.cpp:165–172` | `createTensorExtLiteCapability` builds a `tcrv.exec.capability` op and sets `"kind"` = `getTensorExtLiteFragmentCapabilityKind()` | A | **KEEP** — **verified it builds a CapabilityOp, not a TargetOp.** This is the ONLY C++ producer of an exec `"kind"` attr; it writes a capability-fact kind → stays. (Confirms `target_kind`/`region_kind` have **zero** C++ producers.) |

### lit / test fixtures

Target-op `kind =` → **`target_kind =`**:

| file:line | occurrence | decision |
|---|---|---|
| `test/Dialect/Exec/verify.mlir:155,168,175,185,212,221,233` | `tcrv.exec.target @… {… kind = "profile" …}` | **→ target_kind** |
| `test/Dialect/Exec/verify.mlir:161` | expected-error `…both non-empty string attributes 'id' and 'kind'` | **→ `'target_kind'`** (tracks constant-derived msg at ExecOps.cpp:596) |
| `test/Dialect/Exec/verify.mlir:206` | expected-error `…with non-empty id and kind` | **→ judgment** (tracks hardcoded prose ExecOps.cpp:676) |
| `test/Dialect/Exec/capability-relations-attr.mlir:40` | `tcrv.exec.target @typed_target {… kind = "build-policy" …}` | **→ target_kind**; re-derive any round-trip CHECK for `@typed_target` (l.37 CHECK-LABEL) |
| `test/Transforms/ExecutionPlanning/execution-planning-pipeline-offload.mlir:66` | `tcrv.exec.target @module_offload_scalar_profile {… kind = "profile" …}` | **→ target_kind** |
| `test/Support/CapabilityModelTest.cpp:74,81,92` | embedded MLIR: `tcrv.exec.target … kind = "profile"` | **→ target_kind** (C++ raw-string fixtures) |
| `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir:9` | `tcrv.exec.target @rvv_profile {… kind = "profile" …}` (INPUT) | **→ target_kind** |

Region-op `kind =` → **`region_kind =`**:

| file:line | occurrence | decision |
|---|---|---|
| `test/Dialect/Exec/verify.mlir:16,401` | `tcrv.exec.region attributes {kind = "extension-resource" …}` | **→ region_kind** |
| `test/Dialect/Exec/verify.mlir:391` | expected-error `requires non-empty string attribute 'kind'` (region-missing-kind case, kernel `@missing_region_kind` l.388, region l.392 has no kind) | **→ `'region_kind'`** (tracks ExecOps.cpp:950) |
| `test/Dialect/Exec/basic.mlir:65` | `tcrv.exec.region attributes {kind = "extension-resource", name=…, purpose=…}` | **→ region_kind** |
| `test/Dialect/Exec/basic.mlir:62` | `// CHECK-SAME: kind = "extension-resource"` (region round-trip) | **→ `region_kind`** |

Fixtures that STAY `kind` (capability-fact, meaning A) — **do NOT touch**:

- All `tcrv.exec.capability … kind = …` in every file. High-count in `verify.mlir` (`kind = "isa-vector"`, `"toolchain"`, etc.) and `CapabilityModelTest.cpp:86–131,537–545`, `execution-planning-pipeline-offload.mlir:8–157`, `capability-relations-attr.mlir:12–47` (except l.40 target), `basic.mlir:11,16` (+ CHECK-SAME l.10,15).
- `verify.mlir:130` expected-error `requires non-empty string attribute 'kind'` — this is the **CapabilityOp** missing-kind case (ExecOps.cpp:632) → **KEEP `'kind'`**. (Contrast l.391 which is the region case → region_kind. Same message text, different op — must be classified individually.)
- `verify.mlir:216` `tcrv.exec.capability @shadowed_module_target {id = "local.shadow", kind = "profile"}` — value is `"profile"` but it is a **capability op** → **KEEP** attr name `kind`. (Value ≠ attr; do not be fooled by the `"profile"` value.)
- `explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir:11` `tcrv.exec.capability @scalar_fallback {… kind = "fallback"}` → KEEP.

Fixtures that STAY `kind` (emitted artifact string, NOT an MLIR attr) — **do NOT touch**:

- `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir:47,48,69,70` — PLAN/HEADER CHECK lines assert on `…;id=rvv.profile.rv64gcv;kind=profile;rvv=provides`. The `kind=` token is **emitter-hardcoded** (`RVVEmitCRoutePlanning.cpp:115,127`) and its value flows through `getCapabilityProviderKind`→descriptor, unchanged by the attr rename. These lines **must stay byte-identical**. (This is the byte-exact linchpin: renaming the input attr on l.9 does NOT change the exported artifact bytes — provided the coupled C++ reads in summary §4 are renamed in lockstep.)

Fixtures with RVV op `kind =` (meaning C) — **do NOT touch**:

- `explicit-selected-body-artifact…:28` `tcrv_rvv.macc … kind = "add" …` and any other `tcrv_rvv.*` op kind in fixtures.

---

## OUT OF SCOPE bucket (meaning C) — do NOT rename

~250+ `kind` / `getKind()` occurrences are RVV/IME/extension **operation-classification** and are unrelated to the exec op-classification and capability-fact axes:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (253 case-insensitive matches).
- `lib/Dialect/RVV/IR/RVVDialect*.cpp`, `RVVDialectWideningOps.cpp` (`getKind() != "ggml_*_block_dot"`, `"signed_widening_product"`, `"add"`, …), `RVVDialectArithmeticOps/ReductionOps/StoreOps/ControlOps.cpp`.
- `lib/Plugin/RVV/**` (front-doors & selected-body owners `state.addAttribute("kind", …)` for op identity), `lib/Plugin/RVV/EmitC/**` (`RVVEmitCRouteAnalysis.cpp`, `RVVEmitCRouteFamilyDerivation.cpp`, etc. — `compareOp.getKind()`, `move.getKind()`, …).
- `lib/Conversion/RVV/**` (`RVVToEmitC.cpp`, `RVVToEmitCDeferredDequant.cpp`, `RVVToEmitCBlockQuantLinear.cpp:5224` reads an RVV op `"kind"`).
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp:478,505` (`blockDot->getAttrOfType("kind")` → `fc.kindMetadataKey` artifact) — RVV block-dot op kind, not exec.
- `handoff_kind` (TensorExtLiteSourceFrontDoor.cpp:178) — compound name, extension attr, KEEP.

These are listed so the implementer can positively exclude them; a naive `s/"kind"/"target_kind"/` would corrupt all of them.

**Remaining `.td` files verified out of scope** (opened, not assumed):
- `include/TianChenRV/Dialect/{Offload,Toy,TensorExtLite,Template}/IR/*Ops.td` — each has one `StrAttr:$handoff_kind` on an extension handoff op (compound name, meaning C). No bare exec/capability `kind`.
- `include/TianChenRV/Transforms/Passes.td:286,834` — `kind` appears only in pass-description prose ("RVV-kind", "kind, callback, and runtime ABI parameter"), not as an op attribute. N/A.
