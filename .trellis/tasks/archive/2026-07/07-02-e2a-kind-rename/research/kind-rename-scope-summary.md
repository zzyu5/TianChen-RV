# Research: `kind` rename — scope summary & edit sequence

- **Query**: Is this a bounded mechanical rename? What splits KEEP vs RENAME, what regenerates, what could turn it into something larger?
- **Scope**: internal
- **Date**: 2026-07-02
- **Repo HEAD**: `b128e3d5` (all anchors re-grepped here; the `7d781994` in the task is stale).
- Companion: `kind-rename-map.md` (per-occurrence master table).

---

## 1. Total occurrences — KEEP vs RENAME

In-scope exec/capability universe (excluding meaning-C extension noise):

| Bucket | Count | Where |
|---|---|---|
| **KEEP — capability-fact `kind` (A)** | 1 ODS arg + ~16 C++ sites + ~60+ fixture lines | `CapabilityOp` `$kind` (ExecOps.td:112); `CapabilityOp::verify` (ExecOps.cpp:630,632); descriptor accessors (CapabilityModel.h:41, ExtensionPlugin.h:49); `capability.getKind()` throughout plugins/transforms; `TensorExtLiteSourceFrontDoor.cpp:172` (writes capability op); all `tcrv.exec.capability … kind =` fixtures |
| **RENAME → `target_kind` (B)** | 0 ODS · **9 C++ read/verify sites (in 3 files)** · 0 producers · ~13 fixture lines | ExecOps.cpp:{321,591,596,602,612,676,730}; CapabilityProviderComposition.cpp:{204,229}; CapabilityModel.cpp:{45(add),257,295}; + target-op fixtures |
| **RENAME → `region_kind` (B')** | **1 ODS arg** · 2 C++ verify sites · ~4 fixture lines | ExecOps.td:224; ExecOps.cpp:{948,950}; region-op fixtures |
| **SHARED CONSTANT to split** | 2 declarations | `kKindAttrName("kind")` at ExecOps.cpp:36 and CapabilityProviderComposition.cpp:16 |
| **OUT OF SCOPE — extension op-kind (C)** | ~250+ | RVV/IME `.td` + `lib/{Dialect,Plugin,Conversion}/RVV/**` — **must NOT be touched** |

**Verdict: this IS a bounded, mechanical rename.** Total in-scope C++ edit sites = 11 read/verify locations across exactly **3 core files** (`ExecOps.cpp`, `CapabilityProviderComposition.cpp`, `CapabilityModel.cpp`) + 1 shared-constant split, + 1 ODS line, + ~17 fixture lines across 6 test files. There are **zero builders/setters** to change (see §2). The only thing that stops it being a pure `sed` is the shared constant and three coupling points (§3, §4, §5).

---

## 2. Is `region_kind` actually needed? YES — and a key asymmetry

- **`region_kind` IS in scope.** A region-classification `kind` exists today: ODS `RegionOp $kind` (ExecOps.td:224), verifier `RegionOp::verify` (ExecOps.cpp:948), and fixtures (verify.mlir:16,401; basic.mlir:65). So both `target_kind` and `region_kind` are real.
- **Asymmetry the implementer must know:**
  - `region_kind` is **ODS-declared** → its rename edits ODS and **regenerates `ExecOps.{h,cpp}.inc`** (generated accessor `getKind()`→`getRegionKind()`, generated attr-name string). But no C++ calls `RegionOp::getKind()` (verified: the verifier reads by string constant, and no external `RegionOp(...).getKind()` consumer exists), so no accessor-call breakage.
  - `target_kind` is **NOT ODS-declared** (TargetOp's `kind` is a discardable attr-dict attribute). Its rename is **pure C++-string-constant + fixtures, regenerating no `.inc`.**
- **Zero C++ producers for either.** The only C++ that *writes* an exec `"kind"` attr is `TensorExtLiteSourceFrontDoor.cpp:172`, and it writes a **CapabilityOp** (meaning A, KEEP). Target/region kind values only ever arrive from parsed IR/fixtures and are read by the 3 core files. So there is no builder/`state.addAttribute("kind", …)`/setter to rename on the target/region side.

---

## 3. ODS → generated `.inc` → dependent tests (byte-exact gate)

- **ODS file touched: exactly one** — `include/TianChenRV/Dialect/Exec/IR/ExecOps.td` (only line 224, `RegionOp $kind`→`$region_kind`). `CapabilityOp $kind` (line 112) stays; no other ODS changes.
- **Regenerated artifacts:** `build/**/ExecOps.h.inc`, `build/**/ExecOps.cpp.inc` (RegionOp accessor + attr-name string). The `target_kind` half regenerates **nothing** (not ODS).
- **Build caveat (project memory `build-incremental-unreliable`):** this tree's `ExecOps.cpp.inc` regenerates every build and `tcrv-opt` sometimes fails to relink. **The byte-exact gate must use a forced/clean rebuild** and BEFORE==AFTER on dependent artifacts; do not trust incremental. Do not target stale absolute fingerprints.
- **Dependent lit/unit tests (all must be re-run green after rebuild):**
  - `test/Dialect/Exec/verify.mlir` — verifier + expected-error text (target l.161→`target_kind`; region l.391→`region_kind`; capability l.130 stays `kind`; prose l.206 judgment).
  - `test/Dialect/Exec/basic.mlir` — round-trip printer (region CHECK-SAME l.62→`region_kind`).
  - `test/Dialect/Exec/capability-relations-attr.mlir` — round-trip on `tcrv.exec.target @typed_target` (l.40→`target_kind`; re-derive its CHECK).
  - `test/Support/CapabilityModelTest.cpp` — C++ unit test with embedded MLIR (target l.74,81,92→`target_kind`).
  - `test/Transforms/ExecutionPlanning/execution-planning-pipeline-offload.mlir` — target l.66→`target_kind`.
  - `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir` — **export golden**: INPUT target l.9→`target_kind`; **PLAN/HEADER output lines 47,48,69,70 must stay byte-identical** (`;kind=profile` is emitter-hardcoded, see §4).
- **No golden/objdump test asserts on the MLIR attribute name.** objdump/byte goldens assert on emitted instruction/artifact bytes, not on `kind`. The single artifact string that contains `kind=` (`;kind=profile`) is produced by a hardcoded emitter literal, not the attr name, so it survives the rename. This means the rename is expected to be **byte-neutral on all exported artifacts** — that is the pass condition of the gate.

---

## 4. Must-land-atomically coupling (or the export golden breaks)

Renaming the fixture attr `kind→target_kind` on target ops is byte-safe **only if these C++ target-kind READERS are renamed in the same commit**:

- `CapabilityModel.cpp:257` and `:295` — `makeDescriptor(target, getStringAttr(target, "kind"))` → `"target_kind"`.
- `CapabilityProviderComposition.cpp:229` — `getCapabilityProviderKind` TargetOp branch reads `kKindAttrName` → `kTargetKindAttrName`.

Why: the target op's `kind`/`target_kind` value is routed **into the `CapabilityDescriptor.kind` field**, which the RVV route-planning mirror prints as the hardcoded `";kind=" << capability.getKind()` (`RVVEmitCRoutePlanning.cpp:115,127`). If a fixture renames the attr to `target_kind` but any of these readers still looks up `"kind"`, the descriptor's kind goes **empty** → the exported PLAN/HEADER lines print `kind=` (empty) → `explicit-selected-body-artifact*.mlir` breaks. Rename fixture + these three readers together, atomically.

Also **required in the same change** (behavior/byte preservation): `isCoreCapabilityAttribute` (`CapabilityModel.cpp:45`) must **add `"target_kind"`** (keep `"kind"`). It runs at l.87 `collectCapabilityProperties` over target-provider attrs; if `target_kind` isn't recognized as a core attr it leaks into the capability *property* map and can alter mirrors/exports.

---

## 5. Genuinely ambiguous / needs a human-or-spec decision

1. **`[Semantic-latent]` Target kind flows into the capability-fact descriptor.** Today `getCapabilityProviderKind`/`makeDescriptor` store the **target op-classification value** into `CapabilityDescriptor.kind` (the [S-1] closed-enum field). Spec is in tension: `tcrv-exec-contract.md:121,146` say a provider target carries both `id` and `kind` (structured provider), while `:133` says a **profile-provider carries `provides`, NOT a capability-fact `kind`**. A *mechanical* rename preserves current behavior (read under new name, still store into `descriptor.kind`) and is byte-neutral. But it does **not** resolve whether a target's `target_kind` should feed the capability-fact axis at all. **Recommendation: do the mechanical rename only; flag the provides-vs-kind semantic cleanup as a separate, out-of-scope task.** Do not change data flow here.

2. **`[Ambiguity/Prose]` Hardcoded error-message wording.** Three error strings say "id and kind" / "id/kind" as prose, not via the constant: `ExecOps.cpp:612`, `ExecOps.cpp:676`, `CapabilityProviderComposition.cpp:83`. Constant-derived messages (ExecOps.cpp:596 target, :950 region) auto-update when the constant is split and their expected-error CHECKs (`verify.mlir:161`, `:391`) **must** update. For the hardcoded prose, it is a **judgment call** whether to reword `kind`→`target_kind` for consistency (then `verify.mlir:206` and the l.161 wording must track). Recommendation: reword to `target_kind` for consistency and update the two matching CHECKs; either way it is a wording choice, not a behavior change.

3. **`[Same-text, different-op]` `verify.mlir:130` vs `:391`.** Both expected-errors read `requires non-empty string attribute 'kind'`, but l.130 is the **CapabilityOp** case (stays `kind`) and l.391 is the **RegionOp** case (→`region_kind`). They must be classified per-op, not by text match — a text-based find/replace would wrongly touch both.

No other occurrence is ambiguous; every remaining site classifies cleanly by which op/struct it operates on.

---

## 6. Recommended edit sequence

Split into a **region sub-sequence** (touches ODS+`.inc`) and a **target sub-sequence** (pure C+++fixtures), then one gate.

**A. Region (`region_kind`) — ODS-first:**
1. ODS: `ExecOps.td:224` `$kind`→`$region_kind`.
2. Forced/clean rebuild so `ExecOps.{h,cpp}.inc` regenerate (per §3 build caveat).
3. C++: split constant — add `kRegionKindAttrName("region_kind")`; repoint `ExecOps.cpp:948,950`.
4. Fixtures: `verify.mlir:16,401` + `:391` expected-error; `basic.mlir:65` + `:62` CHECK-SAME.

**B. Target (`target_kind`) — C++ constant + fixtures (no `.inc`), land atomically per §4:**
5. C++: add `kTargetKindAttrName("target_kind")`; repoint `ExecOps.cpp:{321,591,596,602,730}` and (prose) `612,676`; `CapabilityProviderComposition.cpp:{16→or add,204,229}` and (prose) `83`; `CapabilityModel.cpp:{257,295}` string literals; **add `"target_kind"` to `isCoreCapabilityAttribute` (l.45)**.
6. Leave `kKindAttrName("kind")` in place for CapabilityOp (ExecOps.cpp:630,632) and all descriptor `getKind()` — untouched.
7. Fixtures: target ops in `verify.mlir:{155,168,175,185,212,221,233}` + `:161` (`target_kind`) + `:206` (prose judgment); `capability-relations-attr.mlir:40` (+CHECK); `execution-planning-pipeline-offload.mlir:66`; `CapabilityModelTest.cpp:{74,81,92}`; `explicit-selected-body-artifact*.mlir:9` **only** (leave l.47,48,69,70 untouched — §4).

**C. Gate:**
8. Forced/clean rebuild; run the 6 dependent tests (§3) + the full exec/capability lit suite; confirm the export golden `explicit-selected-body-artifact*` PLAN/HEADER bytes are **unchanged** (byte-neutral is the pass condition).
9. Do NOT touch the ~250+ meaning-C RVV/IME `kind` sites; a targeted grep of the diff should show changes confined to `Dialect/Exec/`, `Support/CapabilityModel.*`, and the 6 test files (+ regenerated `Exec*.inc`).

No build/CI/falsifier script depends on the attribute spelling (checked `.trellis/scripts/`, `scripts/` — no grep-on-`kind` falsifier); the spec's [F-1] reference is conceptual.
