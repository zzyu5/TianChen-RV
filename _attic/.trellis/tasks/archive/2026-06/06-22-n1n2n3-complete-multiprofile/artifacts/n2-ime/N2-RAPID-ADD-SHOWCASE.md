# N2 — adding a second hardware family (IME) by reuse, not core surgery

*A 5-minute, standalone account of the N2 claim: how the Spacemit X60 IME (a non-RVV RISC-V matrix
extension) became a second hardware family in this compiler with **zero core lines** — and why that is a
mature-compiler abstraction, not string-matching. Every number here is mechanically checkable; anchors given.*

## §0 — Thesis
Adding IME cost **0 lines in any core selection/materialization pass** (`lib/Transforms/*`). The core never
learned what "IME" is. The family lives entirely in **1918 isolated plugin lines** + **9 additive wiring
lines**, and is selected by a **derived capability fact**, never a family-name string. The proof that the
wiring was "paid once" is not asserted — it is shown by **two later incremental ops** (a 2nd and 3rd IME op)
that each touched **zero** core/wiring lines.

## §1 — The abstraction: what a family plugin must provide (the "mature compiler" answer)
A new family does **not** edit the core. It implements one `ExtensionPlugin` vtable + one extension dialect and
supplies exactly these hooks; the **generic core passes call them by iterating the plugin registry — they never
branch on family identity** (the same shape as LLVM's `TargetMachine`/`MCTargetDesc` registry):

| Hook (IMEExtensionPlugin.cpp) | What the family supplies | Who calls it |
|---|---|---|
| `registerDialects` (:350) | its ODS extension dialect | generic dialect registration |
| `supportsOperation` / `collectVariantProposals` (:355 / :375) | propose variants **iff the capability fact is present** | `registry.collectVariantProposals` (VariantMaterialization.cpp:596) |
| `verifyVariantLegality` (:392) | plugin-owned fail-closed legality (I7) | generic legality pass |
| `estimateVariantCost` (:438) | the family's cost voice | generic measured/static selection |
| `buildVariantEmissionPlan` / `materializeSelectedLoweringBoundary` (:482 / :537) | emit the family's boundary op | generic materialization |
| `BackendEmissionRegistry` driver (IMEBackendEmissionDriver.cpp) | lower the boundary op via the common EmitC route | `tryConvertModuleWithRegisteredBackend` (EmitCLowerableMaterialization.cpp:268) |

The core dispatches **generically over the registry** — `collectVariantProposals(request,…)` and
`tryConvertModuleWithRegisteredBackend(module)`. There is **no `if (family==…)` anywhere** in core.

## §2 — The 6 reused interfaces (RVV uses them; IME reuses them UNMODIFIED — modified count = 0)
1. `TCRVEmitCLowerableOpInterface` — declared in IME ODS exactly like RVV (`IMEOps.td:36`); the generic EmitC
   route keys on the interface, not the dialect name.
2. `ExtensionPluginRegistry` / `ExtensionBundleRegistry` — IME registers via the same
   `ExtensionPluginRegistrationFn` signature every family uses.
3. `BackendEmissionRegistry` — IME's driver body is **byte-for-byte** RVV's pattern.
4. `CapabilityDescriptor` / `lookupProviderByID` — the shared capability model RVV also queries; neither touches it.
5. `tcrv.exec` variant/dispatch/legality/boundary orchestration — IME wrote **no** orchestration; it filled
   plugin vtable hooks called BY the generic core passes.
6. Common EmitC→C++ route — stock `mlir-translate --mlir-to-cpp`; IME added no exporter.

## §3 — Capability-FACT dispatch, NOT string-matching (the load-bearing anti-string-match section)
The discriminator the core uses is the **presence of a derived capability provider**, keyed on the family's
OWN id, never a family name the core spells:
```
request.getCapabilities().lookupProviderByID("spacemit.ime")->isAvailable()
```
- `"spacemit.ime"` is defined **exactly once** (`IMEExtensionPlugin.cpp:25`) — the plugin queries *itself*; the
  core names no family.
- That provider exists only because **`deriveIMEMatmulCapability` (:137)** derived it from *validated ISA
  evidence*: the march token **`xsmtvdotii`** ⇒ `vmadot` assembles; VLEN=256 / SEW=8 ⇒ the **4×4×8 int32 MAC
  fragment**. The fragment shape is **computed from capability**, not hard-coded — change VLEN and a different
  shape is derived (an N1 capability hook living inside N2).
- **The immature alternative, named and proven absent:**
  `if (op.getName().contains("matmul") && target.march.contains("spacemit")) emitVmadot()` — exists in **no
  core pass**: `grep -rinE '\bime\b|spacemit|vmadot' lib/Transforms/` ⇒ **0** hits.
- **The complementary-arm negative test** `ime-mma-capability-absent-negative.mlir`: an RVV-but-not-IME target
  does NOT satisfy `lookupProviderByID("spacemit.ime")`, so IME **declines**, the op routes elsewhere, and **no**
  `vmadot`/`spacemit` token is materialized. Dispatch is fact-gated, not name-hardcoded.

## §4 — Rapid-add quantification
| Axis | Count | Evidence |
|---|---|---|
| **Core LOC changed** (`lib/Transforms/*`) | **0** | numstat of `2eeabff9` (no Transforms file in diff) + grep 0 family tokens + generic registry dispatch |
| Family-local plugin LOC | **1918** (all NEW, isolated) | `lib/Plugin/IME/`, `lib/Dialect/IME/`, `include/.../IME/` |
| Wiring (4 sites) | **9 added / 0 deleted** | 2 Builtin table rows + `add_subdirectory` lines |
| Reused interfaces (modified) | 6 reused / **0 modified** | §2 anchors |
| **2nd op** (`vmadotu`) core + wiring | **0 / 0** | commit `c7069111` |
| **3rd op** (tiled `matmul`) core + wiring | **0 / 0** | commit `0b8c6168` |
| **4th op** (`vmadotsu`, mixed-sign — LIVE 2026-06-24) | **0 core / 0 wiring / 0 new id; +175/−37 family-local** | commits `70bb845a`+`a75adc27`; tree-wide grep 0 core tokens; trellis-check PASS (I3/I5/I7); **silicon bit-exact 16/16 on real X60**, operand convention CONFIRMED (signed-A v0 × unsigned-B v1, 4-way discriminator), real encoding `e210212b`. The COMPILER-EMITTED helper was the thing tested |
| Runs on real silicon | bit-exact 16/16 | SpacemiT GCC15.2, objdump `smt.vmadot`, X60 |

**On the 4th op and the "string-swap" objection (turned into the demonstration):** a near-sibling op
(`vmadotsu`) still costs **0 core lines** and is selected by a derived **signedness FACT**
(`ime_signedness ⇒ vmadotsu`) flowed as DATA through the materialized variant attribute and read back at
boundary time — there is **no `if (name == "vmadotsu")` anywhere in core; the core never sees the mnemonic.**
The cheapness *is* the thesis: a mature plugin abstraction makes a new execution op a **data-flow change, not
core surgery**. (The test architecture even pre-figured the signedness family — `vmadotsu` was the negative
control; promoting it to a real op just rotates which sibling, `vmadotus`, is the rejected discriminator.)

## §5 — In one line
Family addition is fast **because nothing in the core had to learn what "IME" is** — 1918 isolated lines + 9
additive wiring lines + 0 core lines, dispatched by a derived capability fact, proven by the complementary-arm
test, with the 2nd/3rd/4th ops confirming the wiring cost was paid exactly once.
