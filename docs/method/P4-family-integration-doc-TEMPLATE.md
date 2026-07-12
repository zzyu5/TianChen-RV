# [P-4] Family Integration Doc — TEMPLATE

> **Purpose.** The fill-in skeleton for the per-family **integration doc** that
> [P-4] External Integrability requires ("a non-core author can complete one family
> integration from the docs alone"). Copy this file to
> `docs/reports/<date>-<Fam>-family-integration.md`, replace every `<…>` slot, and
> delete the guidance blockquotes. This closes [GAP-P4-NO-P4-TEMPLATE].
>
> **Authoritative protocol contracts referenced below** (read first):
> `.trellis/spec/plugin-protocol/extension-plugin-integration.md` (Standard Flow,
> [P-2] five-piece, [GAP-P4-REGISTER], landing points),
> `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
> ([GAP-P4-TOUCHSET] real touch-set), `.trellis/spec/capability-model/capability-contract.md`
> ([S-1]/[S-2], [GAP-P4-SCHEMA-DIVERGENCE]),
> `docs/method/REPOSITORY-MAP-五大件.md`, `docs/method/FALSIFIER-INDEX.md`.

---

## 0. Family identity

| Field | Value |
|---|---|
| Family name | `<Fam>` |
| MLIR dialect namespace | `weft_<fam>` |
| Dispatch key(s) | `<fam>` |
| Target capability id(s) | `<fam>.<...>` |
| Toolchain / runtime assumptions | `<...>` |
| Maturity | `<real / scaffold / not-yet-built>` |
| Capability-gate family? ([S-8]) | `<yes: owned body uses base ISA, facts gate; / no>` |

## 1. Delivery five-piece ([P-2], the acceptance axis — 缺一不收)

| # | Piece | Landed at | Status |
|---|---|---|---|
| ① | facts + relation rows | `lib/Plugin/<Fam>/<Fam>ExtensionPlugin.cpp` `getCapabilities()` (C++, PluginCapability) | `<…>` |
| ② | legality predicate | `lib/Plugin/<Fam>/<Fam>VariantLegality.cpp` | `<…>` |
| ③ | emission pattern (or explicit "reuse existing") | `<Fam>ConstructionProtocol.cpp` + `<Fam>EmitCRouteProvider.cpp` + `<Fam>BackendEmissionDriver.cpp` | `<…>` |
| ④ | tests (lit byte-exact + per-board objdump golden) | `test/**` | `<…>` |
| ⑤ | ledger row (C2 cost) | `docs/method/C2_marginal_cost_ledger.md` | `<…>` |

> Note the capability-fact **shape** is aspirational vs the fields code consumes
> ([GAP-P4-SCHEMA-DIVERGENCE]): record which fact keys you actually rely on.

## 2. Real touch-set ([GAP-P4-TOUCHSET]) — the 6 roots + registration

> List the concrete files you created/edited. The reference `Template` family =
> 6 directory roots / 20 source files / 6 CMakeLists + 1 shared registration.

```text
lib/Plugin/<Fam>/                      <5 .cpp + CMakeLists>
include/Weft/Plugin/<Fam>/       <4 headers>
lib/Dialect/<Fam>/ (+IR/)              <Dialect.cpp + 2 CMakeLists>
include/Weft/Dialect/<Fam>/(+IR/) <Dialect.h + Ops.td + 2 CMakeLists>
lib/Target/<Fam>/                      <TargetSupportBundle.cpp + CMakeLists>
include/Weft/Target/<Fam>/       <TargetSupportBundle.h>
lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp   <#include + kBuiltinExtensionBundles[] row>  ← [GAP-P4-REGISTER]
schema/family-manifest.v1.json + family-regex.v1.json (+ family-dirs.v1.json)  <territory + dispatch-key rows>
```

- **Registration done?** `register<Fam>ExtensionPlugin` declared in the family
  header AND added to `kBuiltinExtensionBundles[]`: `<yes/no>`.
- **Build links?** `cmake --build build` reaches the family targets
  (`Weft<Fam>Dialect`, `Weft<Fam>Target`): `<yes/no>`.

## 3. Standard-flow conformance

> Confirm the executable flow (spec §Standard Flow): selected `weft.exec` variant
> → typed `weft_<fam>` body → plugin legality → optional realization → plugin-built
> `WEFTEmitCLowerableRoute` → common EmitC → target artifact.

- Typed body / selected boundary: `<...>`
- Route provider builds `WEFTEmitCLowerableRoute`: `<yes/no>`
- Common EmitC only materializes provider payload (no extension semantics in core): `<yes/no>`

## 4. Falsifier six-gate status (机检验收)

> 见 `docs/method/FALSIFIER-INDEX.md`. Fill the gates the family touches.

| Gate | What it checks | Rows added | Status |
|---|---|---|---|
| [F-1] zero-branch | no `if <fam>` in core (`family-regex.v1.json` core_scope) | `<...>` | `<GREEN/RED>` |
| [F-3] containment | PR touches only family territory + allowances (`family-manifest.v1.json`) | `<...>` | `<GREEN/RED>` |
| [F-6] independent-family closure | `implies` ∩ `rvv.*` = ∅ (if independent) | `<...>` | `<n/a or GREEN/RED>` |
| others | `<...>` | `<...>` | `<...>` |

## 5. Evidence

- Compile/lit (byte-exact): `<lit names>`
- Silicon (per-board objdump golden / on-device): `<board id + result, or "compile-tier only">`
- Correctness/perf claims: real hardware only (I8). `<...>`

## 6. External-followability self-check ([P-4])

- [ ] All 8 [GAP-P4-*] gaps navigable from the docs I used (touch-set, register, schema-divergence, five-piece cross-map, landing points, this template, README entry, hook signatures)?
- [ ] Could a non-core author reproduce this integration from spec + this doc alone (no reading of RVV/IME internals)?
- [ ] Named residual gaps (honesty — no "实质胜利"): `<...>`
