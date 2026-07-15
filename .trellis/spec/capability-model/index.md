# Capability Model Specs

Capability model is the first core of Weft-RV MLIR. It turns target facts into MLIR-visible objects that affect passes.

## Pre-Development Checklist

- [ ] Are ISA, uarch, runtime/offload, and toolchain facts represented as queryable capability data?
- [ ] Does this change make at least one pass decision depend on capability data where appropriate?
- [ ] Are capability relations such as require/provide/imply/conflict/dispatch condition explicit?
- [ ] Are unavailable toolchains or runtime APIs represented as legality failures or diagnostics?
- [ ] Are current and future target profiles kept distinct?
- [ ] Is the capability model implemented as C++/MLIR compiler objects, not Python/JSON-only structures?
- [ ] Are capability facts used only to constrain legality/realization, rather than creating RVV dtype/config/body/route identities?
- [ ] If RVV dtype/config affects executable lowering, is it structural in typed `weft_rvv` body or consumed into realized body before route construction?

## Guidelines Index

| Spec | Description |
|---|---|
| [Capability Contract](./capability-contract.md) | Object shape、sources、relations、verifier expectations |
| [Target Profiles](./profiles.md) | RVV main、IME、RISC-V Sophgo/offload profiles |

## Quality Check

- Capability facts are structured records with `kind` from the closed enum `{isa_ext, sub_ext, uarch, policy}`, namespaced `params`, and `provenance`/`trust` fields ([S-1]); not free strings.
- `implies` is a transitive closure (**compile-time today** via the capability model, consumed by the compile-time selection pass; **load-time resolution per [D-2a] is a 目标契约·未实现** target contract, not yet implemented); `conflicts` is fail-closed (executed 编译期 by VariantSelection, `VariantSelection.cpp:667/701`); an unknown/absent fact counts as false ([S-2]).
- Probes emit facts only — no "probed X therefore route Y" ([S-3]); uarch quirks are table-keyed facts, not `if (core == ...)` branches ([S-4]).
- The capability shape is the six-item `schema.def` declaration artifact; family-onboarding PRs must not touch it ([S-5]/[F-2′]).
- A family is a capability-declaration + ownership boundary, not instruction density; "capability-gate families" (e.g. the vector-absent scalar family) are legal ([S-8]).
- Capability cannot be a plain string attached after lowering.
- Capability facts such as VLEN, dtype throughput, preferred LMUL, or toolchain support must not directly become route ids, dtype authority, intrinsic choices, or artifact names.
- If a variant requires an extension, the requirement must be represented in `#weft.requires<...>` or equivalent structured data.
- If a hardware/runtime/toolchain feature is absent, the system must either reject the variant or generate a dispatch/fallback path with diagnostics.
- Capability model behavior requires lit/FileCheck coverage for IR syntax/verification and C++ tests for registry/helper APIs where appropriate.
