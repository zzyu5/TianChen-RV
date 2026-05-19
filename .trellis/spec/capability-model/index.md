# Capability Model Specs

Capability model is the first core of TianChen-RV MLIR. It turns target facts into MLIR-visible objects that affect passes.

## Pre-Development Checklist

- [ ] Are ISA, uarch, runtime/offload, and toolchain facts represented as queryable capability data?
- [ ] Does this change make at least one pass decision depend on capability data where appropriate?
- [ ] Are capability relations such as require/provide/imply/conflict/dispatch condition explicit?
- [ ] Are unavailable toolchains or runtime APIs represented as legality failures or diagnostics?
- [ ] Are current and future target profiles kept distinct?
- [ ] Is the capability model implemented as C++/MLIR compiler objects, not Python/JSON-only structures?
- [ ] Are capability facts used only to constrain legality/realization, rather than creating RVV dtype/config/body/route identities?
- [ ] If RVV dtype/config affects executable lowering, is it structural in typed `tcrv_rvv` body or consumed into realized body before route construction?

## Guidelines Index

| Spec | Description |
|---|---|
| [Capability Contract](./capability-contract.md) | Object shape、sources、relations、verifier expectations |
| [Target Profiles](./profiles.md) | RVV main、K3/IME later、RISC-V Sophgo/offload profiles |

## Quality Check

- Capability cannot be a plain string attached after lowering.
- Capability facts such as VLEN, dtype throughput, preferred LMUL, or toolchain support must not directly become route ids, dtype authority, intrinsic choices, or artifact names.
- If a variant requires an extension, the requirement must be represented in `#tcrv.requires<...>` or equivalent structured data.
- If a hardware/runtime/toolchain feature is absent, the system must either reject the variant or generate a dispatch/fallback path with diagnostics.
- Capability model behavior requires lit/FileCheck coverage for IR syntax/verification and C++ tests for registry/helper APIs where appropriate.
