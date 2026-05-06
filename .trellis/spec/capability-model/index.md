# Capability Model Specs

Capability model is the first core of TianChen-RV MLIR. It turns target facts into MLIR-visible objects that affect passes.

## Pre-Development Checklist

- [ ] Are ISA, uarch, runtime/offload, and toolchain facts represented as queryable capability data?
- [ ] Does this change make at least one pass decision depend on capability data where appropriate?
- [ ] Are capability relations such as require/provide/imply/conflict/dispatch condition explicit?
- [ ] Are unavailable toolchains or runtime APIs represented as legality failures or diagnostics?
- [ ] Are current and future target profiles kept distinct?

## Guidelines Index

| Spec | Description |
|---|---|
| [Capability Contract](./capability-contract.md) | Object shape、sources、relations、verifier expectations |
| [Target Profiles](./profiles.md) | RVV main、K3/IME later、RISC-V Sophgo/offload profiles |

## Quality Check

- Capability cannot be a plain string attached after lowering.
- If a variant requires an extension, the requirement must be represented in `#tcrv.requires<...>` or equivalent structured data.
- If a hardware/runtime/toolchain feature is absent, the system must either reject the variant or generate a dispatch/fallback path with diagnostics.
