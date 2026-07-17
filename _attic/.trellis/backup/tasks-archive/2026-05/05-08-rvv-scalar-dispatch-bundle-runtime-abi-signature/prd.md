# RVV Scalar Dispatch Bundle Runtime ABI Signature

## Goal

Make the compiler-produced target artifact bundle index self-describing at the external C callable ABI boundary for the bounded RVV+scalar i32-vadd dispatch component group.

## Requirements

- Add ordered runtime ABI parameter metadata to target artifact bundle records using the existing C++ `support::RuntimeABIParameter` model.
- Emit stable bundle index fields for the dispatch external ABI signature, including each parameter's `c_name`, `c_type`, `role`, and `ownership`.
- Populate RVV+scalar dispatch source/header/object composite bundle records from the central i32-vadd dispatch runtime ABI contract.
- Validate grouped bundle records fail closed when signature metadata is missing, duplicated by role, mismatched by role/name/type/ownership, or ordered differently across source/header/object records.
- Keep target-specific dispatch ABI identity in RVV/scalar target exporter registration and support runtime ABI contract code; keep grouped-record validation generic in target artifact infrastructure.
- Update the RVV+scalar dispatch e2e Python runner only as a consumer of compiler-emitted bundle index metadata.
- Preserve the existing source-built remote path, bundle-object remote path, component group validation, external ABI validation, and checked capability construction behavior.

## Acceptance Criteria

- [ ] Target artifact bundle index prints explicit ordered `runtime_abi_parameter[...]` metadata for the RVV+scalar dispatch external ABI group.
- [ ] C++ validation rejects missing runtime ABI signatures for grouped external ABI records.
- [ ] C++ validation rejects duplicate parameter roles in grouped signatures.
- [ ] C++ validation rejects mismatched parameter name/type/ownership for the same role.
- [ ] C++ validation rejects mismatched parameter order across grouped source/header/object records.
- [ ] Runner self-test proves bundle mode parses and validates the compiler-emitted signature fields before caller construction.
- [ ] Local build and test commands from the user request pass, or blockers are recorded precisely.
- [ ] Real `ssh rvv` bundle e2e is rerun if available; otherwise no new RVV runtime/correctness claim is made.

## Definition of Done

- C++/MLIR/lit/FileCheck remain the owner of compiler behavior.
- Python remains runner/evidence tooling and bundle-index consumption only.
- Specs are updated only where they describe the new compiler-owned ABI signature contract.
- Trellis task is archived before the final commit.
- One coherent commit is created and the worktree is clean.

## Technical Approach

Extend `TargetArtifactBundleRecord` and `TargetArtifactCompositeExporter` to carry `RuntimeABIParameter` vectors. Single records inherit candidate parameters; RVV+scalar dispatch composite records receive the dispatch ABI signature from `getI32VAddRuntimeABIContract().getDispatchRuntimeABIParameters()`. Bundle validation compares grouped signatures in order and emits stable errors for missing, duplicated, mismatched, or reordered parameters. The bundle index and manifest target-artifact records use the same `runtime_abi_parameter[index]` field shape.

## Out of Scope

- Arbitrary kernels or generic RVV lowering.
- Dynamic runtime loading or runtime ABI decision logic in Python.
- New dialect features or compute operations in `tcrv.exec`.
- New performance measurements.
- Claims not backed by current `ssh rvv` evidence.

## Technical Notes

- Required inspection completed for repo root, git status/log/show, file listing, Trellis task state, specs, support ABI headers, target artifact exporter, emission manifest, RVV/scalar dispatch exporter, RVV/scalar microkernels, translate tool, runner, and focused tests.
- Existing support owner: `support::getI32VAddRuntimeABIContract()`.
- Existing dispatch component group: `rvv-scalar-i32-vadd-dispatch-external-abi.v1`.
- Existing dispatch external ABI name: `rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1`.
