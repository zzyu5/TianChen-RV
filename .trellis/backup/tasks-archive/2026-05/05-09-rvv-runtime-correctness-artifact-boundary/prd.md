# RVV Runtime Correctness Artifact Boundary

## Goal

Make the linalg-fed target artifact bundle expose a durable RVV runtime-correctness handoff boundary instead of stopping at local bundle export. The compiler-owned C++ target/exporter surface should describe how the generated source/header/object artifacts form one external ABI group that an evidence runner can link and execute on the real `ssh rvv` host.

## What I Already Know

- The current HEAD is expected to be `d4624e2 feat: plan linalg vadd artifact bundles`, and initial `git status --short` was clean.
- `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` already runs bounded linalg i32-vadd frontend lowering, the execution planning pipeline, and the target artifact bundle exporter.
- The existing direct RVV bundle emits source/header/object files, but its index records do not yet group those three records as one typed external ABI contract.
- Existing dispatch bundle evidence already consumes `component_group`, `component_role`, `external_abi_name`, and ordered runtime ABI signature metadata from the bundle index.

## Requirements

- Keep core `tcrv.exec` and RVV dialect ODS compute boundaries unchanged unless a public interface extension is truly required.
- Keep RVV-specific semantics in the RVV plugin/target-export surfaces.
- Add the missing C++ target/exporter surface for the direct RVV i32-vadd microkernel bundle runtime handoff.
- The bundle index must make source/header/object records for the selected RVV microkernel share a stable external ABI group and ordered runtime ABI signature.
- The runner may orchestrate bundle export, remote copy, compile, link, and run; it must not implement compiler semantics.
- Runtime/correctness may be claimed only from real `ssh rvv` compile/link/run evidence.

## Acceptance Criteria

- [x] A marked linalg i32-vadd input can be planned and exported into a direct RVV source/header/object bundle whose index exposes a typed external ABI component group.
- [x] A focused lit/FileCheck test proves the compiler/exporter bundle metadata for the linalg-fed RVV path.
- [x] The evidence runner can consume the bundle index, generated header, generated object, and generated external caller without relying on source/header/object file-name heuristics as the contract.
- [x] `git diff --check`, `cmake --build build --target tcrv-translate`, and `cmake --build build --target check-tianchenrv` pass.
- [x] Real `ssh rvv` evidence is run if runtime correctness is claimed; otherwise the exact blocker is recorded.
- [x] The Trellis task is archived before the final commit.

## Out Of Scope

- Generic linalg lowering beyond the existing marked i32-vadd slice.
- New compute operations in `tcrv.exec`.
- Python implementations of compiler IR, passes, plugin registry, target emission, or runtime ABI semantics.
- RVV performance claims.
