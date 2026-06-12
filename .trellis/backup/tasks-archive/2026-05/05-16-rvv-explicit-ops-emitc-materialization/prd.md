# RVV explicit ops to EmitC materialization

## Goal

Build the first bounded post-deletion RVV EmitC materialization path: a hand-authored explicit `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` / `tcrv_rvv.i32_load` / `tcrv_rvv.i32_add` / `tcrv_rvv.i32_store` slice must lower through the common EmitC lowerable route and produce a parseable MLIR EmitC module. The route is a compiler handoff from extension-family IR to EmitC, not a descriptor, direct C source exporter, target artifact route, runtime claim, or hardware evidence claim.

## What I Already Know

- Current HEAD is `d4634a6 chore(rvv): erase finite i64 family`; worktree was clean before this task.
- `.trellis/.current-task` was absent, so this task was created from the Hermes direction brief.
- `.trellis/spec/index.md` defines the stable route as extension-family ops -> EmitC ops -> intrinsic/vendor/runtime C/C++ -> native compiler, with Python limited to tooling and descriptor/direct-C paths invalid as active architecture.
- `.trellis/spec/extension-plugins/rvv-plugin.md` states that the surviving RVV slice includes `setvl`, `with_vl`, i32 load/add/sub/mul/store under `tcrv_rvv`, and that the intended rebuild route is explicit RVV ops -> materialized MLIR EmitC module -> C/C++.
- `.trellis/spec/lowering-runtime/emitc-route.md` defines `TCRVEmitCLowerableOpInterface` as the generated source-op boundary and `TCRVEmitCLowerableRoute` as the hand-written route payload consumed by the common materializer.
- Existing `include/lib/Conversion/EmitC` already has the route payload, route verifier, and materializer that creates `emitc.include`, `emitc.func`, `emitc.verbatim`, and `emitc.call_opaque`.
- Existing RVV ODS currently puts `TCRVEmitCLowerableOpInterface` only on arithmetic ops; this task needs the selected load/add/store slice to participate in the common interface-backed provenance.

## Requirements

- Add or wire a production path that consumes explicit RVV first-slice IR and materializes a parseable MLIR EmitC module through the common `TCRVEmitCLowerableRoute` / materializer API.
- Keep RVV-specific validation, operation ordering, ABI role interpretation, and intrinsic spelling mapping inside RVV-owned code.
- Keep the generic pass/tool boundary extension-neutral: it may route to the origin plugin, but it must not branch on RVV semantics, dtype, LMUL, intrinsic names, or ABI roles.
- Preserve runtime `n` / AVL / VL / ABI role surfaces as IR-derived or ABI-derived data in the route; do not encode them as descriptor metadata or target artifact records.
- Emit route evidence in the produced EmitC module: headers, source ops, `TCRVEmitCLowerableOpInterface` provenance, and `emitc.call_opaque` callees.
- Fail closed for unsupported shapes or malformed bodies before producing an EmitC module.

## Acceptance Criteria

- [ ] A selected/materialized explicit RVV i32 m1 add body can be lowered/materialized to a parseable MLIR EmitC module through the common interface-backed route.
- [ ] The produced EmitC module includes `emitc.include`, an `emitc.func` ABI boundary, and `emitc.call_opaque` calls for setvl, load, add, and store with RVV-owned intrinsic names.
- [ ] The route records source-op/interface provenance derived from typed `tcrv_rvv` ops, not descriptor strings.
- [ ] Negative coverage proves unsupported RVV shapes or malformed/missing body steps fail before EmitC output.
- [ ] No descriptor-driven computation, direct C exporter, source artifact route, target bundle export, compatibility wrapper, i64 family, or broad smoke/report path is reintroduced.

## Out of Scope

- General RVV type/config/lowering system.
- MLIR vector lowering, LLVM/RISC-V lowering, inline asm, native object generation, bundle packaging, target artifact export, or `ssh rvv` runtime/correctness/performance claims.
- Finite i64 families, new dtype families, general LMUL expansion beyond the bounded first-slice coverage needed here.
- Descriptor replacements, descriptor-to-C compatibility, direct handwritten C source generation, or route metadata as semantic authority.
- New high-level frontend lowering or broad execution-planning rebuild.

## Technical Notes

- Read first: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
- Likely code surfaces: `include/TianChenRV/Conversion/EmitC/`, `lib/Conversion/EmitC/`, `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, `lib/Dialect/RVV/IR/RVVDialect.cpp`, `include/TianChenRV/Plugin/ExtensionPlugin.h`, `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, `lib/Transforms/`, and `tools/tcrv-opt/`.
- Focused evidence should be lit/FileCheck and/or C++ test coverage over real MLIR/LLVM stack. Local checks are allowed for materialization; no RVV runtime claim is made without `ssh rvv`.
