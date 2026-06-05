# Stage2 RVV Gearbox bounded candidate-selection contract

## Goal

Expand the existing RVV Gearbox MVP from one static schedule annotation into a
bounded candidate-set plus selected-schedule contract for the existing typed
`dequantize_i32_to_f32` route. The pass must derive the legal candidate
inventory and deterministic selected candidate from selected typed
`tcrv_rvv` body/config/runtime facts, the RVV provider must prove the selected
candidate belongs to that pass-produced legal set before route construction,
and target artifact metadata may mirror only provider-derived Gearbox facts.

This task makes the pass/provider/target boundary more real. It is not runtime
benchmarking, a persistent tuning cache, a new high-level frontend, or a broad
route-surface expansion.

## What I already know

- No `.trellis/.current-task` existed when this round began, so this task was
  created from the Hermes Direction Brief instead of choosing an unrelated
  owner.
- The previous completed Gearbox MVP registered
  `--tcrv-rvv-materialize-gearbox-schedules`, matched the existing typed
  `dequantize_i32_to_f32` body, materialized one static schedule, required
  provider consumption, and exported provider-derived mirrors.
- The next bottleneck is not executable evidence. It is the compiler contract:
  candidate identities, schedule dimensions, selection reason, source/dest
  SEW/LMUL, operation, runtime AVL source, and legality scope must come from
  typed IR/config/capability facts.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires RVV schedule and
  selected-body realization decisions to stay RVV plugin-owned and to be
  consumed before route construction when they affect generated code.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC/export
  to carry provider payloads neutrally; it must not choose RVV schedules or use
  artifact metadata as route authority.
- Memory/context from prior RVV Stage work reinforces the authority chain:
  selected typed `tcrv_rvv` body -> RVV plugin legality/realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact mirror.

## Requirements

- Production pass/provider/target code changes are required. A prompt-only,
  report-only, helper-only, or metadata-only round is not sufficient.
- Keep the proof route bounded to the existing selected typed
  `dequantize_i32_to_f32` body.
- Extend the Gearbox schedule fact surface so the pass records:
  - bounded legal candidate inventory;
  - deterministic selected candidate identity;
  - selection reason;
  - operation and legality scope;
  - source/destination SEW and LMUL;
  - runtime AVL source;
  - schedule dimensions such as unroll and VL policy.
- If only one candidate is currently legal, record exactly that one legal
  candidate and a truthful deterministic selection reason. Do not fake
  benchmark/autotuning alternatives.
- The RVV provider must validate before route construction that:
  - the candidate inventory exists;
  - the selected candidate exists;
  - the selected candidate is a member of the legal candidate set;
  - selected schedule facts agree with the candidate dimensions and typed
    body/config/runtime facts;
  - unsupported or stale schedule dimensions fail closed.
- Target artifact validation must reject stale candidate-set or selected
  schedule mirrors when metadata is touched. Target mirrors are accepted only
  after provider route facts are rebuilt and checked.
- Common EmitC/export remains neutral. It may carry provider payload/mirrors,
  but must not infer or choose Gearbox candidates.
- Do not use route strings, artifact names, q-names, ABI strings, test names,
  or intrinsic spelling as Gearbox authority.

## Acceptance criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      Gearbox candidate-selection task and its spec basis.
- [x] The production Gearbox pass materializes a bounded legal candidate
      inventory plus selected candidate facts for the positive
      `dequantize_i32_to_f32` typed body.
- [x] A focused positive pass test checks candidate inventory, selected
      candidate, selection reason, schedule dimensions, operation, legality
      scope, source/dest SEW/LMUL, and runtime AVL source.
- [x] RVV provider route planning validates selected-candidate membership and
      schedule consistency before building the lowerable route.
- [x] A provider/route-planning test proves selected candidate membership is
      consumed before route construction.
- [x] Negative tests cover missing candidate set, selected candidate not in
      candidate set, unsupported schedule dimension, typed body/config
      mismatch, route-string/artifact-name/intrinsic-spelling authority
      attempts, and stale target mirrors when metadata is touched.
- [x] Target artifact checks, if metadata changes, accept only
      provider-derived Gearbox mirrors and reject stale candidate or selected
      schedule metadata.
- [x] Focused build/lit/C++ tests pass for touched RVV pass/provider/target
      owners.
- [x] `git diff --check`, `git diff --cached --check`, and a bounded
      old-authority/q-name scan over touched files pass.
- [x] `ssh rvv` is reported as not applicable unless this task makes
      executable correctness/runtime/performance claims.
- [x] The Trellis task status, journal, archive state, commit hash, and final
      worktree status are recorded truthfully.

## Completion evidence

- Extended RVV Gearbox schedule constants with
  `candidate_set`, `selected_candidate`, `selection_reason`, and
  `legality_scope` for the bounded
  `dequantize_i32_to_f32` `e32,m1,unroll=1` candidate.
- Updated `--tcrv-rvv-materialize-gearbox-schedules` so the pass materializes
  the candidate inventory and deterministic selected candidate on both
  `tcrv_rvv.with_vl` and `tcrv_rvv.dequantize`, while still fail-closing stale
  preexisting attrs.
- Extended RVV dialect verification allow-lists so the new Gearbox attrs are
  legal only on the intended `with_vl` boundary and dequant operation surfaces.
- Extended dequant provider route facts, route-family plan, route description,
  conversion dtype-policy validation contract, target route-family validation,
  and target header metadata mirrors with the candidate-selection fields.
- Added provider-side membership validation: the selected candidate must belong
  to the pass-produced legal candidate set before route construction.
- Added/updated focused tests:
  - positive pass FileCheck for candidate inventory, selected candidate,
    selection reason, legality scope, dimensions, operation, and runtime AVL
    source;
  - pass negative stale selected candidate;
  - target/provider negative missing candidate set, selected candidate outside
    candidate set, unsupported unroll, stale selected-candidate metadata, and
    stale schedule metadata.
- Checks run:
  - `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test`
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  - `cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-gearbox-dequantize-i32-to-f32|explicit-selected-body-artifact-dequantize-i32-to-f32'`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `./build/bin/tianchenrv-rvv-extension-plugin-test`
  - `git diff --check`
  - bounded old-authority/q-name scan over touched diff
- Self-repair performed:
  - added missing RVV dialect verifier allow-list entries for the new Gearbox
    attrs on `tcrv_rvv.with_vl` and `tcrv_rvv.dequantize`;
  - corrected the C++ target test expectation to match the target validator's
    provider-facts diagnostic boundary.
- `ssh rvv` was not run because this task changes pass/provider/target
  contract metadata only and makes no executable correctness/runtime/performance
  claim.

## Out of scope

- Runtime benchmarking or performance claims.
- Persistent tuning cache, assembly feedback loop, cross-kernel autotuning, or
  global tuning database.
- New high-level Linalg/Vector/StableHLO frontend.
- Common/core RVV schedule selection.
- Direct C/intrinsic macro autotune as route authority.
- Broad route-surface expansion or one-intrinsic wrapper growth.
- q-name, benchmark-name, artifact-name, ABI-string, or intrinsic-spelling
  authority.
- Treating the v3 Gearbox artifact as final design authority.

## Technical notes

- Required specs/reference context:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/tasks/archive/2026-06/06-05-rvv-gearbox-autotuning-pass-reference/`,
  and `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`.
- Main code owners to inspect:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  and `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- Focused tests to inspect/update:
  `test/Transforms/RVV/rvv-gearbox-dequantize-i32-to-f32.mlir`,
  `test/Transforms/RVV/rvv-gearbox-dequantize-i32-to-f32-negative.mlir`,
  and
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`.

## Decision

Implement the candidate-selection contract as a bounded extension of the
existing Gearbox pass/provider surface. Candidate enumeration remains static and
truthful for now: if current legality admits only one schedule, the legal set
contains one candidate and the selected schedule references that candidate.
Later work can add more legal candidates, cost models, profile data, or
runtime/performance evidence behind the same membership contract.
