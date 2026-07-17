# RVV vlenb capability capacity in variant decisions

## Goal

Promote bounded RVV `vlenb` / i32 lane-capacity evidence from probe/profile data
into the C++ capability model and make the RVV plugin consume that structured
fact during proposal, legality, and cost metadata. The compiler path that
becomes more real is:

```text
sanitized ssh-rvv probe fact
  -> plugin-local C++ RVV capability profile
  -> TargetCapabilitySet
  -> rvv-plugin proposal attributes and legality checks
  -> capability-aware preference metadata
```

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD at task start is `997ed9d feat: add plan-and-export bundle front door`.
- The previous completed round added the plan-and-export target artifact bundle
  front door and real bounded `ssh rvv` bundle external ABI evidence.
- There is no active Trellis task before this round; this task was created for
  the single-worker continuation requested by the prompt.
- Existing RVV proposal metadata already includes typed policy,
  `tcrv_rvv.required_march`, finite lowering descriptor, and descriptor-local
  `tcrv_rvv.element_count`.
- Existing RVV capability profile preserves architecture, hart count, toolchain
  availability, selected march/mabi, and compile-run digests, but it does not
  yet expose `vlenb` or derived i32 m1 lane capacity as compiler-queryable C++
  capability facts.

## Requirements

- Keep core compiler implementation in C++ / MLIR / TableGen / CMake; Python may
  only collect/replay sanitized probe facts.
- Add bounded `vlenb` and derived i32 m1 lane-capacity facts to the RVV evidence
  tooling and replay helper without making Python own capability legality,
  proposal, selection, lowering, or emission.
- Add plugin-local C++ RVV capability profile support for these facts as
  structured `TargetCapabilitySet` capabilities/properties.
- Make `RVVExtensionPlugin` consume the structured capacity fact when available
  by attaching plugin-owned variant metadata and validating it during legality.
- Keep `tcrv_rvv.element_count` descriptor-local; do not describe it as high
  level shape, AVL, VL, or global problem size.
- Do not add generic RVV lowering, dynamic runtime integration, performance
  claims, or new target-family branches in core orchestration.

## Acceptance Criteria

- [x] C++ RVV capability profile tests prove vlenb/lane facts are preserved in
      deterministic capability records and can drive RVV proposal metadata.
- [x] RVV legality rejects stale or mismatched plugin-owned vlenb/lane metadata
      when a variant claims capacity facts that are not supported by the
      `TargetCapabilitySet`.
- [x] RVV cost/preference metadata remains plugin-local and may name capacity as
      a heuristic input without claiming runtime performance.
- [x] Python probe and replay self-tests cover the new sanitized capability
      fields and preserve the C++ decision boundary.
- [x] Local build/checks pass, or any missing toolchain is reported with the
      exact missing tool.
- [x] Any `ssh rvv` claim is reported only as hardware/toolchain evidence unless
      a generated compiler artifact is actually compiled and run.

## Completion Evidence

- `git diff --check` passed.
- `python3 -m py_compile scripts/rvv_remote_probe.py scripts/rvv_probe_to_mlir.py` passed.
- `python3 scripts/rvv_remote_probe.py --self-test` passed.
- `python3 scripts/rvv_probe_to_mlir.py --self-test` passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 128/128 tests.
- Real `ssh rvv` probe hardware/toolchain evidence passed under
  `artifacts/tmp/rvv_probe/20260508T-vlenb-capacity/`; it reports
  `vlenb_bytes = 16`, `i32_m1_lane_count = 4`, `hart_count = 64`, and
  compile/run success for the bounded probe program.
- Replaying that live probe artifact through
  `scripts/rvv_probe_to_mlir.py` and `tcrv-opt --tcrv-execution-planning-pipeline`
  produced `rvv.vlenb_bytes`, `rvv.i32_m1_lane_count`,
  `tcrv_rvv.vlenb_bytes`, `tcrv_rvv.i32_m1_lanes`, and the RVV preference
  explanation that explicitly says it is not a runtime performance claim.

## Out Of Scope

- Generic RVV lowering, arbitrary kernels, new compute dialect semantics,
  performance measurement, dynamic loading, or runtime integration.
- Moving RVV-specific logic into generic core passes or target artifact routing.
- Treating descriptor-local `element_count` as runtime `n`, AVL, VL, tensor
  shape, or hardware capacity.

## Technical Notes

- Latest supervisor audit/review inspected:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0065-20260508T120248Z/`.
- Relevant specs inspected: capability model, RVV plugin, variant pipeline,
  lowering/runtime boundary, testing, and implementation stack.
- Expected primary code surfaces: `RVVCapabilityProfile`, `RVVExtensionPlugin`,
  RVV probe/replay scripts, focused plugin/script tests, and matching specs.
