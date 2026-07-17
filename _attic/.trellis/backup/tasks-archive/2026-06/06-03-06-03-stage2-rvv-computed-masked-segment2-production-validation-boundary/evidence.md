# Stage2 RVV computed masked segment2 production validation boundary evidence

## Summary

Completed the bounded provider-to-target validation boundary for computed-mask
segment2 load/unit-store, store/unit-load, and update/unit-load routes.

The production diff adds a provider-owned
`RVVComputedMaskSegment2MemoryRouteFacts` surface in the RVV provider layer,
derives runtime ABI/binding/header/type/mask/segment/update facts from that
surface, and rewires target artifact validation to consume those facts instead
of carrying duplicate target-local segment2 constants.

## Changed Surfaces

- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `.trellis/spec/lowering-runtime/emitc-route.md`

## Checks

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-segment2'` from `build/test`
- [OK] `git diff --check`
- [OK] Bounded old-authority scan over touched production files and task PRD.

## Bounded Authority Scan

The bounded scan checked touched production files and task PRD for legacy or
non-authoritative route sources including `RVVI32M1`, `rvv-i32m1`,
`tcrv_rvv.i32_`, `!tcrv_rvv.i32m1`, `source-front-door`,
`source-artifact`, `direct-C`, `source-export`, `descriptor`, route-id,
artifact-name, metadata-only, and provider mirror drift.

Hits were limited to:

- task PRD forbidden-pattern text;
- existing provider mirror and rebuilt route-id validation text;
- an existing fail-closed `tcrv_rvv.i32_` check outside this new segment2 fact
  surface.

No new source-front-door, descriptor, direct-C/source-export, or legacy i32
positive route authority was introduced.

## Runtime Evidence

No new `ssh rvv` run was performed in this round because the production change
tightens provider/target validation only. It does not change generated runtime
ABI behavior, emitted segment2 statement semantics, or executable harness
behavior. Existing archived computed-mask segment2 load/store/update ABI tasks
remain the runtime evidence source for those routes.

## Finish State

Code/spec verification is complete. The task is archived, and the code, spec,
task evidence, and workspace journal are ready for the coherent closeout commit.
