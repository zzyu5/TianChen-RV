# Standalone dequantize-i32-to-f32 RVV->emitc conversion — evidence

Stage 3 换心 final RVV gap. HEAD b270dcb3. Closes the last RVV family still on
the legacy string-machine body-emission path.

## What changed (lib/Conversion/RVV/RVVToEmitC.cpp)

- `emitDequantizeChain(...)`: extracted the bare vfcvt_f_x_v + vfmul_vf core from
  `emitDequantize` (shared by the clamp epilogue and the new standalone body).
  `emitDequantize` keeps its `bodyHasSelect` guard as the fail-closed net.
- `emitLoad` / `emitStore`: gained a defaulted `mlir::Value extraOffset = {}` —
  when set, emit the SECOND pointer add (`ptr2 = ptr + extraOffset`) matching the
  legacy unrolled `v18 = base+i; v19 = v18+vl0` two-add form. Existing callers pass
  null (one add) and are unaffected.
- `isStandaloneDequantBody(scope)`: detector — a with_vl scope whose ONLY compute
  is exactly one i32m1 load -> one dequantize (kind `i32_to_f32_scaled`, i32->f32)
  -> one f32m1 store, nothing else.
- `emitStandaloneDequantBody(...)`: emits the Gearbox-unrolled (u<unroll>) two-slice
  runtime-avl setvl loop. Step vlmax*unroll, bound n; each slice recomputes the
  remaining AVL fresh from (n - i) minus the prior slices' runtime VLs; pointer
  offset = running sum of prior slice VLs. No separate scalar tail loop (the
  two-slice remaining-VL setvl covers the tail). `unroll` read from the realized
  scope's `tcrv_rvv.gearbox.unroll` schedule fact; absent/non-positive -> fail
  closed -> legacy fallback. Dispatched at the scope handler right after
  `isLowPrecisionDequantBody`.

## Byte-identity vs legacy golden

`tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans`
then `tcrv-translate --tcrv-rvv-emitc-to-cpp` (conversion-gated export) produces a
body BYTE-IDENTICAL to the fresh-HEAD legacy materializer output (the legacy path
the family fell back to before this change). The cpp that ran on ssh rvv hardware
(`materialized_rvv_emitc.cpp`) is also byte-identical to that legacy golden
(`legacy-golden-body.cpp`).

## ssh rvv hardware lamp (no --dry-run)

PASS op=dequantize_i32_to_f32 counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 tolerance=1e-05
ssh_evidence=true, status=success, f32_abs_tolerance=1e-05. 20/20 cases ok against
the independent reference oracle (ssh-rvv-evidence.json).

## Poison-test (TargetArtifactExport.cpp:2157, before buildSelectedEmitCArtifactRoute)

Inserting `return createStringError(... "POISON")` before the ONLY reachable entry
to the legacy per-family statement-plan owners, rebuilding, and running lit gives
EXACTLY 15 reds (down 0 RVV families touched):

  3 environmental (pre-existing):
    Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test
    Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test
    Scripts/rvv-generated-bundle-abi-e2e-self-test.test
  12 NON-RVV (legacy path is still legitimately theirs):
    Plugin/construction-protocol-common.test   (aborts at the Template consumer — non-RVV)
    Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-toy.mlir
    Target/Template/template-emitc-to-cpp-compile.test
    Target/Template/template-emitc-to-cpp.mlir
    Target/Template/template-target-artifact-object.mlir
    Target/TensorExtLite/tensorext-lite-materialized-target-artifact-bundle.mlir
    Target/TensorExtLite/tensorext-lite-materialized-target-artifact-object.mlir
    Target/TensorExtLite/tensorext-lite-runtime-abi-harness.test
    Target/TensorExtLite/tensorext-lite-source-front-door-emitc-to-cpp.mlir
    Target/TensorExtLite/tensorext-lite-source-front-door-target-artifact-bundle.mlir
    Target/TensorExtLite/tensorext-lite-target-artifact-header.mlir
    Target/Toy/toy-materialized-target-artifact-object.mlir

dequantize-i32-to-f32 (Target fixture + e2e dry-run) is NOT poison-red — it now
converts.

construction-protocol-common (test/Plugin/ConstructionProtocolCommonTest.cpp)
exercises ZERO RVV bodies. It drives the COMMON construction model with a
SYNTHETIC non-RVV family `tcrv_template_consumer.compute_sentinel`
(parseArtifactFixtureModule, line 818-849) through the SAME export gate
(`materializeSelectedEmitCArtifactModule`, the function carrying the poison at
:2161). The RVV-only conversion patterns cannot legalize a tcrv_template_consumer
body, so it correctly falls to the (poisoned) legacy path → stdout
`TemplateConsumer materialized EmitC module: POISON`. There is no RVV portion of
this test (verified by source read: no tcrv_rvv.load/with_vl/dequantize/setvl op
anywhere in the fixture) — so its poison-red is a non-RVV family on the legacy
path, NOT an RVV body. ALL RVV families now lower through the live RVV->emitc
DialectConversion.

Note: the author's predicted "malformed-body negatives remain red" under poison did
NOT materialize — those stayed green, which is correct: they fail-closed BEFORE the
export gate (verifier/legality rejection), so poisoning the gate doesn't change them.

Poison REVERTED; clean rebuild + fresh-link (tcrv-opt newer than libs) -> full lit
589 tests, EXACTLY the 3 environmental reds (honest-green). +1 vs HEAD baseline of
588 = the new structural test test/Conversion/RVV/rvv-to-emitc-dequantize-i32-to-f32.mlir.
