# Stage-1 Removal Plan — RVV Fail-Closed Legacy Vector Source Front Door

Task: `06-12-mlir-audit-refactor`
Synthesized from 6 structured investigation findings (q1 ref-map, q2 pipeline-impact, q3 family-asymmetry, q4 ods-td, q5 test-impact, q6 direct-c/build) and re-verified against the working tree.

## Verdict: `removal-with-test-deletions`

The fail-closed legacy pass `tcrv-rvv-fail-closed-legacy-vector-source-front-door` is dead on every supported path (it only emits an error and `signalPassFailure`; it is registered `ExplicitOnly`, so it is gated OUT of the default `tcrv-source-artifact-front-door-pipeline`). It has **no live, non-fail-closed dependency** and **no shared-infra entanglement**. Removal is clean BUT requires deleting 4 lit fixtures and editing 3 files plus 2 Target bundle tests that reuse the fixtures as generic input.

### CRITICAL SCOPING CAVEAT (do NOT widen to a file-level delete)

`lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` (1,777 lines) is a **mixed file**. The same `.cpp`/`.h` also defines THREE LIVE, actively-materializing family passes with passing `MATERIALIZED`/`HEADER` lit tests:

- `tcrv-rvv-materialize-vector-binary-source-front-door`
- `tcrv-rvv-materialize-vector-compare-select-source-front-door`
- `tcrv-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door`

A file-level delete (or removing the CMake entry / header) would silently regress these. **This plan removes ONLY the fail-closed pass class + its three private helpers + its sole factory + its single registration site. The file, header, and CMake entry STAY.**

Shared helpers `populateRVVVectorSourceFrontDoorDependentDialects`, `hasForeignLoweringSeedAttr`, `failLegacyMaterializer`, `requireLegacySourceOnlyModule`, `kSourceKernelAttrName` are called by BOTH the fail-closed pass AND the materialize family passes (`getDependentDialects` at `.cpp:1657` and `.cpp:1711` both call `populateRVVVectorSourceFrontDoorDependentDialects`). Removing `failLegacyMaterializer` / `requireLegacySourceOnlyModule` is only safe AFTER confirming no remaining in-file caller — verify by grep, do not assume.

---

## Evidence Table (file:line)

| Claim | Evidence | Source finding |
|---|---|---|
| Fail-closed pass is pure error-emit + `signalPassFailure`, no materialization | `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp:1642-1691` (error text :1684-1688) | q1,q2,q5 (verified) |
| Pass class is hand-written `PassWrapper`, NOT TableGen (absent from Passes.td, no GEN_PASS) | `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp:1642-1645` | q2 |
| Sole factory `createFailClosedRVVLegacyVectorSourceFrontDoorPass` | `.cpp:1738-1741`; decl `include/.../RVVVectorSourceFrontDoor.h:20-21` | q1,q2 (verified) |
| Sole registration site (push_back), `ExplicitOnly` policy | `lib/Plugin/RVV/RVVExtensionPlugin.cpp:428-434`; KEEP `return registerRVVVectorSourceFrontDoorFamilyPasses(...)` at :435 | q1,q2,q6 (verified) |
| `ExplicitOnly` is gated OUT of default pipeline (`continue` before addPass) | `lib/Transforms/ExecutionPlanningPipeline.cpp:64-69`; eligibility true only for `Eligible` `include/.../ExtensionPlugin.h:82-85` | q1,q2 |
| Default pipeline PIPE-FAIL is the exec-plan coherence check, NOT the fail-closed pass — proves it is not in the pipeline | `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir:3,30-31` | q2 (verified) |
| THREE LIVE materialize family passes co-host the same file — MUST survive | `.cpp:1693-1734` (family pass + factory); registry `.cpp:130-169`; emit `.cpp:1588-1635` | q3,q5 (verified) |
| Header declares 5 symbols; 4 back LIVE passes — header stays | `include/.../RVVVectorSourceFrontDoor.h:18-37` | q1 (verified) |
| Shared dialect-dep helper called by BOTH fail-closed and family pass | `.cpp:1657` and `.cpp:1711` | verified |
| No ODS op/attr tied to the front door; td refs are doc-string disclaimers only (cosmetic) | `RVVOps.td:24,147,4167` | q4 |
| Only `RVVExtensionPlugin.cpp` references front-door `.cpp` symbols externally | grep lib/include/tools: hits only `RVVExtensionPlugin.cpp`, `RVVVectorSourceFrontDoor.cpp/.h` | q1,q6 (verified) |
| Registry-listing C++ test asserts `rvvIndex>=0` on the fail-closed arg AND order `rvv<toy<tensorextLite` | `test/Plugin/ToyExtensionPluginTest.cpp:305-308,322-325,334` | q1,q2,q5 (verified) |
| 4 i32m1 fixtures invoke the fail-closed flag in their OWN RUN lines | base+`-sub`+`-mul`:1,2; `-negative`:1 (`--verify-diagnostics`) | q5 (verified) |
| 2 Target bundle tests reuse base/`-sub`/`-mul` fixtures as GENERIC input (not `-negative`) | `source-artifact-bundle-front-door-rvv.mlir:2,6,10`; `...-fail-closed.mlir:2`; both assert generic coherence/registration failure | q5 (verified) |
| Scripts `.test` files reference the arg only via `--implicit-check-not` (NOT-checks) — stay satisfied, NO edit | selected-body/pre-realized/lmul-m2 dry-run `:5-7`; fail-closed `.test:11,16` | q5 (verified) |
| Both `containsForbiddenDirectCMarker` copies are LIVE I7 guards in anon namespaces | `RVVTargetSupportBundle.cpp:443,471,553,622,1762`; `TargetArtifactExport.cpp:1384,1408,1418,1806`; `core-invariants.md:31` | q6 |

---

## Conflict reconciliation

- **q1 `removal_impact` proposes deleting the whole `.cpp`/`.h` and CMake entry.** This CONFLICTS with q3 and q5, which prove the file co-hosts three LIVE materialize passes. **Resolved conservatively in favor of q3/q5: do NOT delete the file/header/CMake entry.** q1's own findings table (claim 11) and `blocks_removal=false` reasoning already acknowledge the live materializers; only its summary over-reached. Verified directly in the working tree (`.cpp:1693-1734`, header :23-35).
- **q6 `removal_impact` text also references a whole-`.cpp` delete + header/CMake removal.** Same resolution: superseded by the narrow scope. q6's load-bearing contribution is the direct-C recommendation and the link-break analysis, both retained.
- No finding reported a genuine live (non-fail-closed) dependency or shared-infra blocker. All six set `blocks_removal=false`. No verdict-level conflict.

---

## files_to_delete (source)

None. The `.cpp`, `.h`, and CMake entry all stay (they host live passes).

## files_to_edit

1. `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
   - Remove class `FailClosedRVVLegacyVectorSourceFrontDoorPass` (`:1642-1691`).
   - Remove factory `createFailClosedRVVLegacyVectorSourceFrontDoorPass` (`:1738-1741`).
   - After removal, grep the file for `failLegacyMaterializer` and `requireLegacySourceOnlyModule`; if the fail-closed pass was their ONLY remaining caller, remove those helper definitions too (and `hasForeignLoweringSeedAttr` / `kSourceKernelAttrName` only if they likewise become unreferenced). Do NOT remove `populateRVVVectorSourceFrontDoorDependentDialects` (still used by family passes at `:1711`). Keep whatever the materialize passes and `materializeRVVVectorSourceFrontDoorFamily` still reference.
2. `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`
   - Remove the declaration `createFailClosedRVVLegacyVectorSourceFrontDoorPass()` (`:20-21`). Keep the other 4 symbols.
3. `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
   - Remove the `out.push_back(...)` fail-closed registration block (`:428-434`).
   - KEEP `return registerRVVVectorSourceFrontDoorFamilyPasses(kRVVPluginName, out);` (`:435`) — it registers the live family passes.
   - Keep the `#include` of the header (`:13`) — still needed for the family register call.
4. `RVVOps.td` (OPTIONAL, cosmetic) — the doc-string disclaimers at `:24,147,4167` may be left or de-stale'd; not load-bearing, no build impact. Defer unless trivially clean.

## tests_to_delete (4 lit fixtures — their only RUN target is the fail-closed flag)

- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir`

(Each invokes `--tcrv-rvv-fail-closed-legacy-vector-source-front-door` and asserts only the disabled/`Unknown command line argument` message. `-negative` is self-contained — not reused elsewhere.)

## tests_to_update (3)

1. `test/Plugin/ToyExtensionPluginTest.cpp` — in `runBuiltinSourceFrontDoorCollectionTest`: remove the `rvvIndex` lookup keyed on the fail-closed arg (`:305-308`) and its `rvvIndex>=0` assertion (`:322-325`). Adjust the ordering assertion (`:334`): either drop the RVV term and assert `toyIndex < tensorextLiteIndex`, OR re-anchor `rvvIndex` to a surviving RVV family arg (e.g. `tcrv-rvv-materialize-vector-binary-source-front-door`) since the RVV family passes still register before Toy. Prefer re-anchoring to keep the rvv<toy<tensorextLite coverage intact.
2. `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir` — RUN lines `:2,6,10` consume the deleted base/`-sub`/`-mul` fixtures as `%S/../../Transforms/RVV/...` input. These assert GENERIC pipeline coherence failure (`:17 ... requires at least one tcrv.exec.kernel`), not fail-closed behavior. Retarget to a small inline generic source MLIR (the fixture bodies have no front-door-specific ops — a plain vector func suffices) OR add a co-located minimal `.mlir` input under `TargetArtifactBundleExport/`. Do NOT let it dangle on a deleted path.
3. `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-fail-closed.mlir` — RUN line `:2` consumes the deleted base i32m1 fixture. It asserts the generic `requires at least one registered source front-door pass` (`:17`) under `--tcrv-disable-builtin-plugins`, not fail-closed behavior. Same retarget as above.

## tests UNAFFECTED (do NOT touch)

- Scripts dry-run `.test` files (`selected-body`, `pre-realized`, `lmul-m2`) and `...-fail-closed.test` — reference the arg only via `--implicit-check-not` / `-NOT`; NOT-checks remain satisfied after removal.
- The three `tcrv-rvv-materialize-vector-*` MATERIALIZED/HEADER tests + their `-negative.mlir` + `rvv-vector-source-front-door-family-registry-negative.mlir` — exercise the LIVE family passes.
- Toy / TensorExtLite source-front-door tests (family-owned). `test/Dialect/RVV/dataflow.mlir:6` (prose comment only).

## blockers

None. All six findings: `blocks_removal=false`. No live non-fail-closed dependency; no shared-infra entanglement; no ODS orphan.

Watch-items (not blockers):
- The mixed-file scope discipline above (never delete the file/header/CMake entry).
- Helper-removal must be grep-gated, not assumed, because helpers are shared with the live family passes.
- None of the affected tests are in `baseline-lit-failures.txt` — they all currently PASS, so any breakage is a fresh regression, not pre-existing noise. (`Plugin/rvv-extension-plugin.test` IS a baseline failure but has zero front-door references.)

---

## direct_c_recommendation: DEDUPE-INTO-SHARED-HELPER

The two `containsForbiddenDirectCMarker` copies (`RVVTargetSupportBundle.cpp`, `TargetArtifactExport.cpp`, both in anon namespaces) are LIVE I7 fail-closed guards wired into reachable artifact-metadata validation that rejects direct-C / source-export / descriptor metadata (`core-invariants.md:31`). They are NOT dead and MUST NOT be removed. They are unrelated to the front-door pass removal and should be handled as an INDEPENDENT cleanup: dedupe the two identical copies into one shared helper. Do this in a SEPARATE change, not coupled to this removal.

---

## risk_summary

Low. No supported-path behavior changes (the fail-closed pass was already gated out of the default pipeline as `ExplicitOnly`). The only real failure modes are mechanical: (1) widening scope to a file/header/CMake delete and silently killing the three live materialize family passes — guarded by the scoping caveat; (2) removing a shared helper still used by a family pass and breaking the link — guarded by grep-before-remove; (3) leaving the two Target bundle tests dangling on deleted fixture paths — guarded by retarget-to-inline. Net coverage loss is nil: every deleted test asserts only the fail-closed/disabled message or a generic-pipeline failure reproducible with any source input.

---

## ordered_steps (build green + lit no-worse-than-baseline at every step)

1. Capture baseline: confirm working-tree lit failures match `baseline-lit-failures.txt`; confirm the 4 fixtures, `ToyExtensionPluginTest.cpp`, and the 2 Target bundle tests currently PASS.
2. Edit `lib/Plugin/RVV/RVVExtensionPlugin.cpp`: remove the fail-closed `out.push_back` block (`:428-434`); keep the `return registerRVVVectorSourceFrontDoorFamilyPasses(...)` (`:435`) and the `#include` (`:13`).
3. Edit `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`: remove class `FailClosedRVVLegacyVectorSourceFrontDoorPass` (`:1642-1691`) and factory (`:1738-1741`). Then grep the file for `failLegacyMaterializer`, `requireLegacySourceOnlyModule`, `hasForeignLoweringSeedAttr`, `kSourceKernelAttrName`; remove only those now-unreferenced. Do NOT remove `populateRVVVectorSourceFrontDoorDependentDialects`.
4. Edit `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`: remove the `createFailClosedRVVLegacyVectorSourceFrontDoorPass()` decl (`:20-21`); keep the other 4 symbols.
5. Build. Resolve any link/compile error from a dangling reference or an over-removed helper before proceeding. The build MUST be green here.
6. Update `test/Plugin/ToyExtensionPluginTest.cpp`: drop the fail-closed `rvvIndex` lookup/assert and re-anchor `rvvIndex` to a surviving RVV family arg (preferred) or relax the order assertion to `toyIndex < tensorextLiteIndex`.
7. Retarget the 2 Target bundle tests (`source-artifact-bundle-front-door-rvv.mlir`, `source-artifact-bundle-front-door-fail-closed.mlir`) off the about-to-be-deleted i32m1 fixtures to an inline / co-located minimal generic source MLIR; keep their FAIL/DISABLED/NO-MATCH check prefixes.
8. Delete the 4 i32m1 fixtures (base, `-sub`, `-mul`, `-negative`).
9. Run lit. Confirm: no new failures vs baseline; the 4 deleted fixtures are gone; the 3 updated tests pass; the three `tcrv-rvv-materialize-vector-*` family tests still PASS.
10. (Optional) De-stale `RVVOps.td` doc-string disclaimers (`:24,147,4167`). Cosmetic; no build/test impact.
11. (Separate change) DEDUPE the two `containsForbiddenDirectCMarker` copies into one shared helper — do not couple to this removal.
