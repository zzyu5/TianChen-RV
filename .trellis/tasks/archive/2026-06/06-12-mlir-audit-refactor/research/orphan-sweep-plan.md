# Orphan-Sweep Deletion Plan — task 06-12-mlir-audit-refactor

Synthesis of 5 structured sweeps (s1–s5) over the dead weight that commit `d0934210`
("excise golden-string validation and dead golden tests", -81,401 lines) orphaned,
re-verified against the **current** tree (HEAD = `8d022042`).

## Verdict: substantial-dead-weight-found

A single large, grep-proven-closed dead cluster (~4,800 lines) survives in the RVV
EmitC route-planning subsystem, plus a small (~90-line) zero-reference tail in the
construction-protocol layer. Net realistic deletable now: **~4,900 lines**, rising to
**~5,000** if the surviving golden-style gtest is pruned in lockstep.

> HONESTY NOTE: `d0934210` took the single biggest chunk (the 17.5k validator + 64k of
> golden tests). What it *orphaned* but did not itself remove is the route-validation /
> metadata-mirror **contract** cluster those golden tests/validator were the sole
> consumer of — that is the real remaining prize. Everything else is small.

## Tree state caveats (the sweeps analyzed a stale snapshot)

Two commits have already landed **after** `d0934210` and must not be re-attempted:

- `f418cdf9` — synced stale lit checks; **baseline is now 3 reds, not 13**
  (539 passed / 3 failed; the 3 reds are gate4 measurement-evidence JSON dry-run/self-test
  that cannot pass from a clean checkout).
- `8d022042` — removed the dead fail-closed RVV vector-source front-door pass
  (`FailClosedRVVLegacyVectorSourceFrontDoorPass`). `RVVVectorSourceFrontDoor.cpp` is
  out of scope here and already cleaned.

## Cross-check result: dropped / corrected candidates

| Candidate | Source | Action | Reason |
|---|---|---|---|
| 7 "stale" `target_artifact_validator` CHECK-DAG lines in `test/Scripts/*widening*dry-run.test` | s5 | **DROP** | The field is emitted by the **Python harness** `scripts/rvv_generated_bundle_abi_e2e.py` (5 sites), NOT by the deleted C++ file. The `WDOT-DAG` lines are active (RUN line uses `--check-prefix=WDOT`) and self-consistent. The 9 tests are `REQUIRES: tianchenrv-local-rvv-object-clang` → UNSUPPORTED in this env, hence not red. Removing the lines is a behavioral test change, not orphan cleanup. |
| Whole `RVVTargetSupportBundle.cpp` "84% dead string validation" | audit premise / s1 | **DROP** | `validateRVVSelectedBodyTargetArtifactCandidate` chain is the LIVE `candidateValidationFn` (invoked `TargetArtifactExport.cpp:1701-1703` and `:3174-3183`); the ~1,000-line evidence table is the LIVE header-emission source (`TargetArtifactExport.cpp:1653-1661`). Not dead. |
| `validateRVVTargetArtifactRouteFamily{ProviderFacts,CandidateMirrors}` | s3 | **N/A** | Already deleted by `d0934210` with both call sites; zero residual refs. Nothing to do. |
| `verifyRVVRuntimeABIValueRoleOpInterface` (sibling of the dead role-op wrappers) | s2 | **PRESERVE** | LIVE: 2 call sites `RVVEmitCRoutePlanning.cpp:1923,14259`. |
| `get*RouteFacts` / `add*RouteFamilyMetadataMirrors` (naming-prefix neighbours of the dead cluster) | s5 | **PRESERVE** | LIVE emission/mirror-write path with real callers. Must NOT be swept up. |
| s2 k* backing constants (e.g. `kArchetype` behind `getRVVConstructionArchetype`) | s2/s1 | **PRESERVE the constants** | Getters are dead but their constants are shared with live manifest/spec literals. Delete the getter bodies only. |

## Deletion batches

### Batch A — RVV route-validation / metadata-mirror contract cluster (the prize)
- **Files:**
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` (~91 contract refs, bodies 29–150 lines each)
  - `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp` (~11)
  - `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp` (~6)
  - `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` (~12)
  - `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp` (~14)
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` (~116 contract-token lines: structs/enums/decls)
- **Symbols (by family, 23 top-level getters + their dedicated builders/append-helpers, ~69 cluster symbols):**
  `get/build*RouteValidationContract`, `get/build*RouteMetadataMirrorContract`,
  `append*ValidationTypeMapping`, `append*ValidationHeaders`,
  `append*ValidationRuntimeABIRoles`, `append*ValidationConsumerLabel`, and their
  header structs (`RVV*RouteValidationContract`, `RVV*RouteMetadataMirrorContract[Set]`,
  `RVV*ValidationTypeMapping`) + enums (`RVV*RouteValidationKind`, `*BoundaryContract`).
- **approx_lines:** ~4,800 (≈4,695 .cpp bodies + ~116 header)
- **risk:** medium
- **Evidence (closed dead graph):** all 23 top-level getters have 0 external callers; the
  only 3 with internal callers (`getRVVComputedMaskIndexedMemoryRouteValidationContract`@26951,
  `getRVVComputedMaskStridedMemoryRouteValidationContract`@27273,
  `getRVVUnitStrideMaskedMemoryRouteValidationContract`@26624) are called **only** from
  sibling cluster members (`getRVVCompositeGatherMAccScatterRouteValidationContract`@27090,
  `getRVVComputedMaskIndexedMemoryRouteMetadataMirrorContract`@28036,
  `getRVVComputedMaskStridedMemoryRouteMetadataMirrorContract`@28318,
  `getRVVUnitStrideMaskedMemoryRouteMetadataMirrorContract`@27917), each of which has 0
  external callers → fully closed. Sole historic consumer = deleted
  `RVVTargetArtifactRouteFamilyValidation.cpp`.
- **HAZARD:** dead getters are interleaved line-by-line with LIVE `get*RouteFacts` and
  `add*RouteFamilyMetadataMirrors` in the same 44k-line file. **Delete by symbol, never by
  line range.** Remove header decls/structs/enums in lockstep.
- **verify:** `ninja` (or cmake build) green, then full lit
  `cd /home/kingdom/phdworks/TianchenRV/build/test && python3 /usr/lib/llvm-20/build/utils/lit.py .`
  expecting exactly the 3 known baseline reds, 0 new failures.

### Batch B — construction-protocol zero-reference orphans (deletable now)
- **Files:**
  - `lib/Plugin/Construction/ConstructionProtocol.cpp` + `…/ConstructionProtocol.h`
  - `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp` + `…/RVVConstructionProtocol.h`
- **Symbols (all confirmed 2 refs = def + decl, zero callers):**
  `isValidCIdentifier` (ConstructionProtocol.cpp:321-333),
  `emitTypedRoleGraphRealization` (:1014-1028),
  the 5 role-op wrappers `verifyRVV{Load,Compute,Store,SetVL,WithVL}RoleOpInterface`
  (RVVConstructionProtocol.cpp:6037-6055),
  13 string-mirror getters (`getRVVConstructionArchetype`, `getRVVConstructionSemanticRoleGraph`,
  `getRVVConstructionInterfaceRealization`, `getRVVTypedRoleRealizationSummary`,
  `getRVVConstructionEvidenceProfile`, `getRVVSelectedBodyTypedComputeOpMetadataName`,
  `getRVVRuntimeABINameMetadataName`, `getRVVRuntimeABIContractMetadataName`,
  `getRVVTypedRoleAttrName`, `getRVVRoleOrderAttrName`, `getRVVSourceRoleAttrName`,
  `getRVVRoleSpecificInterfaceAttrName`, `getRVVRoleOpBoundaryStatus`; 4193-4333),
  `lookupRVVSelectedBodyConstructionRouteByTypedComputeOpName` (5977-5984).
- **approx_lines:** ~90
- **risk:** low
- **PRESERVE alongside:** `verifyRVVRuntimeABIValueRoleOpInterface` (LIVE), the shared
  `verifyRVVRoleOperationInterface` (LIVE, 7 refs), and the k* backing constants (shared).
- **verify:** same build + full lit recipe; 0 new failures.

### Batch C — orphaned route-ID getters (trivial)
- **Files:** `lib/Target/RVV/RVVTargetSupportBundle.cpp:1863-1869`;
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h:18-19`
- **Symbols:** `getRVVMaterializedEmitCTargetArtifactRouteID`,
  `getRVVMaterializedEmitCHeaderArtifactRouteID` (+ 2 header decls)
- **approx_lines:** ~9
- **risk:** none — grep over `lib/ include/ tools/ test/` finds only def + decl; sole prior
  consumer was the deleted `TargetArtifactExportTest.cpp`. Live route IDs are obtained
  directly from `plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID()` /
  `getRVVTargetMapping().headerRouteID`, never via these wrappers.
- **verify:** build green; full lit 0 new failures.

### Batch D (OPTIONAL / gated) — construction-protocol test-only tier + its test
- **Files:** `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp` (+ header) AND
  `test/Plugin/ConstructionProtocolCommonTest.cpp` (must be pruned in the SAME commit).
- **Symbols (LIB=0, only non-def ref is the surviving golden-style gtest):**
  `getRVVSelectedBodyConstructionRoutes`, 3× `getRVVSelectedBodyExecutableRoleSteps` overloads,
  `getRVVTypedRoleGraphRealization`, `verifyRVVConstructionManifest`,
  `verifyRVVTypedRoleGraphRealization`, `verifyRVVSelectedBodyConstructionPlanMapping`
  + its sole callee `lookupRVVSelectedBodyConstructionRouteByEmitCRouteID`.
- **approx_lines:** ~95 (lib) — NOT counted in the headline total; deletable only if the gtest
  goes too. `verifyRVVConstructionManifest`/`verifyRVVTypedRoleGraphRealization` are thin
  wrappers; their live callees (`verifySelectedBodyRoutes`, `getRVVConstructionValidationSpec`)
  stay via `verifyRVVConstructionProtocolReadyUncached` — delete only the wrapper bodies.
- **risk:** low (deterministic build break if test not pruned in lockstep).
- **DECISION REQUIRED:** `ConstructionProtocolCommonTest.cpp` is a golden-style self-check
  (already in the git-modified set; was op-count-realigned by `f418cdf9`). Pruning it is a
  policy call analogous to deleting the other golden tests in `d0934210`. Default: **defer**.
- **verify:** build green; full lit; the gtest target must not regress (it is being removed).

## Preservation boundary (LIVE — do NOT touch)

- **I8 export path:** `exportMaterializedRVVEmitCToCpp`, `compileRVVGeneratedSourceToObject`
  (clang `-target riscv64 -march=rv64gcv` shell-out), `mlir::emitc::translateToCpp` delegation,
  `getRVVSelectedBodyArtifactConfig`/`…AdapterConfig`,
  `exportRVVSelectedBody{Target,Header}Artifact`,
  `registerRVVSelectedBodyTargetArtifactExporter`,
  `registerRVVTargetSupportTargetTranslateRoutes`,
  `configureRVVTargetSupportExtensionBundle`
  (`RVVTargetSupportBundle.cpp:1654-1907`; wired at `RVVExtensionPlugin.cpp:712,717`).
- **LIVE candidate validation chain** rooted at `validateRVVSelectedBodyTargetArtifactCandidate`
  (= `config.candidateValidationFn`, invoked `TargetArtifactExport.cpp:1701-1703` and
  `:3174-3183`) and every transitive callee (`requireCandidateMetadataMirror`,
  `validateRVVRouteMetadataMirrorsSelectedBody`,
  `validateRVVConfigArtifactMetadataMirrorsSelectedBody`,
  `validateRVVRuntimeAVLVLArtifactMetadata`, etc.).
- **I7 fail-closed guards:** `containsForbiddenDirectCMarker` +
  `rejectForbiddenRVVArtifactMetadata` (`RVVTargetSupportBundle.cpp:443-489`);
  export-core `containsForbiddenDirectCMarker` /
  `rejectForbiddenMaterializedEmitCHeaderMetadata` (`TargetArtifactExport.cpp:1384-1426`).
- **LIVE header-emission evidence table:** `buildRVVSelectedBodyHeaderMetadataEvidence` +
  `append*` helpers (`RVVTargetSupportBundle.cpp:628-1651`) → `config.metadataEvidence` →
  provenance comments in the real `.h` (`TargetArtifactExport.cpp:1653-1661`).
- **Plugin-registration conformance gate (I7):** `verifyRVVConstructionProtocolReady[Uncached]`
  + reachable subgraph (`RVVConstructionProtocol.cpp:4489-4890`), invoked via
  `ExtensionPlugin.cpp:706` → `RVVExtensionPlugin.cpp:423`; plus
  `verifySelectedLoweringBoundaryConformance` (`RVVExtensionPlugin.cpp:325`).
- **LIVE EmitC route-planning consumers:** `verifyRVVRoleOperationInterface`,
  `verifyRVVRuntimeABIValueRoleOpInterface`, `verifyRVVSelectedBodySelectedRoleSequence`,
  `verifyRVVSelectedBodyConstructionRouteMapping`,
  `lookupRVVSelectedBodyConstructionRouteByOperationMnemonic`,
  and all `get*RouteFacts` / `add*RouteFamilyMetadataMirrors` emitters.
- **LIVE CapabilityModel:** `CapabilityDescriptor` + `makeDescriptor` and all accessors
  (`CapabilityModel.{h,cpp}`; ~15 consumer files across legality/cost/dispatch/plugins).
- **Test that gates Batch D:** `test/Plugin/ConstructionProtocolCommonTest.cpp` (do not touch
  unless executing Batch D).

## Ordered steps

1. Capture baseline: build green + full lit → confirm exactly 3 reds (gate4 dry-run/self-test).
2. Batch C (trivial, risk none): delete the 2 route-ID getters + 2 decls. Build + lit.
3. Batch B (risk low): delete the ~90 lines of zero-ref construction-protocol orphans
   (keep `verifyRVVRuntimeABIValueRoleOpInterface` + k* constants). Build + lit.
4. Batch A (the prize, risk medium): delete the contract cluster **by symbol**, removing
   header structs/enums/decls in lockstep; keep `get*RouteFacts` /
   `add*RouteFamilyMetadataMirrors`. Build (full link) + full lit. Commit only after green +
   no new lit failures.
5. (Optional, requires policy decision) Batch D: prune the test-only tier together with
   `ConstructionProtocolCommonTest.cpp` cases.
6. Do NOT touch the s5 `target_artifact_validator` test lines, `RVVVectorSourceFrontDoor.cpp`,
   or the export/I7/CapabilityModel boundary.

## Risk summary

Batch A is the only medium-risk item: large, correct-by-grep, but interleaved with live code
in a 44k-line file, so it must be removed symbol-by-symbol with header decls in lockstep and a
full re-link to catch any missed reference. Batches B/C are low/no risk (proven zero callers).
The headline correction is that s5's 7 "stale test" lines are **not** dead — they are produced
by the Python harness and the owning tests are `REQUIRES`-gated UNSUPPORTED — so they are
excluded. Realistic deletable-now total ≈ 4,900 lines (≈5,000 with Batch D).

## Open questions (need build-assisted check before/while deleting)

1. **Batch A link-completeness:** after symbol-level deletion, a full **link** (not just
   `-fsyntax-only`) is required — header structs/enums shared by any missed live reference
   would only surface at link or as a compile error in a sibling TU.
2. **k* constant transitive death (Batch B):** for each deleted string-mirror getter, confirm
   its backing `k*` constant is still referenced by live manifest/spec literals (sampled:
   `kArchetype` is shared → keep). A second-pass `-Wunused-const-variable` recompile after
   Batch B would flag any constant that became dead and could be removed for extra savings.
3. **Batch D policy:** is pruning `ConstructionProtocolCommonTest.cpp` (a surviving
   golden-style self-check) in scope for this 去伪 pass, or out-of-bounds like the other
   behavioral changes? Defaulted to defer.
