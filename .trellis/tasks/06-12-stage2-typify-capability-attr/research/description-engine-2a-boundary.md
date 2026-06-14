# Description-engine retirement STAGE 2 MOVE 2A — STEP 1 boundary measurement

HEAD: fc9aa69f. File: `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` (33148 lines).
Target premise (from task): retire the "~94% never-consumed I4-mirror" `low_precision_resource.*`
synthesis in `analyzeRVVSelectedBodyRoute`, claimed "asserted ONLY by fixtures, read by ZERO
live code, NOT in the exported .h that positive artifact fixtures check."

## VERDICT: STOP for RE-AUTHORIZATION (not entanglement). NO deletion executed.

This synthesis **IS retireable (c)-class I4-mirror** — zero live value-readers (RVVToEmitC 0 refs;
the export path just serializes; the bundle path is `actual==expected` mirror-validation; the
plugin only writes). N3 is untouched by retiring it (the gearbox reads a *different* producer — the
IR-attr channel). So under I4 the deletion is architecturally clean.

**The block is OPERATIONAL, not classificatory:** the task's premise is **factually false in a
load-bearing way.** It authorized this believing the keys are "read by ZERO live code, asserted
ONLY by fixtures, and *bundle reads the genuine (a) metadata not these*." But ~18 of these keys are
**exported `.h` header body lines** (HEADER 156-173) + an **I8 header-evidence schema**
(RVVTargetSupportBundle.cpp:823+) + a bundle mirror-validator. Retiring the symbol therefore
**shrinks the exported `.h` header** — an artifact-contract change that exceeds the authorized
"remove fixture-only comment lines." Acting on the original authorization would be acting on a
factual error. → re-authorize under the corrected facts, then execute.

## The three distinct `low_precision_resource.*` channels (137 refs in the file; key name ≠ channel)

1. **N3 IR-attr channel (OFF-LIMITS).** Attr-name constants in `include/.../RVVGearboxSchedule.h`
   (`kRVVLowPrecisionResource*AttrName`). Stamped onto `tcrv_rvv.with_vl` by the realization /
   gearbox schedule pass and **READ BACK to shape the tune**:
   - `lib/Plugin/RVV/RVVGearboxSchedules.cpp:543-546, 750-751, 892, 1624` —
     `op->getAttrOfType<StringAttr/IntegerAttr>(kRVVLowPrecisionResource…)` then branches on the
     value (vector_register_budget, remediation budget). This IS N3 tune authority.
   - `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp:36-40 copyLowPrecisionResourceAttrs`
     propagates them. Also `RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`.
   - Gearbox lit (`test/Transforms/RVV/rvv-gearbox-widening-product-reduce-dequantize-f32.mlir`)
     STALE/UNSUPPORTED RUN lines inject adversarial values and assert the schedule pass REJECTS
     them — proving the channel is honest (rejects artifact-name-derived values). N3 — never touch.

2. **Description-engine synthesis → ArtifactMetadataEntry (the deletion target).**
   `appendRVVLowPrecisionStableResourceCompilerFactMetadata` (RVVEmitCRoutePlanning.cpp:31669-~31898,
   ~230 lines) + its sibling `appendRVVLowPrecisionPrimitivePayloadMirrorMetadata` (31615-31667) +
   `makeRVVLowPrecisionStableResourceCompilerFacts` (header inline) + the inline pushes at
   32628-32750 inside `getRVVSelectedBodyConfigArtifactMetadata` (31900-…). Uses its OWN hardcoded
   string literals, builds `support::ArtifactMetadataEntry` for the route description.
   Sole caller of the append fn: RVVEmitCRoutePlanning.cpp:32625 (self).
   **Where it goes (NOT fixture-only):**
   - `RVVExtensionPlugin.cpp:636-638` → `out.addArtifactMetadata(key,value)` (WRITER, no branch).
   - `RVVTargetSupportBundle.cpp:490 validateRVVConfigArtifactMetadataMirrorsSelectedBody`
     (`actual == expected`, MIRROR-VALIDATES-MIRROR — classic dead-mirror tell).
   - `RVVTargetSupportBundle.cpp:600-613 appendRVVConfigVLMetadataEvidence` +
     static schema table `RVVTargetSupportBundle.cpp:823+` → declares these keys as **I8 header
     evidence** (which keys the exported .h MUST carry).
   - **EXPORTED INTO THE GENUINE `.h`**: e.g.
     `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
     RUN line `tcrv-translate --tcrv-export-target-header-artifact | FileCheck --check-prefix=HEADER`,
     HEADER lines 156-173: `// HEADER: tianchenrv.rvv.low_precision_resource.*: …` interleaved with
     unambiguously-genuine artifact bytes (runtime_abi_order 153, c_type_mapping 177,
     dequantization_relation 154, gearbox_producer_scope 174 — same synthesis, same .h).
   - PLAN emission-plan fixtures (`// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.*"...}`)
     and STALE-*-MIRROR header-export guards (lines 21-22) assert these at the artifact boundary.
   **Classification: case (c) I4-mirror — no compute/dispatch/legality consumer reads its VALUES
   (I4: artifact metadata / manifest IS mirror, even when it lands in the .h). "Reaches the .h"
   is BLAST-RADIUS, not authority. The premise's error is only that it assumed these were NOT in
   the exported .h; they are (~18 keys, HEADER 156-173) + I8 header-evidence schema.**

3. **Selection authority `RVVLowPrecisionContractionResourceSelection` (case (b), KEEP).**
   100+ refs across `RVVLowPrecisionPerformancePolicy.cpp`, `RVVEmitCContractionRouteFamilyPlanOwners.cpp`
   (`derive*` / `verify*` / `populate*FromCandidate` / `*FromPassFacts`). This is the candidate
   selection / performance-feedback / remediation **logic** — the gearbox decision input. The
   description synthesis serializes a snapshot of it; the realization (N3) consumes it. KEEP.

## Why no clean (c) subset can be split off

- The synthesis is ONE symbol feeding `getRVVSelectedBodyConfigArtifactMetadata`; dead-mirror guide
  mandates symbol-level (not line) deletion. Can't drop "only the keys that never reach HEADER" —
  they share the one function and the one emission pass.
- ~18 of the keys appear verbatim in the exported `.h` HEADER block (genuine artifact bytes); the
  rest appear in PLAN / description / bundle-mirror-validate. Deleting the symbol drops HEADER +
  PLAN + STALE-*-MIRROR + bundle-validate together — i.e., it changes exported `.h` header bytes.
- The task's own gate: "genuine artifact bytes MUST stay byte-identical EXCEPT removed vestigial
  COMMENT lines; if a genuine assertion moves → crossed into (a) → STOP." HEADER `low_precision_*`
  lines are real `.h` body bytes, not comment lines. → STOP condition met.

## Honest recommendation (for human / next move)

The synthesis IS I4-mirror by the consumer test (no compute/dispatch/legality reads its *values*;
the no-read fixture `rvv-to-emitc-packed-i4-dequant-i5-no-read-resource-mirror.mlir` proves the
conversion ignores them, RVVToEmitC.cpp has 0 refs). So retiring it is *defensible* under I4 — but
its blast radius is the **exported `.h` header schema + the bundle mirror-validator + ~6 artifact
fixtures' HEADER/PLAN/STALE-MIRROR blocks**, not "fixture-only PLAN-DAG/STALE comment lines."
That is materially bigger and different from the task's stated premise, and it is a deliberate
"shrink the exported header surface" decision (drop ~18 header-evidence keys + their I8 schema +
the mirror-validator), which is a product/contract call for the human — NOT a quiet 2A deletion.

If approved later, the clean closed-graph would be: delete
`appendRVVLowPrecisionStableResourceCompilerFactMetadata` + `appendRVVLowPrecisionPrimitivePayloadMirrorMetadata`
+ the 32628-32750 inline pushes + `makeRVVLowPrecisionStableResourceCompilerFacts` +
`RVVLowPrecisionStableResourceCompilerFacts`/`isRVV…FactsEqual`, then the bundle
`validateRVVConfigArtifactMetadataMirrorsSelectedBody` + the `appendRVVConfigVLMetadataEvidence`
`low_precision_resource` schema rows + re-baseline HEADER/PLAN/STALE-MIRROR fixture blocks — while
NOT touching the N3 IR-attr channel (RVVGearboxSchedule.h constants, RVVGearboxSchedules.cpp reads,
realization owners) or the selection authority. That is a separate, human-approved sub-move.
