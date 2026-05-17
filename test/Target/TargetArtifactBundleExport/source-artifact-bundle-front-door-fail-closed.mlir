// RUN: rm -rf %t.disabled.bundle && mkdir %t.disabled.bundle
// RUN: not tcrv-translate --tcrv-disable-builtin-plugins --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.disabled.bundle %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: not test -e %t.disabled.bundle/tianchenrv-target-artifact-bundle.index

// RUN: rm -rf %t.nomatch.bundle && mkdir %t.nomatch.bundle
// RUN: not tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.nomatch.bundle %S/../../Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-stale-toy.mlir 2>&1 | FileCheck %s --check-prefix=NO-MATCH --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: not test -e %t.nomatch.bundle/tianchenrv-target-artifact-bundle.index

// RUN: rm -rf %t.noartifact.bundle && mkdir %t.noartifact.bundle
// RUN: not tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.noartifact.bundle %s 2>&1 | FileCheck %s --check-prefix=NO-ARTIFACT --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: not test -e %t.noartifact.bundle/tianchenrv-target-artifact-bundle.index

// RUN: rm -rf %t.missing.bundle
// RUN: not tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.missing.bundle %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir 2>&1 | FileCheck %s --check-prefix=MISSING-DIR --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: not test -e %t.missing.bundle/tianchenrv-target-artifact-bundle.index

// DISABLED: TianChen-RV source-artifact bundle front door requires at least one registered source front-door pass

// NO-MATCH: TianChen-RV execution plan coherence check failed for kernel <missing>: requires at least one tcrv.exec.kernel
// NO-MATCH: TianChen-RV source-artifact bundle front door failed during source-artifact front-door pipeline

// NO-ARTIFACT: TianChen-RV source-artifact bundle front door failed during source-artifact front-door pipeline

// MISSING-DIR: TianChen-RV target artifact bundle export failed: output directory must already exist

module @source_artifact_bundle_front_door_no_artifact_input {
  tcrv.exec.kernel @source_front_door_no_artifact {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}
