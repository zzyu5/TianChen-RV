// RUN: rm -rf %t.disabled.bundle && mkdir %t.disabled.bundle
// RUN: not weft-translate --weft-disable-builtin-plugins --weft-source-artifact-bundle-front-door --weft-target-artifact-bundle-output-dir=%t.disabled.bundle %S/Inputs/generic-vector-source.mlir 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not="weft.target_artifact_bundle_export: complete"
// RUN: not test -e %t.disabled.bundle/weft-target-artifact-bundle.index

// RUN: rm -rf %t.nomatch.bundle && mkdir %t.nomatch.bundle
// RUN: not weft-translate --weft-source-artifact-bundle-front-door --weft-target-artifact-bundle-output-dir=%t.nomatch.bundle %S/../../Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-stale-toy.mlir 2>&1 | FileCheck %s --check-prefix=NO-MATCH --implicit-check-not="weft.target_artifact_bundle_export: complete"
// RUN: not test -e %t.nomatch.bundle/weft-target-artifact-bundle.index

// RUN: rm -rf %t.noartifact.bundle && mkdir %t.noartifact.bundle
// RUN: not weft-translate --weft-source-artifact-bundle-front-door --weft-target-artifact-bundle-output-dir=%t.noartifact.bundle %s 2>&1 | FileCheck %s --check-prefix=NO-ARTIFACT --implicit-check-not="weft.target_artifact_bundle_export: complete"
// RUN: not test -e %t.noartifact.bundle/weft-target-artifact-bundle.index

// RUN: rm -rf %t.missing.bundle
// RUN: not weft-translate --weft-source-artifact-bundle-front-door --weft-target-artifact-bundle-output-dir=%t.missing.bundle %S/../../Transforms/Toy/toy-template-source-front-door.mlir 2>&1 | FileCheck %s --check-prefix=MISSING-DIR --implicit-check-not="weft.target_artifact_bundle_export: complete"
// RUN: not test -e %t.missing.bundle/weft-target-artifact-bundle.index

// DISABLED: Weft-RV source-artifact bundle front door requires at least one registered source front-door pass

// NO-MATCH: Weft-RV execution plan coherence check failed for kernel <missing>: requires at least one weft.exec.kernel
// NO-MATCH: Weft-RV source-artifact bundle front door failed during source-artifact front-door pipeline

// NO-ARTIFACT: Weft-RV source-artifact bundle front door failed during source-artifact front-door pipeline

// MISSING-DIR: Weft-RV target artifact bundle export failed: output directory must already exist

module @source_artifact_bundle_front_door_no_artifact_input {
  weft.exec.kernel @source_front_door_no_artifact {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}
