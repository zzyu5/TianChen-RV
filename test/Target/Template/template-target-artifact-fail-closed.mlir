// RUN: not weft-translate --weft-export-target-artifact %s 2>&1 | FileCheck %s --check-prefix=NONMATERIALIZED --implicit-check-not="weft.target_artifact_bundle_export: complete"
// RUN: sed 's/template-extension-compute-skeleton-emitc-route/template-extension-stale-route/g' %S/template-target-artifact-object.mlir | not weft-translate --weft-export-target-artifact 2>&1 | FileCheck %s --check-prefix=STALE --implicit-check-not="weft.target_artifact_bundle_export: complete"

module {
  weft.exec.kernel @template_emitc_kernel {
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }
}

// NONMATERIALIZED: Weft-RV execution plan coherence check failed
// NONMATERIALIZED: requires a selected dispatch or selected-path diagnostic surface

// STALE: Weft-RV execution plan coherence check failed
// STALE: unknown target artifact export route id 'template-extension-stale-route'
