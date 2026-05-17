// RUN: not tcrv-translate --tcrv-export-target-artifact %s 2>&1 | FileCheck %s --check-prefix=NONMATERIALIZED --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | sed 's/template-extension-compute-skeleton-emitc-route/template-extension-stale-route/g' | not tcrv-translate --tcrv-export-target-artifact 2>&1 | FileCheck %s --check-prefix=STALE --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"

module {
  tcrv.exec.kernel @template_emitc_kernel {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }
}

// NONMATERIALIZED: TianChen-RV execution plan coherence check failed
// NONMATERIALIZED: requires a selected dispatch or selected-path diagnostic surface

// STALE: TianChen-RV execution plan coherence check failed
// STALE: unknown target artifact export route id 'template-extension-stale-route'
