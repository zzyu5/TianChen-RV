// RUN: rm -rf %t.no-viable.bundle && mkdir %t.no-viable.bundle
// RUN: not tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.no-viable.bundle %s 2>&1 | FileCheck %s --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: not test -e %t.no-viable.bundle/tianchenrv-target-artifact-bundle.index

module @plan_and_export_target_artifact_bundle_no_viable_input {
  tcrv.exec.kernel @plan_and_export_no_viable {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}

// CHECK: error: TianChen-RV plugin variant materialization for kernel @plan_and_export_no_viable collected no viable plugin proposals; recoverable plugin declines in registration order: rvv-plugin:
// CHECK-SAME: capability id 'rvv' requires preserved property 'architecture'
// CHECK: error: TianChen-RV plan-and-export target artifact bundle failed during execution planning pipeline
