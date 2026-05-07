// RUN: not tcrv-opt %s --tcrv-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_missing_property_evidence {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}

// CHECK: error: TianChen-RV RVV extension plugin first slice failed: capability id 'rvv' requires preserved property 'architecture'
