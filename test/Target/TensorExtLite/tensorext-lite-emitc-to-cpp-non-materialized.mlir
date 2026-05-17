// RUN: not tcrv-translate --tcrv-tensorext-lite-emitc-to-cpp %s 2>&1 | FileCheck %s --implicit-check-not="tcrv_tensorext_lite_config" --implicit-check-not="tcrv_tensorext_lite_tile_mma"

module attributes {
  tcrv_tensorext_lite.source_front_door = "fragment_mma_template",
  tcrv_tensorext_lite.source_kernel = "tensorext_lite_header_export"
} {
}

// CHECK: TensorExtLite materialized EmitC C/C++ emitter bridge failed
// CHECK-SAME: stale TensorExtLite source-front-door metadata is not accepted
// CHECK-SAME: run the source-artifact front-door pipeline before target translation
