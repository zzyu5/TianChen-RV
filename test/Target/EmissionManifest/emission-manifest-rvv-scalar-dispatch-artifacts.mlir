// RUN: tcrv-opt %S/../RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-emission-manifest | FileCheck %s --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=runtime_success --implicit-check-not=password --implicit-check-not=token

// CHECK: tianchenrv.emission_manifest.version: 1
// CHECK: module: "rvv_scalar_dispatch_input"
// CHECK-LABEL: kernel @dispatch_vadd
// CHECK: selected_surface: dispatch
// CHECK: dispatch_case[0]: @rvv_first_slice
// CHECK: dispatch_fallback: @scalar_fallback_first_slice

// CHECK: target_artifacts:
// CHECK: artifact[0]:
// CHECK: selected_surface: "dispatch"
// CHECK: component[0]:
// CHECK: selected_variant: @rvv_first_slice
// CHECK: role: "dispatch case"
// CHECK: component[1]:
// CHECK: selected_variant: @scalar_fallback_first_slice
// CHECK: role: "dispatch fallback"
// CHECK: artifact_kind: "runtime-callable-c-source"
// CHECK: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"
// CHECK: owner: "rvv-scalar-dispatch-target"
// CHECK: generic_front_door_selectable: true
// CHECK: selectable_via: "tcrv-export-target-source-artifact"
// CHECK: direct_helper_route: true
// CHECK: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// CHECK: runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// CHECK: evidence_role: "compiler-artifact"

// CHECK: artifact[1]:
// CHECK: selected_surface: "dispatch"
// CHECK: artifact_kind: "runtime-callable-c-header"
// CHECK: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"
// CHECK: owner: "rvv-scalar-dispatch-target"
// CHECK: generic_front_door_selectable: true
// CHECK: selectable_via: "tcrv-export-target-header-artifact"
// CHECK: direct_helper_route: true
// CHECK: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// CHECK: runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// CHECK: evidence_role: "header-declaration"

// CHECK: artifact[2]:
// CHECK: selected_surface: "dispatch"
// CHECK: artifact_kind: "riscv-elf-relocatable-object"
// CHECK: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"
// CHECK: owner: "rvv-scalar-dispatch-target"
// CHECK: generic_front_door_selectable: true
// CHECK: selectable_via: "tcrv-export-target-artifact"
// CHECK: direct_helper_route: true
// CHECK: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// CHECK: runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// CHECK: evidence_role: "relocatable-object"

// CHECK: path[0]:
// CHECK: selected_variant: @rvv_first_slice
// CHECK: role: "dispatch case"
// CHECK: origin: "rvv-plugin"
// CHECK: runtime_abi_parameters:
// CHECK: c_name: "lhs"
// CHECK: role: "lhs-input-buffer"
// CHECK: c_name: "n"
// CHECK: role: "runtime-element-count"
// CHECK: required_capabilities: [@rvv]
// CHECK: path[1]:
// CHECK: selected_variant: @scalar_fallback_first_slice
// CHECK: role: "dispatch fallback"
// CHECK: origin: "scalar-plugin"
// CHECK: runtime_abi_parameters:
// CHECK: c_name: "lhs"
// CHECK: role: "lhs-input-buffer"
// CHECK: c_name: "n"
// CHECK: role: "runtime-element-count"
// CHECK: required_capabilities: [@scalar_fallback]
// CHECK: fallback_role: "conservative"
