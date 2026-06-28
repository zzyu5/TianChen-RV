// REQUIRES: tianchenrv-local-rvv-object-clang
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-opt %S/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="source-front-door" --implicit-check-not="tcrv_rvv.i32_" < %t.bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="source-front-door" --implicit-check-not="tcrv_rvv.i32_" < %t.bundle/tianchenrv-target-artifact-bundle.index

// This target artifact bundle test intentionally starts from the existing
// pre-realized selected-body fixture. It proves the selected-boundary producer,
// provider-built route, and target artifact validator reach the generated
// object/header bundle before runtime evidence is collected by the script.

// BUNDLE-STDOUT: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_pre_dual_cmp_mask_select_kernel_pre_rvv_dual_cmp_mask_select

// HEADER: #ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER-DAG: tianchenrv.rvv.selected_variant: @pre_rvv_dual_cmp_mask_select
// HEADER-DAG: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1
// HEADER-DAG: tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *cmp_lhs_a role=lhs-input-buffer ownership=target-export-abi-owned
// HEADER-DAG: tianchenrv.rvv.runtime_abi_parameter[1]: int32_t rhs_scalar_a role=rhs-scalar-value ownership=target-export-abi-owned
// HEADER-DAG: tianchenrv.rvv.runtime_abi_parameter[2]: const int32_t *cmp_lhs_b role=rhs-input-buffer ownership=target-export-abi-owned
// HEADER-DAG: tianchenrv.rvv.runtime_abi_parameter[3]: int32_t rhs_scalar_b role=rhs-secondary-scalar-value ownership=target-export-abi-owned
// HEADER-DAG: tianchenrv.rvv.runtime_abi_parameter[4]: const int32_t *true_value role=true-value-input-buffer ownership=target-export-abi-owned
// HEADER-DAG: tianchenrv.rvv.runtime_abi_parameter[5]: const int32_t *false_value role=false-value-input-buffer ownership=target-export-abi-owned
// HEADER-DAG: tianchenrv.rvv.runtime_abi_parameter[6]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER-DAG: tianchenrv.rvv.runtime_abi_parameter[7]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER-DAG: tianchenrv.rvv.runtime_vl_contract: rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1
// HEADER-DAG: tianchenrv.rvv.runtime_abi_order: cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n
// HEADER-DAG: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-dual-cmp-mask-and-select-plan-validated
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:rs_dual_cmp_mask_select.v1
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:rs_dual_cmp_mask_select.v1;cmp_lhs_a=lhs-input-buffer:cmp_lhs_a:abi|ld|cmp|and|hdr;rhs_scalar_a=rhs-scalar-value:rhs_scalar_a:abi|splat|cmp|hdr;cmp_lhs_b=rhs-input-buffer:cmp_lhs_b:abi|ld|cmp|and|hdr;rhs_scalar_b=rhs-secondary-scalar-value:rhs_scalar_b:abi|splat|cmp|hdr;true_value=true-value-input-buffer:true_value:abi|ld|sel|hdr;false_value=false-value-input-buffer:false_value:abi|ld|sel|hdr;out=output-buffer:out:abi|st|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr
// HEADER-DAG: tianchenrv.rvv.computed_mask_select_mask_producer_source: dual-runtime-scalar-splat-compare-rhs-mask-and
// HEADER-DAG: tianchenrv.rvv.mask_role: predicate-mask-produced-by-mask-and
// HEADER-DAG: tianchenrv.rvv.mask_source: mask-and-of-two-runtime-scalar-compare-produced-masks
// HEADER-DAG: tianchenrv.rvv.mask_memory_form: composed-compare-produced-mask
// HEADER-DAG: tianchenrv.rvv.mask_composition: and
// HEADER-DAG: tianchenrv.rvv.select_layout: select-true-value-when-mask-else-false-value
// HEADER: void tcrv_emitc_pre_dual_cmp_mask_select_kernel_pre_rvv_dual_cmp_mask_select(const int32_t *cmp_lhs_a, int32_t rhs_scalar_a, const int32_t *cmp_lhs_b, int32_t rhs_scalar_b, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 2
// INDEX: artifact[0]:
// INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1"
// INDEX: selected_variant: @pre_rvv_dual_cmp_mask_select
// INDEX: route: "rvv-generic-typed-body-emitc-route-family"
// INDEX: owner: "rvv-plugin"
// INDEX: runtime_abi_parameter_count: 8
// INDEX: c_name: "cmp_lhs_a"
// INDEX: role: "lhs-input-buffer"
// INDEX: c_name: "rhs_scalar_a"
// INDEX: role: "rhs-scalar-value"
// INDEX: c_name: "cmp_lhs_b"
// INDEX: role: "rhs-input-buffer"
// INDEX: c_name: "rhs_scalar_b"
// INDEX: role: "rhs-secondary-scalar-value"
// INDEX: c_name: "true_value"
// INDEX: role: "true-value-input-buffer"
// INDEX: c_name: "false_value"
// INDEX: role: "false-value-input-buffer"
// INDEX: c_name: "out"
// INDEX: role: "output-buffer"
// INDEX: c_name: "n"
// INDEX: role: "runtime-element-count"
// INDEX: key: "rvv_emitc_lowerable_route"
// INDEX: value: "rvv-generic-runtime-scalar-dual-cmp-mask-and-select-emitc-route"
// INDEX: key: "rvv_selected_body_operation"
// INDEX: value: "runtime_scalar_dual_cmp_mask_and_select"
// INDEX: key: "tcrv_rvv.runtime_abi_order"
// INDEX: value: "cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n"
// INDEX: key: "tcrv_rvv.route_operand_binding_plan"
// INDEX: value: "rvv-route-operand-binding:rs_dual_cmp_mask_select.v1"
// INDEX: key: "tcrv_rvv.provider_supported_mirror"
// INDEX: value: "provider_supported_mirror:rvv-runtime-scalar-dual-cmp-mask-and-select-plan-validated"
// INDEX: key: "tcrv_rvv.computed_mask_select_mask_producer_source"
// INDEX: value: "dual-runtime-scalar-splat-compare-rhs-mask-and"
// INDEX: key: "tcrv_rvv.mask_composition"
// INDEX: value: "and"
// INDEX: key: "tcrv_rvv.select_layout"
// INDEX: value: "select-true-value-when-mask-else-false-value"
