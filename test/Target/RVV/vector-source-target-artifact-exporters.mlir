// REQUIRES: tianchenrv-local-rvv-object-clang
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z57tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add"
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel" < %t.bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" < %t.bundle/tianchenrv-target-artifact-bundle.index

// This is source-level target-artifact bridge coverage. It intentionally starts
// from func/scf/vector/arith source MLIR, then uses the production
// source-artifact pipeline before each target exporter. It is not a lower-level
// selected/materialized RVV IR fixture.

module {
  func.func @vector_source(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// PLAN-LABEL: tcrv.exec.kernel @vector_source_kernel
// PLAN: tcrv.exec.capability @rvv
// PLAN-SAME: id = "rvv"
// PLAN-SAME: kind = "isa-vector"
// PLAN-LABEL: tcrv.exec.variant @vector_source_rvv_i32_add
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: requires = [@rvv]
// PLAN: = tcrv_rvv.runtime_abi_value
// PLAN-SAME: c_name = "lhs"
// PLAN-SAME: role = "lhs-input-buffer"
// PLAN: = tcrv_rvv.runtime_abi_value
// PLAN-SAME: c_name = "rhs"
// PLAN-SAME: role = "rhs-input-buffer"
// PLAN: = tcrv_rvv.runtime_abi_value
// PLAN-SAME: c_name = "out"
// PLAN-SAME: role = "output-buffer"
// PLAN: = tcrv_rvv.runtime_abi_value
// PLAN-SAME: c_name = "n"
// PLAN-SAME: role = "runtime-element-count"
// PLAN: tcrv_rvv.with_vl
// PLAN: tcrv_rvv.i32_add
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: artifact_metadata = [{key = "rvv_emitc_lowerable_route", value = "rvv-i32m1-add-emitc-route"}
// PLAN-SAME: {key = "rvv_arithmetic_op", value = "add"}
// PLAN-SAME: {key = "rvv_construction_protocol", value = "extension-family-construction-protocol.v1"}
// PLAN-SAME: {key = "rvv_extension_archetype", value = "rvv-finite-binary"}
// PLAN-SAME: {key = "rvv_semantic_role_graph", value = "runtime_abi->configure->scope->load->compute->store"}
// PLAN-SAME: {key = "rvv_emitc_route_mapping", value = "rvv-i32m1-arithmetic-emitc-route-family"}
// PLAN-SAME: {key = "rvv_runtime_abi_contract", value = "rvv-i32m1-arithmetic-callable-c-abi-family.v1"}
// PLAN-SAME: {key = "rvv_bundle_component_group", value = "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1"}
// PLAN-SAME: {key = "rvv_object_handoff", value = "materialized-emitc-cpp-rvv-intrinsic-object"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: lowering_pipeline = "rvv-i32m1-arithmetic-emitc-route-family"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: required_capabilities = [@rvv]
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_parameters = [{c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"}, {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"}, {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"}, {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}]
// PLAN-SAME: runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @vector_source_rvv_i32_add

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add

// HEADER: #ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.rvv.origin_plugin: rvv-plugin
// HEADER: tianchenrv.rvv.selected_variant: @vector_source_rvv_i32_add
// HEADER: tianchenrv.rvv.selected_route: rvv-i32m1-arithmetic-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-i32m1-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[1]: const int32_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[2]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[3]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.construction_protocol: extension-family-construction-protocol.v1
// HEADER: tianchenrv.rvv.extension_archetype: rvv-finite-binary
// HEADER: tianchenrv.rvv.semantic_role_graph: runtime_abi->configure->scope->load->compute->store
// HEADER: tianchenrv.rvv.common_interface_realization: runtime_abi/resource+emitc
// HEADER: tianchenrv.rvv.typed_role_realization: runtime_abi:tcrv_rvv.runtime_abi_value
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-i32m1-arithmetic-emitc-route-family
// HEADER: tianchenrv.rvv.evidence_profile: parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_target_artifact|ssh_rvv_required_for_runtime_claims
// HEADER: tianchenrv.rvv.runtime_abi_contract: rvv-i32m1-arithmetic-callable-c-abi-family.v1
// HEADER: tianchenrv.rvv.bundle_component_group: rvv-i32m1-arithmetic-materialized-emitc-bundle.v1
// HEADER: tianchenrv.rvv.object_handoff: materialized-emitc-cpp-rvv-intrinsic-object
// HEADER: tianchenrv.rvv.runtime_avl_source: runtime_abi:n
// HEADER: tianchenrv.rvv.config_contract: rvv-i32m1-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.runtime_vl_contract: rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1
// HEADER: tianchenrv.rvv.vl_def: tcrv_rvv.setvl
// HEADER: tianchenrv.rvv.vl_scope: tcrv_rvv.with_vl
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs,out,n
// HEADER: tianchenrv.rvv.emitc_loop: emitc.for
// HEADER: tianchenrv.rvv.multi_vl: supported
// HEADER: extern "C" {
// HEADER: void tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// BUNDLE-STDOUT: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// BUNDLE-INDEX: tianchenrv.target_artifact_bundle.version: 1
// BUNDLE-INDEX: bundle_status: "complete"
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX-LABEL: artifact[0]:
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @vector_source_rvv_i32_add
// BUNDLE-INDEX: role: "dispatch case"
// BUNDLE-INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family"
// BUNDLE-INDEX: owner: "rvv-plugin"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 4
// BUNDLE-INDEX: c_name: "lhs"
// BUNDLE-INDEX: c_name: "rhs"
// BUNDLE-INDEX: c_name: "out"
// BUNDLE-INDEX: c_name: "n"
// BUNDLE-INDEX: key: "rvv_emitc_lowerable_route"
// BUNDLE-INDEX: value: "rvv-i32m1-add-emitc-route"
// BUNDLE-INDEX: key: "rvv_arithmetic_op"
// BUNDLE-INDEX: value: "add"
// BUNDLE-INDEX: key: "rvv_construction_protocol"
// BUNDLE-INDEX: value: "extension-family-construction-protocol.v1"
// BUNDLE-INDEX: key: "rvv_extension_archetype"
// BUNDLE-INDEX: value: "rvv-finite-binary"
// BUNDLE-INDEX: key: "rvv_common_interface_realization"
// BUNDLE-INDEX: key: "rvv_emitc_route_mapping"
// BUNDLE-INDEX: value: "rvv-i32m1-arithmetic-emitc-route-family"
// BUNDLE-INDEX: key: "rvv_runtime_abi_contract"
// BUNDLE-INDEX: value: "rvv-i32m1-arithmetic-callable-c-abi-family.v1"
// BUNDLE-INDEX: key: "rvv_object_handoff"
// BUNDLE-INDEX: value: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: key: "tcrv_rvv.runtime_abi_order"
// BUNDLE-INDEX: value: "lhs,rhs,out,n"
// BUNDLE-INDEX: key: "tcrv_rvv.emitc_loop"
// BUNDLE-INDEX: value: "emitc.for"
// BUNDLE-INDEX: key: "tcrv_rvv.multi_vl"
// BUNDLE-INDEX: value: "supported"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX-LABEL: artifact[1]:
// BUNDLE-INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family.header"
// BUNDLE-INDEX: evidence_role: "header-declaration"
