// REQUIRES: weft-local-rvv-object-clang
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z57weft_emitc_vector_source_kernel_vector_source_rvv_i32_add"
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/rvv_selected_body_operation\", value = \"add\"/s//rvv_selected_body_operation\", value = \"sub\"/' | not weft-translate --weft-export-target-artifact 2>&1 | FileCheck %s --check-prefix=STALE-OP --implicit-check-not="Format: elf64"
// RUN: weft-opt %s --weft-materialize-emission-plans | sed 's/rvv-generic-binary-add-emitc-route/rvv-generic-binary-sub-emitc-route/' | not weft-translate --weft-export-target-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ROUTE --implicit-check-not="Format: elf64"
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact-bundle --weft-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel" < %t.bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/weft-target-artifact-bundle.index

// This is lower-level materialized-module exporter coverage. It intentionally
// starts from selected/materialized RVV IR and does not exercise the source
// artifact front-door workflow.

module {
  weft.exec.kernel @vector_source_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @vector_source_rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-0:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %1 = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-1:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %2 = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-2:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %3 = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "source-arg-3:n", role = "runtime-element-count"} : index
      %4 = weft_rvv.setvl %3 {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %4 attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "dispatch case", selected_variant = @vector_source_rvv_i32_add, sew = 32 : i64, source_kernel = "vector_source_kernel", status = "selected-lowering-boundary"} {
        %5 = weft_rvv.load %0, %4 : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %6 = weft_rvv.load %1, %4 : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %7 = weft_rvv.binary %5, %6, %4 {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %2, %7, %4 : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @vector_source_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @vector_source_rvv_i32_add {origin = "rvv-plugin", policy = "source-pattern-selected-rvv-case"}
      weft.exec.fallback @vector_source_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "source-pattern-conservative-fallback-envelope"}
    }
  }
}

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: weft_emitc_vector_source_kernel_vector_source_rvv_i32_add

// STALE-OP: RVV materialized EmitC target artifact bridge failed
// STALE-OP: rvv_selected_body_operation provenance must mirror selected typed RVV body operation 'add'
// STALE-OP-SAME: sub

// STALE-ROUTE: RVV materialized EmitC target artifact bridge failed
// STALE-ROUTE: rvv_emitc_lowerable_route provenance must mirror selected typed RVV body route 'rvv-generic-binary-add-emitc-route'
// STALE-ROUTE-SAME: rvv-generic-binary-sub-emitc-route

// HEADER: #ifndef WEFT_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: weft.rvv.origin_plugin: rvv-plugin
// HEADER: weft.rvv.selected_variant: @vector_source_rvv_i32_add
// HEADER: weft.rvv.selected_route: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-binary-add-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_parameter[0]: const int32_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned
// HEADER: weft.rvv.runtime_abi_parameter[1]: const int32_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned
// HEADER: weft.rvv.runtime_abi_parameter[2]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER: weft.rvv.runtime_abi_parameter[3]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER: weft.rvv.construction_protocol: extension-family-construction-protocol.v1
// HEADER: weft.rvv.extension_archetype: rvv-generic-typed-body
// HEADER: weft.rvv.semantic_role_graph: runtime_abi->configure->scope->load->compute->store
// HEADER: weft.rvv.common_interface_realization: runtime_abi/resource+emitc
// HEADER: weft.rvv.typed_role_realization: runtime_abi:weft_rvv.runtime_abi_value
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.target_artifact_route: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.target_artifact_kind: riscv-elf-relocatable-object
// HEADER: weft.rvv.evidence_profile: parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_target_artifact|ssh_rvv_required_for_runtime_claims
// HEADER: weft.rvv.bundle_component_group: rvv-generic-typed-body-materialized-emitc-bundle.v1
// HEADER: weft.rvv.object_handoff: materialized-emitc-cpp-rvv-intrinsic-object
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.sew: 32
// HEADER: weft.rvv.lmul: m1
// HEADER: weft.rvv.tail_policy: agnostic
// HEADER: weft.rvv.mask_policy: agnostic
// HEADER: weft.rvv.runtime_vl_contract: rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1
// HEADER: weft.rvv.runtime_avl_source: runtime_abi:n
// HEADER: weft.rvv.vl_def: weft_rvv.setvl
// HEADER: weft.rvv.vl_scope: weft_rvv.with_vl
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs,out,n
// HEADER: weft.rvv.emitc_loop: emitc.for
// HEADER: weft.rvv.remaining_avl: n-offset
// HEADER: weft.rvv.pointer_advance: offset
// HEADER: weft.rvv.multi_vl: supported
// HEADER: void weft_emitc_vector_source_kernel_vector_source_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// BUNDLE-STDOUT: weft.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "weft-target-artifact-bundle.index"

// BUNDLE-INDEX: weft.target_artifact_bundle.version: 1
// BUNDLE-INDEX: bundle_status: "complete"
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: artifact[0]:
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "rvv-generic-binary-add-callable-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @vector_source_rvv_i32_add
// BUNDLE-INDEX: role: "dispatch case"
// BUNDLE-INDEX: route: "rvv-generic-typed-body-emitc-route-family"
// BUNDLE-INDEX: owner: "rvv-plugin"
// BUNDLE-INDEX: runtime_abi_name: "rvv-generic-binary-add-callable-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 4
// BUNDLE-INDEX: key: "rvv_emitc_lowerable_route"
// BUNDLE-INDEX: value: "rvv-generic-binary-add-emitc-route"
// BUNDLE-INDEX: key: "rvv_construction_protocol"
// BUNDLE-INDEX: value: "extension-family-construction-protocol.v1"
// BUNDLE-INDEX: key: "rvv_common_interface_realization"
// BUNDLE-INDEX: key: "rvv_emitc_route_mapping"
// BUNDLE-INDEX: value: "rvv-generic-typed-body-emitc-route-family"
// BUNDLE-INDEX: key: "rvv_target_artifact_route"
// BUNDLE-INDEX: value: "rvv-generic-typed-body-emitc-route-family"
// BUNDLE-INDEX: key: "rvv_target_artifact_kind"
// BUNDLE-INDEX: value: "riscv-elf-relocatable-object"
// BUNDLE-INDEX: key: "rvv_runtime_abi_name"
// BUNDLE-INDEX: value: "rvv-generic-binary-add-callable-c-abi.v1"
// BUNDLE-INDEX: key: "rvv_runtime_abi_contract"
// BUNDLE-INDEX: value: "rvv-generic-binary-add-callable-c-abi"
// BUNDLE-INDEX: key: "rvv_object_handoff"
// BUNDLE-INDEX: value: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: key: "weft_rvv.emitc_loop"
// BUNDLE-INDEX: value: "emitc.for"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: route: "rvv-generic-typed-body-emitc-route-family.header"
// BUNDLE-INDEX: evidence_role: "header-declaration"
