// REQUIRES: tianchenrv-local-rvv-object-clang
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z57tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add"
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.i32_add/s//tcrv_rvv.i32_sub/' | not tcrv-translate --tcrv-export-target-artifact 2>&1 | FileCheck %s --check-prefix=STALE-OP --implicit-check-not="Format: elf64"
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel" < %t.bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

// This is lower-level materialized-module exporter coverage. It intentionally
// starts from selected/materialized RVV IR and does not exercise the source
// artifact front-door workflow.

module {
  tcrv.exec.kernel @vector_source_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @vector_source_rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-0:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %1 = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-1:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %2 = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-2:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %3 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "source-arg-3:n", role = "runtime-element-count"} : index
      %4 = tcrv_rvv.setvl %3 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %4 attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %5 = tcrv_rvv.i32_load %0, %4 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %6 = tcrv_rvv.i32_load %1, %4 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %7 = tcrv_rvv.i32_add %5, %6, %4 : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %2, %7, %4 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @vector_source_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @vector_source_rvv_i32_add {origin = "rvv-plugin", policy = "source-pattern-selected-rvv-case"}
      tcrv.exec.fallback @vector_source_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "source-pattern-conservative-fallback-envelope"}
    }
  }
}

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add

// STALE-OP: RVV materialized EmitC target artifact bridge failed
// STALE-OP: selected RVV i32m1 arithmetic route expected i32_add but variant body contains i32_sub

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
// HEADER: tianchenrv.rvv.sew: 32
// HEADER: tianchenrv.rvv.lmul: m1
// HEADER: tianchenrv.rvv.runtime_vl_contract: rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1
// HEADER: tianchenrv.rvv.vl_def: tcrv_rvv.setvl
// HEADER: tianchenrv.rvv.vl_scope: tcrv_rvv.with_vl
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs,out,n
// HEADER: tianchenrv.rvv.emitc_loop: emitc.for
// HEADER: tianchenrv.rvv.multi_vl: supported
// HEADER: void tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// BUNDLE-STDOUT: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// BUNDLE-INDEX: tianchenrv.target_artifact_bundle.version: 1
// BUNDLE-INDEX: bundle_status: "complete"
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: artifact[0]:
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @vector_source_rvv_i32_add
// BUNDLE-INDEX: role: "dispatch case"
// BUNDLE-INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family"
// BUNDLE-INDEX: owner: "rvv-plugin"
// BUNDLE-INDEX: runtime_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 4
// BUNDLE-INDEX: key: "rvv_emitc_lowerable_route"
// BUNDLE-INDEX: value: "rvv-i32m1-add-emitc-route"
// BUNDLE-INDEX: key: "rvv_construction_protocol"
// BUNDLE-INDEX: value: "extension-family-construction-protocol.v1"
// BUNDLE-INDEX: key: "rvv_common_interface_realization"
// BUNDLE-INDEX: key: "rvv_emitc_route_mapping"
// BUNDLE-INDEX: value: "rvv-i32m1-arithmetic-emitc-route-family"
// BUNDLE-INDEX: key: "rvv_runtime_abi_contract"
// BUNDLE-INDEX: value: "rvv-i32m1-arithmetic-callable-c-abi-family.v1"
// BUNDLE-INDEX: key: "rvv_object_handoff"
// BUNDLE-INDEX: value: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: key: "tcrv_rvv.emitc_loop"
// BUNDLE-INDEX: value: "emitc.for"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family.header"
// BUNDLE-INDEX: evidence_role: "header-declaration"
