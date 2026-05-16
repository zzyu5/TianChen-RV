// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.generic.o
// RUN: file %t.generic.o | FileCheck %s --check-prefix=OBJECT
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="riscv_vector.h" --implicit-check-not="int main"
// RUN: tcrv-translate --tcrv-rvv-i32m1-add-header %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="riscv_vector.h" --implicit-check-not="int main"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STATUS
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="riscv_vector.h" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-tcrv-rvv-i32m1-add-callable-c-header.h
// RUN: file %t.bundle/artifact-0-riscv-elf-relocatable-object-tcrv-rvv-i32m1-add-riscv-elf-object.o | FileCheck %s --check-prefix=OBJECT
// RUN: tcrv-translate --tcrv-rvv-i32m1-add-object %s > %t.direct.o
// RUN: file %t.direct.o | FileCheck %s --check-prefix=OBJECT
// RUN: not tcrv-translate --tcrv-rvv-emitc-to-cpp %s 2>&1 | FileCheck %s --check-prefix=NONEMITC

module {
  tcrv.exec.kernel @rvv_i32_add_kernel {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {
      message = "selected explicit RVV i32m1 add",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      severity = "note",
      status = "selected",
      target = @rvv_i32_add
    }
  }
}

// PLAN: tcrv.exec.diagnostic {artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: lowering_pipeline = "tcrv-rvv-i32m1-add-riscv-elf-object"
// PLAN-SAME: runtime_abi = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_parameters = [{c_name = "lhs"
// PLAN-SAME: c_type = "const int32_t *"
// PLAN-SAME: ownership = "target-export-abi-owned"
// PLAN-SAME: role = "lhs-input-buffer"
// PLAN-SAME: c_name = "rhs"
// PLAN-SAME: c_type = "const int32_t *"
// PLAN-SAME: role = "rhs-input-buffer"
// PLAN-SAME: c_name = "out"
// PLAN-SAME: c_type = "int32_t *"
// PLAN-SAME: role = "output-buffer"
// PLAN-SAME: c_name = "n"
// PLAN-SAME: c_type = "size_t"
// PLAN-SAME: role = "runtime-element-count"
// PLAN-SAME: runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_i32_add

// OBJECT: ELF 64-bit LSB relocatable
// OBJECT-SAME: RISC-V

// HEADER: #ifndef TIANCHENRV_RVV_I32M1_ADD_CALLABLE_TCRV_EMITC_RVV_I32_ADD_KERNEL_RVV_I32_ADD_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: extern "C" {
// HEADER: tianchenrv.runtime_abi_name: rvv-i32m1-add-callable-c-abi.v1
// HEADER: tianchenrv.runtime_glue_role: emitc-cpp-rvv-intrinsic-runtime-glue
// HEADER: tianchenrv.object_route: tcrv-rvv-i32m1-add-riscv-elf-object
// HEADER: tianchenrv.header_route: tcrv-rvv-i32m1-add-callable-c-header
// HEADER: tianchenrv.runtime_abi_parameter[0]: lhs : const int32_t * : lhs-input-buffer
// HEADER: tianchenrv.runtime_abi_parameter[1]: rhs : const int32_t * : rhs-input-buffer
// HEADER: tianchenrv.runtime_abi_parameter[2]: out : int32_t * : output-buffer
// HEADER: tianchenrv.runtime_abi_parameter[3]: n : size_t : runtime-element-count
// HEADER: void tcrv_emitc_rvv_i32_add_kernel_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// BUNDLE-STATUS: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STATUS: index_file: "tianchenrv-target-artifact-bundle.index"

// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-tcrv-rvv-i32m1-add-riscv-elf-object.o"
// BUNDLE-INDEX: component_group: "rvv-i32m1-add-callable-artifact-bundle.v1"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @rvv_i32_add
// BUNDLE-INDEX: artifact_kind: "riscv-elf-relocatable-object"
// BUNDLE-INDEX: route: "tcrv-rvv-i32m1-add-riscv-elf-object"
// BUNDLE-INDEX: runtime_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter[0]:
// BUNDLE-INDEX: c_name: "lhs"
// BUNDLE-INDEX: runtime_abi_parameter[1]:
// BUNDLE-INDEX: c_name: "rhs"
// BUNDLE-INDEX: runtime_abi_parameter[2]:
// BUNDLE-INDEX: c_name: "out"
// BUNDLE-INDEX: runtime_abi_parameter[3]:
// BUNDLE-INDEX: c_name: "n"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-rvv-i32m1-add-callable-c-header.h"
// BUNDLE-INDEX: component_group: "rvv-i32m1-add-callable-artifact-bundle.v1"
// BUNDLE-INDEX: component_role: "header"
// BUNDLE-INDEX: external_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: artifact_kind: "runtime-callable-c-header"
// BUNDLE-INDEX: route: "tcrv-rvv-i32m1-add-callable-c-header"
// BUNDLE-INDEX: owner: "rvv-plugin"
// BUNDLE-INDEX: runtime_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: evidence_role: "header-declaration"

// NONEMITC: requires an already materialized EmitC module
// NONEMITC: found non-EmitC op 'tcrv.exec.capability'
