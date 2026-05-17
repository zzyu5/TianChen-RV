// REQUIRES: tianchenrv-local-rvv-object-clang
// RUN: rm -f %t.o
// RUN: tcrv-translate --tcrv-export-target-artifact %s > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z57tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add"
// RUN: tcrv-translate --tcrv-export-target-header-artifact %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle %s | FileCheck %s --check-prefix=BUNDLE-STDOUT
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
    tcrv.exec.diagnostic {artifact_kind = "riscv-elf-relocatable-object", artifact_metadata = [{key = "rvv_emitc_lowerable_route", value = "rvv-i32m1-add-emitc-route"}, {key = "rvv_arithmetic_op", value = "add"}, {key = "tcrv_rvv.config_contract", value = "rvv-i32m1-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}, {key = "tcrv_rvv.sew", value = "32"}, {key = "tcrv_rvv.lmul", value = "m1"}, {key = "tcrv_rvv.tail_policy", value = "agnostic"}, {key = "tcrv_rvv.mask_policy", value = "agnostic"}, {key = "tcrv_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}, {key = "tcrv_rvv.runtime_avl_source", value = "runtime_abi:n"}, {key = "tcrv_rvv.vl_def", value = "tcrv_rvv.setvl"}, {key = "tcrv_rvv.vl_scope", value = "tcrv_rvv.with_vl"}, {key = "tcrv_rvv.vl_uses", value = "emitc_for,with_vl,i32_load,i32_load,i32_arithmetic,i32_store"}, {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}, {key = "tcrv_rvv.runtime_avl_abi_parameter", value = "n"}, {key = "tcrv_rvv.emitc_loop", value = "emitc.for"}, {key = "tcrv_rvv.loop_induction", value = "offset"}, {key = "tcrv_rvv.loop_step", value = "full_chunk_vl"}, {key = "tcrv_rvv.remaining_avl", value = "n-offset"}, {key = "tcrv_rvv.pointer_advance", value = "offset"}, {key = "tcrv_rvv.bounded_slice", value = "multi-vl-i32m1-arithmetic"}, {key = "tcrv_rvv.multi_vl", value = "supported"}], emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object", lowering_boundary = "tcrv_rvv.with_vl", lowering_pipeline = "rvv-i32m1-arithmetic-emitc-route-family", message = "RVV selected i32m1 arithmetic route materializes a verified EmitC module through the common TCRVEmitCLowerableRoute materializer, then uses the MLIR EmitC C/C++ emitter before RISC-V object packaging", origin = "rvv-plugin", plan_kind = "plugin-emission-plan", reason = "emission_plan", required_capabilities = [@rvv], role = "dispatch case", runtime_abi = "rvv-i32m1-add-callable-c-abi.v1", runtime_abi_kind = "plugin-owned-runtime-abi", runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1", runtime_abi_parameters = [{c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"}, {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"}, {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"}, {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}], runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue", severity = "info", status = "supported", target = @vector_source_rvv_i32_add}
    tcrv.exec.diagnostic {artifact_kind = "unsupported-emission-diagnostic", emission_kind = "scalar-fallback-unsupported-emission", lowering_pipeline = "scalar-fallback-no-materialized-emitc-route", message = "scalar fallback first slice has no materialized extension-family body, EmitC lowering, runtime ABI, target artifact route, or legacy metadata emission route", origin = "scalar-plugin", plan_kind = "plugin-emission-plan", reason = "emission_plan", required_capabilities = [@scalar_fallback], role = "dispatch fallback", runtime_abi = "scalar-fallback-no-runtime-abi", runtime_abi_kind = "unsupported-plugin-runtime-abi", runtime_abi_name = "unsupported-emission-runtime-abi", runtime_glue_role = "no-runtime-glue-unsupported", severity = "error", status = "unsupported", target = @vector_source_scalar_fallback}
  }
}

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
// HEADER: tianchenrv.rvv.runtime_avl_source: runtime_abi:n
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
// BUNDLE-INDEX: key: "rvv_emitc_lowerable_route"
// BUNDLE-INDEX: value: "rvv-i32m1-add-emitc-route"
// BUNDLE-INDEX: key: "tcrv_rvv.emitc_loop"
// BUNDLE-INDEX: value: "emitc.for"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family.header"
// BUNDLE-INDEX: evidence_role: "header-declaration"
