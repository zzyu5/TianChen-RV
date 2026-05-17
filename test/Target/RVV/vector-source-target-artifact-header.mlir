// RUN: tcrv-opt %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"
// RUN: tcrv-opt %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-SUB --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"
// RUN: tcrv-opt %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-MUL --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"

// HEADER: #ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER: #define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.rvv.materialized_emitc_header.version: 1
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
// HEADER: tianchenrv.rvv.runtime_avl_abi_parameter: n
// HEADER: tianchenrv.rvv.vl_def: tcrv_rvv.setvl
// HEADER: tianchenrv.rvv.vl_scope: tcrv_rvv.with_vl
// HEADER: tianchenrv.rvv.emitc_loop: emitc.for
// HEADER: tianchenrv.rvv.loop_induction: offset
// HEADER: tianchenrv.rvv.loop_step: full_chunk_vl
// HEADER: tianchenrv.rvv.remaining_avl: n-offset
// HEADER: tianchenrv.rvv.pointer_advance: offset
// HEADER: tianchenrv.rvv.bounded_slice: multi-vl-i32m1-arithmetic
// HEADER: tianchenrv.rvv.multi_vl: supported
// HEADER: #ifdef __cplusplus
// HEADER: extern "C" {
// HEADER: #endif
// HEADER: void tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #ifdef __cplusplus
// HEADER: } /* extern "C" */
// HEADER: #endif
// HEADER: #endif

// HEADER-SUB: #ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER-SUB: tianchenrv.rvv.origin_plugin: rvv-plugin
// HEADER-SUB: tianchenrv.rvv.selected_variant: @vector_source_sub_rvv_i32_sub
// HEADER-SUB: tianchenrv.rvv.selected_route: rvv-i32m1-arithmetic-emitc-route-family
// HEADER-SUB: tianchenrv.rvv.runtime_abi_name: rvv-i32m1-sub-callable-c-abi.v1
// HEADER-SUB: tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned
// HEADER-SUB: tianchenrv.rvv.runtime_abi_parameter[1]: const int32_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned
// HEADER-SUB: tianchenrv.rvv.runtime_abi_parameter[2]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER-SUB: tianchenrv.rvv.runtime_abi_parameter[3]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER-SUB: tianchenrv.rvv.bounded_slice: multi-vl-i32m1-arithmetic
// HEADER-SUB: tianchenrv.rvv.multi_vl: supported
// HEADER-SUB: void tcrv_emitc_vector_source_sub_kernel_vector_source_sub_rvv_i32_sub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER-SUB: #endif

// HEADER-MUL: #ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER-MUL: tianchenrv.rvv.origin_plugin: rvv-plugin
// HEADER-MUL: tianchenrv.rvv.selected_variant: @vector_source_mul_rvv_i32_mul
// HEADER-MUL: tianchenrv.rvv.selected_route: rvv-i32m1-arithmetic-emitc-route-family
// HEADER-MUL: tianchenrv.rvv.runtime_abi_name: rvv-i32m1-mul-callable-c-abi.v1
// HEADER-MUL: tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned
// HEADER-MUL: tianchenrv.rvv.runtime_abi_parameter[1]: const int32_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned
// HEADER-MUL: tianchenrv.rvv.runtime_abi_parameter[2]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER-MUL: tianchenrv.rvv.runtime_abi_parameter[3]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER-MUL: tianchenrv.rvv.bounded_slice: multi-vl-i32m1-arithmetic
// HEADER-MUL: tianchenrv.rvv.multi_vl: supported
// HEADER-MUL: void tcrv_emitc_vector_source_mul_kernel_vector_source_mul_rvv_i32_mul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER-MUL: #endif
