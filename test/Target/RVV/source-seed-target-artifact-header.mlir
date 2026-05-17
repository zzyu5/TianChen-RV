// RUN: tcrv-opt %S/../../Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir --tcrv-source-seed-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel"

// HEADER: #ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER: #define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.rvv.materialized_emitc_header.version: 1
// HEADER: tianchenrv.rvv.origin_plugin: rvv-plugin
// HEADER: tianchenrv.rvv.selected_variant: @seed_rvv_i32_add
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
// HEADER: void tcrv_emitc_seed_kernel_seed_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #endif
