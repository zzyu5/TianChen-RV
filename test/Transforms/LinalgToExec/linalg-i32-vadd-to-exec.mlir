// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=lowering_descriptor --implicit-check-not=int64_t --implicit-check-not=__riscv_vadd_vv_i64m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not=__riscv --implicit-check-not=riscv_vector --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.linalg-i32-vadd.o
// RUN: llvm-readobj --file-headers --symbols %t.linalg-i32-vadd.o | FileCheck %s --check-prefix=OBJECT --implicit-check-not="Name: main"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '/^[[:space:]]*tcrv_rvv.i32_vadd_microkernel attributes/,/^    }/d' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-BODY --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed -e '/tcrv_rvv.lowering_boundary/s/emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface", //' -e '/tcrv_rvv.lowering_boundary/s/emitc_source_op = "tcrv_rvv.i32_add", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_dtype = "i32", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_family = "i32-vadd", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_microkernel_op = "tcrv_rvv.i32_vadd_microkernel", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_operator = "add", //' -e '/tcrv_rvv.lowering_boundary/s/selected_binary_source_kind = "frontend-lowering", //' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-BOUNDARY-SOURCE --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed -e '/tcrv_rvv.i32_vadd_microkernel attributes/s/emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface", //' -e '/tcrv_rvv.i32_vadd_microkernel attributes/s/emitc_source_op = "tcrv_rvv.i32_add", //' -e '/tcrv_rvv.i32_vadd_microkernel attributes/s/selected_binary_dtype = "i32", //' -e '/tcrv_rvv.i32_vadd_microkernel attributes/s/selected_binary_family = "i32-vadd", //' -e '/tcrv_rvv.i32_vadd_microkernel attributes/s/selected_binary_microkernel_op = "tcrv_rvv.i32_vadd_microkernel", //' -e '/tcrv_rvv.i32_vadd_microkernel attributes/s/selected_binary_operator = "add", //' -e '/tcrv_rvv.i32_vadd_microkernel attributes/s/selected_binary_source_kind = "frontend-lowering", //' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-OP-SOURCE --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '/tcrv_rvv.i32_vadd_microkernel attributes/s/selected_binary_family = "i32-vadd"/selected_binary_family = "i32-vsub"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-OP-SOURCE --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/tcrv_rvv.selected_vector_suffix = "i32m1"/s//tcrv_rvv.selected_vector_suffix = "i32m2"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CONFIG --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/value = "runtime-element-count"/s//value = "descriptor-element-count"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-AVL --implicit-check-not="#include <riscv_vector.h>"

#map = affine_map<(d0) -> (d0)>

module {
  // LOWER-LABEL: tcrv.exec.target @frontend_rvv_scalar_profile
  // PIPE-LABEL: tcrv.exec.target @frontend_rvv_scalar_profile
  tcrv.exec.target @frontend_rvv_scalar_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march", "scalar.fallback"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_i32_vadd",
        tcrv_frontend_target = @frontend_rvv_scalar_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %sum = arith.addi %a, %b : i32
      linalg.yield %sum : i32
    }
    return
  }
}

// LOWER-LABEL: tcrv.exec.kernel @frontend_i32_vadd
// LOWER-SAME: target = @frontend_rvv_scalar_profile
// LOWER-SAME: tcrv_frontend_lowering = "i32-vadd"
// LOWER: tcrv.exec.mem_window @abi_lhs_input_buffer
// LOWER-SAME: abi_role = "lhs-input-buffer"
// LOWER-SAME: access = "read"
// LOWER-SAME: c_type = "const int32_t *"
// LOWER-SAME: ownership = "target-export-abi-owned"
// LOWER-SAME: purpose = "runtime-abi-buffer"
// LOWER: tcrv.exec.mem_window @abi_rhs_input_buffer
// LOWER-SAME: abi_role = "rhs-input-buffer"
// LOWER-SAME: access = "read"
// LOWER-SAME: c_type = "const int32_t *"
// LOWER: tcrv.exec.mem_window @abi_output_buffer
// LOWER-SAME: abi_role = "output-buffer"
// LOWER-SAME: access = "write"
// LOWER-SAME: c_type = "int32_t *"
// LOWER: tcrv.exec.runtime_param @abi_runtime_element_count
// LOWER-SAME: abi_role = "runtime-element-count"
// LOWER-SAME: c_name = "n"
// LOWER-SAME: c_type = "size_t"
// LOWER-SAME: ownership = "target-export-abi-owned"
// LOWER-SAME: purpose = "runtime-abi-scalar"

// PIPE-LABEL: tcrv.exec.kernel @frontend_i32_vadd
// PIPE-SAME: target = @frontend_rvv_scalar_profile
// PIPE-SAME: tcrv_frontend_lowering = "i32-vadd"
// PIPE: tcrv.exec.mem_window @abi_lhs_input_buffer
// PIPE: tcrv.exec.runtime_param @abi_runtime_element_count
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: requires = [@frontend_rvv_scalar_profile]
// PIPE-SAME: tcrv_rvv.base_i32_m1_lanes = 4 : i64
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-NOT: tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"
// PIPE-SAME: tcrv_rvv.required_march = "rv64gcv"
// PIPE-SAME: tcrv_rvv.selected_binary_dtype = "i32"
// PIPE-SAME: tcrv_rvv.selected_binary_family = "i32-vadd"
// PIPE-SAME: tcrv_rvv.selected_binary_operator = "add"
// PIPE-SAME: tcrv_rvv.selected_binary_source_kind = "frontend-lowering"
// PIPE-SAME: tcrv_rvv.selected_setvl_suffix = "e32m1"
// PIPE-SAME: tcrv_rvv.selected_vector_lmul = "m1"
// PIPE-SAME: tcrv_rvv.selected_vector_shape = "i32m1"
// PIPE-SAME: tcrv_rvv.selected_vector_suffix = "i32m1"
// PIPE-SAME: tcrv_rvv.selected_vector_type = "vint32m1_t"
// PIPE-SAME: tcrv_rvv.vlenb_bytes = 16 : i64
// PIPE: tcrv.exec.variant @scalar_fallback_first_slice
// PIPE-SAME: fallback_role = "conservative"
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-SAME: requires = [@frontend_rvv_scalar_profile]
// PIPE-NOT: tcrv_scalar.lowering_descriptor
// PIPE-NOT: tcrv_scalar.element_count
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: reason = "variant-selected"
// PIPE-SAME: selection_kind = "static-variant"
// PIPE-SAME: target = @rvv_first_slice
// PIPE: tcrv_rvv.lowering_boundary
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: required_capabilities = [@frontend_rvv_scalar_profile]
// PIPE-SAME: role = "direct variant"
// PIPE-SAME: selected_binary_dtype = "i32"
// PIPE-SAME: selected_binary_family = "i32-vadd"
// PIPE-SAME: selected_binary_microkernel_op = "tcrv_rvv.i32_vadd_microkernel"
// PIPE-SAME: selected_binary_operator = "add"
// PIPE-SAME: selected_binary_source_kind = "frontend-lowering"
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_lmul = "m1"
// PIPE-SAME: selected_vector_shape = "i32m1"
// PIPE-SAME: selected_vector_type = "vint32m1_t"
// PIPE-SAME: source_kernel = "frontend_i32_vadd"
// PIPE: tcrv_rvv.i32_vadd_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface"
// PIPE-SAME: emitc_source_op = "tcrv_rvv.i32_add"
// PIPE-SAME: required_capabilities = [@frontend_rvv_scalar_profile]
// PIPE-SAME: role = "direct variant"
// PIPE-SAME: selected_binary_dtype = "i32"
// PIPE-SAME: selected_binary_family = "i32-vadd"
// PIPE-SAME: selected_binary_microkernel_op = "tcrv_rvv.i32_vadd_microkernel"
// PIPE-SAME: selected_binary_operator = "add"
// PIPE-SAME: selected_binary_source_kind = "frontend-lowering"
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_lmul = "m1"
// PIPE-SAME: selected_vector_shape = "i32m1"
// PIPE-SAME: selected_vector_suffix = "i32m1"
// PIPE-SAME: selected_vector_type = "vint32m1_t"
// PIPE-SAME: source_kernel = "frontend_i32_vadd"
// PIPE: tcrv_rvv.setvl
// PIPE-SAME: lmul = "m1"
// PIPE-SAME: sew = 32 : i64
// PIPE: tcrv_rvv.with_vl
// PIPE-SAME: lmul = "m1"
// PIPE: tcrv_rvv.i32_load
// PIPE-SAME: buffer_role = "lhs-input-buffer"
// PIPE: tcrv_rvv.i32_load
// PIPE-SAME: buffer_role = "rhs-input-buffer"
// PIPE: tcrv_rvv.i32_add
// PIPE: tcrv_rvv.i32_store
// PIPE-SAME: buffer_role = "output-buffer"
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
// PIPE-SAME: reason = "emission_plan"
// PIPE-SAME: runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1"
// PIPE-SAME: selected_plan_metadata = [{name = "tcrv_rvv.selected_vector_shape"
// PIPE-SAME: value = "i32m1"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_avl_source"
// PIPE-SAME: value = "runtime-element-count-abi-parameter"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_avl_role"
// PIPE-SAME: value = "runtime-element-count"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_family"
// PIPE-SAME: value = "i32-vadd"}
// PIPE-SAME: {name = "tcrv_rvv.emitc_source_op"
// PIPE-SAME: value = "tcrv_rvv.i32_add"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_element_count_c_name"
// PIPE-SAME: value = "n"}
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @rvv_first_slice

// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vadd_microkernel */
// SOURCE: /* arithmetic_family: i32-vadd */
// SOURCE: /* selected_binary_config: dtype=i32, family=i32-vadd
// SOURCE-SAME: runtime_element_count_c_name=n
// SOURCE-SAME: selected_role=direct variant */
// SOURCE: /* selected_runtime_vl_boundary: runtime_element_count_c_name=n
// SOURCE-SAME: runtime_avl_source=runtime-element-count-abi-parameter
// SOURCE-SAME: runtime_vl_source=tcrv_rvv.setvl
// SOURCE-SAME: runtime_vl_scope=tcrv_rvv.with_vl
// SOURCE: /* control_plane_runtime_avl: body index argument maps to target/export-owned runtime n ABI parameter */
// SOURCE: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_add -> tcrv_rvv.i32_store */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_route: tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++ */
// SOURCE: /* emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// SOURCE: /* emitc_materialization_boundary: verified MLIR EmitC module with emitc.include, emitc.func, emitc.if, emitc.call_opaque, and emitc.call before MLIR Cpp emitter production source output */
// SOURCE: /* emitc_c_source_authority: MLIR EmitC module translated by mlir::emitc::translateToCpp */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: /* emitc_route_source_ops: tcrv_rvv.setvl tcrv_rvv.with_vl tcrv_rvv.i32_load tcrv_rvv.i32_load tcrv_rvv.i32_add tcrv_rvv.i32_store */
// SOURCE: /* emitc.call_opaque[3]: __riscv_vadd_vv_i32m1 from tcrv_rvv.i32_add */
// SOURCE: /* emitc.call_opaque_boundary[3]: source_role=compute, operands=3, result=sum_vec:vint32m1_t, op_interface=TCRVEmitCLowerableOpInterface */
// SOURCE: /* selected_vector_shape_config: shape=i32m1, sew=32, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i32_m1.sew32 rvv.i32_m1.lmul_m1 rvv.i32_m1.tail_policy.agnostic rvv.i32_m1.mask_policy.agnostic */
// SOURCE: /* selected_config_emission_authority: vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1, setvl_intrinsic=__riscv_vsetvl_e32m1, load_intrinsic=__riscv_vle32_v_i32m1, arithmetic_intrinsic=__riscv_vadd_vv_i32m1, store_intrinsic=__riscv_vse32_v_i32m1, tail_policy=agnostic, mask_policy=agnostic, source=RVVBinarySelectedConfigContract */
// SOURCE: /* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
// SOURCE: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// SOURCE: /* runtime_callable_abi: void tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n) */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice__tcrv_emitc_body
// SOURCE: __riscv_vsetvl_e32m1
// SOURCE: __riscv_vle32_v_i32m1
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: __riscv_vse32_v_i32m1
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice

// HEADER: /* selected_body_authority: tcrv_rvv.i32_vadd_microkernel */
// HEADER: /* selected_binary_config: dtype=i32, family=i32-vadd
// HEADER-SAME: runtime_element_count_c_name=n
// HEADER-SAME: selected_role=direct variant */
// HEADER: /* selected_runtime_vl_boundary: runtime_element_count_c_name=n
// HEADER-SAME: runtime_avl_source=runtime-element-count-abi-parameter
// HEADER-SAME: runtime_vl_source=tcrv_rvv.setvl
// HEADER-SAME: runtime_vl_scope=tcrv_rvv.with_vl
// HEADER: /* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
// HEADER: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// HEADER: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// HEADER: /* runtime_callable_abi: void tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n) */
// HEADER: #ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL_FRONTEND_I32_VADD_RVV_FIRST_SLICE_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: void tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #endif /* TIANCHENRV_RVV_I32_VADD_MICROKERNEL_FRONTEND_I32_VADD_RVV_FIRST_SLICE_H */

// OBJECT: Format: elf64-littleriscv
// OBJECT: Type: Relocatable
// OBJECT: Machine: EM_RISCV
// OBJECT: Name: tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice

// MISSING-BODY: selected RVV path @rvv_first_slice as direct variant requires exactly one matching RVV i32 microkernel

// MISSING-BOUNDARY-SOURCE: tcrv_rvv.lowering_boundary for @rvv_first_slice requires selected RVV binary source identity before target artifact export

// MISSING-OP-SOURCE: tcrv_rvv.i32_vadd_microkernel requires op-owned selected RVV binary source identity for frontend-lowering before target artifact export

// STALE-OP-SOURCE: selected RVV binary source identity attribute 'selected_binary_family' value 'i32-vsub' must match family 'i32-vadd'

// STALE-CONFIG: selected RVV variant @rvv_first_slice
// STALE-CONFIG-SAME: selected vector-shape vector suffix must be 'i32m1'

// STALE-AVL: route id 'tcrv-export-rvv-microkernel-c'
// STALE-AVL-SAME: selected_plan_metadata 'tcrv_rvv.runtime_avl_role' must use value 'runtime-element-count'
