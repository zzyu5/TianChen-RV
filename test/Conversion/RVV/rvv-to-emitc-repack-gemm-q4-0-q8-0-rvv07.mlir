// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector | FileCheck %s --check-prefix=STAMP
// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOFRAC
// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=PASS4

// The GEMM arm of the SECOND repack divergence axis (ISA GENERATION). RVV0.7.1
// (XuanTie xtheadvector) has NO fractional LMUL, so the GEMM repack's default
// fractional core (i8mf2) is rejected. The stamp pins integer_core_lmul="m1" AND
// half_lanes=16; the emitter lowers the WHOLE-LMUL chain (i8m1 -> i16m2 -> i32m4
// -> f32m4, scale f16m2) with the strip loop running ONCE (numHalves=1, one
// 16-lane strip). Byte-identical math to the RVV1.0 form (the 16-way interleaved
// repack reads the same bytes), using the whole LMUL the ISA supports. NO mf2.

module {
  tcrv.exec.kernel @ggml_repack_gemm_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemm_q4_0_q8_0 %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q4_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// The stamp pins BOTH the whole-LMUL core anchor AND its mandatory one-strip
// width: integer_core_lmul="m1" overrides the absent attribute, half_lanes=16
// overrides the authored 8.
// STAMP: tcrv_rvv.repack_gemm_q4_0_q8_0
// STAMP-SAME: half_lanes = 16 : i64
// STAMP-SAME: integer_core_lmul = "m1"

// CHECK-NOT: tcrv_rvv.repack_gemm_q4_0_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
// The active vl is the 16-lane whole-LMUL strip width, NOT 8.
// CHECK: literal "16"
// The whole-LMUL i8m1 sub-load + plain sign-extension decode (NO vxor).
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vsll_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// The 4x16 lane-wise vwmacc i16m2 product (the GEMM processes 4 activation columns).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// The lo/hi combine vwadd_vv_i32m4, the f16m2 weight-scale vle16, and the
// whole-LMUL f32m4 scale fold + store.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m4"
// CHECK: call_opaque "__riscv_vle16_v_f16m2"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: return

// RVV0.7.1 has NO fractional LMUL -- the whole emission must be fraction-free.
// NOFRAC-NOT: mf2

// The whole-LMUL chain (i8m1/i16m2/i32m4/f32m4) saturates the 32-vreg file if all
// 4 activation columns are held live at once -> GCC spilled the 4th f32m4 acc
// under the e16,m2 nibble vtype (zeroing only 8 of 16 lanes -> col-3 upper-lane
// garbage). The fix folds ONE column per pass: FOUR independent block loops, each
// re-decoding the shared weight nibbles, dropping the live set to the bit-exact
// GEMV's ~12-14 vregs -> no spill. Pin the 4-pass structure by the per-pass weight
// re-decode count (the single-pass buggy form re-decoded ONCE). The store count is
// NOT a guard -- both forms emit 4 stores; only the re-decode count discriminates.
// PASS4-COUNT-4: call_opaque "__riscv_vle8_v_i8m1"
// PASS4-NOT: call_opaque "__riscv_vle8_v_i8m1"
