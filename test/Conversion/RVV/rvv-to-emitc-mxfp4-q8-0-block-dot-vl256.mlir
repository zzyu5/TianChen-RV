// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The VLEN256 (`_vl256`) shape of the ggml MXFP4 x Q8_0 codebook block dot-product: the
// SAME structured kernel as the VLEN128 form (test rvv-to-emitc-mxfp4-q8-0-block-dot.mlir)
// -- including the unchanged structured E8M0 -> fp32 HALF scale reconstruction -- but the
// i8 integer-core anchor moves m1 -> mf2 driven by the VLEN CAPABILITY fact
// (integer_core_lmul = "mf2", minimum_vlen = 256). At VLEN256 mf2's VLMAX = 16 = a FULL
// mf2 register (the 16-lane nibble-pair half-block exactly fills it; at VLEN128 mf2 ->
// VLMAX 8 < 16 breaks the codebook gather, which the verifier gates on minimum_vlen >=
// 256), and multi_block_factor = 2 keeps two independent blocks in flight. NO memory
// layout / algorithm change vs the m1 form -- only the LMUL spelling + unroll move (pure
// backend Win-A lowering); the vwredsum reduction destination + seed STAY m1.
//
// NON-NULL divergence proof: the mf2 / vwmul i16m1 / vwredsum i16m1 bytes here DIFFER
// from the VLEN128 m1 / vwmul i16m2 / vwredsum i16m2 emit.

module {
  tcrv.exec.kernel @ggml_vec_dot_mxfp4_q8_0_kernel_vl256 {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_mxfp4_q8_0_kernel_vl256", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>, integer_core_lmul = "mf2", multi_block_factor = 2 : i64, strip_elision = "elided", minimum_vlen = 256 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_vl256_ggml_vec_dot_mxfp4_q8_0(
// The codebook table broadcast-loaded at the mf2 anchor (16 entries fill a full mf2
// register at VLEN256).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// The m1-form strip spellings must be ABSENT.
// CHECK-NOT: call_opaque "__riscv_vsetvl_e8m1"
// CHECK-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// The structured E8M0 -> fp32 HALF scale reconstruction is UNCHANGED (the FP4-class
// scale decode is scalar, outside the vector core, identical to the m1 form).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The mf2 half-block setvl + the packed weight u8 load + the two q8 signed halves.
// CHECK: call_opaque "__riscv_vsetvl_e8mf2"
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// The codebook nibble split + gather at mf2.
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// CHECK: call_opaque "__riscv_vrgather_vv_i8mf2"
// CHECK: call_opaque "__riscv_vrgather_vv_i8mf2"
// The widened i16 product one step wider than mf2: i16m1 (VLEN128 form emits i16m2).
// CHECK: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m1"
// The reduction destination + seed STAY m1: vwredsum_vs_i16m1_i32m1, NOT _i16m2_.
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m1_i32m1"
// CHECK-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The *s store.
// CHECK: subscript
// CHECK: assign
