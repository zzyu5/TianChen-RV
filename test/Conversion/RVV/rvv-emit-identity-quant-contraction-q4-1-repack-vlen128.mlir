// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MIN
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=UNSIGNED

// G3-lode-flat 曳光弹 -- q4_1 gemm_tile (decode) FRONT-DOOR construction on the
// SUPPORTED RVV1.0 regime (march=rv64gcv => Zvl128b => VLEN128). The abstract,
// algorithm-UNCOMMITTED tcrv_rvv.quant_contraction op (q4_1 / decode) is
// AUTO-LOWERED by --tcrv-rvv-lower-quant-contraction: the in-compiler selection
// picks REPACK (deriveMinimumVLEN(rv64gcv)=128) and the C1 bridge CONSTRUCTS the
// typed tcrv_rvv.typed_repack_gemv_loop_body region carrying the SHARED q4_0
// decomposed bricks -- the per-block lane-wise integer CORE
// (repack_lane_wise_q4_x_i8_dot, stamping weight_nibble_unsigned = the q4_1 RAW
// nibble [0,15] decode, NO offset-binary -8) + the two per-strip dual-fp16 scale
// FOLDs (repack_dual_fp16_scale_fold, stamping the single MIN-fold offset pair
// weight_min_byte_offset=32 / activation_sum_byte_offset=2). --tcrv-rvv-lower-to-
// emitc then lowers it to the mf2/half_lanes=8 two-8-lane-halves repack-GEVM kernel.
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the q4_1 repack region
// through the SAME front door as q4_0 (previously q4_1 was direct-emitter
// dispatch-wired). NO perf/e2e claim -- the kernel is lit-emitted, NOT run.

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = tcrv_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_1_q8_1, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q4_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 20 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected q4_1 repack-GEVM is lowered through
// the SAME front door as q4_0.
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemv_q4_1_q8_1 %
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(
// The per-group weight base vx + x*nb*320 (block_q4_1x16 stride 320 = 16 d + 16 m
// + 256 nibbles).
// CHECK: literal "320"
// The two 8-lane f32m2 accumulators (rows 0..7, 8..15) -- the mf2/half_lanes=8 form.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The unsigned RAW-nibble load + lane-wise vwmacc accumulate.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The q4_1 UNSIGNED-nibble decode: vand(0x0F)/vsrl(0x04)/vreinterpret peel, NO
// offset-binary vsll/vsra sign-extension.
// UNSIGNED: call_opaque "__riscv_vand_vx_u8mf2"
// UNSIGNED: call_opaque "__riscv_vsrl_vx_u8mf2"
// UNSIGNED: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// UNSIGNED-NOT: __riscv_vsll_vx_i8mf2

// The q4_1 single MIN-fold: the SECOND vfwmul (m_x * s_y) folded by vfadd AFTER the
// dual-fp16 scale vfmacc.
// MIN: call_opaque "__riscv_vfmacc_vv_f32m2"
// MIN: call_opaque "__riscv_vfwmul_vf_f32m2"
// MIN: call_opaque "__riscv_vfadd_vv_f32m2"
