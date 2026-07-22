// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=QH
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MIN

// G3-lode-flat FLAT-3 -- q5_1 gemm_tile (decode) FRONT-DOOR construction on the
// SUPPORTED RVV1.0 regime (march=rv64gcv => Zvl128b => VLEN128). The abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op (q5_1 / decode) is
// AUTO-LOWERED by --weft-rvv-lower-quant-contraction: the in-compiler selection
// picks REPACK (deriveMinimumVLEN(rv64gcv)=128) and the C1 bridge CONSTRUCTS the
// typed weft_rvv.typed_repack_gemv_loop_body region carrying the SHARED q4_0
// decomposed bricks -- the per-block lane-wise integer CORE
// (repack_lane_wise_q4_x_i8_dot, stamping weight_nibble_unsigned +
// weight_qh_byte_offset=320 with NO weight_offset_bias = the q5_1 UNSIGNED 5-bit
// `(nibble) | (qh_bit<<4)` decode in [0,31]) + the two per-strip dual-fp16 scale
// FOLDs (repack_dual_fp16_scale_fold, WITH the q4_1 MIN offset pair). q5_1 =
// q5_0's qh gather (unsigned/no -16) + q4_1's min fold. --weft-rvv-lower-to-emitc
// then lowers it to the mf2/half_lanes=8 two-8-lane-halves repack-GEVM kernel.
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the q5_1 repack region
// through the SAME front door as q4_0/q4_1/q5_0 (previously q5_1 was direct-emitter
// dispatch-wired, GEVM-only). NO perf/e2e claim -- the kernel is lit-emitted, NOT run.

module {
  weft.exec.kernel @ggml_vec_dot_q5_1_q8_1_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q5_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q5_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min-five-bit", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 24 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected q5_1 repack-GEVM is lowered through
// the SAME front door as q4_0/q4_1/q5_0.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_q5_1_q8_1 %
// CHECK: emitc.func @weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
// The per-group weight base vx + x*nb*384 (block_q5_1x16 stride 384 = 16 d + 16 m
// + 256 nibbles + 64 transposed qh bytes).
// CHECK: literal "384"
// The two 8-lane f32m2 accumulators (rows 0..7, 8..15) -- the mf2/half_lanes=8 form.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The unsigned RAW-nibble load + lane-wise vwmacc accumulate.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The q5_1 5th-bit (qh) assembly, REDESIGN-B native-mask form: the RAW unsigned
// nibble is extracted (vand low), the transposed qh mask bits are loaded DIRECTLY
// per strip (vlm_v_b16), and -- UNLIKE q5_0 (offset-binary vsub) -- the 5th bit is
// added with the RAW (NON-inverted) mask via ONE masked add (vadd_vx_u8mf2_mu
// by 16), reinterpreted u8->i8. The UNSIGNED [0,31] weight feeds vwmacc directly
// (the asymmetric bias lives in the MIN fold); NO offset-binary vsub.
// The OLD per-lane expand chain (vsrl_vv / vsll / vncvt / vor) is RETIRED.
// QH: call_opaque "__riscv_vand_vx_u8mf2"
// QH: call_opaque "__riscv_vlm_v_b16"
// QH: call_opaque "__riscv_vadd_vx_u8mf2_mu"
// QH: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// QH-NOT: __riscv_vsub_vx_i8mf2
// QH-NOT: __riscv_vor_vv_u8mf2
// QH-NOT: __riscv_vncvt_x_x_w_u8mf2
// QH-NOT: __riscv_vsrl_vv_u16m1

// The q5_1 fold carries the q4_1 MIN term: the dual-fp16 d scale tree
// (vfwmul(d) / vfcvt / vfmacc) PLUS the second vfwmul(m_x) + vfadd min correction.
// MIN: call_opaque "__riscv_vfwmul_vf_f32m2"
// MIN: call_opaque "__riscv_vfmacc_vv_f32m2"
// MIN: call_opaque "__riscv_vfadd_vv_f32m2"
