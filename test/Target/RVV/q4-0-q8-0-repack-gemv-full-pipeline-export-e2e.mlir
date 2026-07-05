// The q4_0 16x1-REPACKED GEVM front-door CONSTRUCTION + CORE emit closure (the
// M-FLAT REPACK flip: gemm_tile/q4_0/rvv dispatch-wired -> constructed). Unlike the
// flat / super-block block-dot routes (which are SOURCE-FUNC front doors flowing
// through the shared monolithic-block-dot emission-plan route), the repacked GEVM
// is the option-2 quant_contraction BRIDGE route: --tcrv-rvv-lower-quant-contraction
// CONSTRUCTS the typed tcrv_rvv.typed_repack_gemv_loop_body REGION (the monolithic
// tcrv_rvv.repack_gemv_q4_0_q8_0 op is retired) out of the two decomposed inner
// bricks -- the per-block lane-wise integer CORE (tcrv_rvv.repack_lane_wise_q4_x_i8_dot,
// numHalves per-strip sumi) + the per-strip dual-fp16 scale FOLDs
// (tcrv_rvv.repack_dual_fp16_scale_fold) -- around a per-strip LANE-WISE f32 VECTOR
// accumulator, then --tcrv-rvv-lower-to-emitc lowers it to the byte-exact repacked
// GEVM kernel. This is the input the E5 strong-route auto-readout
// (.trellis/scripts/e5_strong_readout.py) walks to derive constructed (STRONG):
// the realized with_vl body is the decomposed region (a real dot-product +
// fused-reduce primitive, NO opaque *_block_dot hand helper). NO board / NO perf
// claim -- coverage/representation maturity; numbers pending-hardware (ssh rvv).

// CONSTRUCTION: the front door (VLEN128 => Zvl128b => repack SELECTED) constructs the
// region + its decomposed bricks from the REAL quant_contraction request (NOT test-
// authored).
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT

// CORE emit: the constructed region lowers to the mf2 / half_lanes=8 two-8-lane-
// halves repacked GEVM kernel (byte-exact to the pre-retirement monolith, proven at
// Phase B).
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_0_q8_0_repack_gemv attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0_repack_gemv, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// The abstract quant_contraction op is GONE; the repack-SELECTED cell is CONSTRUCTED
// as the typed region + its decomposed bricks, carrying the block_q4_0x16 x16 facts
// + the DECLARED OUTPUT CONTRACT. NO monolithic repack op, NO block-dot op.
// CONSTRUCT-NOT: tcrv_rvv.quant_contraction
// CONSTRUCT-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0
// CONSTRUCT-NOT: tcrv_rvv.q4_0_q8_0_block_dot
// CONSTRUCT: tcrv_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: half_lanes = 8 : i64
// CONSTRUCT-SAME: tcrv_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: tcrv_rvv.path_materialization = "realized"
// CONSTRUCT-SAME: tcrv_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 288 : i64
// CONSTRUCT: tcrv_rvv.repack_lane_wise_q4_x_i8_dot
// CONSTRUCT: tcrv_rvv.repack_dual_fp16_scale_fold
// CONSTRUCT: tcrv_rvv.repack_dual_fp16_scale_fold
// CONSTRUCT: tcrv_rvv.typed_repack_gemv_loop_yield

// The CORE EmitC lowering: the two 8-lane f32m2 accumulators, the two disjoint
// repacked i8mf2 sub-loads, the lane-wise vwmacc accumulate (NO cross-lane
// reduction wall), and the two per-strip vector stores.
// EMIT-NOT: unrealized_conversion_cast
// EMIT: emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel_ggml_vec_dot_q4_0_q8_0_repack_gemv(
// EMIT: literal "288"
// EMIT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: call_opaque "__riscv_vfmv_v_f_f32m2"
// EMIT: call_opaque "__riscv_vle8_v_i8mf2"
// EMIT: call_opaque "__riscv_vle8_v_i8mf2"
// EMIT: call_opaque "__riscv_vwmacc_vx_i16m1"
// EMIT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT: call_opaque "__riscv_vse32_v_f32m2"
// EMIT: return
