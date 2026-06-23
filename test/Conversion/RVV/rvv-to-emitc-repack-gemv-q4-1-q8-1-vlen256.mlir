// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=ONESTRIP

// The N1 capability-divergence axis applied to the FAMILY-B q4_1 repack GEMV: the
// SAME typed tcrv_rvv.repack_gemv_q4_1_q8_1 op (authored half_lanes=8) is
// re-stamped to half_lanes=16 by --tcrv-rvv-materialize-repack-strip-width
// =march=rv64gcv_zvl256b, deriving the guaranteed minimum VLEN (256) -> e16m1
// strip width 16 through the SAME plugin-local capability authority
// (deriveMinimumVLEN) the q4_0 sibling uses. The emitter then tiles the
// 16-block-as-lane group into ONE 16-lane strip instead of two disjoint 8-lane
// halves -- divergence by construction from a single capability fact, and the new
// q4_1 GEMV participates in the SAME gearbox selection as q4_0. Both arms are
// numerically PASS on real K1 VLEN=256 silicon (norm ~6e-6 vs scalar reference
// AND vs the in-tree q4_1 block-dot cross-check).

module {
  tcrv.exec.kernel @ggml_repack_gemv_q4_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q4_1_q8_1, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q4_1_q8_1 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q4_1_q8_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", qk = 32 : i64, weight_block_stride = 320 : i64, activation_block_stride = 36 : i64, weight_quant_byte_offset = 64 : i64, activation_quant_byte_offset = 4 : i64, weight_min_byte_offset = 32 : i64, activation_sum_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_q4_1_q8_1 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q4_1_q8_1_kernel_ggml_repack_gemv_q4_1_q8_1(
// The active vl is the 16-lane e16m1 strip width (VLEN=256), NOT 8.
// CHECK: literal "16"
// The OUTER weight-column-GROUP loop over nc/16, per-group weight base (stride 320).
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "320"
// ONE 16-lane f32 accumulator (rows 0..15): a SINGLE vfmv_v_f_f32m2(0.0f, 16).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb; plain q8_1 activation stride 36.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "36"
// ONE 16-lane i16 lo + ONE 16-lane i16 hi accumulator seed (NOT four 8-lane).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The nibble-step loop; ONE disjoint UNSIGNED repacked sub-load (rows 0..15),
// then the UNSIGNED nibble decode (vand 0x0F / vsrl 0x04 -> reinterpret).
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// TWO lane-wise vwmacc (low qs[i], high qs[16+i]) for the single 16-lane strip.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// ONE lo/hi combine vwadd_vv_i32m2.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// ONE 16-lane weight-delta vle16 (b[l].d[0]) + ONE 16-lane weight-MIN vle16
// (b[l].m[0]) + the single raw _Float16 act delta + act scaled-sum.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "*(const _Float16 *)"
// The scale term (vfwmul d_x*d_y / vfcvt / vfmacc) THEN the LANE-WISE MIN term
// (vfwmul m_x*s_y / vfadd), for the single 16-lane strip.
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// ONE 16-lane vector store vse32 (s + x*16 + 0; rows 0..15 in one strip).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Divergence-by-construction is a COUNT: the 1x16 strip emits EXACTLY ONE of each
// reduction the 2x8 path emits twice. A leaked second strip would add a second
// store / accumulator / delta-load.
// ONESTRIP-COUNT-1: call_opaque "__riscv_vfmv_v_f_f32m2"
// ONESTRIP-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// ONESTRIP-COUNT-1: call_opaque "__riscv_vse32_v_f32m2"
// ONESTRIP-NOT: call_opaque "__riscv_vse32_v_f32m2"
