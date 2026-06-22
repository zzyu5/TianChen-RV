// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector | FileCheck %s --check-prefix=STAMP
// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOFRAC
// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=ONESTRIP

// The SECOND repack divergence axis: ISA GENERATION. The pre-ratification
// RVV0.7.1 generation (XuanTie xtheadvector on the C920) has NO fractional LMUL,
// so the repack's default fractional core (i8mf2) is REJECTED by the XuanTie
// toolchain. --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector
// derives the RVV-generation fact (deriveRVVVersion == RVV0p7) through the SAME
// plugin-local capability authority and stamps integer_core_lmul="m1" AND
// half_lanes=16. The emitter then lowers the WHOLE-LMUL chain (i8m1 -> i16m2 ->
// i32m4 -> f32m4, scale f16m2) as ONE 16-lane strip -- byte-identical math to
// the VLEN=256 fractional one-strip form (the 16-way interleaved repack reads
// the same bytes; the i8m1 strip is 16 i8 lanes at VLEN=128), but using the
// whole LMUL the ISA supports. NO mf2 appears anywhere in the emission.
//
// The companion test rvv-to-emitc-repack-gemv-q4-0-q8-0-vlen256.mlir pins the
// RVV1.0 one-strip (mf2,16) arm; rvv-to-emitc-repack-gemv-q4-0-q8-0.mlir pins
// the RVV1.0 two-halves (mf2,8) arm; the verifier accept(m1,16)/reject(m1,8)
// cases live in the Dialect repack-gemv dataflow test.

module {
  tcrv.exec.kernel @ggml_repack_gemv_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q4_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q4_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// The stamp pins BOTH the whole-LMUL core anchor AND its mandatory one-strip
// width: integer_core_lmul="m1" overrides the absent attribute, half_lanes=16
// overrides the authored 8 (the i8m1 strip is 16 i8 lanes at VLEN=128).
// STAMP: tcrv_rvv.repack_gemv_q4_0_q8_0
// STAMP-SAME: half_lanes = 16 : i64
// STAMP-SAME: integer_core_lmul = "m1"

// CHECK-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(
// The active vl is the 16-lane whole-LMUL strip width, NOT 8.
// CHECK: literal "16"
// The OUTER weight-column-GROUP loop over nc/16, per-group weight base (stride 288).
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "288"
// ONE 16-lane f32m4 accumulator (rows 0..15): a SINGLE vfmv_v_f_f32m4(0.0f, 16).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"
// The inner contraction-block loop over nb; plain q8_0 activation stride 34.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "34"
// ONE 16-lane i16m2 lo + ONE 16-lane i16m2 hi accumulator seed (NOT four 8-lane).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m2"
// The nibble-step loop; ONE disjoint repacked i8m1 sub-load (rows 0..15 at
// qs[i*16+0]), then the plain sign-extension decode (NO vxor):
// b_lo=vsra(vsll(b,4),4); b_hi=vsra(b,4) -- all at whole-LMUL i8m1.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vsll_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// TWO lane-wise vwmacc i16m2 (low qs[i], high qs[16+i]) for the single 16-lane strip.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// ONE lo/hi combine vwadd_vv_i32m4.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m4"
// ONE 16-lane f16m2 weight-scale vle16 (b[l].d[0]) + the single raw _Float16 act scale.
// CHECK: call_opaque "__riscv_vle16_v_f16m2"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m4"
// ONE 16-lane f32m4 vector store vse32 (s + x*16 + 0; rows 0..15 in one strip).
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: return

// RVV0.7.1 has NO fractional LMUL -- the XuanTie toolchain rejects any mf2/mf4/
// mf8 type. The whole emission must be fraction-free: not one mf2 spelling.
// NOFRAC-NOT: mf2

// Divergence-by-construction is a COUNT, not a presence: the 1x16 whole-LMUL
// strip emits EXACTLY ONE of each reduction the 2x8 path emits twice. A leaked
// second strip (a stray 2x8 fallback) would add a second store / accumulator /
// scale-load. The COUNT-1/NOT pairs are ordered by emission position.
// ONESTRIP-COUNT-1: call_opaque "__riscv_vfmv_v_f_f32m4"
// ONESTRIP-NOT: call_opaque "__riscv_vfmv_v_f_f32m4"
// ONESTRIP-COUNT-1: call_opaque "__riscv_vle16_v_f16m2"
// ONESTRIP-NOT: call_opaque "__riscv_vle16_v_f16m2"
// ONESTRIP-COUNT-1: call_opaque "__riscv_vse32_v_f32m4"
// ONESTRIP-NOT: call_opaque "__riscv_vse32_v_f32m4"
