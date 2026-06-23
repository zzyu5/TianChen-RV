// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-materialize-repack-strip-width=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=ONESTRIP

// The N3-reframed PRIZE seam on the q8_0 repack hot path: a compiler-automatic
// SELECTION. The SAME typed tcrv_rvv.repack_gemv_q8_0_q8_0 op (authored
// half_lanes=8) is re-stamped to half_lanes=16 by
// --tcrv-rvv-materialize-repack-strip-width=march=rv64gcv_zvl256b, deriving the
// guaranteed minimum VLEN (256) -> e16m1 strip width 16 through the SAME
// plugin-local capability authority (deriveMinimumVLEN). The emitter then tiles
// the 16-block-as-lane group into ONE 16-lane strip instead of two disjoint
// 8-lane halves -- divergence by construction from a single capability fact.
//
// This is BYTE-SAFE only because the repack is 16-way interleaved (block_q8_0x16:
// 512 int8 qs[] bytes = 16 blocks-as-lanes, byte i = block(i%16) offset(i/16)),
// so the one 16-lane strip at VLEN=256 reads byte-identical repacked data to the
// two 8-lane halves at VLEN=128. The companion test
// rvv-to-emitc-repack-gemv-q8-0-q8-0.mlir pins the VLEN=128 (two-halves) arm.

module {
  tcrv.exec.kernel @ggml_repack_gemv_q8_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q8_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q8_0_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q8_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q8_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 544 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_q8_0_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel_ggml_repack_gemv_q8_0_q8_0(
// The active vl is the 16-lane e16m1 strip width (VLEN=256), NOT 8.
// CHECK: literal "16"
// The OUTER weight-column-GROUP loop over nc/16, per-group weight base (stride 544).
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "544"
// ONE 16-lane f32 accumulator (rows 0..15): a SINGLE vfmv_v_f_f32m2(0.0f, 16).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb; plain q8_0 activation stride 34.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "34"
// ONE 16-lane i32 accumulator seed (NOT two 8-lane), vmv_v_x_i32m2.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The 32-position contraction loop; ONE disjoint int8 sub-load (rows 0..15 at
// qs[i*16+0]) read DIRECTLY (NO vsll/vsra nibble decode).
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// ONE lane-wise vwmul (i8 x i8 -> i16) then ONE i32 vwadd_wv accumulate for the
// single 16-lane strip.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// ONE 16-lane weight-scale vle16 (b[l].d[0]) + the single raw _Float16 act scale.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// ONE 16-lane vector store vse32 (s + x*16 + 0; rows 0..15 in one strip).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Divergence-by-construction is a COUNT, not a presence: the 1x16 strip emits
// EXACTLY ONE of each reduction the 2x8 path emits twice. A leaked second strip
// (a stray 2x8 fallback) would add a second store / accumulator / scale-load.
// The COUNT-1/NOT pairs are ordered by emission position (accumulator seed first,
// then the weight-scale load, then the output store) so each NOT scans forward
// over the remaining input.
// ONESTRIP-COUNT-1: call_opaque "__riscv_vfmv_v_f_f32m2"
// ONESTRIP-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// ONESTRIP-COUNT-1: call_opaque "__riscv_vle16_v_f16m1"
// ONESTRIP-NOT: call_opaque "__riscv_vle16_v_f16m1"
// ONESTRIP-COUNT-1: call_opaque "__riscv_vse32_v_f32m2"
// ONESTRIP-NOT: call_opaque "__riscv_vse32_v_f32m2"
