// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml q4_0 16x1-REPACKED GEMV (decode) hot kernel (the validated
// vlen128-q4_0-16x1 ggml_gemv_q4_0_16x1_q8_0 path) as STRUCTURED emitc IR (I5;
// ZERO raw() strings). The single typed op tcrv_rvv.repack_gemv_q4_0_q8_0 lowers
// to the BLOCK-AS-LANE single-output-column matmul: the 16 interleaved weight
// rows of a group occupy 16 vector lanes, the dot accumulates LANE-WISE via
// vwmacc (NO cross-lane vredsum wall), and at VLEN=128 each 16-block group is
// processed as TWO disjoint contiguous 8-lane halves (rows 0..7 then rows 8..15)
// inline (NO half loop, NO nr/4 row loop). The activation is ONE plain block_q8_0
// stream (stride 34, quants at +2) -- a single activation column. The repacked
// nibbles already carry the offset-binary ^0x88 bias, so the decode is a plain
// vsll/vsra sign-extension (NO in-kernel vxor). Structural fidelity to the
// patch's node sequence is the byte-exactness proof: the patch is already
// byte-exact vs _generic (artifact under
// .trellis/tasks/06-18-.../artifacts/e2e-repack-gemm/).
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob.

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

// CHECK-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(
// The block count nb = n / 32.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*288 (block_q4_0x16 stride 288).
// CHECK: mul %[[X]], %{{.*}}
// CHECK: literal "288"
// The two 8-lane f32 accumulators (rows 0..7, 8..15): two vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-block plain q8_0 activation base al = a + l*34 (stride 34).
// CHECK: literal "34"
// The 4x i16 lane accumulator seeds vmv_v_x_i16m1(0, 8) (a_lo,a_hi,b_lo,b_hi).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The nibble-step loop over the 16 weight bytes.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// Two disjoint repacked sub-loads (rows 0..7 at qs[i*16+0], rows 8..15 at
// qs[i*16+8]), then the plain sign-extension decode of each (NO vxor): b_lo =
// vsra(vsll(b,4),4); b_hi = vsra(b,4).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// The lane-wise vwmacc accumulate against the SINGLE-column scalar q8 quants
// (NO cross-lane vredsum) -- low qs[i] and high qs[16+i], for both 8-lane halves
// (4 vwmacc per nibble step).
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// The lo/hi combine vwadd_vv_i32m2 for both halves.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// The vector scale fold: vle16 the two 8-lane weight scale halves (b[l].d[0]
// and b[l].d[8]), read the SINGLE raw _Float16 activation scale once, then per
// half vfwmul / vfcvt the i32 accumulator / vfmacc into the f32 accumulator.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The single-column two-half vector store vse32_v_f32m2 (s+x*16+0, s+x*16+8;
// NO bs row stride, NO per-block scalar *s store).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: the
// dot accumulates LANE-WISE via vwmacc, so NO vredsum / vwredsum (any SEW or
// signedness) appears anywhere in the emitted GEMV.
// NOWALL-NOT: redsum
