// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml q4_0 16x1-REPACKED GEMM hot kernel (the validated vlen128-q4_0-16x1
// ggml_gemm_q4_0_16x1_q8_0 path) as STRUCTURED emitc IR (I5; ZERO raw() strings).
// The single typed op tcrv_rvv.repack_gemm_q4_0_q8_0 lowers to the BLOCK-AS-LANE
// matmul: the 16 interleaved weight rows of a group occupy 16 vector lanes, the
// dot accumulates LANE-WISE via vwmacc (NO cross-lane vredsum wall), and at
// VLEN=128 each 16-block group is processed as TWO disjoint contiguous 8-lane
// halves (rows 0..7 then rows 8..15). The repacked nibbles already carry the
// offset-binary ^0x88 bias, so the decode is a plain vsll/vsra sign-extension
// (NO in-kernel vxor). Structural fidelity to the patch's node sequence is the
// byte-exactness proof: the patch is already byte-exact vs _generic (artifact
// under .trellis/tasks/06-18-.../artifacts/e2e-repack-gemm/).
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob.

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
        %g = tcrv_rvv.repack_gemm_q4_0_q8_0 %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q4_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemm_q4_0_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
// The block count nb = n / 32.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The row-group count nr/4 (%arg4 = nr) and column-group count nc/16 (%arg5 = nc).
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop over nr/4.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// Per-group activation base vy + y*nb*136 (block_q8_0x4 stride 136).
// CHECK: mul %[[Y]], %{{.*}}
// CHECK: literal "136"
// The weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*288 (block_q4_0x16 stride 288).
// CHECK: literal "288"
// The two-halves loop over the 8-lane halves (rows 0..7, 8..15).
// CHECK: for %[[H:.*]] = %{{.*}} to %{{.*}} step
// The 4x8 f32 accumulator set: four vfmv_v_f_f32m2(0.0f, 8) seeds.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The 4x{lo,hi} i16 lane accumulator seeds vmv_v_x_i16m1(0, 8).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The nibble-step loop over the 16 weight bytes.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// The contiguous repacked sub-load + the plain sign-extension decode (NO vxor):
// b_lo = vsra(vsll(b,4),4); b_hi = vsra(b,4).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// The lane-wise vwmacc accumulate against the scalar q8 quants (NO cross-lane
// vredsum) -- the low-half then high-half products per activation column.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// The lo/hi combine vwadd_vv_i32m2.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// The vector scale fold: vle16 the 8 weight scales, vfwmul by the raw _Float16
// activation scale, vfcvt the i32 accumulator, vfmacc into the f32 accumulator.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The 4x8 vector store vse32_v_f32m2 (NO per-block scalar *s store).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
