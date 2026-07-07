// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml tq2_0 x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the FIRST TERNARY
// (BitNet-class) block-as-lane sibling of the oracle-verified K-quant repacked GEVMs
// -- as STRUCTURED emitc IR (I5; ZERO raw() strings). The single typed op
// tcrv_rvv.repack_gemv_tq2_0_q8_K lowers to the BLOCK-AS-LANE single-output-column
// matmul: the 16 interleaved weight columns of a group occupy 16 vector lanes, the
// dot accumulates LANE-WISE via vwmacc (NO cross-lane vredsum wall), and at VLEN=128
// each 16-block group is processed as TWO disjoint 8-lane halves. The activation is
// ONE plain block_q8_K stream (stride 292, fp32 d at +0, int8 quants at +4). tq2_0 is
// LINEAR: the weight is a 2-BIT TERNARY trit, ((byte >> {0,2,4,6}) & 3) - 1 in
// {-1,0,1} (the vand 0x03 peel + the `vsub_vx_i8` by 1 ternary bias the q2_K repack
// LACKS), the whole 256-element super-block carries ONE fp16 super-block scale, and
// the dot folds LANE-WISE into ONE i32 accumulator per strip (a per-super-half i16
// partial widened via vwadd_wv) -- NO per-sub-block scale, NO dmin, NO bsums, NO min
// term. block_tq2_0x16 stride 1056, d at +0, qs at +32.

module {
  tcrv.exec.kernel @ggml_repack_gemv_tq2_0_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_tq2_0_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_tq2_0_q8_K", scale_model = "superblock-d.fp16-single-scale-2bit-ternary-nomin", qk = 256 : i64, weight_block_stride = 1056 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_tq2_0_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_tq2_0_q8_K_kernel_ggml_repack_gemv_tq2_0_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*1056 (block_tq2_0x16 stride 1056).
// CHECK: literal "1056"
// The two 8-lane f32 accumulators (cols 0..7, 8..15): vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-block plain q8_K activation base al = a + l*292 (stride 292).
// CHECK: literal "292"
// The activation super-block delta d_y is a single fp32 SCALAR (NOT fp16).
// CHECK: call_opaque "*(const float *)"
// The per-strip i32 accumulator (single, no min) seed vmv_v_x_i32m2, and the
// per-super-half i16 partial seed vmv_v_x_i16m1.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The tq2_0 2-bit TERNARY weight assembly: vle8 the qs byte strip ONCE, vand 0x03
// (the low 2-bit lane), reinterpret to SIGNED i8, then vsub_vx_i8 by 1 -- the `-1`
// TERNARY BIAS (trit in {-1,0,1}) the q2_K repack (unsigned [0,3]) LACKS.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The lane-wise integer dot against the single-column q8 quants (NO vredsum), a
// 16-bit partial vwmacc_vx, and the higher 2-bit lanes peeled by vsrl {2,4,6}.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// The per-super-half fold widens the i16 partial into the i32 accumulator: vwadd_wv.
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The end-of-block SINGLE-scale fold: vle16 the fp16 d strip, vfwcvt, vfmul by d_y,
// vfcvt the i32 sumi, then vfmacc the single d term (NO vfnmsac MIN term).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The single-column two-half vector store vse32_v_f32m2 (s+x*16+0, s+x*16+8).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall, and the
// LINEAR ternary fold has NO min term (no vfnmsac) and NO per-sub-block scale
// (no vwmacc_vv_i32).
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
