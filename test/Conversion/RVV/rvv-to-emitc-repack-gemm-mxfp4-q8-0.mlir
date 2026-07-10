// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=E8M0
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 retirement_batch 4 mxfp4 codebook front-door: the ggml mxfp4 x q8_0 16x1-REPACKED GEMM
// (prefill) hot kernel -- the CODEBOOK + E8M0 prefill sibling of the mxfp4 GEVM -- is now
// CONSTRUCTED through the typed-region FRONT DOOR, NOT the retired monolithic
// emitRepackGemmMxfp4Q8 direct emitter. The tcrv_rvv.typed_repack_gemm_loop_body region
// (fold_model "codebook_flat_e8m0_scale") carries the tcrv_rvv.repack_gemm_codebook_core
// integer-core BRICK (decode_model "mxfp4" + the 16-entry doubled-E2M1 codebook), block_index
// + strip_row_offset tied (anti-bypass). The lowering GATES the emit on those ties, then
// RE-EMITS the byte-exact mxfp4 GEMM body via emitTypedRepackGemmLoopBody's codebook branch ->
// emitRepackCodebookGemmBodyMxfp4 (byte-identical to the retired direct emitter; PLAIN untiled
// -- the memory-gather decode already sits at the <=32-vreg cliff, so S6 tiling is a structural
// no-op). The 4-bit nibble is an INDEX into the 16-entry codebook, decoded by the memory
// vluxei16 GATHER (NOT a register vrgather), AMORTIZED once per 16-weight group and reused
// across the 4 interleaved block_q8_0x4 activation columns. The per-column weight scale is ONE
// E8M0 shared-exponent byte reconstructed to 2^(e-128) by bit arithmetic, i32 accumulator, NO
// min. block_mxfp4x16 stride 272 (E8M0 @0, qs @16); block_q8_0x4 stride 136, qs @8.

module {
  tcrv.exec.kernel @ggml_repack_gemm_mxfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemm_mxfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "flat.e8m0-single-scale-codebook-4col-nomin", qk = 32 : i64, weight_block_stride = 272 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 16 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "codebook_flat_e8m0_scale"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">, %acc2: !tcrv_rvv.vector<f32, "m2">, %acc3: !tcrv_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied CODEBOOK GEMM integer-core BRICK:
          // per-block lane-wise mxfp4 memory-gather dot across the 4 interleaved columns ->
          // the columnsPerPass (4) per-column i32 sumi. The typed emitter re-emits the whole
          // byte-exact mxfp4 GEMM body from this brick's identity + its codebook; the yield
          // passes through the carried-in per-column accs.
          %sumi:4 = tcrv_rvv.repack_gemm_codebook_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_codebook_core", decode_model = "mxfp4", weight_quant_byte_offset = 16 : i64, activation_quant_byte_offset = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: tcrv_rvv.repack_gemm_codebook_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_mxfp4_q8_0_kernel_ggml_repack_gemm_mxfp4_q8_0(
// CHECK: verbatim "static const int8_t tcrv_mxfp4_repack_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1
// The block count nb = n/32, the row-group count nr/4 (%arg4), the column-group
// count nc/16 (%arg5).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-ROW-group loop over nr/4, the interleaved q8_0x4 base
// (a + y*nb*136 stride 136).
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "136"
// The weight-column-GROUP loop; weight base vx + x*nb*272.
// CHECK: literal "272"

// ===== The SHARED (amortized) memory codebook GATHER (vluxei16). =====
// GATHER: call_opaque "__riscv_vle8_v_u8mf2"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: literal "tcrv_mxfp4_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL E8M0 shared-exponent scale reconstruction (2^(e-128)). =====
// E8M0: call_opaque "__riscv_vzext_vf4_u32m2"
// E8M0: call_opaque "__riscv_vand_vx_u32m2"
// E8M0: call_opaque "__riscv_vsll_vv_u32m2"
// E8M0: call_opaque "__riscv_vmsltu_vx_u32m2_b16"
// E8M0: call_opaque "__riscv_vmerge_vvm_u32m2"
// E8M0: call_opaque "__riscv_vreinterpret_v_u32m2_f32m2"
// E8M0: call_opaque "__riscv_vfmul_vf_f32m2"

// The per-column fold (vfcvt the i32 sumi, vfmacc) and per-(row,col) store through the
// runtime output row stride bs.
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall (NO redsum); the flat codebook
// GEMM fold has NO min (no vfnmsac) and NO per-sub-block scale (no vwmacc_vv_i32); the E8M0
// scale is bit-reconstructed, NOT an fp16 delta (no vfwmul/vle16 weight-scale); the decode is a
// MEMORY gather, NOT a register vrgather. Also NO trailing dead result-token vmv (the region is
// result-less, unlike the retired direct emitter).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
// NOWALL-NOT: vfwmul
// NOWALL-NOT: vle16
