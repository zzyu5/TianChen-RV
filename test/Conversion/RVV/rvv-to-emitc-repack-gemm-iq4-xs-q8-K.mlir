// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M2-后 iq4_xs codebook front-door: the ggml iq4_xs x q8_K 16x1-REPACKED GEMM (prefill)
// hot kernel -- the SUPER-BLOCK CODEBOOK prefill sibling of iq4_nl -- is now CONSTRUCTED
// through the typed-region FRONT DOOR, NOT the retired monolithic emitRepackGemmIq4XsQ8K
// direct emitter. The weft_rvv.typed_repack_gemm_loop_body region (fold_model
// "codebook_superblock_signed6_no_min") carries the SHARED weft_rvv.repack_gemm_codebook_core
// integer-core BRICK (decode_model "iq4_xs" + the SAME 16-entry non-linear int8 codebook
// iq4_nl uses), block_index + strip_row_offset tied (anti-bypass) and named off the
// loop-body's own weight / activation ABI bases. The codebook memory GATHER (vluxei16) +
// 6-bit SIGNED per-sub-block scale (vsub 32, vsext_vf4, i32 vmacc, NO min) are AMORTIZED
// once per 16-weight group and reused across the 4 interleaved block_q8_Kx4 activation
// columns (stride 1168, qs at +16, bsums UNREAD). The GEMM ships PLAIN/UNTILED (iq4_xs sits
// at the <=32-vreg cliff, S6 tiling a structural no-op). block_iq4_xsx16 stride 2176.

module {
  weft.exec.kernel @ggml_repack_gemm_iq4_xs_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq4_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq4_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq4_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock.fp16-signed6-scale-codebook-4col-nomin", qk = 256 : i64, weight_block_stride = 2176 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 128 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 64 : i64, weight_scales_high_byte_offset = 32 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "codebook_superblock_signed6_no_min", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied CODEBOOK GEMM integer-core BRICK (SHARED
          // with iq4_nl): per-block lane-wise iq4_xs memory-gather dot + 6-bit signed
          // sub-block-scale vmacc fold across the 4 interleaved columns -> the columnsPerPass
          // (4) per-column i32 sumi. The typed emitter re-emits the whole byte-exact iq4_xs
          // GEMM body from this brick's identity + its codebook + the loop body's
          // scales_l/scales_h/n_subblocks; the yield passes through the carried-in accs.
          %sumi:4 = weft_rvv.repack_gemm_codebook_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_codebook_core", decode_model = "iq4_xs", weight_quant_byte_offset = 128 : i64, activation_quant_byte_offset = 16 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemm_codebook_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq4_xs_q8_K_kernel_ggml_repack_gemm_iq4_xs_q8_K(
// The 16-entry NON-LINEAR int8 codebook decl (the SAME kvalues_iq4nl table).
// CHECK: verbatim "static const int8_t weft_iq4_xs_repack_kvalues[16] = {-127, -104
// The row-group count nr/4 and the column-group count nc/16.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The interleaved block_q8_Kx4 activation stride 1168; the 4 per-column d_y fp32 scalars.
// CHECK: literal "1168"
// CHECK: call_opaque "*(const float *)"

// ===== The SHARED signed-scale unpack + memory codebook GATHER (vluxei16). =====
// GATHER: call_opaque "__riscv_vsub_vx_i8mf2"
// GATHER: call_opaque "__riscv_vsext_vf4_i32m2"
// GATHER: literal "weft_iq4_xs_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"
// The SIGNED scale weights the i32 sub-block dot into the block accumulator.
// GATHER: call_opaque "__riscv_vmacc_vv_i32m2"

// The end-of-block per-column fold: vfmul the widened weight d strip by fp32 d_y, vfcvt
// the i32 sumi, vfmacc (NO min term), then the per-column two-half vector store.
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Codebook decode is a MEMORY gather (vluxei16), NOT a register vrgather. iq4_xs is
// scale-ONLY: NO min term (no vfnmsac), NO activation bsums read, NO cross-lane
// reduction wall. Also NO trailing dead result-token vmv (the region is result-less).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
