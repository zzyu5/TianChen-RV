// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M2 iq4_nl codebook front-door: the ggml iq4_nl x q8_0 16x1-REPACKED GEMM (prefill) hot
// kernel -- the CODEBOOK prefill sibling of the iq4_nl GEVM -- is now CONSTRUCTED through
// the typed-region FRONT DOOR, NOT the retired monolithic emitRepackGemmIq4NlQ80 direct
// emitter. The weft_rvv.typed_repack_gemm_loop_body region (fold_model
// "codebook_flat_single_scale") carries the weft_rvv.repack_gemm_codebook_core integer-core
// BRICK (decode_model "iq4_nl" + the 16-entry codebook), block_index + strip_row_offset
// tied (anti-bypass). The lowering GATES the emit on those ties, then RE-EMITS the byte-exact
// iq4_nl GEMM body via emitTypedRepackGemmLoopBody's codebook branch ->
// emitRepackCodebookGemmBodyIq4Nl (byte-identical to the retired direct emitter; PLAIN
// untiled -- iq4_nl already sits at the <=32-vreg cliff, so S6 tiling is a structural no-op).
// The 4-bit nibble is an INDEX into the 16-entry codebook, decoded by the memory vluxei16
// GATHER (NOT a register vrgather, NOT fake-linear), AMORTIZED once per 16-weight group and
// reused across the 4 interleaved block_q8_0x4 activation columns. Single fp16 scale per
// column, i32 accumulator, NO min. block_iq4_nlx16 stride 288; block_q8_0x4 stride 136, qs @8.

module {
  weft.exec.kernel @ggml_repack_gemm_iq4_nl_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq4_nl_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq4_nl_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq4_nl_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "flat.fp16-single-scale-codebook-4col-nomin", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "codebook_flat_single_scale"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied CODEBOOK GEMM integer-core BRICK:
          // per-block lane-wise iq4_nl memory-gather dot across the 4 interleaved columns ->
          // the columnsPerPass (4) per-column i32 sumi. The typed emitter re-emits the whole
          // byte-exact iq4_nl GEMM body from this brick's identity + its codebook; the yield
          // passes through the carried-in per-column accs.
          %sumi:4 = weft_rvv.repack_gemm_codebook_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_codebook_core", decode_model = "iq4_nl", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(
// CHECK: verbatim "static const int8_t weft_iq4_nl_repack_kvalues[16] = {-127, -104
// The block count nb = n/32, the row-group count nr/4 (%arg4), the column-group
// count nc/16 (%arg5).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-ROW-group loop over nr/4, the interleaved q8_0x4 base
// (a + y*nb*136 stride 136).
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "136"
// The weight-column-GROUP loop; weight base vx + x*nb*288.
// CHECK: literal "288"

// ===== The SHARED (amortized) memory codebook GATHER (vluxei16). =====
// GATHER: call_opaque "__riscv_vle8_v_u8mf2"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: literal "weft_iq4_nl_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// The per-column single-scale fold (vfwmul d_x*d_y_c, vfmacc) and per-(row,col)
// store through the runtime output row stride bs.
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall (NO redsum); the flat
// codebook GEMM fold has NO min (no vfnmsac) and NO per-sub-block scale (no vwmacc_vv_i32);
// the decode is a MEMORY gather, NOT a register vrgather. Also NO trailing dead
// result-token vmv (the region is result-less, unlike the retired direct emitter).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
