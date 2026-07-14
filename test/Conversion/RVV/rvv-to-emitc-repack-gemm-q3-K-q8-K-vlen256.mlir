// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=QHMASK
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=RETIRED

// G8 stage-2 [QH-MASK] DOUBLE-BOARD construction evidence (VLEN256 board) -- the q3_K x
// q8_K 16x1-REPACKED PREFILL GEMM that consumes the SHARED native-interleaved-static
// mask helper (emitNativeMaskStaticBitBias), at VLEN256. Complements the VLEN128 sibling
// (rvv-to-emitc-repack-gemm-q3-K-q8-K.mlir, half_lanes = 8) exactly as the [ROLL] q6_K
// VLEN256 fixture complements its VLEN128 auto sibling: the VLEN256 board is expressed by
// half_lanes = 16 (numHalves = 1, a SINGLE 16-lane weight-decode strip vs the VLEN128 two
// 8-lane strips). This asserts the [QH-MASK] helper CONSTRUCTS the native-mask single-bit
// bias CORRECTLY at VLEN256: the vand(0x03) low plane + the vand(1<<p) isolate + the
// per-lane vmseq==0 predicate + the FUSED vadd_vx_i8mf2_mu(-4) all still emit byte-exact,
// with the vbool width AUTO-COMPUTED from the mf2 LMUL (mf2 => b16 => vbool16_t; 8/LMUL).
// The QH-MASK op suffixes are VLEN-INVARIANT (u8mf2 / b16); only the strip COUNT halves
// (512 native-mask sites vs the VLEN128 1024) -- i.e. the native-mask construction is
// board-invariant while the surrounding strip loop tracks half_lanes.
//
// NUMERIC STATUS: this lit checks the LOWERED STRUCTURE only. Numeric bit-exact-vs-ggml is
// proven SEPARATELY by the board oracle (independent scalar q3_K dequant-matmul reference,
// byte-exact-integer GREEN). Pending-hardware for the VLEN256 board (ssh rvv is VLEN128).

module {
  weft.exec.kernel @ggml_repack_gemm_q3_K_q8_K_vlen256_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q3_K_q8_K_vlen256 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q3_K_q8_K_vlen256, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q3_K_q8_K_vlen256_kernel", status = "selected-lowering-boundary"} {
        // half_lanes = 16 => VLEN256 board: a SINGLE 16-lane weight-decode strip.
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 16 : i64, fold_model = "kquant_single_scale_no_min"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          %sumi:4 = weft_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q3_K", weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemm_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q3_K_q8_K_vlen256_kernel_ggml_repack_gemm_q3_K_q8_K_vlen256(
// The three outer runtime loops (row-group, column-group, contraction-block).
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step

// [QH-MASK] VLEN256 native-mask construction: the q3_K 3-bit SUBTRACTIVE weight assembly
// via the SHARED helper -- qs/hmask strip loads, the 2-bit low plane (vand 0x03) to signed
// i8, then the SINGLE hmask high bit isolated IN PLACE (vand 1<<p) + lifted to a per-lane
// vbool by vmseq==0, with the -4 SUBTRACTIVE bias FUSED into ONE masked op
// vadd_vx_i8mf2_mu(mask, base, base, -4). The vbool width is AUTO-COMPUTED from the mf2
// LMUL: mf2 => 8/LMUL = 16 => b16 => vbool16_t (VLEN-invariant; identical to the VLEN128
// sibling's suffix -- only the strip COUNT differs). Byte-exact to the retired
// vsll 2 | vor | vsub 4 expand chain.
// QHMASK: call_opaque "__riscv_vle8_v_u8mf2"
// QHMASK: literal "0x03"
// QHMASK: call_opaque "__riscv_vand_vx_u8mf2"
// QHMASK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// QHMASK: call_opaque "__riscv_vand_vx_u8mf2"
// QHMASK: call_opaque "__riscv_vmseq_vx_u8mf2_b16"
// QHMASK-SAME: -> !emitc.opaque<"vbool16_t">
// QHMASK: literal "-4"
// QHMASK: call_opaque "__riscv_vadd_vx_i8mf2_mu"
// QHMASK-SAME: !emitc.opaque<"vbool16_t">

// The OLD per-lane hmask expand chain (vsll<<2 | vor | vsub 4) is fully RETIRED by the
// SHARED [QH-MASK] native-mask helper -- none of those three ops survive at VLEN256 either.
// RETIRED-NOT: __riscv_vsll_vx_u8mf2
// RETIRED-NOT: __riscv_vor_vv_u8mf2
// RETIRED-NOT: __riscv_vsub_vx_i8mf2
