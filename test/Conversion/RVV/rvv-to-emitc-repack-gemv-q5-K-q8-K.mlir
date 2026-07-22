// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T3 format4: the ggml q5_K x q8_K 16x1-REPACKED single-output-column GEVM (decode)
// hot kernel is now CONSTRUCTED through the typed-region FRONT DOOR (the q4_0 / ternary /
// q4_K / q2_K typed_repack precedent), NOT the retired monolithic emitRepackGemvQ5KQ8K
// direct emitter. The weft_rvv.typed_repack_gemv_loop_body region (fold_model
// "kquant_dmin_bsums_min", SHARED with q4_K) carries the weft_rvv.repack_gemv_kquant_core
// integer-core BRICK (decode_model "q5_K"), block_index tied (anti-bypass) and named off the
// loop-body's own weight / activation ABI bases. The lowering GATES the emit on that
// anti-bypass tie, then RE-EMITS the byte-exact q5_K GEVM body via
// emitTypedRepackGemvLoopBody's K-quant MIN branch -> emitRepackKQuantGemvBodyQ5K. q5_K ==
// q4_K + the qh 5th-bit plane: the ONLY delta vs q4_K is the qh inject (each 4-bit nibble
// lifted to a 5-bit value in [0,31] by OR-ing qh[i]'s per-sub-block high bit <<4); the q4_K
// dual d/dmin + bsums-min fold + 6-bit scale/min unpack are SHARED. block_q5_Kx16 stride
// 2816, qs at +768, qh at +256, dmin at +32, 6-bit scales at +64; plain block_q8_K activation
// stride 292 (bsums at +260). VLEN=128 mf2 => TWO 8-lane strips.
//
// NUMERIC STATUS: this lit checks the LOWERED STRUCTURE only. Numeric correctness is proven
// SEPARATELY by the board oracle (independent scalar q5_K dequant-matmul reference,
// bounded-norm GREEN, same IEEE-reassociation caveat as q4_K).

module {
  weft.exec.kernel @ggml_repack_gemv_q5_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q5_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q5_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q5_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 4 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 260 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "kquant_dmin_bsums_min", main_term_form = "unrolled"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied q5_K integer-core BRICK: per-block lane-wise q5_K dot ->
          // the numHalves (2) per-strip i32 sumi. The typed emitter re-emits the whole
          // byte-exact q5_K body (the q4_K nibble + qh 5th-bit inject + 6-bit scale/min
          // unpack + dual d/dmin + bsums-min fold) from this brick's identity; the yield
          // passes the accs through.
          %sumi:2 = weft_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_kquant_core", decode_model = "q5_K", weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemv_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K(
// Per-group weight base vx + x*nb*2816 (block_q5_Kx16 stride 2816).
// CHECK: literal "2816"
// The two 8-lane f32m2 accumulators (mf2 / half_lanes=8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The plain q8_K activation base al = a + l*292 (stride 292), inside the block loop.
// CHECK: literal "292"
// The q5_K 6-bit scale/min unpack (SHARED, IDENTICAL to q4_K's get_scale_min_k4 dance):
// vle8 low+high bytes, vand 0x0F / vsrl 4, j-dependent high bits, vor, vzext to i16.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_u8mf2"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// The q5_K 5-bit weight assembly: the q4_K nibble (vand 0x0F / vsrl 4) PLUS a SECOND vle8
// loading the qh strip, vand 0x01 masking the selected per-sub-block bit (a mask q4_K
// NEVER emits), vsll 4, vor merging onto the nibble -> a 5-bit value in [0,31], then
// reinterpret to a SIGNED i8 lane.
// CHECK: literal "0x01"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// The lane-wise integer dot + scale-weighted i32 promote (NO vredsum).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The MIN term: read the per-block int16 bsums, fold via vwmacc, subtract via vfnmsac.
// CHECK: call_opaque "*(const int16_t *)"
// The end-of-block dual fold: vfmul_vf (d_x*d_y), vfcvt, vfmacc (main), vfnmsac (MIN).
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall.
// NOWALL-NOT: redsum
