// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=E8M0
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 retirement_batch 4 mxfp4 codebook front-door: the ggml mxfp4 x q8_0 16x1-REPACKED GEVM
// (decode) hot kernel -- the FP4-CODEBOOK + E8M0 (micro-exponent) block-as-lane sibling of
// the iq4_nl repacked GEVM -- is now CONSTRUCTED through the typed-region FRONT DOOR (the
// q4_0 / ternary / K-quant / iq4_nl typed_repack precedent), NOT the retired monolithic
// emitRepackGemvMxfp4Q8 direct emitter. The tcrv_rvv.typed_repack_gemv_loop_body region
// (fold_model "codebook_flat_e8m0_scale") carries the tcrv_rvv.repack_gemv_codebook_core
// integer-core BRICK (decode_model "mxfp4" + the 16-entry doubled-E2M1 int8 codebook
// DenseI8ArrayAttr), block_index-tied (anti-bypass) and named off the loop-body's own weight
// / activation ABI bases. The lowering GATES the emit on that brick's anti-bypass ties, then
// RE-EMITS the byte-exact mxfp4 GEVM body via emitTypedRepackGemvLoopBody's codebook branch
// -> emitRepackCodebookGemvBodyMxfp4 (byte-identical to the retired direct emitter). The
// 4-bit weight nibble is an INDEX into the 16-entry codebook, decoded by a REAL MEMORY
// codebook GATHER (vzext -> vluxei16 -- NOT a register vrgather); the per-column weight scale
// is ONE E8M0 shared-exponent BYTE reconstructed to 2^(e-128) by bit arithmetic (NOT an fp16
// delta). The dot accumulates LANE-WISE in an i32 accumulator, NO min. block_mxfp4x16 stride
// 272 (E8M0 e @0, qs @16); activation is one plain block_q8_0 (stride 34, qs at +2).
// VLEN=128 => TWO strips.

module {
  tcrv.exec.kernel @ggml_repack_gemv_mxfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_mxfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "flat.e8m0-single-scale-codebook-nomin", qk = 32 : i64, weight_block_stride = 272 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 16 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "codebook_flat_e8m0_scale"} {
        ^bb0(%block_index: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">):
          // The block_index-tied CODEBOOK integer-core BRICK: per-block lane-wise mxfp4
          // memory-gather dot -> the numHalves (2) per-strip i32 sumi. The typed emitter
          // re-emits the whole byte-exact mxfp4 body (codebook gather + i32 dot + E8M0 scale
          // fold) from this brick's identity + its 16-entry codebook; the yield passes through
          // the carried-in accs.
          %sumi:2 = tcrv_rvv.repack_gemv_codebook_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_codebook_core", decode_model = "mxfp4", weight_quant_byte_offset = 16 : i64, activation_quant_byte_offset = 2 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: tcrv_rvv.repack_gemv_codebook_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_mxfp4_q8_0_kernel_ggml_repack_gemv_mxfp4_q8_0(
// The 16-entry doubled-E2M1 int8 fp4 codebook decl (the kvalues_mxfp4 table).
// CHECK: verbatim "static const int8_t tcrv_mxfp4_repack_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1
// The block count nb = n / 32 and the column-group count nc/16.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop; per-group weight base vx + x*nb*272.
// CHECK: literal "272"
// The two 8-lane f32 accumulators: vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop; per-block plain q8_0 base al = a + l*34.
// CHECK: literal "34"
// The per-strip i32 accumulator seed.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"

// ===== The REAL fp4-codebook GATHER (memory vluxei16, NOT register vrgather). =====
// GATHER: call_opaque "__riscv_vle8_v_u8mf2"
// GATHER: call_opaque "__riscv_vand_vx_u8mf2"
// GATHER: literal "0x0F"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: literal "tcrv_mxfp4_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vsrl_vx_u8mf2"
// GATHER: literal "0x04"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL E8M0 shared-exponent scale reconstruction (2^(e-128)). =====
// E8M0: call_opaque "__riscv_vzext_vf4_u32m2"
// E8M0: literal "0x1F"
// E8M0: call_opaque "__riscv_vand_vx_u32m2"
// E8M0: literal "0x00200000"
// E8M0: call_opaque "__riscv_vmv_v_x_u32m2"
// E8M0: call_opaque "__riscv_vsll_vv_u32m2"
// E8M0: call_opaque "__riscv_vsub_vx_u32m2"
// E8M0: literal "23"
// E8M0: call_opaque "__riscv_vsll_vx_u32m2"
// E8M0: call_opaque "__riscv_vmsltu_vx_u32m2_b16"
// E8M0: call_opaque "__riscv_vmerge_vvm_u32m2"
// E8M0: call_opaque "__riscv_vreinterpret_v_u32m2_f32m2"
// E8M0: call_opaque "__riscv_vfmul_vf_f32m2"

// The end-of-block fold: vfcvt the i32 sumi, then vfmacc (NO min term, NO dmin).
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The codebook decode is a MEMORY gather (vluxei16), NOT a register vrgather; the E8M0 scale
// is reconstructed by bit-arithmetic, NOT read as an fp16 delta (no vle16/vfwmul weight-scale
// load). The block-as-lane repack erases the cross-lane reduction wall; the flat codebook
// fold has NO min (no vfnmsac) and NO per-sub-block scale (no vwmacc_vv_i32). Also NO trailing
// dead result-token vmv (the region is result-less, unlike the retired direct emitter).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
// NOWALL-NOT: vfwmul
// NOWALL-NOT: vle16
