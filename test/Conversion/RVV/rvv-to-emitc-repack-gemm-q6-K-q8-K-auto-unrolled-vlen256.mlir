// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=UNROLLED

// G8 stage-2 [ROLL] MEASURED-GATE AUTO-PREDICATE fixture (VLEN256 board).
//
// The SECOND board of the missing double-board auto-predicate coverage (audit §五 档 B#4:
// "ROLL 无 VLEN256 ... auto 谓词从无 fixture 走过"). SAME q6_K x q8_K 16x1-REPACKED PREFILL
// GEMM region as the VLEN128 auto fixture but at VLEN256 (half_lanes = 16 => e16m1 16-lane
// strip, numHalves = 16/16 = 1 single strip, HALF the VLEN128 strip count). NO
// main_term_form stamp => the RVVRepackScheduleFormula MEASURED-GATE decides.
//
// The VLEN256 single-strip main-term code volume (proxy 1024 = numHalves 1 x nSuperHalves
// 2 x mHalves 2 x mGroup 16 x columnsPerPass 4 x lanes 4) still EXCEEDS the 192 budget so
// it is roll-ELIGIBLE, but with the measured table EMPTY the default is UNROLLED. This is
// exactly the [GAP-KQUANT-VLEN256-UNROLL-VS-ROLLED] carve-back candidate: Stage-2 asserts
// NOTHING about the VLEN256 crossover -- it conservatively keeps the WHOLE family unrolled
// and defers the VLEN split to on-board A/B (the carve-back may KEEP VLEN256 unrolled once
// measured). BYTE-EXACT to the shipped emit by construction.
module {
  weft.exec.kernel @ggml_repack_gemm_q6_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q6_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q6-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-signed8-scale-16-subblocks-6bit-4col-nomin", qk = 256 : i64, weight_block_stride = 3360 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 1312 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 16 : i64, integer_core_lmul = "mf2", fold_model = "kquant_single_scale_no_min", main_term_form = "unrolled", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          %sumi:4 = weft_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q6_K", weight_quant_byte_offset = 1312 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemm_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(
// The THREE outer runtime loops -- the AUTO-default (unrolled) form has NO fourth inner
// per-position runtime loop at VLEN256 either.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The fully-UNROLLED main term (single 16-lane strip), inline vwmacc.
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// ===== AUTO-PREDICATE GUARD (VLEN256): measured-gate default is UNROLLED; the ROLLED-only
// resident i16 partial round-trip MUST NOT appear. [GAP-KQUANT-VLEN256-UNROLL-VS-ROLLED]
// carve-back: VLEN256 stays unrolled at Stage-2 (no projection), so these stay absent.
// UNROLLED-NOT: assign %{{.*}} : !emitc.opaque<"vint16m1_t">
// UNROLLED-NOT: load %{{.*}} : <!emitc.opaque<"vint16m1_t">>
