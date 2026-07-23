// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=FULLI8
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MF2

// G3-lode-flat FLAT-4 收官格 -- q8_0 gemm_tile (PREFILL) FRONT-DOOR construction on
// the SUPPORTED RVV1.0 regime (march=rv64gcv => Zvl128b => VLEN128). The abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op (q8_0 / PREFILL) is
// AUTO-LOWERED: the in-compiler selection picks REPACK on the HONEST q8_0 fact set
// -- block_dot_memory_bound = true (fact 2b: q8_0's LEAN block-dot is bandwidth-
// bound, 34 bytes/block ~= 1 byte/weight, so repack's x16 stream removes redundant
// MEMORY traffic) with NO forced block_dot_compute_heavy -- and -- because
// m_regime == "prefill" -- the C1 bridge CONSTRUCTS the typed weft_rvv.typed_repack_gemm_loop_body
// REGION carrying the SHARED q4_0 GEMM bricks -- the one-strip N-column integer CORE
// (repack_gemm_lane_wise_q4_x_i8_dot, stamping the NEW weight_full_i8 selector = the
// FULL signed int8 decode: NO nibble unpack; qk=32 positions/block, vle8 i8 strip at
// qs[i*16 + roff] + per-position/per-column vwmul i8xi8 -> i16 + vwadd_wv into an i32
// in-block accumulator) + the four per-column dual-fp16 scale FOLDs
// (repack_gemm_dual_fp16_scale_fold, d-ONLY, NO min). It reconstructs the
// block_q8_0x16 x16 weight facts (stride 544, quants @32) AND the block_q8_0x4
// INTERLEAVED activation facts (stride 136, quants @8), MATERIALIZES the two GEMM ABI
// values (nr, bs). This GEMM is NET-NEW construction (no q8_0 GEMM direct emitter ever
// existed).
//
// With no qualified residual winner, the analytic resource prior selects
// integer_core_lmul=mf2, half_lanes=8, and f32m2 accumulators.

module {
  weft.exec.kernel @ggml_gemm_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_gemm_q8_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q8_0", scale_model = "dual-fp16-per-block-d_x.d_y-full-i8", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_memory_bound = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected q8_0 repack-GEMM is lowered through the
// SAME front door as q4_0.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_q8_0_q8_0 %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK: emitc.func @weft_emitc_ggml_gemm_q8_0_q8_0_kernel_ggml_gemm_q8_0_q8_0(
// Analytic scheduling initializes the f32m2 accumulator before the group loops.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The selected loop order forms the weight base before the activation base.
// CHECK: literal "544"
// CHECK: literal "136"
// The dual-fp16 scale fold and the 16-lane vector store.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The q8_0 FULL-int8 integer core (GEMM RUNTIME-strip form): the SIGNED vle8 i8 strip
// load (NO nibble unpack), the per-position/per-column vwmul (i8xi8 -> i16), and the
// i32 IN-BLOCK accumulate vwadd_wv (NO i16 vwmacc + lo/hi vwadd_vv combine).
// FULLI8: call_opaque "__riscv_vle8_v_i8mf2"
// FULLI8: call_opaque "__riscv_vwmul_vx_i16m1"
// FULLI8: call_opaque "__riscv_vwadd_wv_i32m2"
// FULLI8-NOT: __riscv_vwmacc_vx_i16m1
// FULLI8-NOT: __riscv_vand_vx_u8mf2
// FULLI8-NOT: __riscv_vfadd_vv_f32m2

// The analytic mf2 route is present and the removed m1 winner is absent.
// MF2: __riscv_vfmv_v_f_f32m2
// MF2: __riscv_vle8_v_i8mf2
// MF2-NOT: __riscv_vfmv_v_f_f32m4
// MF2-NOT: __riscv_vle8_v_i8m1
// MF2-NOT: redsum
