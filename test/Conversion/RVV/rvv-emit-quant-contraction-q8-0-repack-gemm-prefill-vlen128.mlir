// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=FULLI8
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=WIDE

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
// [GAP-P1]-loosen (2026-07-19): the repack ACCUMULATOR LMUL is the board-MEASURED
// WIDE m1 whole-LMUL chain (integer_core_lmul m1, half_lanes 16, columnsPerPass 1,
// f32m4) -- lookupRepackMeasuredM1Faster(kNibbleQ80ScaleModel)=true. On rvv VLEN128
// the deployed q8_0 repack-GEMM m1 chain is ~1.40x faster than the mf2 default
// (columnsPerPass=4, f32m2) and byte-exact (3-arm mism=0) -- the wide 16-lane strip
// beats the mf2 4-column amortization. Every OTHER format stays mf2. NO board number
// is PROJECTED here (see experiments/active/r51f-gapp1-q8-deployed-board/FINDING.md);
// the per-format selector judgment lives in
// rvv-repack-accumulator-lmul-measured-gate-q8.mlir.

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
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_gemm_q8_0_q8_0, sew = 32 : i64, source_kernel = "ggml_gemm_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
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
// The per-group activation base vy + y*nb*136 (block_q8_0x4 stride 136) and the
// per-group weight base vx + x*nb*544 (block_q8_0x16 stride 544).
// CHECK: literal "136"
// CHECK: literal "544"
// The WIDE m1 whole-LMUL f32m4 accumulator (columnsPerPass == 1: ONE 16-lane strip,
// the board-measured accumulator flip -- vs the old mf2 4x8 f32m2 4-column pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"
// The dual-fp16 scale fold and the 16-lane vector store.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: return

// The q8_0 FULL-int8 integer core (GEMM RUNTIME-strip form): the SIGNED vle8 i8 strip
// load (NO nibble unpack), the per-position/per-column vwmul (i8xi8 -> i16), and the
// i32 IN-BLOCK accumulate vwadd_wv (NO i16 vwmacc + lo/hi vwadd_vv combine).
// FULLI8: call_opaque "__riscv_vle8_v_i8m1"
// FULLI8: call_opaque "__riscv_vwmul_vx_i16m2"
// FULLI8: call_opaque "__riscv_vwadd_wv_i32m4"
// FULLI8-NOT: __riscv_vwmacc_vx_i16m1
// FULLI8-NOT: __riscv_vand_vx_u8mf2
// FULLI8-NOT: __riscv_vfadd_vv_f32m2

// POST-[GAP-P1]-loosen the RVV1.0 q8_0 repack-GEMM DEPLOYS the WIDE m1 whole-LMUL
// chain (board-measured faster, byte-exact): the f32m4 accumulator + i8m1 strip ARE
// present, the mf2 f32m2/i8mf2 spellings are GONE, and there is still NO cross-lane
// vredsum reduction wall (lane-wise fold, not a reduce).
// WIDE: __riscv_vfmv_v_f_f32m4
// WIDE: __riscv_vle8_v_i8m1
// WIDE-NOT: __riscv_vfmv_v_f_f32m2
// WIDE-NOT: __riscv_vle8_v_i8mf2
// WIDE-NOT: redsum
