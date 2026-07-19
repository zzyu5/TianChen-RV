// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=M1
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMF2

// OPTION-2 M1 -- EMIT-IDENTITY on the SUPPORTED RVV1.0 regime (march=rv64gcv =>
// Zvl128b => VLEN128). The abstract, algorithm-UNCOMMITTED
// weft_rvv.quant_contraction op (q4_0 / decode) is AUTO-LOWERED by
// --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection picks
// REPACK (deriveMinimumVLEN(rv64gcv)=128 => the q4_0-vlen128-decode keep) and the
// C1 bridge CONSTRUCTS the typed weft_rvv.typed_repack_gemv_loop_body region.
//
// [GAP-P1]-loosen r51g board sweep flipped q4_0 (kNibbleQ40ScaleModel) to the WIDE
// m1 whole-LMUL chain: selectRepackAccumulatorLMUL consults lookupRepackMeasuredM1
// Faster, and q4_0 now has a board-MEASURED row (DEPLOYED repack GEVM 2.3-2.5x /
// GEMM 1.24x faster than mf2 @rvv VLEN128, spill-free, byte-exact 3-arm -- see
// experiments/active/r51g-b1-repack-family-sweep/FINDING.md). So the region carries
// half_lanes=16 => integer_core_lmul "m1", numHalves==1 (one 16-lane strip), PLUS
// the emitter-INERT audit attrs. --weft-rvv-lower-to-emitc lowers it to the m1
// whole-LMUL (i8m1 -> i16m2 -> i32m4, single 16-lane f32m4 strip) repack-GEMV.
//
// THIS proves the COMPILER AUTO-SELECTS the board-measured m1 chain (registration-
// as-DATA: a data row flips it, no per-format C++ switch). The m1 emit is byte-exact
// to the mf2 default (the LMUL flip does not change arithmetic; board 3-arm mism=0).
//
// SCOPE: EMIT-IDENTITY ONLY. The m1 whole-LMUL form is now the board-MEASURED
// DEPLOYED q4_0 chain on RVV1.0 (VLEN128). NO extra perf claim in this fixture --
// the kernel is lit-emitted here; the perf is in the r51g FINDING.

module {
  weft.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected repack-GEMV is lowered to the SAME
// repack kernel the direct repack op produces (the audit attrs are emitter-inert).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_q4_0_q8_0 %
// CHECK-NOT: weft_rvv.q4_0_q8_0_block_dot
// CHECK-NOT: unrealized_conversion_cast
// The repack-GEMV export signature (the SAME 8-value ABI wrapper the abstract op
// carried: n,s,bs,vx,bx,vy,by,nrc -- unused values stay as params).
// CHECK: emitc.func @weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
// The per-group weight base vx + x*nb*288 (block_q4_0x16 stride 288).
// CHECK: literal "288"
// The single 16-lane f32m4 accumulator -- the board-measured m1/half_lanes=16 form.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"
// The repacked i8m1 whole-LMUL load then lane-wise vwmacc into i16m2 (NO cross-lane
// reduction wall) then vwadd into i32m4.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m4"
// The single 16-lane vector store vse32_v_f32m4.
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: return

// The board-MEASURED RVV1.0 whole-LMUL m1 form: the i8m1 strip is PRESENT.
// M1: call_opaque "__riscv_vle8_v_i8m1"

// This is the m1 (single 16-lane strip) chain -- the mf2 (two 8-lane halves)
// spellings must NOT appear (the flip is a REAL per-format LMUL change).
// NOMF2-NOT: __riscv_vle8_v_i8mf2
// NOMF2-NOT: __riscv_vfmv_v_f_f32m2
// NOMF2-NOT: __riscv_vwmacc_vx_i16m1
