// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=FULLI8
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMIN

// G3-lode-flat FLAT-4 收官格 -- q8_0 gemm_tile (decode) FRONT-DOOR construction on
// the SUPPORTED RVV1.0 regime (march=rv64gcv => Zvl128b => VLEN128). The abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op (q8_0 / decode) is
// AUTO-LOWERED by --weft-rvv-lower-quant-contraction: the in-compiler selection
// picks REPACK (deriveMinimumVLEN(rv64gcv)=128) on the HONEST q8_0 fact set --
// block_dot_memory_bound = true (fact 2b: q8_0's LEAN block-dot streams 34
// bytes/block ~= 1 byte/weight, the widest linear quant, so it is bandwidth-bound
// and repack's x16 stream removes redundant MEMORY traffic) with NO forced
// block_dot_compute_heavy -- and the C1 bridge CONSTRUCTS the
// typed weft_rvv.typed_repack_gemv_loop_body region carrying the SHARED lane-wise
// integer CORE brick (repack_lane_wise_q4_x_i8_dot) stamping the NEW weight_full_i8
// selector (the FULL signed int8 decode: NO nibble unpack; qk=32 positions/block,
// vle8 i8 strip + per-position vwmul i8xi8 -> i16 + vwadd_wv into an i32 in-block
// accumulator) + the two per-strip dual-fp16 scale FOLDs (repack_dual_fp16_scale_fold,
// d-ONLY, NO min). --weft-rvv-lower-to-emitc then lowers it to the mf2/half_lanes=8
// two-8-lane-halves repack-GEVM kernel.
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the q8_0 repack region
// through the SAME front door as q4_0/q4_1/q5_0/q5_1 (previously q8_0 was
// direct-emitter dispatch-wired -- the SIMPLEST flat variant, the LAST FLAT-4). NO
// perf/e2e claim -- the kernel is lit-emitted, NOT run.

module {
  weft.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q8_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q8_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q8_0", scale_model = "dual-fp16-per-block-d_x.d_y-full-i8", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_memory_bound = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected q8_0 repack-GEVM is lowered through
// the SAME front door as q4_0.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_q8_0_q8_0 %
// CHECK: emitc.func @weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(
// The per-group weight base vx + x*nb*544 (block_q8_0x16 stride 544 = 16 d + 512
// full int8 quants).
// CHECK: literal "544"
// The two 8-lane f32m2 accumulators (rows 0..7, 8..15) -- the mf2/half_lanes=8 form.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The q8_0 FULL-int8 integer core: the SIGNED vle8 i8 strip load (NO nibble unpack
// -- NO vand/vsrl/vsll/vreinterpret decode), the per-position vwmul (i8xi8 -> i16),
// and the i32 IN-BLOCK accumulate vwadd_wv (REPLACING q4_0's i16 vwmacc + lo/hi
// vwadd_vv combine -- full int8 products overflow i16).
// FULLI8: call_opaque "__riscv_vle8_v_i8mf2"
// FULLI8: call_opaque "__riscv_vwmul_vx_i16m1"
// FULLI8: call_opaque "__riscv_vwadd_wv_i32m2"
// The q8_0 core does NOT unpack nibbles: no vwmacc, no unsigned-nibble peel.
// FULLI8-NOT: __riscv_vwmacc_vx_i16m1
// FULLI8-NOT: __riscv_vand_vx_u8mf2

// The q8_0 fold is d-ONLY (the q4_0 dual-fp16 scale tree, NO min): vfwmul(d) /
// vfcvt / vfmacc, with NO second vfwmul(m_x) + vfadd min correction.
// NOMIN: call_opaque "__riscv_vfwmul_vf_f32m2"
// NOMIN: call_opaque "__riscv_vfmacc_vv_f32m2"
// NOMIN-NOT: __riscv_vfadd_vv_f32m2
