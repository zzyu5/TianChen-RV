// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The VLEN256 (`_vl256`) shape of the ggml IQ4_NL x Q8_0 codebook block dot-product:
// the SAME structured kernel as the VLEN128 form (test rvv-to-emitc-iq4-nl-q8-0-block-
// dot.mlir), but the i8 integer-core anchor moves from m1 to mf2 driven by the VLEN
// CAPABILITY fact (integer_core_lmul = "mf2", minimum_vlen = 256). This is the ggml
// `_vl256` lowering: at VLEN256 mf2's VLMAX = 16 = a FULL mf2 register (the 16-lane
// nibble-pair half-block exactly fills it; at VLEN128 mf2 -> VLMAX 8 < 16 would break
// the codebook gather, which is why the verifier gates mf2 on minimum_vlen >= 256), and
// the multi_block_factor = 2 unroll keeps two independent blocks in flight to fill the
// wider register file. NO memory layout / algorithm change vs the m1 form -- only the
// LMUL spelling + unroll move (pure backend Win-A lowering): the i8 source/gather/vsetvl
// are spelled at mf2, the widened i16 product at m1 (one step wider than mf2), and the
// vwredsum reduction destination + seed STAY m1 (a 16-element reduce is an m1 reduce).
//
// This file is the NON-NULL divergence proof: the emitted bytes here (mf2 / vwmul i16m1
// / vwredsum i16m1) DIFFER from the VLEN128 emit (m1 / vwmul i16m2 / vwredsum i16m2), so
// the VLEN capability fact drives a genuinely different lowering, not a Win-C no-op.

module {
  tcrv.exec.kernel @ggml_vec_dot_iq4_nl_q8_0_kernel_vl256 {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_iq4_nl_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_iq4_nl_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_iq4_nl_q8_0_kernel_vl256", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.iq4_nl_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_nl_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>, integer_core_lmul = "mf2", multi_block_factor = 2 : i64, strip_elision = "elided", minimum_vlen = 256 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_vl256_ggml_vec_dot_iq4_nl_q8_0(
// The codebook table is broadcast-loaded ONCE at the mf2 anchor (16 entries fill a full
// mf2 register at VLEN256 -- the gather can index all 16). This is the FIRST divergence
// from the m1 form (which loads the table at i8m1).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// The m1-form i8m1 table load + strip spellings must be ABSENT in the mf2 core.
// CHECK-NOT: call_opaque "__riscv_vsetvl_e8m1"
// CHECK-NOT: call_opaque "__riscv_vle8_v_u8m1"
// CHECK-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// The mf2 half-block setvl + the packed weight u8 load + the two q8 signed halves, all
// at the mf2 anchor (the i8 source LMUL the VLEN capability fact selected).
// CHECK: call_opaque "__riscv_vsetvl_e8mf2"
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// The codebook nibble split + gather at mf2 (NOT m1).
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// CHECK: call_opaque "__riscv_vrgather_vv_i8mf2"
// CHECK: call_opaque "__riscv_vrgather_vv_i8mf2"
// The widened i16 product is ONE step wider than the mf2 source: i16m1 (the VLEN128 form
// emits i16m2 -- this is the load-bearing byte divergence).
// CHECK: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m1"
// The vwredsum reduction destination + seed STAY m1 (a 16-element reduce is an m1
// reduce regardless of the i8 anchor): vwredsum_vs_i16m1_i32m1, NOT _i16m2_.
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m1_i32m1"
// CHECK-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The 2-block unroll emits BOTH independent integer cores before either fp32 fold (the
// fold stays in strict ascending block order for fp byte-exactness): two mf2 gathers per
// main-loop body.
// CHECK: expression : !emitc.opaque<"float">
// The *s store.
// CHECK: subscript
// CHECK: assign
