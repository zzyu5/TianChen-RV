// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/fold_model = "sumi_times_scales"/fold_model = "unsupported_fold"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADFOLD

// M1 byte-exact evidence (HISTORICAL, EMPIRICAL): at milestone M1 (commit
// 197a7a5b, BEFORE the monolith was retired) the typed
// tcrv_rvv.typed_flat_block_dot_loop_body codebook branch was proven byte-identical
// to the monolithic tcrv_rvv.iq4_nl_q8_0_block_dot emit by an EMPTY canonicalized
// region-vs-monolith `diff` on a forced clean rebuild (like the q4_0 repack Phase-B).
// The typed body reuses the SAME shared emitFlatBlockCore (-> emitFlatIntegerCore
// CodebookGatherNibble) + emitFlatFold (SumiTimesScales) the monolith drove, so the
// emitted KERNEL BODY is byte-identical BY CONSTRUCTION. At M2 the monolith op +
// emitter were RETIRED (the typed body is now the SOLE representation, front-door
// constructed), so the live diff can no longer parse the retired-op reference; the
// M1 empirical proof stands as the byte-exactness record. The FileCheck below
// asserts the codebook emit directly.

// Anti-bypass / "byte-exact is contingent on the offset" divergence: rewiring the
// codebook-gather product's HIGH q8 activation load quant offset (18 -> 20) genuinely
// moves the emitted high-half load address (the load offset is SOURCED from the
// region load op, not silently hardcoded), so the offset-20 emit is NO LONGER
// byte-identical to the offset-18 baseline (the literal "20" is absent from it).
// RUN: sed 's/quant_byte_offset = 18 : i64/quant_byte_offset = 20 : i64/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=HIGHOFF

// The FULL iq4_nl (ggml IQ4_NL x Q8_0) typed flat block-dot loop body -- the CODEBOOK
// (2nd primitive class) sibling of the q4_0/q5_0 typed loops. M1 constructed-WEAK
// byte-exact increment (NOT a flip): the region carries the 16-entry non-linear int8
// codebook table broadcast brick + the asymmetric codebook-gather packed-i4 x
// plain-i8 product brick (both certified by the [L-8] loop-body allowlist), and the
// lowering reuses the SHARED emitFlatBlockCore + emitFlatFold the monolithic iq4_nl
// path drives. The with_vl is SEW32 m1 (the standalone_reduce codebook-gather
// framing). Numerical bit-exact-vs-ggml is pending-hardware.

module {
  tcrv.exec.kernel @ggml_vec_dot_iq4_nl_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_iq4_nl_q8_0_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %2 = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %3 = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %4 = tcrv_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %5 = tcrv_rvv.setvl %0 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %5 attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_iq4_nl_q8_0_block_dot, sew = 32 : i64, source_kernel = "ggml_vec_dot_iq4_nl_q8_0_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_flat_block_dot_loop_body %2, %3, %1, %0 attributes {activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m1", kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, strip_elision = "elided", weight_block_stride = 18 : i64} {
        ^bb0(%arg0: index, %arg1: f32):
          %6 = tcrv_rvv.block_fp16_scale_product %2, %3 block %arg0 : index {kind = "dual_fp16_per_block_scale_product", lhs_block_stride = 18 : i64, rhs_block_stride = 34 : i64, scale_model = "dual-fp16-per-block-d_x.d_y"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value -> f32
          %7 = tcrv_rvv.codebook_table_broadcast {codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>, table_symbol = "tcrv_iq4_nl_kvalues"} : !tcrv_rvv.vector<i8, "m1">
          %8 = tcrv_rvv.load %2, %5 block %arg0 : index {block_stride = 18 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "m1">
          %9 = tcrv_rvv.load %3, %5 block %arg0 : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
          %10 = tcrv_rvv.load %3, %5 block %arg0 : index {block_stride = 34 : i64, quant_byte_offset = 18 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
          %11 = tcrv_rvv.codebook_gather_x_i8_product %8, %9, %10, %7, %5 {kind = "signed_codebook_gather_x_i8_product", product_relation = "codebook-gather-i8-x-i8x2-to-i16"} : !tcrv_rvv.vector<ui8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
          %12 = tcrv_rvv.standalone_reduce %11, %4, %5 {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %13 = tcrv_rvv.typed_vector_lane0_to_scalar_extract %12, %5 {extract_relation = "i32m1-lane0-to-scalar-i32", kind = "vector_lane0_to_scalar_i32_extract"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> i32
          %14 = tcrv_rvv.block_computed_scale_dequant %13, %6 {dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32", kind = "computed_scale_sumi_dequant"} : i32, f32 -> f32
          %15 = tcrv_rvv.cross_block_f32_accumulate %arg1, %14 {accumulate_order = "strict-ascending-block-carried", kind = "cross_block_f32_scalar_accumulate"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %15 : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot(

// The 16-entry non-linear codebook as a structured static const int8_t[16] decl,
// emitted BEFORE the sumf accumulator (byte-exact to the monolith's decl position).
// CHECK: verbatim "static const int8_t tcrv_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The codebook table broadcast-loaded ONCE (above the block loop), i8m1.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The outer block loop; per-block address arithmetic vx + ib*18, vy + ib*34.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: literal "18"
// CHECK: literal "34"
// The two scalar fp16->fp32 reads (d_x, d_y).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The m1 half-block setvl + the packed weight u8 load + the two q8 signed halves.
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The CODEBOOK class mechanism: split the nibble into the two UNSIGNED index lanes,
// then GATHER each through the broadcast table (NOT the q4_0 offset-binary decode).
// CHECK: call_opaque "__riscv_vand_vx_u8m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: call_opaque "__riscv_vrgather_vv_i8m1"
// The q4_0 offset-binary decode chain must be ABSENT (this is a codebook gather).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8
// The shared asymmetric widening product (i8m1 x i8m1 -> i16m2): low then + high.
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// The per-block reduce seed + vwredsum into i32m1 + the scalar extract.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The SumiTimesScales fold: (float)sumi FIRST, then * (d_x * d_y), then sumf + that.
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store.
// CHECK: subscript
// CHECK: assign

// Fail-closed on the fold_model (I7).
// BADFOLD: currently supports only fold_model

// Anti-bypass: the mutated high-half quant offset propagates into the emitted load
// address (literal "20" is absent from the byte-exact offset-18 baseline emit).
// HIGHOFF: literal "20" : !emitc.opaque<"size_t">
