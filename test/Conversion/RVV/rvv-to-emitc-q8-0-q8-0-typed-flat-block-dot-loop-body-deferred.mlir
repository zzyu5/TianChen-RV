// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --implicit-check-not=vfredusum --implicit-check-not=vfmacc --implicit-check-not=vwmacc
// The SAME typed body with fold_structure REMOVED falls back to the per-block
// default schedule (mbf==4 elided): scalar SeparatedLeftAssoc folds, NO
// vfredosum. This locks that fold_structure is the knob that toggles the fold
// SCHEDULE and that the default path is unchanged (zero-regression demonstration).
// RUN: sed 's/, fold_structure = "deferred-ordered"//' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=PERBLOCK --implicit-check-not=vfredosum

// M-FLAT P2c: the q8_0 (sumi_times_scales) typed flat block-dot LOOP body with
// fold_structure = "deferred-ordered" -- the batched-vector cross-block fold
// schedule for the SAME pinned §1 oracle
// (measurement/浮点折叠oracle.md). fold_structure is ORTHOGONAL to
// fold_model (which fixes the arithmetic tree) + multi_block_factor / strip_elision
// (which drive the integer-core schedule): it selects HOW the fold is issued.
// For B = multi_block_factor = 4 blocks: PHASE A runs the four region-sourced
// integer cores (two i8m2 loads -> vwmul -> vwredsum -> lane0 extract -> scalar
// sumi_k) and packs the four sumi into ONE i32m1 vector via vslide1down (vl=B, so
// the final lane i == block i in STRICT ASCENDING order), plus a vlse16 strided
// load of the four fp16 d_x / d_y scales (block stride 34) widened f16->f32
// (exact). PHASE B converts sumi (int32->f32, exact since |sumi| < 2^24), does
// vfmul x2 per lane (t_b = ((sumi_b*d_x_b)*d_y_b), SeparatedLeftAssoc §1: NO
// d_x*d_y premultiply, NO FMA contraction -- vfmul/vfredosum are separate ops),
// then folds the four terms with ONE vfredosum.vs SEEDED by the running sumf.
// RVV vfredosum.vs reduces lane-ascending, seed-first, so
// ((sumf+t_0)+t_1)+t_2)+t_3 = the §1 serial left-fold BYTE-FOR-BYTE (bit-exact by
// construction, NOT approximation). Batch chaining (each batch seeds vfredosum
// with the prior sumf) + the nb % 4 robust scalar tail (the SAME §1
// SeparatedLeftAssoc scalar fold) reconstruct the full §1 fold. The unordered
// vfredusum and the FMA-contracting vfmacc are BANNED (implicit-check-not above):
// only the ordered vfredosum.vs is bit-exact. Numerical bit-exact-vs-the-§1-oracle
// on hardware is pending-hardware; the perf overlap mechanism is hypothesized.

module {
  weft.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_q8_0_q8_0_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %zero_seed = weft_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q8_0_q8_0_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "elided", multi_block_factor = 4 : i64, fold_structure = "deferred-ordered"} {
        ^bb0(%block_index: index, %acc: f32):
          %dd = weft_rvv.block_fp16_scale_product %vx, %vy block %block_index : index {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          %wv = weft_rvv.load %vx, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
          %av = weft_rvv.load %vy, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
          %prod = weft_rvv.widening_product %wv, %av, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !weft_rvv.vector<i8, "m2">, !weft_rvv.vector<i8, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m4">
          %red = weft_rvv.standalone_reduce %prod, %zero_seed, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m4">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          %sumi = weft_rvv.typed_vector_lane0_to_scalar_extract %red, %vl {kind = "vector_lane0_to_scalar_i32_extract", extract_relation = "i32m1-lane0-to-scalar-i32"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> i32
          %bterm = weft_rvv.block_computed_scale_dequant %sumi, %dd {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, f32 -> f32
          %acc_next = weft_rvv.cross_block_f32_accumulate %acc, %bterm {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
// The function-scoped fp32 accumulator + block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The by-4 batch main loop bound nb_main = nb - nb % 4.
// CHECK: %[[REM:.*]] = rem %{{.*}}, %{{.*}}
// CHECK: %[[MAIN:.*]] = sub %{{.*}}, %[[REM]]
// CHECK: for %{{.*}} = %{{.*}} to %[[MAIN]] step
// --- PHASE A: the sumi vector seeded 0 (i32m1), then the four region-sourced
// --- integer cores (each vwmul -> vwredsum -> lane0 extract) packed by four
// --- vslide1down slides (vl=B): the final lane i == block i, ascending. ---
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// CHECK: %[[SLIDE0:.*]] = call_opaque "__riscv_vslide1down_vx_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: call_opaque "__riscv_vslide1down_vx_i32m1"(%[[SLIDE0]]
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: call_opaque "__riscv_vslide1down_vx_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: %[[SUMIVEC:.*]] = call_opaque "__riscv_vslide1down_vx_i32m1"
// --- PHASE A scales: the four d_x then four d_y fp16 scales strided-loaded
// --- (vlse16, block stride) and widened f16->f32 (exact). ---
// CHECK: call_opaque "__riscv_vlse16_v_f16mf2"
// CHECK: %[[DXVEC:.*]] = call_opaque "__riscv_vfwcvt_f_f_v_f32m1"
// CHECK: call_opaque "__riscv_vlse16_v_f16mf2"
// CHECK: %[[DYVEC:.*]] = call_opaque "__riscv_vfwcvt_f_f_v_f32m1"
// --- PHASE B: bit-exact §1. The seed for the ORDERED reduction is the running
// --- sumf (load -> vfmv_v_f), so vs1[0] carries the prior accumulator. sumi is
// --- int32->f32 (exact), then vfmul x2 (SEPARATE roundings, NO premultiply / NO
// --- FMA), then ONE vfredosum.vs seeded by that sumf-vector = the §1 serial
// --- left-fold. The result extracts back into the SAME sumf lvalue. ---
// CHECK: %[[SUMFCUR:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vfmv_v_f_f32m1"(%[[SUMFCUR]]
// CHECK: %[[SUMIF:.*]] = call_opaque "__riscv_vfcvt_f_x_v_f32m1"(%[[SUMIVEC]]
// CHECK: %[[T0:.*]] = call_opaque "__riscv_vfmul_vv_f32m1"(%[[SUMIF]], %[[DXVEC]]
// CHECK: %[[T1:.*]] = call_opaque "__riscv_vfmul_vv_f32m1"(%[[T0]], %[[DYVEC]]
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vfredosum_vs_f32m1_f32m1"(%[[T1]], %[[SEED]]
// CHECK: %[[OUT:.*]] = call_opaque "__riscv_vfmv_f_s_f32m1_f32"(%[[RED]]
// CHECK: assign %[[OUT]] : !emitc.opaque<"float"> to %[[SUMF]]
// --- the nb % 4 robust single-block SCALAR tail: the SAME §1 SeparatedLeftAssoc
// --- fold (cast/mul/mul/add), NOT a vfredosum -- so batch + tail = the full §1
// --- serial fold. ---
// CHECK: for %{{.*}} = %[[MAIN]] to %{{.*}} step
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: %[[TC:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[TT:.*]] = mul %[[TC]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[TT2:.*]] = mul %[[TT]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[TACC:.*]] = add %{{.*}}, %[[TT2]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: assign %[[TACC]] : !emitc.opaque<"float"> to %[[SUMF]]
// The *s scalar store through the output pointer.
// CHECK: subscript
// CHECK: return

// The per-block default schedule (fold_structure removed): the mbf==4 elided
// scaffold emits four elided cores then four SCALAR SeparatedLeftAssoc folds and
// a scalar tail -- NO vfredosum anywhere (implicit-check-not), NO vslide1down
// packing, NO vlse16 scale gather. The knob toggles the fold schedule; the
// default is the existing per-block emit.
// PERBLOCK: emitc.func @weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
// PERBLOCK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// PERBLOCK: %[[PBC:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// PERBLOCK: %[[PBT:.*]] = mul %[[PBC]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// PERBLOCK: %[[PBT2:.*]] = mul %[[PBT]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// PERBLOCK: add %{{.*}}, %[[PBT2]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// PERBLOCK: return
