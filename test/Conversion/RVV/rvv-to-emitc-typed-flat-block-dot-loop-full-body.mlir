// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/%acc, %bterm/%acc, %dd/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REGION
// RUN: sed 's/%wv, %av, %vl/%av, %av, %vl/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=CHAIN

// M-FLAT loop-scaffold step 5a + W4 -- the FULL q8_0 typed body with the vector
// integer core wired IN-REGION. The region-carrying
// tcrv_rvv.typed_flat_block_dot_loop_body now holds the COMPLETE per-block
// chain: brick 1 (the per-block `d_x*d_y` fp16 scale product over the
// block_index induction variable) -> the vector integer core (two per-block i8
// loads -> tcrv_rvv.widening_product signed i8m2xi8m2->i16m4 ->
// tcrv_rvv.standalone_reduce ->tcrv_rvv.typed_vector_lane0_to_scalar_extract
// into the scalar i32 sumi) -> brick 2 (tcrv_rvv.block_computed_scale_dequant,
// `(float)sumi * scale` on THAT sumi) -> brick 3
// (tcrv_rvv.cross_block_f32_accumulate, `sumf + term`) -> the typed yield. The
// lowering recognizes this chain REGION-DRIVEN (the presence of brick 2 marks a
// full body), GATES the emit on every integer-core link (W4), then drives the
// FULL-body emit OP-BY-OP from the region ops' OPERANDS (W4): the per-block base
// address is built from the LOAD op's block_stride/quant_byte_offset, the
// widening product reads its lhs/rhs load results, the reduce reads its product,
// the extract reads its reduce, brick 1's two fp16 reads share the load's
// per-block base (memoized on the (buffer, block_index) SSA pair), and brick
// 1/2/3 lower to the PINNED SeparatedLeftAssoc fold. The whole emitted body
// (loop skeleton + the plain i8xi8 widening-product/reduce integer core + the
// pinned fp32 fold `t=(float)sumi*d_x; t=t*d_y; sumf=sumf+t`) is SOURCED from the
// region ops, not re-derived from the loop-body attrs. The fold conforms to
// [testing/flat-block-dot-fp-fold-oracle.md §1] (separate statements, no
// premultiply, no FMA) and is therefore INTENTIONALLY no longer byte-exact vs
// the monolith's fused `sumf + (float)sumi*(d_x*d_y)` (the sanctioned gate
// migration, q8_0 only; the monolith retires later).
//
// This is an emit-consistency ("CORE == emission-plans") lit that locks the
// WHOLE body (not just the step-1 skeleton): the pinned fold is emitted as
// SEPARATE cast/mul/mul/add emitc statements (NOT one emitc.expression) so clang
// cannot contract the (t*d_y)+sumf into fmaf, AND the sumf LOAD lands AFTER the
// block integer core (not at the loop top). The gated integer-core ops are
// lowered OP-BY-OP from their operands: each load's vle8 address arithmetic is
// built from the load op's block_stride/quant_byte_offset (see the operand-flow
// CHECKs below pinning stride 34 into the base mul and quant offset 2 into the
// load add), the vwmul/vwredsum/vmv_x_s consume their predecessor op results
// through the valueMap, and the fold tail (brick 1's d_x/d_y + brick 2 + brick 3)
// lowers to the separated cast/mul/mul/add via the reused emitFlatFold (fed the
// operand-derived d_x/d_y + sumi). Numerical
// bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here. Nothing is
// wired to a q8_0 dispatch and no format state is flipped (that is step 5b/6);
// the provenance verbatim comments deliberately carry THIS op's identity.

module {
  tcrv.exec.kernel @rvv_flat_full_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_flat_full_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      // The per-block reduce seed (0) -- standalone_reduce takes a runtime ABI
      // scalar seed. Each block reduces fresh; cross-block fold is brick 3 (f32).
      %zero_seed = tcrv_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_flat_full_body, sew = 8 : i64, source_kernel = "rvv_flat_full_body_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "elided"} {
        ^bb0(%block_index: index, %acc: f32):
          // brick 1: the per-block d_x*d_y fp16 scale over the block_index.
          // (SSA named %dd -- d_x*d_y -- to avoid the lit %s/%t RUN-line rewrite.)
          %dd = tcrv_rvv.block_fp16_scale_product %vx, %vy block %block_index : index {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value -> f32
          // W4 integer core: two per-block i8 loads (base + ib*stride + quant_off).
          %wv = tcrv_rvv.load %vx, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
          %av = tcrv_rvv.load %vy, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
          // vector dot: signed widening product (i8m2 x i8m2 -> i16m4).
          %prod = tcrv_rvv.widening_product %wv, %av, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
          // reduce i16m4 -> i32m1 lane0, then extract lane0 -> scalar i32 sumi.
          %red = tcrv_rvv.standalone_reduce %prod, %zero_seed, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %sumi = tcrv_rvv.typed_vector_lane0_to_scalar_extract %red, %vl {kind = "vector_lane0_to_scalar_i32_extract", extract_relation = "i32m1-lane0-to-scalar-i32"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> i32
          // brick 2: (float)sumi * scale (the per-block dequant term %bterm).
          %bterm = tcrv_rvv.block_computed_scale_dequant %sumi, %dd {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, f32 -> f32
          // brick 3: sumf + term (cross-block fp32 fold, strict ascending).
          %acc_next = tcrv_rvv.cross_block_f32_accumulate %acc, %bterm {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_flat_full_body_kernel_rvv_flat_full_body(
// The loop skeleton: the mutable `float sumf = 0.0f;` accumulator + nb = n/32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: literal "0.0f" : !emitc.opaque<"float">
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK: %[[NB:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: for %[[IB:.*]] = %{{.*}} to %[[NB]] step %{{.*}}  : !emitc.opaque<"size_t"> {

// Per-block address arithmetic -- OPERAND-FLOW positive: the block base mul is
// built from the LOAD op's block_stride (34), NOT re-derived from a descriptor.
// The base pointer + ib*34 is byte-exact to the q8_0 monolith's blockBaseValue.
// CHECK: %[[STRIDEX:.*]] = literal "34" : !emitc.opaque<"size_t">
// CHECK: mul %[[IB]], %[[STRIDEX]]
// CHECK: %[[XB:.*]] = add %{{.*}}, %{{.*}} : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">)
// CHECK: %[[STRIDEY:.*]] = literal "34" : !emitc.opaque<"size_t">
// CHECK: mul %[[IB]], %[[STRIDEY]]
// CHECK: %[[YB:.*]] = add %{{.*}}, %{{.*}} : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">)
// The two per-block fp16->fp32 scale reads (d_x then d_y).
// CHECK: %[[DX:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[XB]])
// CHECK: %[[DY:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[YB]])

// The plain i8m2 x i8m2 widening-product / reduce integer core (NO nibble
// decode) re-emitted byte-exact by the SHARED emitFlatBlockCore.
// CHECK: %[[SUMI:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: assign %{{.*}} : !emitc.opaque<"int32_t"> to %[[SUMI]]
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// OPERAND-FLOW positive: the weight load address adds the LOAD op's
// quant_byte_offset (2) to block_base_x, then the i8* cast + vle8 -- the load's
// quant offset attr drives the address, not a descriptor.
// CHECK: %[[QOFF:.*]] = literal "2" : !emitc.opaque<"size_t">
// CHECK: add %[[XB]], %[[QOFF]]
// CHECK: %[[VX:.*]] = call_opaque "__riscv_vle8_v_i8m2"
// CHECK: %[[VY:.*]] = call_opaque "__riscv_vle8_v_i8m2"
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16m4"(%[[VX]], %[[VY]]
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8
// The plain-i8 whole-block core reduces ONE vwmul product (no two-nibble-half
// vwmacc): the q8_0 integer-core assertion migrated from the monolith emit fixture.
// CHECK-NOT: call_opaque "__riscv_vwmacc_vv_i16
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"(%[[PROD]]
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vmv_x_s_i32m1_i32"
// CHECK: assign %[[RED]] : !emitc.opaque<"int32_t"> to %[[SUMI]]

// The PINNED SeparatedLeftAssoc fp32 fold [testing/flat-block-dot-fp-fold-oracle.md
// §1]: sumi loaded, sumf loaded AFTER the core, then SEPARATE cast/mul/mul/add
// emitc statements (NOT a fused emitc.expression) so clang cannot contract
// (t*d_y)+sumf into fmaf. The tree is ((sumi*d_x)*d_y) -- NO d_x*d_y premultiply
// (the first mul takes the sumi cast and d_x, not d_x*d_y). The assign takes the
// ADD result DIRECTLY -- a fused expression would assign the expression result,
// so this locks the separation (and the q8_0 typed emit is INTENTIONALLY no
// longer byte-exact vs the monolith's `sumi*(d_x*d_y)`).
// CHECK: %[[SI:.*]] = load %[[SUMI]] : <!emitc.opaque<"int32_t">>
// CHECK: %[[SF:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK: %[[CF:.*]] = cast %[[SI]] : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[T:.*]] = mul %[[CF]], %[[DX]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[T2:.*]] = mul %[[T]], %[[DY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[ACC:.*]] = add %[[SF]], %[[T2]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: assign %[[ACC]] : !emitc.opaque<"float"> to %[[SUMF]] : <!emitc.opaque<"float">>

// *s = sumf; the structured scalar store, then return.
// CHECK: subscript
// CHECK: return

// REGION-DRIVEN (anti-cosmetic) negative: rewiring brick 3 to fold on brick 1's
// per-block SCALE (`%acc, %term` -> `%acc, %scale`) -- skipping brick 2 -- breaks
// the region SSA chain the gate checks (brick 3 term must be brick 2's dequant
// term). With the loop-body attrs UNCHANGED, the emit fails closed: the emission
// is driven by the region op wiring, not re-derived from the attrs alone.
// REGION: failed to legalize operation 'tcrv.exec.variant'

// CHAIN-DRIVEN (W4) negative: misdirecting the vector integer core's widening
// product operands (`%wv, %av` -> `%av, %av`, dropping the weight-load result
// from the dot) parses+verifies fine but breaks the integer-core link the W4
// gate checks (the widening product operands must be the two per-block load
// results). With the loop-body attrs UNCHANGED, the emit fails closed -- proving
// the emit tracks the region integer-core chain (pre-W4 the chain was ignored
// and this misdirect still emitted the byte-exact core).
// CHAIN: failed to legalize operation 'tcrv.exec.variant'
