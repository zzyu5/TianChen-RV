// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/lhs_block_stride = 34 : i64, //' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADSTRIDE
// RUN: sed 's/block %block_index :/block %n :/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADBLOCKIDX

// M-FLAT loop-scaffold step 3/6 -- the LOOP-CAPABLE per-block-source form of
// brick 1 (weft_rvv.block_fp16_scale_product). This closes the root cause the
// two prior single-block STOPs kept naming: the brick 1 verifier hard-required
// its scale bases to be IMPORTED ABI block-0 pointers, so it could not read the
// per-block `base + ib*stride` header inside the flat block loop. Step 3
// additively extends brick 1 with an OPTIONAL block_index operand (the enclosing
// weft_rvv.typed_flat_block_dot_loop_body region's induction variable) plus
// lhs/rhs_block_stride (I64) attrs, so each per-block d_x/d_y header is now a
// typed `base + block_index*stride (+ byte_offset)` expression.
//
// This is an emit-consistency ("CORE == emission-plans") lit that locks the
// per-block address + fp16 read BYTE-EXACT to the monolithic emitFlatBlockDot's
// blockBaseValue (RVVToEmitCBlockQuantLinear.cpp:5451-5462, blockOffset 0 -- a
// size_t emitc.mul of the induction variable by the stride then a pointer
// emitc.add onto the imported base) + fp16ReadAt (:5467-5476 -- the sanctioned
// `(float)*(const _Float16 *)` call_opaque). The imported ABI bases stay the
// loop-invariant block-0 pointers (the base-import contract is NOT relaxed); the
// per-block form only ADDS the loop offset. The extended brick 1 is placed
// inside the step-1 loop op region and its computed scale feeds brick 3's
// cross-block fold. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv),
// not tested here. The single-block (block_index absent) form is unchanged and
// covered by rvv-to-emitc-block-fp16-scale-product.mlir (backward compat).

module {
  weft.exec.kernel @rvv_flat_per_block_source_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_flat_per_block_source attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_flat_per_block_source, sew = 8 : i64, source_kernel = "rvv_flat_per_block_source_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", multi_block_factor = 1 : i64, strip_elision = "elided", fold_structure = "per-block", numerics_tier = "strict"} {
        ^bb0(%block_index: index, %acc: f32):
          // brick 1, loop-capable per-block-source form: read d_x/d_y at
          // `base + block_index*stride` (AoS stride 34 = q8_0 block size).
          %scale = weft_rvv.block_fp16_scale_product %vx, %vy block %block_index : index {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          // brick 3 folds the block-carried acc with the per-block scale term.
          %acc_next = weft_rvv.cross_block_f32_accumulate %acc, %scale {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_flat_per_block_source_kernel_rvv_flat_per_block_source(

// The loop scaffold (step 1): the mutable `float sumf = 0.0f;` accumulator.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: literal "0.0f" : !emitc.opaque<"float">
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]] : <!emitc.opaque<"float">>
// size_t nb = n / QK;
// CHECK: %[[NB:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer block loop; %[[IB]] is the block_index induction variable that
// brick 1's per-block address arithmetic multiplies by the stride.
// CHECK: for %[[IB:.*]] = %{{.*}} to %[[NB]] step %{{.*}}  : !emitc.opaque<"size_t"> {
// The carried-IN acc -> a LOAD of the sumf lvalue at the TOP.
// CHECK: %[[ACCIN:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>

// brick 1 per-block LHS base: xb = vx + block_index*stride  -- byte-exact to
// blockBaseValue (a size_t emitc.mul of the induction variable then a pointer
// emitc.add onto the imported base).
// CHECK: %[[OFFX:.*]] = mul %[[IB]], %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[XB:.*]] = add %{{.*}}, %[[OFFX]] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
// brick 1 per-block RHS base: yb = vy + block_index*stride.
// CHECK: %[[OFFY:.*]] = mul %[[IB]], %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[YB:.*]] = add %{{.*}}, %[[OFFY]] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>

// The two per-block fp16->fp32 reads at the computed block bases -- byte-exact
// to fp16ReadAt: the SAME `(float)*(const _Float16 *)` call_opaque the
// monolithic block-dot emits inline for d_x / d_y.
// CHECK: %[[DX:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[XB]]) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
// CHECK: %[[DY:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[YB]]) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>) -> !emitc.opaque<"float">
// The per-block scale product d_x * d_y (ggml q8_0 order: scales FIRST).
// CHECK: %[[SCALE:.*]] = mul %[[DX]], %[[DY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">

// brick 3 folds acc-in + the per-block scale term (locks the scale as the term).
// CHECK: %[[NEXT:.*]] = add %[[ACCIN]], %[[SCALE]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// The carried-OUT acc_next -> an emitc.assign back into the sumf lvalue at the
// BOTTOM (the SSA-acc <-> emitc-variable mapping).
// CHECK: assign %[[NEXT]] : !emitc.opaque<"float"> to %[[SUMF]] : <!emitc.opaque<"float">>
// *s = sumf; the structured scalar store, then return.
// CHECK: subscript
// CHECK: return

// The bounded per-block surface is fail-closed (I7): block_index present hard-
// requires both AoS block strides, and structurally requires block_index to be
// the enclosing loop op's induction variable (region argument 0).
// BADSTRIDE: requires both lhs_block_stride and rhs_block_stride when block_index is present
// BADBLOCKIDX: requires block_index to be the induction variable (region argument 0) of an enclosing
