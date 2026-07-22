// RUN: not weft-opt %s --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=INCOMPLETE
// RUN: sed 's/lhs_block_stride = 34 : i64, //' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADSTRIDE
// RUN: sed 's/block %block_index :/block %n :/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADBLOCKIDX

// A loop-capable block_fp16_scale_product is a valid mechanism component, but
// brick 1 + brick 3 is not a complete flat kernel. Formula construction must
// not infer a leaf from fold_model or lower this historical scaffold as a
// production fallback. Complete q8_0 bodies are covered by
// rvv-to-emitc-typed-flat-block-dot-loop-full-body.mlir.

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
          %scale = weft_rvv.block_fp16_scale_product %vx, %vy block %block_index : index {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          %acc_next = weft_rvv.cross_block_f32_accumulate %acc, %scale {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// INCOMPLETE: flat block-dot formula cannot classify the typed mechanism body
// BADSTRIDE: requires both lhs_block_stride and rhs_block_stride when block_index is present
// BADBLOCKIDX: requires block_index to be the induction variable (region argument 0) of an enclosing
