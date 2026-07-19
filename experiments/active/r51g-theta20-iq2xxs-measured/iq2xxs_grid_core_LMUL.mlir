// iq2_xxs super-block SCALAR-accumulator GRID-of-8 decode vec_dot, EXPLICIT
// integer_core_lmul = "__LMUL__" (m2 = the VLEN128 anchor / emitter default; m1 = the
// VLEN256 anchor, requires minimum_vlen=256). Lowered to CORE EmitC via
// --weft-rvv-lower-to-emitc, then to C via mlir-translate --mlir-to-cpp. The two emits
// differ in the vint8<core>_t / vint16<wide>_t widths (m2 core -> m4 wide gather; m1 core
// -> m2 wide gather). BOARD FINDING (refuting the a-priori "same arithmetic, different
// width" guess): the pair-batched `vget_v_i8<2core>_i8<core>` register-group geometry ties
// each anchor's i8-strip VLMAX to the 32-element sub-block, so the anchors are VLEN-PINNED
// -- m2 is byte-exact ONLY at VLEN128, m1 ONLY at VLEN256 (each silent-wrong at the other
// VLEN). This is a VLEN-CORRECTNESS selector, NOT a same-VLEN performance gearbox.
module {
  weft.exec.kernel @iq2_xxs_grid_core_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @iq2_xxs_grid_core attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @iq2_xxs_grid_core, sew = 32 : i64, source_kernel = "iq2_xxs_grid_core_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          %bsum = weft_rvv.iq2_xxs_q8_k_grid_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_iq2_xxs_q8_k_grid_core", scale_model = "per-sub-block-aux1-scale-grid-of-8-codebook-signs64-sign-plane-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_qs_byte_offset = 2 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, num_groups = 4 : i64, integer_core_lmul = "__LMUL__"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
          weft_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}
