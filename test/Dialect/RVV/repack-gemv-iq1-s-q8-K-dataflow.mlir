// RUN: weft-opt %s -split-input-file -verify-diagnostics

// C4a-2 iq1_s TERNARY-DELTA grid repack GEVM dataflow / fail-closed pins.
//
// C4a-2 widened TWO closed gates to land iq1_s:
//   (a) the weft::GridDecodePlan registry gained a 4th row ("iq1_s");
//   (b) the loop body's activation_bsums_byte_offset gate, previously q4_K-min-fold
//       ONLY, now also admits the iq1_s "grid_ternary_delta_eighth" delta fold.
// Widening a closed set is where [D-1] "unknown = reject" usually dies quietly, so
// these cases pin that it did NOT: each gate still rejects everything it rejected
// before, and admits the ONE new named thing and nothing else.

module {
  weft.exec.kernel @repack_gemv_iq1_s_grid_accepts_delta_fold_with_bsums {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_iq1_s_grid_accepts_delta_fold_with_bsums", status = "selected-lowering-boundary"} {
        // The POSITIVE control: the iq1_s delta fold DOES read the activation bsums, so
        // the bsums attr is legal here. Without this case the negative cases below could
        // pass for the trivial reason that the attr is rejected everywhere.
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-ternary-delta-singlescale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1312 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 288 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_ternary_delta_eighth"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // weight_sign_byte_offset 160 is the +-1 DELTA strip: a TernaryDelta row has no
          // sign plane, so the SLOT addresses the delta strip.
          %sumi:2 = weft_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq1_s", weight_quant_byte_offset = 288 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @repack_gemv_grid_still_rejects_unknown_ternary_decode_model {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_grid_still_rejects_unknown_ternary_decode_model", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-ternary-delta-singlescale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1312 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 288 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_ternary_delta_eighth"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // [D-1]: adding a ROW does not open the registry. This probe has now been walked
          // forward THREE times, each time by the line that registered the model it named
          // -- which is itself the finding: a probe that names a real-but-unlanded format
          // has a SHELF LIFE, and every landing spends it.
          //   * C4a-2 probed with "iq1_m" (then the most plausible "surely it works too"
          //     guess, since iq1_m rides the SAME 2048-entry iq1s_grid). C4a-3 REGISTERED
          //     iq1_m, so that example stopped testing anything.
          //   * The probe moved to "iq3_xxs". C4a-4 REGISTERED iq3_xxs the same four ways
          //     (row + front door + BOTH leaves + oracle), so it too retired.
          //   * The probe moved to "iq3_s". C4a-5 has now REGISTERED iq3_s the same four
          //     ways -- and iq3_s was the LAST ggml grid sibling.
          // So this probe can no longer name a real format, and it does not pretend to:
          // "iq3_zz" is SYNTHETIC and will never exist. The mechanism under test is
          // unchanged and is the whole point: lookupGridDecodePlan() returns nullptr for
          // any string not in the closed registry, and every caller treats nullptr as
          // REJECT. Registering a 7th row did not open the gate.
          //
          // WHAT THIS PROBE CANNOT TEST -- stated so its green is not read as more than it
          // is. Now that the grid family is complete, no real format sits unregistered, so
          // this case can no longer show the registry refusing something a reader might
          // PLAUSIBLY expect to work; a synthetic string is the EASY case, and this probe
          // got strictly weaker the moment iq3_s landed. It does NOT test that a
          // registered-but-unlowerable row is refused (the leaves' own entry-width and
          // store-scale guards do that), nor that a registered row DECODES correctly (the
          // oracles do), nor anything about the front door, which never sees this string.
          //
          // Note also what this comment does NOT say: nothing about what any future grid
          // format "would need". Three times this family published a confident causal claim
          // about an unlanded row -- "iq3_xxs needs GridEntryWidth::I32x4", "iq3_xxs needs
          // GridSignPlane::Ksigns128", and (right here, in the text this replaces) "iq3_s
          // needs DUAL ls". All three were wrong; all three were in a COMMENT, never in an
          // enum, which is where the rule was looking. iq3_s is in fact SINGLE ls: ggml
          // names two scales per loop iteration but steps ib32 += 2 and spends one on each
          // 32-element sub-block. The prediction came from reading ggml's variable NAMES
          // instead of its loop STRIDE. This probe now asserts a mechanism, not a forecast.
          // expected-error @+1 {{only accepts a decode_model registered in the closed grid decode plan registry}}
          %sumi:2 = weft_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq3_zz", weight_quant_byte_offset = 288 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @repack_gemv_grid_rejects_bsums_outside_the_delta_fold {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_grid_rejects_bsums_outside_the_delta_fold", status = "selected-lowering-boundary"} {
        // [D-1]: C4a-2 widened the bsums gate by exactly ONE named fold. The iq2_xxs grid
        // fold does NOT read bsums, so the attr is STILL rejected here -- the gate was
        // widened, not opened. This is the case that would go quietly green if the gate
        // had been relaxed to "any grid fold".
        // expected-error @+1 {{does not accept activation_bsums_byte_offset for this fold_model}}
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_single_scale_eighth"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          %sumi:2 = weft_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq2_xxs", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @repack_gemv_grid_rejects_unknown_ternary_fold_model {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_grid_rejects_unknown_ternary_fold_model", status = "selected-lowering-boundary"} {
        // [D-1]: the fold_model set is closed too -- a near-miss spelling is rejected.
        // expected-error @+1 {{currently supports only fold_model}}
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-ternary-delta-singlescale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1312 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 288 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "grid_ternary_delta"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          %sumi:2 = weft_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq1_s", weight_quant_byte_offset = 288 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 160 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}
