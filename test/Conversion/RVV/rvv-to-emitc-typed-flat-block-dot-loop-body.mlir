// RUN: not weft-opt %s --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=INCOMPLETE
// RUN: sed 's/kind = "typed_flat_block_dot_loop_body"/kind = "plain_block_dot_loop"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "sumi_times_scales"/fold_model = "unsupported_fold"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/integer_core_lmul = "m2"/integer_core_lmul = "m8"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADLMUL

// A loop/yield shell with a runtime stub term is not a constructed kernel.
// Formula construction rejects it before emission instead of selecting q8_0
// from kind or fold_model. Full mechanism-body loop and accumulator behavior is
// covered by the full-body and source-front-door tests.

module {
  weft.exec.kernel @rvv_typed_flat_block_dot_loop_body_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_typed_flat_block_dot_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %term = weft_rvv.runtime_abi_value {c_name = "stub_term", c_type = "float", ownership = "target-export-abi-owned", purpose = "loop-body:stub-term", role = "lower-bound-scalar-value"} : f32
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", multi_block_factor = 1 : i64, strip_elision = "elided", fold_structure = "per-block", numerics_tier = "strict"} {
        ^bb0(%block_index: index, %acc: f32):
          %acc_next = weft_rvv.cross_block_f32_accumulate %acc, %term {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// INCOMPLETE: flat block-dot formula cannot classify the typed mechanism body
// BADKIND: currently supports only kind "typed_flat_block_dot_loop_body"
// BADFOLD: currently supports only fold_model "sumi_times_scales"
// BADLMUL: only accepts integer_core_lmul "m1", "m2", or "mf4"
