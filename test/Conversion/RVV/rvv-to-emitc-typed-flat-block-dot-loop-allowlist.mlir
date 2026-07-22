// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's|// R1 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's|// R2 ||' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=REJECT-NESTED

// M-FLAT loop-scaffold step 4/6 -- the loop-aware ALLOWLIST RECURSIVE validator
// (the strong-form [L-8] gate; core-invariants I5/I7). It recursively walks the
// weft_rvv.typed_flat_block_dot_loop_body region tree and asserts every op is a
// member of the forward-compatible typed-primitive allowlist (bricks 1/2/3, the
// vector integer-core primitives, the step-2 scalar-lane-extract bridge, the
// setvl/with_vl/load/store structural ops, and the loop op/yield). It FAIL-CLOSES
// on any op outside the allowlist -- in particular an opaque emitc helper leaking
// into the region at ANY nesting depth -- which is the machine-checkable
// provenance basis a future q8_0 constructed(strong) flip depends on.
//
// This is a validator (checking logic), NOT an emitter: byte-exactness does not
// apply. It is the DUAL of the single-block rejectMixedPreRealizedContractionBody
// BLOCKLIST (which does NOT recurse and would wrongly reject the brick BODY if the
// bricks were fed into it -- the anti-write trap); this new walk is additive and
// never touches rejectMixed, so the 3 single-block strong paths are unchanged.
//
// The validator runs at the pass boundary BEFORE the applyPartialConversion
// driver (the loop op flows through THIS lowering pass, never the pre-realized
// realization owner), so the fail-closed diagnostic reaches stderr reliably.
//
// - The base module (no sed) is the POSITIVE case: a legal loop body (brick 3 +
//   the typed yield) passes the validator and lowers.
// - `sed 's|// R1 ||'` injects a TOP-LEVEL opaque emitc.verbatim into the loop
//   region -> fail-closed reject (REJECT).
// - `sed 's|// R2 ||'` injects an opaque emitc.verbatim NESTED inside an
//   allowlisted with_vl inside the loop region -> fail-closed reject that ONLY
//   the RECURSIVE walk can catch (the non-recursive single-block blocklist could
//   not) (REJECT-NESTED).

module {
  weft.exec.kernel @rvv_flat_loop_allowlist_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_flat_loop_allowlist attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %term = weft_rvv.runtime_abi_value {c_name = "stub_term", c_type = "float", ownership = "target-export-abi-owned", purpose = "loop-body:stub-term", role = "lower-bound-scalar-value"} : f32
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_flat_loop_allowlist, sew = 8 : i64, source_kernel = "rvv_flat_loop_allowlist_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", multi_block_factor = 1 : i64, strip_elision = "elided", fold_structure = "per-block", numerics_tier = "strict"} {
        ^bb0(%block_index: index, %acc: f32):
          // R1 %r1 = arith.constant 0.0 : f32
          // R2 %r2vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
          // R2 weft_rvv.with_vl %r2vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
          // R2   %r2 = weft_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %r2vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, integer_core_lmul = "mf4", multi_block_factor = 1 : i64, strip_elision = "robust"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // R2 } : !weft_rvv.vl
          %acc_next = weft_rvv.cross_block_f32_accumulate %acc, %term {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// The legal body (brick 3 + typed yield) passes the allowlist validator and
// lowers to the byte-exact loop skeleton (the emitc.func exists).
// CHECK-NOT: weft_rvv.
// CHECK: emitc.func @weft_emitc_rvv_flat_loop_allowlist_kernel_rvv_flat_loop_allowlist(
// CHECK: return

// A TOP-LEVEL non-allowlist op (any op that is not a typed pattern-library
// primitive) in the loop region is fail-closed rejected -- the allowlist default
// is deny (naming the offending op).
// REJECT: 'arith.constant' op is not in the M-FLAT typed flat block-dot loop-body allowlist

// The hand-written monolithic *_block_dot helper (whose own verifier requires it
// to live directly inside a with_vl body -- exactly the constructed-strong body's
// vector-core scope) NESTED inside an allowlisted with_vl inside the loop region
// is fail-closed rejected ONLY because the validator walk is RECURSIVE (the
// non-recursive single-block blocklist could not see one level deeper). This is
// the literal [L-8] opaque-helper leak the strong-form gate exists to catch.
// REJECT-NESTED: 'weft_rvv.q4_0_q8_0_block_dot' op is not in the M-FLAT typed flat block-dot loop-body allowlist
