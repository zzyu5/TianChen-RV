// W2 / [GAP-P1] / §3.4 JUDGMENT EXPERIMENT (A-line, anti-fake-refactor guard) -- the
// repack FAMILY/WIDTH selector (the strip width half_lanes) is now an EXPLICIT NAMED
// CLOSED FORM f(minimumVLEN, weightInterleave) in the gearbox header
// (getRVVRepackStripHalfLanes @ RVVGearboxSchedule.h), NOT a duplicated local
// deriveRepackHalfLanes helper. The strip-width materializer pass
// (MaterializeRVVRepackStripWidth) AND the repack front door (RVVLowerQuantContraction:
// deriveRepackHalfLanes) both CALL the one closed form; this lit proves it is
// LOAD-BEARING -- the emitted half_lanes tracks the width-closed-form output.
//
// This is the [GAP-P1] strip-width axis -- NOT the reduction integer-core LMUL selector
// (that is MIG-5's getRVVEffectiveWidthInvariantLMUL, judged separately in
// rvv-reduction-integer-core-width-invariant-closedform-judgment.mlir; DO NOT conflate).
// The two are DUALS of theta = f(VLEN): the reduction core holds the register-group
// width CONSTANT by FLIPPING LMUL (m2@128 <-> m1@256); the repack core holds the LMUL
// constant (mf2 default) and GROWS the e16m1 strip WIDTH with the capability VLEN.
//
// PINNED VERDICT-FLIP (re-judgment) RECORD -- the width follows the TYPED CAPABILITY
// fact (the provider op's
// minimum_vlen), read off the in-IR capability object, NEVER a -march re-parse and
// NEVER a cost-model static_order. The reason the closed form returns is
// `capability_strip_width` (RVVRepackStripWidthReason::CapabilityStripWidth):
//   * provider minimum_vlen 256  =>  half_lanes 16 (one 16-lane e16m1 strip)
//   * provider minimum_vlen 128  =>  half_lanes  8 (two 8-lane halves)
// half_lanes = min(VLEN/16, weightInterleave). A WIDER guaranteed VLEN affords a WIDER
// strip, so the family/width output FLIPS with the capability -- if a stale duplicated
// helper or a hardcoded literal were still driving it, the flip could not track the
// closed form.
//
// DECISIVE anti-fake-green (the 2x2 capability x -march matrix below): the -march is
// driven CONFLICTING with the provider on purpose. Flip -march and watch theta try to
// follow == FALSE-GREEN: if theta tracked -march it would be the false-green this task
// forbids. It does NOT -- theta tracks the provider capability fact.
//   provider256 + march zvl128b (march alone => 8)  => half_lanes 16  (capability wins)
//   provider128 + march zvl128b (march alone => 8)  => half_lanes  8  (deployed, byte-exact)
//   provider256 + march zvl512b (march alone => 16) => half_lanes 16  (march flipped, theta same)
//   provider128 + march zvl512b (march alone => 16) => half_lanes  8  (theta IGNORES march)
// The two provider128 cells prove BYTE-EXACT deployment (provider==march==128 => the
// authored 8 is unchanged) AND march-inertness (march zvl512b would give 16 but theta
// stays 8 because it reads the provider128 fact). CORE==PROD: the deployed board
// (provider==march) emits the same half_lanes as before this closed-form refactor.

// RUN 1: -march zvl128b threaded straight at the consumer. The provider fact WINS.
// RUN: weft-opt %s --split-input-file \
// RUN:   --weft-rvv-materialize-repack-strip-width=march=rv64gcv_zvl128b \
// RUN: | FileCheck %s --check-prefix=M128

// RUN 2: a DIFFERENT conflicting -march (zvl512b => march alone would give 16). The
// theta must NOT track it -- the provider fact is authoritative, so provider128 stays
// 8 (NOT the 16 a -march re-parse would produce). march-inertness == bypass dead.
// RUN: weft-opt %s --split-input-file \
// RUN:   --weft-rvv-materialize-repack-strip-width=march=rv64gcv_zvl512b \
// RUN: | FileCheck %s --check-prefix=M512

// ============================ SPLIT 1: provider VLEN 256 ====================
module {
  weft.exec.kernel @strip_provider_vlen256 {
    // Provider carries the TYPED minimum_vlen fact (256), independent of -march.
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 256 : i64}
    weft.exec.variant @rvv_body attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_body, sew = 32 : i64, source_kernel = "strip_provider_vlen256", status = "selected-lowering-boundary"} {
        %g = weft_rvv.repack_gemv_q8_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q8_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 544 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The strip width follows the PROVIDER minimum_vlen=256 (half_lanes 16), NOT the
// conflicting -march (128 => would be 8, 512 => would be 16). Both -march runs => 16.
// M128: weft.exec.kernel @strip_provider_vlen256
// M128: weft_rvv.repack_gemv_q8_0_q8_0
// M128-SAME: half_lanes = 16 : i64
//
// M512: weft.exec.kernel @strip_provider_vlen256
// M512: weft_rvv.repack_gemv_q8_0_q8_0
// M512-SAME: half_lanes = 16 : i64

// The split separator below starts an INDEPENDENT module with its OWN provider fact,
// so --split-input-file re-runs the pass per split (each reads its own capability
// provider, not a module-wide first-provider).

// -----

// ============================ SPLIT 2: provider VLEN 128 ====================
module {
  weft.exec.kernel @strip_provider_vlen128 {
    // Provider carries minimum_vlen 128: the DEPLOYED board fact.
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 128 : i64}
    weft.exec.variant @rvv_body attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_body, sew = 32 : i64, source_kernel = "strip_provider_vlen128", status = "selected-lowering-boundary"} {
        %g = weft_rvv.repack_gemv_q8_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q8_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 544 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The strip width follows the PROVIDER minimum_vlen=128 (half_lanes 8). Under -march
// zvl512b (which ALONE would give 16) it STAYS 8 -- theta reads the provider fact,
// NOT the -march re-parse. This is the deployed byte-exact cell (provider==march==128
// => the authored 8 is unchanged, CORE==PROD) AND the march-inertness witness.
// M128: weft.exec.kernel @strip_provider_vlen128
// M128: weft_rvv.repack_gemv_q8_0_q8_0
// M128-SAME: half_lanes = 8 : i64
//
// M512: weft.exec.kernel @strip_provider_vlen128
// M512: weft_rvv.repack_gemv_q8_0_q8_0
// M512-SAME: half_lanes = 8 : i64
