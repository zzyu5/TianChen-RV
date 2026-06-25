// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// N2 plugin RAPID-ADD: the FOURTH IME execution op — the MIXED-SIGN int8->int32
// `vmadotsu` MAC (signed A * unsigned B, the canonical quantized case: signed
// activations * unsigned weights). Same VLEN-derived 4x4x8 envelope as
// tcrv.ime.mma, but a genuinely different emitted instruction (vmadotsu=0xe210232b
// vs vmadot's 0xe210312b and vmadotu's 0xe210012b) with mixed-sign numeric
// semantics. The fail-closed verifier (I7) admits ONLY the validated IME1
// mixed-sign int8->int32 `vmadotsu` envelope. This op rides the SAME
// mnemonic-generic verifier (verifyIMEMACBoundary) and the SAME spacemit.ime
// capability fact as the other three — the cheapness IS the N2 thesis.
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_mma_su_valid
  tcrv.exec.kernel @ime_mma_su_valid {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available"
    }
    tcrv.exec.variant @ime_vmadotsu_mma_slice attributes {
      origin = "ime-plugin",
      requires = [@spacemit_ime]
    } {
    }
    // CHECK: tcrv_ime.mma_su {accum_bits = 32 : i64
    // CHECK-SAME: available_harts = "0-3"
    // CHECK-SAME: elem_in_bits = 8 : i64
    // CHECK-SAME: ime_op = "vmadotsu"
    // CHECK-SAME: mac_k = 8 : i64
    // CHECK-SAME: mac_m = 4 : i64
    // CHECK-SAME: mac_n = 4 : i64
    // CHECK-SAME: origin = "ime-plugin"
    // CHECK-SAME: required_capabilities = [@spacemit_ime]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @ime_vmadotsu_mma_slice
    // CHECK-SAME: source_kernel = "ime_mma_su_valid"
    // CHECK-SAME: status = "role-op-boundary"
    tcrv_ime.mma_su {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadotsu_mma_slice,
      source_kernel = "ime_mma_su_valid",
      ime_op = "vmadotsu",
      elem_in_bits = 8 : i64,
      accum_bits = 32 : i64,
      mac_m = 4 : i64,
      mac_n = 4 : i64,
      mac_k = 8 : i64,
      available_harts = "0-3"
    }
  }
}

// -----

// Fail-closed (I7): tcrv.ime.mma_su is the MIXED-SIGN surface, so the signed
// `vmadot` mnemonic is rejected — the mixed-sign op only models `vmadotsu`. This
// is the load-bearing axis: the ops are distinguished by the admitted
// instruction, not by a renamed op.
module {
  tcrv.exec.kernel @ime_mma_su_wrong_op {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotsu_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{ime_op must be 'vmadotsu'; this op only models the validated IME1 int8->int32 MAC instruction of its signedness}}
    tcrv_ime.mma_su {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotsu_mma_slice, source_kernel = "ime_mma_su_wrong_op", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the reversed-order mixed-sign sibling `vmadotus` (unsigned A
// * signed B) is NOT the modeled mixed-sign form and is rejected — only the
// validated `vmadotsu` (signed A * unsigned B) envelope is admitted.
module {
  tcrv.exec.kernel @ime_mma_su_wrong_order {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotsu_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{ime_op must be 'vmadotsu'; this op only models the validated IME1 int8->int32 MAC instruction of its signedness}}
    tcrv_ime.mma_su {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotsu_mma_slice, source_kernel = "ime_mma_su_wrong_order", ime_op = "vmadotus", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the IME1 vmadotsu consumes int8 inputs; an out-of-envelope
// element width is rejected.
module {
  tcrv.exec.kernel @ime_mma_su_wrong_elem {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotsu_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{elem_in_bits must be 8 (IME1 vmadotsu consumes int8 inputs)}}
    tcrv_ime.mma_su {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotsu_mma_slice, source_kernel = "ime_mma_su_wrong_elem", ime_op = "vmadotsu", elem_in_bits = 16 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// The mixed-sign op is an IME EXECUTION boundary, not a high-level matmul/tile op:
// generic tensor/tile/benchmark or unknown attributes are rejected.
module {
  tcrv.exec.kernel @ime_mma_su_unknown_attr {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotsu_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'layout'}}
    tcrv_ime.mma_su {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotsu_mma_slice, source_kernel = "ime_mma_su_unknown_attr", ime_op = "vmadotsu", layout = "generic_tile", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}
