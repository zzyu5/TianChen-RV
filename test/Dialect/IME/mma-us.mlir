// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// N2 plugin RAPID-ADD: the SIXTH IME execution op — the OTHER MIXED-SIGN
// int8->int32 `vmadotus` MAC (unsigned A * signed B), the reversed-order sibling
// of tcrv.ime.mma_su that COMPLETES the signedness family {ss=vmadot, uu=vmadotu,
// su=vmadotsu, us=vmadotus}. Same VLEN-derived 4x4x8 envelope as tcrv.ime.mma,
// but a genuinely different emitted instruction (vmadotus=0xe210112b vs vmadotsu's
// 0xe210212b, vmadot's 0xe210312b and vmadotu's 0xe210012b) with the OTHER
// mixed-sign numeric semantics. The fail-closed verifier (I7) admits ONLY the
// validated IME1 mixed-sign int8->int32 `vmadotus` envelope. This op rides the
// SAME mnemonic-generic verifier (verifyIMEMACBoundary) and the SAME spacemit.ime
// capability fact as the other five — the cheapness IS the N2 thesis.
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_mma_us_valid
  tcrv.exec.kernel @ime_mma_us_valid {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available"
    }
    tcrv.exec.variant @ime_vmadotus_mma_slice attributes {
      origin = "ime-plugin",
      requires = [@spacemit_ime]
    } {
    }
    // CHECK: tcrv_ime.mma_us {accum_bits = 32 : i64
    // CHECK-SAME: available_harts = "0-3"
    // CHECK-SAME: elem_in_bits = 8 : i64
    // CHECK-SAME: ime_op = "vmadotus"
    // CHECK-SAME: mac_k = 8 : i64
    // CHECK-SAME: mac_m = 4 : i64
    // CHECK-SAME: mac_n = 4 : i64
    // CHECK-SAME: origin = "ime-plugin"
    // CHECK-SAME: required_capabilities = [@spacemit_ime]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @ime_vmadotus_mma_slice
    // CHECK-SAME: source_kernel = "ime_mma_us_valid"
    // CHECK-SAME: status = "role-op-boundary"
    tcrv_ime.mma_us {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadotus_mma_slice,
      source_kernel = "ime_mma_us_valid",
      ime_op = "vmadotus",
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

// Fail-closed (I7): tcrv.ime.mma_us is the REVERSED-ORDER MIXED-SIGN surface, so
// the `vmadotsu` mnemonic (the OTHER mixed-sign sibling, signed A * unsigned B) is
// rejected — the us op only models `vmadotus`. This is the rotated negative
// control: the ops are distinguished by the admitted instruction, not by a
// renamed op.
module {
  tcrv.exec.kernel @ime_mma_us_wrong_op {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotus_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{ime_op must be 'vmadotus'; this op only models the validated IME1 int8->int32 MAC instruction of its signedness}}
    tcrv_ime.mma_us {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotus_mma_slice, source_kernel = "ime_mma_us_wrong_op", ime_op = "vmadotsu", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the pure-signed `vmadot` mnemonic is rejected — only the
// validated `vmadotus` (unsigned A * signed B) envelope is admitted.
module {
  tcrv.exec.kernel @ime_mma_us_wrong_signed {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotus_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{ime_op must be 'vmadotus'; this op only models the validated IME1 int8->int32 MAC instruction of its signedness}}
    tcrv_ime.mma_us {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotus_mma_slice, source_kernel = "ime_mma_us_wrong_signed", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the IME1 vmadotus consumes int8 inputs; an out-of-envelope
// element width is rejected.
module {
  tcrv.exec.kernel @ime_mma_us_wrong_elem {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotus_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{elem_in_bits must be 8 (IME1 vmadotus consumes int8 inputs)}}
    tcrv_ime.mma_us {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotus_mma_slice, source_kernel = "ime_mma_us_wrong_elem", ime_op = "vmadotus", elem_in_bits = 16 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// The reversed-order mixed-sign op is an IME EXECUTION boundary, not a high-level
// matmul/tile op: generic tensor/tile/benchmark or unknown attributes are rejected.
module {
  tcrv.exec.kernel @ime_mma_us_unknown_attr {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotus_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'layout'}}
    tcrv_ime.mma_us {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotus_mma_slice, source_kernel = "ime_mma_us_unknown_attr", ime_op = "vmadotus", layout = "generic_tile", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}
